"""Warm-start models from a JEPA or IB-JEPA checkpoint (--initialization)."""

import os

import torch
import torch.nn as nn


def resolve_initialization_path(init_path):
    if init_path is None or str(init_path).strip().lower() in ("", "none"):
        return None

    p = str(init_path).replace("\\", os.sep).strip()
    here = os.path.dirname(os.path.abspath(__file__))
    parent = os.path.normpath(os.path.join(here, ".."))
    candidates = [
        os.path.normpath(p),
        os.path.normpath(os.path.join("Baselines", p.lstrip(os.sep))),
        os.path.normpath(os.path.join(parent, p.lstrip(os.sep))),
        os.path.normpath(os.path.join(parent, "Baselines", p.lstrip(os.sep))),
    ]
    seen = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        if os.path.isfile(path):
            return path

    raise FileNotFoundError(
        f"Initialization checkpoint not found: {init_path!r} "
        f"(tried: {', '.join(seen)})"
    )


def _is_jepa_checkpoint(ckpt):
    return "mu_head.weight" not in ckpt["models"]["context"]


def _init_gaussian_heads(mu_head, logvar_head):
    with torch.no_grad():
        if mu_head.weight.ndim == 2 and mu_head.weight.shape[0] == mu_head.weight.shape[1]:
            nn.init.eye_(mu_head.weight)
        else:
            nn.init.xavier_uniform_(mu_head.weight)
        nn.init.zeros_(mu_head.bias)
        nn.init.zeros_(logvar_head.weight)
        nn.init.zeros_(logvar_head.bias)


def _init_logvar_head(logvar_head):
    with torch.no_grad():
        nn.init.zeros_(logvar_head.weight)
        nn.init.zeros_(logvar_head.bias)


def _load_matching_keys(model, src_sd, skip_prefixes=()):
    dst_sd = model.state_dict()
    for key, val in src_sd.items():
        if any(key.startswith(prefix) for prefix in skip_prefixes):
            continue
        if key not in dst_sd:
            continue
        if dst_sd[key].shape != val.shape:
            continue
        dst_sd[key] = val
    model.load_state_dict(dst_sd, strict=False)


def _load_encoder_trunk_from_jepa(jepa_sd, encoder, init_heads=True):
    model_sd = encoder.state_dict()
    for key, val in jepa_sd.items():
        if key not in model_sd:
            continue
        if model_sd[key].shape != val.shape:
            raise ValueError(
                f"Shape mismatch for encoder '{key}': checkpoint {tuple(val.shape)} vs "
                f"model {tuple(model_sd[key].shape)}. Match --embed_dim / --num_patches."
            )
        model_sd[key] = val
    encoder.load_state_dict(model_sd, strict=False)
    if init_heads and hasattr(encoder, "mu_head") and hasattr(encoder, "logvar_head"):
        _init_gaussian_heads(encoder.mu_head, encoder.logvar_head)


def _load_deterministic_encoder_from_probabilistic(src_sd, encoder):
    _load_matching_keys(
        encoder,
        src_sd,
        skip_prefixes=("mu_head", "logvar_head"),
    )


def _load_ib_predictor_from_jepa(jepa_pred_sd, predictor):
    pred_sd = predictor.state_dict()
    for key, val in jepa_pred_sd.items():
        if key.startswith("out_proj"):
            continue
        if key not in pred_sd:
            continue
        if pred_sd[key].shape != val.shape:
            raise ValueError(
                f"Shape mismatch for predictor '{key}': checkpoint {tuple(val.shape)} vs "
                f"model {tuple(pred_sd[key].shape)}."
            )
        pred_sd[key] = val
    predictor.load_state_dict(pred_sd, strict=False)

    if getattr(predictor, "is_critic", False):
        return
    if hasattr(predictor, "fc_mu") and "out_proj.weight" in jepa_pred_sd:
        predictor.fc_mu.weight.data.copy_(jepa_pred_sd["out_proj.weight"])
        predictor.fc_mu.bias.data.copy_(jepa_pred_sd["out_proj.bias"])
        if hasattr(predictor, "fc_logvar"):
            _init_logvar_head(predictor.fc_logvar)


def _load_jepa_predictor_from_ib(ib_pred_sd, predictor):
    pred_sd = predictor.state_dict()
    for key, val in ib_pred_sd.items():
        if key.startswith("fc_"):
            continue
        if key not in pred_sd:
            continue
        if pred_sd[key].shape != val.shape:
            continue
        pred_sd[key] = val
    predictor.load_state_dict(pred_sd, strict=False)
    if hasattr(predictor, "out_proj") and "fc_mu.weight" in ib_pred_sd:
        predictor.out_proj.weight.data.copy_(ib_pred_sd["fc_mu.weight"])
        predictor.out_proj.bias.data.copy_(ib_pred_sd["fc_mu.bias"])


def _load_ib_jepa_models(ckpt, context, target, predictor):
    context.load_state_dict(ckpt["models"]["context"], strict=True)
    target.load_state_dict(ckpt["models"]["target"], strict=True)
    if predictor is not None and "predictor" in ckpt["models"]:
        predictor.load_state_dict(ckpt["models"]["predictor"], strict=True)


def _load_ib_jepa_from_jepa(ckpt, context, target, predictor):
    jepa_models = ckpt["models"]
    _load_encoder_trunk_from_jepa(jepa_models["context"], context)
    _load_encoder_trunk_from_jepa(jepa_models["target"], target)
    if predictor is not None and "predictor" in jepa_models:
        _load_ib_predictor_from_jepa(jepa_models["predictor"], predictor)


def _load_jepa_models(ckpt, context, target, predictor):
    context.load_state_dict(ckpt["models"]["context"], strict=True)
    target.load_state_dict(ckpt["models"]["target"], strict=True)
    if predictor is not None and "predictor" in ckpt["models"]:
        predictor.load_state_dict(ckpt["models"]["predictor"], strict=True)


def _load_jepa_from_ib(ckpt, context, target, predictor):
    _load_deterministic_encoder_from_probabilistic(ckpt["models"]["context"], context)
    _load_deterministic_encoder_from_probabilistic(ckpt["models"]["target"], target)
    if predictor is not None and "predictor" in ckpt["models"]:
        _load_jepa_predictor_from_ib(ckpt["models"]["predictor"], predictor)


def _load_prob_encoders(ckpt, context, target, from_jepa):
    if from_jepa:
        _load_encoder_trunk_from_jepa(ckpt["models"]["context"], context)
        _load_encoder_trunk_from_jepa(ckpt["models"]["target"], target)
    else:
        context.load_state_dict(ckpt["models"]["context"], strict=True)
        target.load_state_dict(ckpt["models"]["target"], strict=True)


def _apply_profile(ckpt, models, profile, from_jepa, kind):
    context = models.get("context")
    target = models.get("target")
    predictor = models.get("predictor")

    if profile == "jepa":
        if from_jepa:
            _load_jepa_models(ckpt, context, target, predictor)
        else:
            _load_jepa_from_ib(ckpt, context, target, predictor)
        return

    if profile == "vjepa":
        # Matched f_θ / f_θ' (trunk + Gaussian heads) + Gaussian predictor.
        if from_jepa:
            _load_encoder_trunk_from_jepa(ckpt["models"]["context"], context, init_heads=True)
            _load_encoder_trunk_from_jepa(ckpt["models"]["target"], target, init_heads=True)
            if predictor is not None and "predictor" in ckpt["models"]:
                _load_ib_predictor_from_jepa(ckpt["models"]["predictor"], predictor)
        else:
            context.load_state_dict(ckpt["models"]["context"], strict=False)
            target.load_state_dict(ckpt["models"]["target"], strict=False)
            if predictor is not None and "predictor" in ckpt["models"]:
                predictor.load_state_dict(ckpt["models"]["predictor"], strict=False)
        return

    if profile == "ib_jepa":
        if from_jepa:
            _load_ib_jepa_from_jepa(ckpt, context, target, predictor)
        else:
            _load_ib_jepa_models(ckpt, context, target, predictor)
        return

    if profile == "mvib":
        _load_prob_encoders(ckpt, context, target, from_jepa)
        return

    if profile == "supervision":
        # Stochastic context + deterministic target + stochastic predictor (+ expander).
        predictor = models.get("predictor")
        expander = models.get("expander")
        if from_jepa:
            _load_encoder_trunk_from_jepa(ckpt["models"]["context"], context, init_heads=True)
            _load_matching_keys(
                target,
                ckpt["models"]["target"],
                skip_prefixes=("mu_head", "logvar_head"),
            )
            if predictor is not None and "predictor" in ckpt["models"]:
                _load_ib_predictor_from_jepa(ckpt["models"]["predictor"], predictor)
        else:
            context.load_state_dict(ckpt["models"]["context"], strict=True)
            _load_matching_keys(
                target,
                ckpt["models"]["target"],
                skip_prefixes=("mu_head", "logvar_head"),
            )
            if predictor is not None and "predictor" in ckpt["models"]:
                predictor.load_state_dict(ckpt["models"]["predictor"], strict=False)
            if expander is not None and "expander" in ckpt["models"]:
                expander.load_state_dict(ckpt["models"]["expander"], strict=False)
        return

    if profile == "vicreg":
        # Single deterministic encoder (+ expander). context and target may be the same module.
        expander = models.get("expander")
        src_key = "context" if "context" in ckpt["models"] else "target"
        src_sd = ckpt["models"][src_key]
        if from_jepa:
            _load_encoder_trunk_from_jepa(src_sd, context, init_heads=False)
        else:
            _load_matching_keys(
                context,
                src_sd,
                skip_prefixes=("mu_head", "logvar_head"),
            )
        if expander is not None and "expander" in ckpt["models"]:
            expander.load_state_dict(ckpt["models"]["expander"], strict=False)
        return

    raise ValueError(f"Unknown initialization profile: {profile!r}")


def maybe_load_initialization(args, models, profile, method_name=None):
    """
    Load weights from --initialization when set.

    Returns True if a checkpoint was loaded.
    """
    init_path = getattr(args, "initialization", None)
    if not init_path:
        return False

    ckpt_path = resolve_initialization_path(init_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    from_jepa = _is_jepa_checkpoint(ckpt)
    kind = "JEPA" if from_jepa else "IB-JEPA"

    _apply_profile(ckpt, models, profile, from_jepa, kind)

    label = method_name or profile
    print(f"{label}: initialized from {kind} checkpoint {ckpt_path}")
    args._loaded_from_initialization = True
    return True
