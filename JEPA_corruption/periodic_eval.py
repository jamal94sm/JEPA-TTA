"""Shared helpers for periodic downstream evaluation during pretraining."""

import Utils as root_utils


def maybe_eval_epoch(
    epoch,
    context_encoder,
    args,
    eval_history,
    extract_features,
    best_acc=-float("inf"),
    models=None,
    opt=None,
    run_dir=None,
    global_step=None,
):
    """
    Run eval every num_ep_for_eval epochs. Appends (epoch, acc) to eval_history.
    If models/run_dir are provided and acc improves, save best.ckpt (by eval accuracy).

    Returns
    -------
    eval_history, best_acc
    """
    epoch_one_based = epoch + 1
    if not root_utils.should_run_eval(epoch_one_based, args):
        return eval_history, best_acc

    acc = float(root_utils.live_eval(context_encoder, args, extract_features))
    eval_history.append((epoch_one_based, acc))
    label = "linear probe" if args.evaluation.lower() == "linear" else f"k-NN (k={args.K})"
    print(f"Eval @ epoch {epoch_one_based}: {label} acc={acc:.4f}")

    if acc > best_acc:
        best_acc = acc
        if models is not None and run_dir is not None:
            step = 0 if global_step is None else global_step
            root_utils.save_best(run_dir, models, opt, epoch, step, best_acc)
            print(f"New best eval acc={best_acc:.4f} → saved best.ckpt")

    return eval_history, best_acc
