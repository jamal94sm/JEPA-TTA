#!/usr/bin/env python3
"""JEPA_corruption_all — standalone project. Run from this folder:

    python main.py
"""
import os
import sys
import warnings

import Datasets
import Utils
from Config import get_arguments
import runner

warnings.filterwarnings("ignore", message="enable_nested_tensor is True")

SEPARATOR = "=" * 40

DATASET_LOADERS = {
    "stl10": Datasets.load_stl10,
    "tiny-imagenet": Datasets.load_tiny_imagenet,
}


class _TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(here)
    if here not in sys.path:
        sys.path.insert(0, here)

    args = get_arguments()
    Utils.set_seed(42)

    dataset_key = Datasets.normalize_dataset_name(args.dataset)
    loader = DATASET_LOADERS.get(dataset_key)
    if loader is None:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    dataset = loader(root=os.path.join(Datasets.data_bank_root(args), dataset_key), args=args)

    results_dir = os.path.join(here, "Results")
    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(results_dir, "Logs_JEPA_corruption_all.txt")

    stdout0, stderr0 = sys.stdout, sys.stderr
    logf = open(log_path, "w", encoding="utf-8", buffering=1)
    sys.stdout = _TeeStream(stdout0, logf)
    try:
        print(f"{SEPARATOR} JEPA_corruption_all {SEPARATOR}")
        print("Visible-patch corruptions: blue | red | green | jitter | noise")
        print("Loss: MSE(z_p, z_2)  |  targets: clean EMA encoder")
        print(f"Logging to {log_path}")
        runner.run(dataset, args)
        print(f"{SEPARATOR} Done {SEPARATOR}")
    finally:
        sys.stdout, sys.stderr = stdout0, stderr0
        logf.close()
