import copy
from pathlib import Path

import yaml

DEFAULTS = {
    "run_name": "experiment",
    "seed": 42,
    "device": "cpu",
    "dataset": {
        "name": "mnist",
        "root": "data",
        "batch_size": 128,
        "num_workers": 0,
    },
    "model": {
        "name": "simple_cnn",
        "widths": [32, 64],
    },
    "training": {
        "epochs": 3,
        "lr": 1e-3,
        "optimizer": "adam",
        "weight_decay": 0.0,
        "log_every": 200,
    },
    "benchmark": {
        "runs": 200,
        "warmup_runs": 20,
        "batch_size": 64,
        "device": "cpu",
    },
    "quantization": {
        "methods": ["fp32", "fp16", "dynamic_int8", "static_int8", "qat_int8"],
        "backend": "auto",
        "qconfig": "per_channel",
        "calib_batches": 32,
        "qat_epochs": 1,
        "qat_lr": 1e-3,
    },
}


def _merge(base, override):
    merged = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _apply_override(config, dotted_key, value):
    keys = dotted_key.split(".")
    node = config
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = value


def load_config(path=None, overrides=None):
    file_config = {}
    if path is not None:
        with open(Path(path)) as handle:
            file_config = yaml.safe_load(handle) or {}
        if not isinstance(file_config, dict):
            raise ValueError(f"config {path} must contain a mapping at the top level")

    config = _merge(DEFAULTS, file_config)
    for dotted_key, value in (overrides or {}).items():
        if value is not None:
            _apply_override(config, dotted_key, value)
    return config
