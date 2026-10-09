import csv
import json
import warnings
import zipfile
from pathlib import Path

import torch


def prepare_run_dir(root, run_name):
    path = Path(root) / run_name
    (path / "variants").mkdir(parents=True, exist_ok=True)
    return path


def baseline_path(run_dir):
    return Path(run_dir) / "baseline.pt"


def variant_dir(run_dir, method):
    path = Path(run_dir) / "variants" / method
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_checkpoint(model, path, meta=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "meta": meta or {}}, path)


def load_checkpoint(path):
    return torch.load(Path(path), map_location="cpu", weights_only=False)


def save_model(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(model, torch.fx.GraphModule):
        model = _as_torchscript(model)
    if isinstance(model, torch.jit.ScriptModule):
        torch.jit.save(model, str(path))
    else:
        torch.save(model, path)


def _as_torchscript(model):
    # pickling converted fx modules loses submodule state, torchscript round-trips cleanly
    try:
        return torch.jit.script(model)
    except Exception as exc:
        warnings.warn(f"could not script model, saving it with pickle instead: {exc}")
        return model


def load_model(path):
    path = Path(path)
    if _is_torchscript(path):
        return torch.jit.load(str(path))
    return torch.load(path, map_location="cpu", weights_only=False)


def _is_torchscript(path):
    with zipfile.ZipFile(path) as archive:
        return any("/code/" in name for name in archive.namelist())


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")


def load_json(path):
    with open(Path(path)) as handle:
        return json.load(handle)


def write_csv(rows, path):
    if not rows:
        raise ValueError("cannot write csv without rows")
    fieldnames = list(rows[0])
    for row in rows[1:]:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
