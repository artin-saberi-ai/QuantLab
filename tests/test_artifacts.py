import torch
from torch import nn

from quantlab.artifacts import (
    baseline_path,
    load_checkpoint,
    load_model,
    prepare_run_dir,
    save_checkpoint,
    save_model,
    variant_dir,
)


def test_prepare_run_dir_layout(tmp_path):
    run_dir = prepare_run_dir(tmp_path, "mnist_cnn")
    assert run_dir == tmp_path / "mnist_cnn"
    assert (run_dir / "variants").is_dir()
    assert variant_dir(run_dir, "fp16") == run_dir / "variants" / "fp16"
    assert (run_dir / "variants" / "fp16").is_dir()
    assert baseline_path(run_dir) == run_dir / "baseline.pt"


def test_checkpoint_roundtrip_keeps_state_and_meta(tmp_path):
    model = nn.Sequential(nn.Linear(4, 2))
    path = tmp_path / "nested" / "baseline.pt"
    save_checkpoint(model, path, meta={"test_accuracy": 0.75})

    payload = load_checkpoint(path)
    assert payload["meta"]["test_accuracy"] == 0.75

    restored = nn.Sequential(nn.Linear(4, 2))
    restored.load_state_dict(payload["model_state"])
    inputs = torch.randn(3, 4)
    assert torch.allclose(restored(inputs), model(inputs))


def test_load_model_dispatches_on_serialization_format(tmp_path):
    eager = nn.Sequential(nn.Linear(4, 2))
    save_model(eager, tmp_path / "eager.pt")
    assert isinstance(load_model(tmp_path / "eager.pt"), nn.Sequential)

    scripted = torch.jit.script(eager)
    save_model(scripted, tmp_path / "scripted.pt")
    restored = load_model(tmp_path / "scripted.pt")
    assert isinstance(restored, torch.jit.ScriptModule)
    assert restored(torch.randn(2, 4)).shape == (2, 2)
