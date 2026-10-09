import dataclasses

import pytest
import torch

from quantlab.artifacts import load_model, save_model
from quantlab.benchmark import parameter_stats
from quantlab.quantization import (
    SUPPORTED_METHODS,
    QuantSpec,
    apply_quantization,
    make_qconfig,
    resolve_backend,
)


def test_spec_accepts_every_supported_method():
    for method in SUPPORTED_METHODS:
        assert QuantSpec(method=method).method == method


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("method", "int4", "unknown quantization method"),
        ("backend", "tensorrt", "unknown backend"),
        ("qconfig", "per_token", "unknown qconfig"),
        ("calib_batches", 0, "calib_batches"),
        ("qat_epochs", 0, "qat_epochs"),
        ("qat_lr", 0.0, "qat_lr"),
    ],
)
def test_spec_rejects_invalid_values(field, value, message):
    with pytest.raises(ValueError, match=message):
        QuantSpec(**{"method": "fp32", field: value})


def test_spec_is_immutable():
    spec = QuantSpec(method="fp32")
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.method = "fp16"


def test_spec_from_config_reads_quantization_section():
    config = {
        "backend": "qnnpack",
        "qconfig": "per_tensor",
        "calib_batches": 8,
        "qat_epochs": 2,
        "qat_lr": 0.01,
    }
    spec = QuantSpec.from_config(config, "static_int8")
    assert spec.method == "static_int8"
    assert spec.backend == "qnnpack"
    assert spec.qconfig == "per_tensor"
    assert spec.calib_batches == 8
    assert spec.qat_epochs == 2
    assert spec.qat_lr == pytest.approx(0.01)


def test_spec_data_requirements():
    assert QuantSpec(method="static_int8").needs_calibration
    assert QuantSpec(method="qat_int8").needs_finetuning
    assert not QuantSpec(method="dynamic_int8").needs_calibration
    assert not QuantSpec(method="fp16").needs_finetuning


def test_apply_quantization_requires_expected_loaders(tiny_cnn):
    with pytest.raises(ValueError, match="calibration loader"):
        apply_quantization(tiny_cnn, QuantSpec(method="static_int8"))
    with pytest.raises(ValueError, match="training loader"):
        apply_quantization(tiny_cnn, QuantSpec(method="qat_int8"))


def test_fp16_converts_model_and_inputs(tiny_cnn):
    quantized = apply_quantization(tiny_cnn, QuantSpec(method="fp16"))
    assert next(quantized.parameters()).dtype == torch.float16
    output = quantized(torch.randn(2, 1, 8, 8).half())
    assert output.dtype == torch.float16
    assert output.shape == (2, 10)


def test_dynamic_int8_replaces_linear_layers(tiny_cnn):
    quantized = apply_quantization(tiny_cnn, QuantSpec(method="dynamic_int8"))
    linear = quantized[-1]
    assert "quantized" in type(linear).__module__
    assert quantized(torch.randn(2, 1, 8, 8)).shape == (2, 10)
    assert type(tiny_cnn[-1]).__module__ == "torch.nn.modules.linear"


def test_static_int8_calibration_uses_configured_batches(tiny_cnn, fake_loader, tmp_path):
    spec = QuantSpec(method="static_int8", calib_batches=2)
    quantized = apply_quantization(tiny_cnn, spec, calib_loader=fake_loader)

    assert isinstance(quantized, torch.fx.GraphModule)
    assert quantized(torch.randn(2, 1, 8, 8)).shape == (2, 10)
    assert parameter_stats(quantized)[0] == parameter_stats(tiny_cnn)[0]

    artifact = tmp_path / "static_int8.pt"
    save_model(quantized, artifact)
    restored = load_model(artifact)
    assert isinstance(restored, torch.jit.ScriptModule)
    assert "quantize_per_tensor" in restored.code
    assert restored(torch.randn(2, 1, 8, 8)).shape == (2, 10)


def test_qat_int8_returns_quantized_model(tiny_cnn, fake_loader):
    spec = QuantSpec(method="qat_int8", qat_epochs=1, qat_lr=1e-3)
    quantized = apply_quantization(tiny_cnn, spec, train_loader=fake_loader)

    assert isinstance(quantized, torch.fx.GraphModule)
    assert quantized(torch.randn(2, 1, 8, 8)).shape == (2, 10)
    assert any(getattr(module, "_packed_params", None) is not None for module in quantized.modules())


def test_make_qconfig_granularity():
    from torch.ao.quantization import MinMaxObserver, PerChannelMinMaxObserver

    per_channel = make_qconfig("per_channel")
    per_tensor = make_qconfig("per_tensor")
    assert isinstance(per_channel.weight(), PerChannelMinMaxObserver)
    assert isinstance(per_tensor.weight(), MinMaxObserver)
    assert per_tensor.weight().qscheme == torch.per_tensor_symmetric


def test_resolve_backend_rejects_unavailable_engine():
    backend = resolve_backend("auto")
    assert backend in torch.backends.quantized.supported_engines
    with pytest.raises(ValueError, match="not available"):
        resolve_backend("cuda_io")
