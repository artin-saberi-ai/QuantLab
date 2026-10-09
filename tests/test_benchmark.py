import pytest
import torch
from torch import nn

from quantlab.benchmark import (
    BenchmarkConfig,
    measure,
    parameter_stats,
    percentile,
    run_benchmark,
    summarize_latencies,
)
from quantlab.quantization import QuantSpec, apply_quantization


class CountingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))
        self.calls = 0

    def forward(self, inputs):
        self.calls += 1
        return inputs * self.scale


class BrokenModel(nn.Module):
    def forward(self, inputs):
        raise RuntimeError("kaboom")


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("runs", 0, "runs"),
        ("warmup_runs", -1, "warmup_runs"),
        ("batch_size", 0, "batch_size"),
        ("input_shape", (1, 8), "input_shape"),
        ("input_shape", (0, 8, 8), "positive"),
    ],
)
def test_config_rejects_invalid_values(field, value, message):
    with pytest.raises(ValueError, match=message):
        BenchmarkConfig(**{field: value})


def test_config_from_dict_converts_shape_and_ignores_unknown_keys():
    config = BenchmarkConfig.from_dict({"runs": 10, "input_shape": [3, 32, 32], "comment": "ignored"})
    assert config.runs == 10
    assert config.input_shape == (3, 32, 32)
    assert config.warmup_runs == 20


def test_summarize_latencies_statistics():
    stats = summarize_latencies([1.0, 2.0, 3.0, 4.0])
    assert stats["mean"] == pytest.approx(2.5)
    assert stats["median"] == pytest.approx(2.5)
    assert stats["min"] == 1.0
    assert stats["max"] == 4.0
    assert stats["p95"] == 4.0


def test_percentile_interpolates_by_nearest_rank():
    values = [10, 20, 30, 40, 50]
    assert percentile(values, 50) == 30
    assert percentile(values, 100) == 50
    assert percentile([], 50) == 0.0


def test_run_benchmark_excludes_warmup_runs():
    model = CountingModel()
    config = BenchmarkConfig(runs=5, warmup_runs=3, batch_size=2, input_shape=(1, 4, 4))
    result = run_benchmark(model, config)

    assert model.calls == 8
    assert result["runs"] == 5
    assert result["warmup_runs"] == 3
    assert len(result["samples_ms"]) == 5
    assert result["latency_ms"]["mean"] > 0
    assert result["latency_ms"]["min"] <= result["latency_ms"]["max"]


def test_run_benchmark_throughput_matches_latency():
    model = CountingModel()
    config = BenchmarkConfig(runs=5, warmup_runs=1, batch_size=8, input_shape=(1, 4, 4))
    result = run_benchmark(model, config)
    expected = config.batch_size * 1000.0 / result["latency_ms"]["mean"]
    assert result["throughput_samples_s"] == pytest.approx(expected)
    assert result["params"] == 1
    assert result["serialized_bytes"] > 0
    assert result["memory"]["process_peak_rss_bytes"] > 0


def test_run_benchmark_casts_input_for_half_models():
    model = nn.Sequential(nn.Flatten(), nn.Linear(16, 4)).half()
    config = BenchmarkConfig(runs=2, warmup_runs=1, batch_size=2, input_shape=(1, 4, 4))
    result = run_benchmark(model, config)
    assert len(result["samples_ms"]) == 2


def test_parameter_stats_counts_packed_quantized_weights():
    model = nn.Sequential(nn.Flatten(), nn.Linear(16, 4))
    quantized = apply_quantization(model, QuantSpec(method="dynamic_int8"))

    assert parameter_stats(quantized)[0] == parameter_stats(model)[0]
    assert parameter_stats(quantized)[1] < parameter_stats(model)[1]


def test_measure_reports_accuracy_and_benchmark(fake_loader):
    model = nn.Sequential(nn.Flatten(), nn.Linear(64, 10))
    config = BenchmarkConfig(runs=3, warmup_runs=1, batch_size=4, input_shape=(1, 8, 8))
    entry = measure(model, fake_loader, config, "fp32")

    assert entry["status"] == "ok"
    assert entry["method"] == "fp32"
    assert 0.0 <= entry["accuracy"] <= 1.0
    assert entry["latency_ms"]["mean"] > 0


def test_measure_records_failures():
    config = BenchmarkConfig(runs=1, warmup_runs=0, batch_size=1, input_shape=(1, 4, 4))
    entry = measure(BrokenModel(), None, config, "fp16")

    assert entry["status"] == "failed"
    assert entry["method"] == "fp16"
    assert "kaboom" in entry["error"]
