import dataclasses
import io
import math
import resource
import statistics
import sys
import time

import torch
from torch.ao.nn.quantized import Conv2d as QuantizedConv2d
from torch.ao.nn.quantized import Linear as QuantizedLinear

from .evaluation import evaluate

PACKED_UNPACKERS = (
    (QuantizedLinear, torch.ops.quantized.linear_unpack),
    (QuantizedConv2d, torch.ops.quantized.conv2d_unpack),
)


@dataclasses.dataclass(frozen=True)
class BenchmarkConfig:
    runs: int = 200
    warmup_runs: int = 20
    batch_size: int = 64
    input_shape: tuple = (1, 28, 28)
    device: str = "cpu"
    seed: int = 0

    def __post_init__(self):
        if self.runs < 1:
            raise ValueError("runs must be >= 1")
        if self.warmup_runs < 0:
            raise ValueError("warmup_runs must be >= 0")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if len(self.input_shape) != 3:
            raise ValueError("input_shape must be (channels, height, width)")
        if any(dim < 1 for dim in self.input_shape):
            raise ValueError("input_shape dimensions must be positive")

    @classmethod
    def from_dict(cls, config):
        values = dict(config)
        if isinstance(values.get("input_shape"), list):
            values["input_shape"] = tuple(values["input_shape"])
        known = {field.name for field in dataclasses.fields(cls)}
        return cls(**{key: value for key, value in values.items() if key in known})


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def percentile(sorted_values, quantile):
    if not sorted_values:
        return 0.0
    index = math.ceil(quantile / 100 * len(sorted_values)) - 1
    return sorted_values[min(max(index, 0), len(sorted_values) - 1)]


def summarize_latencies(samples_ms):
    ordered = sorted(samples_ms)
    return {
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "p95": percentile(ordered, 95),
        "p99": percentile(ordered, 99),
        "min": ordered[0],
        "max": ordered[-1],
        "std": statistics.pstdev(ordered) if len(ordered) > 1 else 0.0,
    }


def _packed_tensors(module):
    packed = getattr(module, "_packed_params", None)
    if packed is None:
        return []
    if isinstance(packed, torch.nn.Module):
        packed = getattr(packed, "_packed_params", packed)
    for module_type, unpack in PACKED_UNPACKERS:
        if isinstance(module, module_type):
            return [tensor for tensor in unpack(packed) if torch.is_tensor(tensor)]
    return []


def parameter_stats(model):
    tensors = list(model.parameters())
    for module in model.modules():
        tensors += _packed_tensors(module)
    params = sum(tensor.numel() for tensor in tensors)
    nbytes = sum(tensor.numel() * tensor.element_size() for tensor in tensors)
    return params, nbytes


def serialized_size_bytes(model):
    buffer = io.BytesIO()
    if isinstance(model, torch.jit.ScriptModule):
        torch.jit.save(model, buffer)
    else:
        torch.save(model, buffer)
    return buffer.getbuffer().nbytes


def reset_peak_memory(device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()


def peak_memory(device):
    memory = {}
    if device.type == "cuda":
        memory["device_peak_bytes"] = torch.cuda.max_memory_allocated()
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    memory["process_peak_rss_bytes"] = peak_rss if sys.platform == "darwin" else peak_rss * 1024
    return memory


def run_benchmark(model, config):
    device = torch.device(config.device)
    model.eval()
    model.to(device)
    reset_peak_memory(device)

    torch.manual_seed(config.seed)
    inputs = torch.randn(config.batch_size, *config.input_shape, device=device)
    param = next(model.parameters(), None)
    if param is not None and param.dtype == torch.float16:
        inputs = inputs.half()

    with torch.no_grad():
        for _ in range(config.warmup_runs):
            model(inputs)
        synchronize(device)

        samples = []
        for _ in range(config.runs):
            start = time.perf_counter()
            model(inputs)
            synchronize(device)
            samples.append((time.perf_counter() - start) * 1000.0)

    latency = summarize_latencies(samples)
    params, param_bytes = parameter_stats(model)
    return {
        "device": str(device),
        "runs": config.runs,
        "warmup_runs": config.warmup_runs,
        "batch_size": config.batch_size,
        "params": params,
        "param_bytes": param_bytes,
        "serialized_bytes": serialized_size_bytes(model),
        "latency_ms": latency,
        "throughput_samples_s": config.batch_size * 1000.0 / latency["mean"] if latency["mean"] > 0 else 0.0,
        "samples_ms": samples,
        "memory": peak_memory(device),
    }


def measure(model, loader, config, method, stats=None):
    entry = {"method": method, "status": "ok"}
    try:
        if loader is not None:
            entry["accuracy"] = evaluate(model, loader, device=config.device)["accuracy"]
        entry.update(run_benchmark(model, config))
        if stats:
            entry["params"] = stats["params"]
            entry["param_bytes"] = stats["param_bytes"]
    except Exception as exc:
        entry = {"method": method, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    return entry


def summary_rows(entries):
    baseline = next(
        (entry for entry in entries if entry.get("method") == "fp32" and entry.get("status") == "ok"),
        None,
    )
    baseline_accuracy = baseline.get("accuracy") if baseline else None
    baseline_size = baseline.get("serialized_bytes") if baseline else None
    baseline_latency = baseline["latency_ms"]["mean"] if baseline else None

    rows = []
    for entry in entries:
        ok = entry.get("status") == "ok"
        row = {
            "method": entry.get("method"),
            "status": entry.get("status"),
            "accuracy": round(entry["accuracy"], 4) if ok and "accuracy" in entry else None,
            "params": entry.get("params") if ok else None,
            "size_mb": round(entry["serialized_bytes"] / 1e6, 3) if ok else None,
            "latency_ms": round(entry["latency_ms"]["mean"], 4) if ok else None,
            "p95_ms": round(entry["latency_ms"]["p95"], 4) if ok else None,
            "throughput_sps": round(entry["throughput_samples_s"], 1) if ok else None,
            "peak_rss_mb": round(entry["memory"]["process_peak_rss_bytes"] / 1e6, 1) if ok else None,
            "accuracy_delta": None,
            "size_ratio": None,
            "speedup": None,
            "error": entry.get("error"),
        }
        if ok and baseline is not None:
            if row["accuracy"] is not None and baseline_accuracy is not None:
                row["accuracy_delta"] = round(row["accuracy"] - baseline_accuracy, 4)
            if row["size_mb"] is not None and baseline_size:
                row["size_ratio"] = round(entry["serialized_bytes"] / baseline_size, 3)
            if row["latency_ms"] and baseline_latency:
                row["speedup"] = round(baseline_latency / entry["latency_ms"]["mean"], 2)
        rows.append(row)
    return rows
