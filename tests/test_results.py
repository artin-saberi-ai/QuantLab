import csv

import pytest

from quantlab.artifacts import load_json, save_json, write_csv
from quantlab.benchmark import summary_rows


def make_entry(method, accuracy, size_bytes, latency_mean, status="ok", error=None):
    entry = {"method": method, "status": status}
    if status == "ok":
        entry.update(
            {
                "accuracy": accuracy,
                "params": 1000,
                "serialized_bytes": size_bytes,
                "latency_ms": {"mean": latency_mean, "p95": latency_mean * 1.25},
                "throughput_samples_s": 64 * 1000 / latency_mean,
                "memory": {"process_peak_rss_bytes": 123_000_000},
            }
        )
    if error is not None:
        entry["error"] = error
    return entry


def test_summary_rows_include_baseline_deltas():
    entries = [
        make_entry("fp32", 0.98, 10_000_000, 10.0),
        make_entry("fp16", 0.975, 5_000_000, 5.0),
    ]
    rows = summary_rows(entries)
    baseline, half = rows

    assert baseline["size_ratio"] == 1.0
    assert baseline["speedup"] == 1.0
    assert baseline["accuracy_delta"] == 0.0
    assert half["size_ratio"] == 0.5
    assert half["speedup"] == 2.0
    assert half["accuracy_delta"] == pytest.approx(-0.005)
    assert half["size_mb"] == 5.0
    assert half["peak_rss_mb"] == 123.0


def test_summary_rows_without_baseline():
    rows = summary_rows([make_entry("fp16", 0.9, 5_000_000, 5.0)])
    assert rows[0]["speedup"] is None
    assert rows[0]["size_ratio"] is None
    assert rows[0]["accuracy_delta"] is None


def test_summary_rows_keep_failed_entries():
    entries = [
        make_entry("fp32", 0.98, 10_000_000, 10.0),
        {"method": "static_int8", "status": "failed", "error": "RuntimeError: unsupported"},
    ]
    rows = summary_rows(entries)
    failed = rows[1]

    assert failed["status"] == "failed"
    assert failed["error"] == "RuntimeError: unsupported"
    assert failed["accuracy"] is None
    assert failed["latency_ms"] is None
    assert failed["speedup"] is None


def test_result_json_roundtrip(tmp_path):
    payload = {"run": "mnist_cnn", "results": [make_entry("fp32", 0.98, 1000, 5.0)]}
    path = tmp_path / "nested" / "results.json"
    save_json(payload, path)

    assert load_json(path) == payload
    assert path.read_text().endswith("\n")


def test_write_csv_rows(tmp_path):
    rows = summary_rows(
        [
            make_entry("fp32", 0.98, 10_000_000, 10.0),
            {"method": "fp16", "status": "failed", "error": "boom"},
        ]
    )
    path = tmp_path / "results.csv"
    write_csv(rows, path)

    with open(path, newline="") as handle:
        parsed = list(csv.DictReader(handle))
    assert len(parsed) == 2
    assert parsed[0]["method"] == "fp32"
    assert float(parsed[0]["size_mb"]) == 10.0
    assert parsed[1]["method"] == "fp16"
    assert parsed[1]["error"] == "boom"


def test_write_csv_rejects_empty_input(tmp_path):
    with pytest.raises(ValueError, match="without rows"):
        write_csv([], tmp_path / "empty.csv")
