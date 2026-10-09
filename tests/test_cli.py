import pytest

from quantlab.benchmark import summary_rows
from quantlab.cli import _format_table, _parse_methods


def test_parse_methods_returns_clean_list():
    assert _parse_methods("fp32, fp16 ,dynamic_int8") == ["fp32", "fp16", "dynamic_int8"]
    assert _parse_methods("") is None
    assert _parse_methods(None) is None


def test_parse_methods_rejects_unknown_method():
    with pytest.raises(SystemExit, match="unknown methods"):
        _parse_methods("fp32,int4")


def test_format_table_renders_headers_and_missing_cells():
    rows = summary_rows(
        [
            {
                "method": "fp32",
                "status": "ok",
                "accuracy": 0.99,
                "params": 1000,
                "serialized_bytes": 2_000_000,
                "latency_ms": {"mean": 2.0, "p95": 2.5},
                "throughput_samples_s": 1000.0,
                "memory": {"process_peak_rss_bytes": 1_000_000},
            },
            {"method": "fp16", "status": "failed", "error": "boom"},
        ]
    )
    table = _format_table(rows)
    lines = table.splitlines()

    assert lines[0].split()[:3] == ["method", "status", "acc"]
    assert any(line.startswith("fp32") and "0.9900" in line for line in lines)
    assert any(line.startswith("fp16") and "failed" in line and "-" in line for line in lines)
