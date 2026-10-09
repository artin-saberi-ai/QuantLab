import pytest

from quantlab.config import DEFAULTS, load_config


def test_load_config_merges_file_over_defaults(tmp_path):
    path = tmp_path / "experiment.yaml"
    path.write_text("run_name: my_run\ntraining:\n  epochs: 7\n")

    config = load_config(path)
    assert config["run_name"] == "my_run"
    assert config["training"]["epochs"] == 7
    assert config["training"]["optimizer"] == "adam"
    assert config["dataset"]["name"] == "mnist"


def test_load_config_applies_dotted_overrides_and_skips_none(tmp_path):
    path = tmp_path / "experiment.yaml"
    path.write_text("run_name: my_run\n")

    config = load_config(path, {"training.epochs": None, "benchmark.runs": 5, "device": "cuda"})
    assert config["training"]["epochs"] == 3
    assert config["benchmark"]["runs"] == 5
    assert config["device"] == "cuda"


def test_load_config_rejects_non_mapping_file(tmp_path):
    path = tmp_path / "experiment.yaml"
    path.write_text("- fp32\n- int8\n")

    with pytest.raises(ValueError, match="mapping"):
        load_config(path)


def test_load_config_does_not_mutate_defaults(tmp_path):
    path = tmp_path / "experiment.yaml"
    path.write_text("training:\n  epochs: 99\nquantization:\n  methods: [fp16]\n")

    load_config(path)
    assert DEFAULTS["training"]["epochs"] == 3
    assert DEFAULTS["quantization"]["methods"] == ["fp32", "fp16", "dynamic_int8", "static_int8", "qat_int8"]
