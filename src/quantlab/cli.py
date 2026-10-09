import argparse
import dataclasses
import datetime
from pathlib import Path

import torch

from .artifacts import (
    baseline_path,
    load_checkpoint,
    load_json,
    load_model,
    prepare_run_dir,
    save_checkpoint,
    save_json,
    save_model,
    variant_dir,
    write_csv,
)
from .benchmark import BenchmarkConfig, measure, parameter_stats, summary_rows
from .config import load_config
from .data import dataset_spec, make_loaders
from .evaluation import evaluate
from .models import build_model
from .quantization import SUPPORTED_METHODS, QuantSpec, apply_quantization, resolve_backend
from .training import set_seed, train

TABLE_COLUMNS = [
    ("method", "method", None),
    ("status", "status", None),
    ("accuracy", "acc", "{:.4f}"),
    ("accuracy_delta", "d_acc", "{:+.4f}"),
    ("params", "params", "{:,.0f}"),
    ("size_mb", "size_mb", "{:.3f}"),
    ("latency_ms", "latency_ms", "{:.3f}"),
    ("p95_ms", "p95_ms", "{:.3f}"),
    ("throughput_sps", "thr/s", "{:,.1f}"),
    ("size_ratio", "size_ratio", "{:.2f}x"),
    ("speedup", "speedup", "{:.2f}x"),
]


def _parse_methods(raw):
    if not raw:
        return None
    methods = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [method for method in methods if method not in SUPPORTED_METHODS]
    if unknown:
        raise SystemExit(f"unknown methods {unknown} (supported: {', '.join(SUPPORTED_METHODS)})")
    return methods


def _build_model(config, data_spec):
    model_config = dict(config["model"])
    name = model_config.pop("name")
    return build_model(
        name,
        in_channels=data_spec["channels"],
        num_classes=data_spec["num_classes"],
        image_size=data_spec["size"],
        **model_config,
    )


def _format_table(rows):
    rendered = []
    for row in rows:
        cells = []
        for key, _, fmt in TABLE_COLUMNS:
            value = row.get(key)
            if value is None:
                cells.append("-")
            elif fmt:
                cells.append(fmt.format(value))
            else:
                cells.append(str(value))
        rendered.append(cells)

    widths = [len(header) for _, header, _ in TABLE_COLUMNS]
    for cells in rendered:
        for index, cell in enumerate(cells):
            widths[index] = max(widths[index], len(cell))

    lines = []
    header_cells = [header for _, header, _ in TABLE_COLUMNS]
    lines.append("  ".join(cell.ljust(widths[index]) for index, cell in enumerate(header_cells)))
    lines.append("  ".join("-" * widths[index] for index in range(len(widths))))
    for cells in rendered:
        lines.append("  ".join(cell.ljust(widths[index]) for index, cell in enumerate(cells)))
    return "\n".join(lines)


def cmd_train(args):
    config = load_config(args.config, {"training.epochs": args.epochs, "device": args.device})
    set_seed(config["seed"])
    data_spec = dataset_spec(config["dataset"]["name"])
    train_loader, test_loader = make_loaders(
        config["dataset"]["name"],
        config["dataset"]["root"],
        config["dataset"]["batch_size"],
        config["dataset"]["num_workers"],
        seed=config["seed"],
    )

    model = _build_model(config, data_spec)
    print(f"training {config['model']['name']} on {config['dataset']['name']} ({config['device']})")
    history = train(
        model,
        train_loader,
        test_loader,
        epochs=config["training"]["epochs"],
        lr=config["training"]["lr"],
        device=config["device"],
        optimizer=config["training"]["optimizer"],
        weight_decay=config["training"]["weight_decay"],
        log_every=config["training"]["log_every"],
    )

    result = evaluate(model, test_loader, device=config["device"])
    run_dir = prepare_run_dir(args.out, config["run_name"])
    save_checkpoint(
        model,
        baseline_path(run_dir),
        meta={"config": config, "test_accuracy": result["accuracy"], "history": history},
    )
    print(f"baseline accuracy {result['accuracy']:.4f} -> {baseline_path(run_dir)}")


def cmd_quantize(args):
    config = load_config(args.config, {"device": args.device})
    methods = _parse_methods(args.methods) or config["quantization"]["methods"]
    set_seed(config["seed"])

    run_dir = Path(args.out) / config["run_name"]
    checkpoint = baseline_path(run_dir)
    if not checkpoint.exists():
        raise SystemExit(f"missing baseline {checkpoint}, run `quantlab train --config {args.config}` first")

    payload = load_checkpoint(checkpoint)
    data_spec = dataset_spec(config["dataset"]["name"])
    model = _build_model(config, data_spec)
    model.load_state_dict(payload["model_state"])
    torch_engine = resolve_backend(config["quantization"]["backend"])

    quant_specs = [QuantSpec.from_config(config["quantization"], method) for method in methods if method != "fp32"]
    if "fp32" in methods:
        print("fp32: using the baseline checkpoint, nothing to quantize")

    needs_data = any(spec.needs_calibration or spec.needs_finetuning for spec in quant_specs)
    train_loader = test_loader = None
    if needs_data:
        train_loader, test_loader = make_loaders(
            config["dataset"]["name"],
            config["dataset"]["root"],
            config["dataset"]["batch_size"],
            config["dataset"]["num_workers"],
            seed=config["seed"],
        )

    for quant_spec in quant_specs:
        method = quant_spec.method
        print(f"{method}: quantizing (engine {torch_engine})")
        variant = apply_quantization(
            model,
            quant_spec,
            calib_loader=test_loader,
            train_loader=train_loader,
            device=config["device"],
        )
        params, param_bytes = parameter_stats(variant)
        target = variant_dir(run_dir, method)
        save_model(variant, target / "model.pt")
        save_json(
            {
                "method": method,
                "spec": dataclasses.asdict(quant_spec),
                "backend": torch_engine,
                "source": str(checkpoint),
                "params": params,
                "param_bytes": param_bytes,
            },
            target / "meta.json",
        )
        print(f"{method}: saved {target / 'model.pt'} ({params:,} parameters)")


def _load_method_model(config, data_spec, run_dir, method):
    if method == "fp32":
        checkpoint = baseline_path(run_dir)
        if not checkpoint.exists():
            return None, None, f"missing baseline {checkpoint}"
        payload = load_checkpoint(checkpoint)
        model = _build_model(config, data_spec)
        model.load_state_dict(payload["model_state"])
        return model.eval(), None, None

    directory = variant_dir(run_dir, method)
    artifact = directory / "model.pt"
    if not artifact.exists():
        return None, None, f"missing artifact {artifact}, run `quantlab quantize --config ... --methods {method}` first"
    meta = load_json(directory / "meta.json") if (directory / "meta.json").exists() else None
    return load_model(artifact), meta, None


def cmd_benchmark(args):
    overrides = {
        "benchmark.runs": args.runs,
        "benchmark.warmup_runs": args.warmup,
        "benchmark.device": args.device,
    }
    config = load_config(args.config, overrides)
    methods = _parse_methods(args.methods) or config["quantization"]["methods"]
    set_seed(config["seed"])

    data_spec = dataset_spec(config["dataset"]["name"])
    run_dir = Path(args.out) / config["run_name"]
    engine = resolve_backend(config["quantization"]["backend"])
    torch.backends.quantized.engine = engine

    _, test_loader = make_loaders(
        config["dataset"]["name"],
        config["dataset"]["root"],
        config["dataset"]["batch_size"],
        config["dataset"]["num_workers"],
        seed=config["seed"],
    )
    bench_config = BenchmarkConfig.from_dict(
        {**config["benchmark"], "input_shape": (data_spec["channels"], data_spec["size"], data_spec["size"])}
    )
    print(
        f"benchmarking {bench_config.runs} runs ({bench_config.warmup_runs} warm-up) "
        f"batch {bench_config.batch_size} on {bench_config.device}"
    )

    entries = []
    for method in methods:
        model, meta, error = _load_method_model(config, data_spec, run_dir, method)
        if model is None:
            print(f"{method}: skipped ({error})")
            entries.append({"method": method, "status": "failed", "error": error})
            continue
        entry = measure(model, test_loader, bench_config, method, stats=meta)
        entries.append(entry)
        if entry["status"] == "ok":
            print(
                f"{method}: acc {entry['accuracy']:.4f} size {entry['serialized_bytes'] / 1e6:.3f} MB "
                f"latency {entry['latency_ms']['mean']:.3f} ms thr {entry['throughput_samples_s']:.0f}/s"
            )
        else:
            print(f"{method}: failed ({entry['error']})")

    results_dir = Path(args.results_dir)
    json_path = results_dir / f"{config['run_name']}.json"
    csv_path = Path(args.csv) if args.csv else results_dir / f"{config['run_name']}.csv"
    save_json(
        {
            "run": config["run_name"],
            "timestamp": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
            "config": config,
            "benchmark": dataclasses.asdict(bench_config),
            "results": entries,
        },
        json_path,
    )
    write_csv(summary_rows(entries), csv_path)
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")


def cmd_compare(args):
    result = load_json(args.results)
    rows = summary_rows(result["results"])
    bench = result.get("benchmark", {})
    print(
        f"run: {result.get('run')}  device: {bench.get('device')}  "
        f"runs: {bench.get('runs')} (warm-up {bench.get('warmup_runs')})  batch: {bench.get('batch_size')}"
    )
    print(_format_table(rows))
    for row in rows:
        if row["status"] != "ok":
            print(f"{row['method']}: {row['error']}")
    if args.csv:
        write_csv(rows, args.csv)
        print(f"wrote {args.csv}")


def build_parser():
    parser = argparse.ArgumentParser(
        prog="quantlab",
        description="Evaluate accuracy, size, memory and latency trade-offs of neural network quantization",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="train an fp32 baseline model")
    train.add_argument("--config", required=True)
    train.add_argument("--epochs", type=int, default=None)
    train.add_argument("--device", default=None)
    train.add_argument("--out", default="artifacts")
    train.set_defaults(func=cmd_train)

    quantize = sub.add_parser("quantize", help="produce quantized variants of a trained baseline")
    quantize.add_argument("--config", required=True)
    quantize.add_argument("--methods", default=None, help="comma separated, defaults to the config")
    quantize.add_argument("--device", default=None)
    quantize.add_argument("--out", default="artifacts")
    quantize.set_defaults(func=cmd_quantize)

    benchmark = sub.add_parser("benchmark", help="benchmark the baseline and its quantized variants")
    benchmark.add_argument("--config", required=True)
    benchmark.add_argument("--methods", default=None, help="comma separated, defaults to the config")
    benchmark.add_argument("--runs", type=int, default=None)
    benchmark.add_argument("--warmup", type=int, default=None)
    benchmark.add_argument("--device", default=None, help="cpu, cuda or mps (default: benchmark.device)")
    benchmark.add_argument("--out", default="artifacts")
    benchmark.add_argument("--results-dir", default="benchmarks")
    benchmark.add_argument("--csv", default=None)
    benchmark.set_defaults(func=cmd_benchmark)

    compare = sub.add_parser("compare", help="print a comparison table from benchmark results")
    compare.add_argument("--results", required=True)
    compare.add_argument("--csv", default=None)
    compare.set_defaults(func=cmd_compare)

    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
