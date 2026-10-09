import copy
import dataclasses
import warnings

import torch
import torch.nn.functional as F
from torch import nn
from torch.ao.quantization import MinMaxObserver, PerChannelMinMaxObserver, QConfig, QConfigMapping
from torch.ao.quantization import quantize_dynamic
from torch.ao.quantization.observer import HistogramObserver
from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx, prepare_qat_fx

warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message="torch.ao.quantization is deprecated",
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="torch.quantize_per_tensor, torch.quantize_per_channel",
)

SUPPORTED_METHODS = ("fp32", "fp16", "dynamic_int8", "static_int8", "qat_int8")
SUPPORTED_BACKENDS = ("auto", "fbgemm", "qnnpack")
SUPPORTED_QCONFIGS = ("per_channel", "per_tensor")


@dataclasses.dataclass(frozen=True)
class QuantSpec:
    method: str
    backend: str = "auto"
    qconfig: str = "per_channel"
    calib_batches: int = 32
    qat_epochs: int = 1
    qat_lr: float = 1e-3

    def __post_init__(self):
        if self.method not in SUPPORTED_METHODS:
            raise ValueError(f"unknown quantization method '{self.method}' (expected one of {SUPPORTED_METHODS})")
        if self.backend not in SUPPORTED_BACKENDS:
            raise ValueError(f"unknown backend '{self.backend}' (expected one of {SUPPORTED_BACKENDS})")
        if self.qconfig not in SUPPORTED_QCONFIGS:
            raise ValueError(f"unknown qconfig '{self.qconfig}' (expected one of {SUPPORTED_QCONFIGS})")
        if self.calib_batches < 1:
            raise ValueError("calib_batches must be >= 1")
        if self.qat_epochs < 1:
            raise ValueError("qat_epochs must be >= 1")
        if self.qat_lr <= 0:
            raise ValueError("qat_lr must be > 0")

    @classmethod
    def from_config(cls, config, method):
        return cls(
            method=method,
            backend=config.get("backend", "auto"),
            qconfig=config.get("qconfig", "per_channel"),
            calib_batches=int(config.get("calib_batches", 32)),
            qat_epochs=int(config.get("qat_epochs", 1)),
            qat_lr=float(config.get("qat_lr", 1e-3)),
        )

    @property
    def needs_calibration(self):
        return self.method == "static_int8"

    @property
    def needs_finetuning(self):
        return self.method == "qat_int8"


def resolve_backend(name):
    if name != "auto":
        if name not in torch.backends.quantized.supported_engines:
            raise ValueError(
                f"backend '{name}' is not available (supported: {torch.backends.quantized.supported_engines})"
            )
        return name
    for candidate in ("fbgemm", "qnnpack"):
        if candidate in torch.backends.quantized.supported_engines:
            return candidate
    raise RuntimeError(f"no quantized engine available (found {torch.backends.quantized.supported_engines})")


def make_qconfig(granularity, for_qat=False):
    if granularity == "per_channel":
        weight = PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric)
    else:
        weight = MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)

    if for_qat:
        activation = HistogramObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_affine)
    else:
        activation = MinMaxObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_affine)
    return QConfig(activation=activation, weight=weight)


def _example_input(loader):
    if loader is None:
        raise ValueError("a data loader is required to trace the model")
    batch = next(iter(loader))
    inputs = batch[0] if isinstance(batch, (tuple, list)) else batch
    return inputs[:1].detach().clone().cpu()


def dynamic_int8(model, backend="auto"):
    torch.backends.quantized.engine = resolve_backend(backend)
    return quantize_dynamic(copy.deepcopy(model).eval().cpu(), {nn.Linear}, dtype=torch.qint8)


def static_int8(model, calib_loader, spec):
    backend = resolve_backend(spec.backend)
    torch.backends.quantized.engine = backend
    prepared = prepare_fx(
        copy.deepcopy(model).eval().cpu(),
        QConfigMapping().set_global(make_qconfig(spec.qconfig)),
        example_inputs=(_example_input(calib_loader),),
    )
    with torch.no_grad():
        for index, batch in enumerate(calib_loader):
            if index >= spec.calib_batches:
                break
            prepared(batch[0])
    return convert_fx(prepared)


def qat_int8(model, train_loader, spec, device="cpu"):
    backend = resolve_backend(spec.backend)
    torch.backends.quantized.engine = backend

    model = copy.deepcopy(model).cpu()
    model.train()
    prepared = prepare_qat_fx(
        model,
        QConfigMapping().set_global(make_qconfig(spec.qconfig, for_qat=True)),
        example_inputs=(_example_input(train_loader),),
    )
    prepared.to(device)
    opt = torch.optim.Adam(prepared.parameters(), lr=spec.qat_lr)
    for _ in range(spec.qat_epochs):
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(prepared(inputs), targets)
            loss.backward()
            opt.step()

    prepared.eval()
    torch.backends.quantized.engine = backend
    return convert_fx(prepared.to("cpu"))


def apply_quantization(model, spec, *, calib_loader=None, train_loader=None, device="cpu"):
    if spec.method == "fp32":
        return model
    if spec.method == "fp16":
        return copy.deepcopy(model).eval().half()
    if spec.method == "dynamic_int8":
        return dynamic_int8(model, backend=spec.backend)
    if spec.method == "static_int8":
        if calib_loader is None:
            raise ValueError("static_int8 requires a calibration loader")
        return static_int8(model, calib_loader, spec)
    if spec.method == "qat_int8":
        if train_loader is None:
            raise ValueError("qat_int8 requires a training loader")
        return qat_int8(model, train_loader, spec, device=device)
    raise ValueError(f"unknown quantization method '{spec.method}'")
