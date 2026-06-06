from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import torch

GPU_UTILIZER_ROOT = Path(__file__).resolve().parents[1]
TTA_DIR = Path(__file__).resolve().parent
EXTERNAL_METHODS_ROOT = TTA_DIR / "external_methods"
ZOA_ROOT = EXTERNAL_METHODS_ROOT / "zoa"
for path in (GPU_UTILIZER_ROOT, ZOA_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from gpu_metrics import MetricConfig, profile_callable  # noqa: E402


FIELDNAMES = [
    "model",
    "algorithm",
    "precision",
    "batch_size",
    "status",
    "failure_reason",
    "error",
    "output_dtype",
    "output_all_finite",
    "cuda_elapsed_ms_total",
    "latency_ms_per_iter",
    "estimated_sm_cycles_per_iter",
    "energy_j_per_iter",
    "energy_j_per_iter_est_from_power",
    "avg_power_w_from_energy",
    "avg_power_w_sampled",
    "energy_source",
]


class PrintLogger:
    def info(self, message):
        print(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test ZOA adaptation with gpu_utilizer AMP FP16 profiling.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--no-idle-check", action="store_true")
    parser.add_argument("--output", default=str(GPU_UTILIZER_ROOT / "Results" / "TTA" / "zoa_adapt_fp16_gpu_utilizer_smoke.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    metric_config = MetricConfig(
        device_index=args.gpu_index,
        allow_nvidia_smi_fallback=False,
        require_idle_gpu=not args.no_idle_check,
        allow_zero_energy_estimate=True,
    )
    rows: List[Dict[str, object]] = []
    for model_name, builder in (
        ("resnet50", build_zoa_resnet),
        ("vit_base_patch16_224", build_zoa_vit),
        ("mobilevit_xxs", build_mobilevit_zoa_like),
    ):
        row = run_case(args, metric_config, model_name, builder, device)
        rows.append(row)
        write_rows(Path(args.output), rows)
        print(f"{model_name} ZOA AMP FP16 status={row['status']} dtype={row['output_dtype']} finite={row['output_all_finite']} reason={row['failure_reason']}")
    print(f"wrote ZOA adaptation FP16 smoke: {Path(args.output).resolve()}")


def run_case(args, metric_config, model_name, builder, device):
    row: Dict[str, object] = {
        "model": model_name,
        "algorithm": algorithm_name(model_name),
        "precision": "amp_fp16",
        "batch_size": args.batch_size,
        "status": "pending",
        "failure_reason": "",
        "error": "",
        "output_dtype": "",
        "output_all_finite": "",
    }
    try:
        adapt_model, image_size = builder(args.batch_size, device)
        x = torch.randn(args.batch_size, 3, image_size, image_size, device=device)
        output_holder = {}

        def run_once():
            with torch.cuda.amp.autocast(dtype=torch.float16):
                output = adapt_model(x)
            output_holder["output"] = output
            return output

        metrics = profile_callable(
            run_once,
            warmup=args.warmup,
            repeat=args.repeat,
            config=metric_config,
            nvtx_name=f"zoa_adapt_fp16_{model_name}",
        )
        output = output_holder["output"]
        row.update(metrics)
        row["output_dtype"] = str(output.dtype)
        row["output_all_finite"] = bool(torch.isfinite(output).all().item())
        row["status"] = "ok" if row["output_all_finite"] else "error"
        row["failure_reason"] = "" if row["output_all_finite"] else "non_finite_output"
    except Exception as exc:
        row["status"] = "error"
        row["failure_reason"] = classify_failure(exc)
        row["error"] = repr(exc)
    finally:
        torch.cuda.empty_cache()
    return row


def dummy_args(**overrides):
    args = SimpleNamespace(
        quant=False,
        max_weight_nums=32,
        alpha_scale=10.0,
        lr_alpha=0.01,
        weight_decay_alpha=0.1,
        lr=1e-4,
        spsa_momentum=0.0,
        weight_decay=0.4,
        steps=1,
        sp_avg=1,
        spsa_c=0.01,
        spsa_c_alpha=0.05,
        domain_t=0.2,
        profile_timing=False,
        _timing_profiled=False,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def build_zoa_resnet(batch_size: int, device):
    import models.resnet as ResNet
    from models.fuse_resnet import FuseResNet
    import tta_library.zoa_resnet as zoa_resnet

    args = dummy_args(lambda_bp=1)
    model = ResNet.resnet50(pretrained=False).to(device)
    fuse = FuseResNet(args, model, logger=PrintLogger())
    fuse = zoa_resnet.configure_model(fuse).to(device)
    fuse.replace_fuse_bn()
    fuse.configure_model()
    params = fuse.collect_params()
    alpha_optimizer = torch.optim.AdamW([{"params": params["alpha"], "lr": args.lr_alpha, "weight_decay": args.weight_decay_alpha}])
    eps_optimizer = torch.optim.SGD(params["epsilon_weight"] + params["epsilon_bias"], args.lr, momentum=args.spsa_momentum, weight_decay=args.weight_decay)
    fuse.set_optimizers(alpha_optimizer, eps_optimizer)
    adapt_model = zoa_resnet.ZOA_ResNet(args, fuse, args.lambda_bp)
    x = torch.randn(batch_size, 3, 224, 224, device=device)
    adapt_model.train_info = resnet_train_info(fuse, x)
    return adapt_model, 224


def resnet_train_info(fuse, x):
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
        features, _ = fuse.forward_features(x)
    means = []
    stds = []
    for feature in features:
        means.append(feature.float().mean(dim=0))
        stds.append(feature.float().std(dim=0))
    return torch.cat(stds, dim=0), torch.cat(means, dim=0)


def build_zoa_vit(batch_size: int, device):
    import timm
    from models.fuse_vit import FuseViT
    import tta_library.zoa_vit as zoa_vit

    args = dummy_args(lambda_bp=30, spsa_c=0.02, domain_t=0.1)
    model = timm.create_model("vit_base_patch16_224", pretrained=False).to(device)
    fuse = FuseViT(args, model, logger=PrintLogger())
    fuse = zoa_vit.configure_model(fuse).to(device)
    fuse.replace_fuse_ln()
    fuse.configure_model()
    params = fuse.collect_params()
    alpha_optimizer = torch.optim.AdamW([{"params": params["alpha"], "lr": args.lr_alpha, "weight_decay": args.weight_decay_alpha}])
    eps_optimizer = torch.optim.SGD(params["epsilon_weight"] + params["epsilon_bias"], args.lr, momentum=args.spsa_momentum, weight_decay=args.weight_decay)
    fuse.set_optimizers(alpha_optimizer, eps_optimizer)
    adapt_model = zoa_vit.ZOA_ViT(args, fuse, args.lambda_bp)
    x = torch.randn(batch_size, 3, 224, 224, device=device)
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
        features = fuse.layers_cls_features(x)
    adapt_model.train_info = torch.std_mean(features.float(), dim=0)
    return adapt_model, 224


def build_mobilevit_zoa_like(batch_size: int, device):
    import timm

    model = timm.create_model("mobilevit_xxs", pretrained=False).to(device)
    adapt_model = MobileViTZOALike(model, lr=1e-4, ck=0.01, fitness_lambda=30.0).to(device)
    x = torch.randn(batch_size, 3, 256, 256, device=device)
    adapt_model.obtain_origin_stat(x)
    return adapt_model, 256


class MobileViTZOALike(torch.nn.Module):
    """ZOA-style MobileViT smoke adapter.

    Policy mirrors the ViT/ResNet ZOA scripts: keep the earliest and latest
    stages stable, and adapt middle-stage normalization affine parameters.
    """

    def __init__(self, model: torch.nn.Module, lr: float, ck: float, fitness_lambda: float):
        super().__init__()
        self.model = model
        self.lr = lr
        self.ck = ck
        self.fitness_lambda = fitness_lambda
        self.params, self.param_names = self.collect_middle_norm_params()
        if not self.params:
            raise RuntimeError("MobileViT ZOA-like adapter found no middle normalization parameters.")
        self.origin_vector = self.parameters_to_vector().detach().clone()
        self.train_mean = None
        self.train_std = None

    def collect_middle_norm_params(self):
        import torch.nn as nn

        params = []
        names = []
        self.model.train()
        self.model.requires_grad_(False)
        for module_name, module in self.model.named_modules():
            # Exclude stem/stages.0/stages.1 as early features, and stages.4/final/head as late classifier-side features.
            if not module_name.startswith(("stages.2", "stages.3")):
                continue
            if not isinstance(module, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                continue
            if isinstance(module, nn.BatchNorm2d):
                module.track_running_stats = False
                module.running_mean = None
                module.running_var = None
            for param_name, param in module.named_parameters(recurse=False):
                if param_name in {"weight", "bias"}:
                    param.requires_grad_(False)
                    params.append(param)
                    names.append(f"{module_name}.{param_name}")
        return params, names

    @torch.no_grad()
    def obtain_origin_stat(self, x: torch.Tensor) -> None:
        with torch.cuda.amp.autocast(dtype=torch.float16):
            feature = self.model.forward_features(x)
        feature = feature.float()
        self.train_mean = feature.mean(dim=(0, 2, 3))
        self.train_std = feature.std(dim=(0, 2, 3))

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        params = self.parameters_to_vector()
        loss0, output = self.forward_and_loss(x)
        perturb = torch.empty_like(params).uniform_(0.5, 1.0)
        signs = torch.randint(0, 2, perturb.shape, device=perturb.device, dtype=torch.bool)
        perturb = torch.where(signs, perturb, -perturb)

        self.vector_to_parameters(params + self.ck * perturb)
        loss1, _ = self.forward_and_loss(x)
        ghat = ((loss1 - loss0) / self.ck) * perturb
        self.vector_to_parameters(params - self.lr * ghat)
        return output

    @torch.no_grad()
    def forward_and_loss(self, x: torch.Tensor):
        with torch.cuda.amp.autocast(dtype=torch.float16):
            feature = self.model.forward_features(x)
            output = self.model.forward_head(feature)
        entropy = softmax_entropy(output.float()).mean()
        feature_f = feature.float()
        mean = feature_f.mean(dim=(0, 2, 3))
        if feature_f.size(0) > 1:
            std_loss = torch.nn.functional.mse_loss(feature_f.std(dim=(0, 2, 3)), self.train_std)
        else:
            std_loss = torch.zeros((), device=feature_f.device)
        mean_loss = torch.nn.functional.mse_loss(mean, self.train_mean)
        return entropy + self.fitness_lambda * (mean_loss + std_loss), output

    def parameters_to_vector(self) -> torch.Tensor:
        return torch.nn.utils.parameters_to_vector(self.params)

    def vector_to_parameters(self, vector: torch.Tensor) -> None:
        torch.nn.utils.vector_to_parameters(vector, self.params)


def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def algorithm_name(model_name: str) -> str:
    if model_name == "resnet50":
        return "zoa_resnet"
    if model_name == "vit_base_patch16_224":
        return "zoa_vit"
    if model_name == "mobilevit_xxs":
        return "zoa_like_mobilevit_middle_norm"
    return "unknown"


def classify_failure(exc: Exception) -> str:
    message = repr(exc).lower()
    if "out of memory" in message:
        return "oom"
    if "nan" in message or "inf" in message:
        return "non_finite"
    return "error"


def write_rows(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
