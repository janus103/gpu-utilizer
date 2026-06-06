from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch.nn as nn


@dataclass(frozen=True)
class ModelInfo:
    model: nn.Module
    source: str
    name: str


def build_model(
    model_name: str,
    model_source: str,
    pretrained: bool,
    eata_root: Path,
    num_classes: int = 1000,
) -> ModelInfo:
    """Build a target model without tying the profiler to one model zoo."""
    source = resolve_model_source(model_name, model_source, eata_root)
    if source == "eata_resnet":
        return build_eata_resnet(model_name, pretrained, eata_root)
    if source == "torchvision":
        return build_torchvision_model(model_name, pretrained, num_classes)
    if source == "timm":
        return build_timm_model(model_name, pretrained, num_classes)
    raise ValueError(f"Unsupported model source: {source}")


def resolve_model_source(model_name: str, model_source: str, eata_root: Path) -> str:
    if model_source != "auto":
        return model_source
    if has_eata_resnet(model_name, eata_root):
        return "eata_resnet"
    if importlib.util.find_spec("timm") is not None:
        return "timm"
    if importlib.util.find_spec("torchvision") is not None:
        return "torchvision"
    raise RuntimeError("Unable to find a model source. Install timm or torchvision.")


def has_eata_resnet(model_name: str, eata_root: Path) -> bool:
    models_dir = eata_root / "models"
    return (models_dir / "Res.py").exists() and model_name.startswith(("resnet", "resnext"))


def build_eata_resnet(model_name: str, pretrained: bool, eata_root: Path) -> ModelInfo:
    eata_root = eata_root.resolve()
    if str(eata_root) not in sys.path:
        sys.path.insert(0, str(eata_root))
    import models.Res as resnet_models  # type: ignore

    if not hasattr(resnet_models, model_name):
        raise ValueError(f"EATA ResNet model '{model_name}' is not available.")
    model = getattr(resnet_models, model_name)(pretrained=pretrained)
    return ModelInfo(model=model, source="eata_resnet", name=model_name)


def build_torchvision_model(model_name: str, pretrained: bool, num_classes: int) -> ModelInfo:
    import torchvision.models as tv_models

    if hasattr(tv_models, "get_model"):
        weights: Optional[str] = "DEFAULT" if pretrained else None
        model = tv_models.get_model(model_name, weights=weights)
        if not pretrained and num_classes != 1000:
            reset_classifier_if_available(model, num_classes)
        return ModelInfo(model=model, source="torchvision", name=model_name)

    if not hasattr(tv_models, model_name):
        raise ValueError(f"torchvision model '{model_name}' is not available.")
    model = getattr(tv_models, model_name)(pretrained=pretrained)
    if not pretrained and num_classes != 1000:
        reset_classifier_if_available(model, num_classes)
    return ModelInfo(model=model, source="torchvision", name=model_name)


def build_timm_model(model_name: str, pretrained: bool, num_classes: int) -> ModelInfo:
    import timm

    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    return ModelInfo(model=model, source="timm", name=model_name)


def reset_classifier_if_available(model: nn.Module, num_classes: int) -> None:
    if hasattr(model, "reset_classifier"):
        model.reset_classifier(num_classes)  # type: ignore[attr-defined]
        return
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return
    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Linear):
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
