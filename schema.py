from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List


UNKNOWN = "-1"

LAYER_TYPES = {
    "standard_conv",
    "depthwise_conv",
    "pointwise_conv",
    "fully_connected",
    "matrix",
    "self_attention",
}

STANDARD_COLUMNS = [
    "layer_seq",
    "layer_type",
    "layer_name",
    "batch",
    "dtype",
    "device",
    "repeat",
    "warmup",
    "in_channels",
    "out_channels",
    "input_h",
    "input_w",
    "kernel_h",
    "kernel_w",
    "stride_h",
    "stride_w",
    "pad_h",
    "pad_w",
    "groups",
    "bias",
    "m",
    "n",
    "k",
    "in_features",
    "out_features",
    "seq_len",
    "embed_dim",
    "num_heads",
    "head_dim",
    "qkv_bias",
    "causal",
]

ALIASES = {
    "id": "layer_seq",
    "layer_id": "layer_seq",
    "seq": "layer_seq",
    "name": "layer_name",
    "op": "layer_type",
    "op_type": "layer_type",
    "type": "layer_type",
    "kind": "layer_type",
    "N": "batch",
    "C": "in_channels",
    "K": "out_channels",
    "H": "input_h",
    "W": "input_w",
    "R": "kernel_h",
    "S": "kernel_w",
    "stride": "stride_h",
    "padding": "pad_h",
    "pad": "pad_h",
}

TYPE_ALIASES = {
    "conv": "standard_conv",
    "conv2d": "standard_conv",
    "standard": "standard_conv",
    "standard_conv": "standard_conv",
    "sc": "standard_conv",
    "dwconv": "depthwise_conv",
    "depthwise": "depthwise_conv",
    "depthwise_conv": "depthwise_conv",
    "dwc": "depthwise_conv",
    "pointwise": "pointwise_conv",
    "pointwise_conv": "pointwise_conv",
    "pwc": "pointwise_conv",
    "1x1": "pointwise_conv",
    "fc": "fully_connected",
    "linear": "fully_connected",
    "fully_connected": "fully_connected",
    "gemm": "matrix",
    "matmul": "matrix",
    "matrix": "matrix",
    "attention": "self_attention",
    "self_attention": "self_attention",
    "mha": "self_attention",
}


@dataclass(frozen=True)
class LayerSpec:
    values: Dict[str, str]

    def get(self, key: str, default: str = UNKNOWN) -> str:
        return self.values.get(key, default)

    def int(self, key: str, default: int = -1) -> int:
        value = self.get(key)
        if value in {"", UNKNOWN, None}:  # type: ignore[comparison-overlap]
            return default
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default

    def bool(self, key: str, default: bool = False) -> bool:
        value = str(self.get(key)).strip().lower()
        if value in {"1", "true", "yes", "y"}:
            return True
        if value in {"0", "false", "no", "n"}:
            return False
        return default

    def as_row(self) -> Dict[str, str]:
        return dict(self.values)


def normalize_key(key: str) -> str:
    key = key.strip()
    return ALIASES.get(key, ALIASES.get(key.lower(), key.lower()))


def normalize_type(value: str) -> str:
    normalized = TYPE_ALIASES.get(str(value).strip().lower(), str(value).strip().lower())
    if normalized not in LAYER_TYPES:
        raise ValueError(f"Unsupported layer_type '{value}'. Supported: {sorted(LAYER_TYPES)}")
    return normalized


def normalize_row(row: Dict[str, str], row_index: int) -> LayerSpec:
    out = {column: UNKNOWN for column in STANDARD_COLUMNS}
    for raw_key, raw_value in row.items():
        if raw_key is None:
            continue
        key = normalize_key(raw_key)
        if key in out:
            out[key] = str(raw_value).strip() if raw_value not in {None, ""} else UNKNOWN

    if out["layer_seq"] == UNKNOWN:
        out["layer_seq"] = str(row_index)
    if out["layer_name"] == UNKNOWN:
        out["layer_name"] = f"layer_{out['layer_seq']}"
    if out["layer_type"] == UNKNOWN:
        out["layer_type"] = infer_layer_type(out)
    else:
        out["layer_type"] = normalize_type(out["layer_type"])

    mirror_pair(out, "stride_h", "stride_w")
    mirror_pair(out, "pad_h", "pad_w")
    mirror_pair(out, "kernel_h", "kernel_w")
    normalize_depthwise_groups(out)
    return LayerSpec(out)


def infer_layer_type(row: Dict[str, str]) -> str:
    if row.get("seq_len", UNKNOWN) != UNKNOWN or row.get("num_heads", UNKNOWN) != UNKNOWN:
        return "self_attention"
    if row.get("m", UNKNOWN) != UNKNOWN and row.get("n", UNKNOWN) != UNKNOWN and row.get("k", UNKNOWN) != UNKNOWN:
        return "matrix"
    if row.get("in_features", UNKNOWN) != UNKNOWN or row.get("out_features", UNKNOWN) != UNKNOWN:
        return "fully_connected"
    if row.get("kernel_h", UNKNOWN) == "1" and row.get("kernel_w", UNKNOWN) in {"1", UNKNOWN}:
        return "pointwise_conv"
    groups = row.get("groups", UNKNOWN)
    in_channels = row.get("in_channels", UNKNOWN)
    out_channels = row.get("out_channels", UNKNOWN)
    if groups != UNKNOWN and groups == in_channels == out_channels:
        return "depthwise_conv"
    return "standard_conv"


def mirror_pair(row: Dict[str, str], first: str, second: str) -> None:
    if row[first] != UNKNOWN and row[second] == UNKNOWN:
        row[second] = row[first]
    elif row[first] == UNKNOWN and row[second] != UNKNOWN:
        row[first] = row[second]


def normalize_depthwise_groups(row: Dict[str, str]) -> None:
    if row["layer_type"] != "depthwise_conv":
        return
    if row["groups"] == UNKNOWN and row["in_channels"] != UNKNOWN:
        row["groups"] = row["in_channels"]
    if row["out_channels"] == UNKNOWN and row["in_channels"] != UNKNOWN:
        row["out_channels"] = row["in_channels"]


def load_layer_csv(path: str | Path) -> List[LayerSpec]:
    csv_path = Path(path)
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Layer CSV has no header: {csv_path}")
        return [normalize_row(row, idx) for idx, row in enumerate(reader)]


def write_normalized_csv(layers: Iterable[LayerSpec], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=STANDARD_COLUMNS)
        writer.writeheader()
        for layer in layers:
            writer.writerow(layer.as_row())
