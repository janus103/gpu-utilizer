#!/usr/bin/env python3
"""
TTA record TSV 파일들을 분석하여, epoch -1 (TTA 이전) 대비 개선된 target_class 수를 corruption별로 집계합니다.

개선 기준: 해당 target_class의 epochs 0~19 중 최대 backbone_acc > epoch -1의 backbone_acc
"""

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

# analyze_tta_record_table.py와 동일한 corruption 순서
CORRUPTION_ORDER = [
    ("Noise", "Gaussian", "gaussian_noise"),
    ("Noise", "Shot", "shot_noise"),
    ("Noise", "Impulse", "impulse_noise"),
    ("Blur", "Defocus", "defocus_blur"),
    ("Blur", "Glass", "glass_blur"),
    ("Blur", "Motion", "motion_blur"),
    ("Blur", "Zoom", "zoom_blur"),
    ("Weather", "Snow", "snow"),
    ("Weather", "Frost", "frost"),
    ("Weather", "Fog", "fog"),
    ("Weather", "Brightness", "brightness"),
    ("Digital", "Contrast", "contrast"),
    ("Digital", "Elastic", "elastic_transform"),
    ("Digital", "Pixelate", "pixelate"),
    ("Digital", "JPEG", "jpeg_compression"),
]

TSV_PATTERN = "{prefix}_{corruption}_5_l2_ep20.tsv"


def count_improved_classes(tsv_path: Path) -> Optional[dict]:
    """
    TSV 파일에서 epoch -1 대비 개선된 target_class 수를 반환합니다.

    Returns:
        dict: {
            "num_classes": int,
            "num_improved": int,
            "pct_improved": float,
            "mean_epoch_minus1": float,  # epoch -1의 backbone_acc 평균
            "mean_best": float,  # 각 class별 epochs -1~19 중 최대값의 평균
        }
        파일이 없거나 형식이 잘못된 경우 None 반환
    """
    if not tsv_path.exists():
        return None

    df = pd.read_csv(tsv_path, sep="\t")
    if "target_class" not in df.columns or "epoch" not in df.columns or "backbone_acc" not in df.columns:
        return None

    # epoch -1 데이터
    df_init = df[df["epoch"] == -1][["target_class", "backbone_acc"]].rename(
        columns={"backbone_acc": "acc_init"}
    )

    # epochs 0~19 중 최대값
    df_tta = df[df["epoch"].between(0, 19)].groupby("target_class")["backbone_acc"].max().reset_index()
    df_tta = df_tta.rename(columns={"backbone_acc": "acc_max_tta"})

    merged = df_init.merge(df_tta, on="target_class", how="inner")
    num_improved = (merged["acc_max_tta"] > merged["acc_init"]).sum()
    num_classes = len(merged)
    pct_improved = (num_improved / num_classes * 100) if num_classes > 0 else 0.0

    # Col +1: Epoch (-1)의 평균
    mean_epoch_minus1 = merged["acc_init"].mean()

    # Col +2: 각 class별 epochs -1~19 중 최대값의 평균
    df_all = df[df["epoch"].between(-1, 19)]
    mean_best = df_all.groupby("target_class")["backbone_acc"].max().mean()

    return {
        "num_classes": num_classes,
        "num_improved": int(num_improved),
        "pct_improved": pct_improved,
        "mean_epoch_minus1": float(mean_epoch_minus1),
        "mean_best": float(mean_best),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TTA record에서 epoch -1 대비 개선된 target_class 수를 corruption별로 집계"
    )
    parser.add_argument(
        "record_dir",
        type=str,
        nargs="?",
        default=None,
        metavar="RECORD_DIR",
        help="TTA record TSV 파일이 있는 디렉토리",
    )
    parser.add_argument(
        "--record_dir",
        type=str,
        dest="record_dir_opt",
        default=None,
        help="TTA record TSV 파일이 있는 디렉토리",
    )
    parser.add_argument(
        "--tsv_prefix",
        type=str,
        default="tta_record",
        help="TSV 파일명 접두사 (기본: tta_record, ViT용: tta_vit_record)",
    )
    args = parser.parse_args()

    record_dir_path = args.record_dir_opt or args.record_dir or "output_tta_record"
    record_dir = Path(record_dir_path)
    if not record_dir.is_dir():
        print(f"Error: 디렉토리를 찾을 수 없습니다: {record_dir}")
        return

    # 테이블 출력
    print("=" * 95)
    print("TTA 개선 분석 - epoch -1 대비 개선된 target_class 수 (corruption별)")
    print("=" * 95)
    print(
        f"{'Category':<10} | {'Corruption':<12} | {'Total':>6} | {'Improved':>8} | {'%':>6} | "
        f"{'Avg(E-1)':>10} | {'Avg(Best)':>10}"
    )
    print("-" * 95)

    for category, display_name, corruption_key in CORRUPTION_ORDER:
        tsv_name = TSV_PATTERN.format(prefix=args.tsv_prefix, corruption=corruption_key)
        tsv_path = record_dir / tsv_name

        result = count_improved_classes(tsv_path)
        if result is None:
            print(
                f"{category:<10} | {display_name:<12} | {'-':>6} | {'-':>8} | {'-':>6} | "
                f"{'-':>10} | {'-':>10}"
            )
        else:
            print(
                f"{category:<10} | {display_name:<12} | {result['num_classes']:>6} | "
                f"{result['num_improved']:>8} | {result['pct_improved']:>5.2f}% | "
                f"{result['mean_epoch_minus1']:>10.4f} | {result['mean_best']:>10.4f}"
            )

    print("=" * 95)


if __name__ == "__main__":
    main()
