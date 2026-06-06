#!/usr/bin/env python3
"""
TSV TTA record 분석 스크립트

output_tta_record 디렉토리 내 TSV 파일을 입력으로 받아 다음 정보를 제공합니다:
- target_class 수 및 1000 대비 등록 비율
- epoch별 (-1~19) backbone_acc 평균
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import pandas as pd


def get_tta_stats(tsv_path: str) -> Optional[dict]:
    """
    TSV 파일을 분석하여 target_class 수와 epoch별 backbone_acc 평균을 반환합니다.

    Returns:
        dict: {
            "num_classes": int,
            "pct": float,
            "epoch_means": dict[int, float]  # epoch -> mean backbone_acc
        }
        파일이 없거나 형식이 잘못된 경우 None 반환
    """
    path = Path(tsv_path)
    if not path.exists():
        return None

    df = pd.read_csv(path, sep="\t")

    if "target_class" not in df.columns or "backbone_acc" not in df.columns:
        return None

    unique_classes = df["target_class"].unique()
    num_classes = len(unique_classes)
    total_expected = 1000
    pct = (num_classes / total_expected) * 100

    expected_epochs = list(range(-1, 20))
    epoch_means = df.groupby("epoch")["backbone_acc"].mean()
    epoch_means_dict = {int(epoch): float(epoch_means[epoch]) for epoch in expected_epochs if epoch in epoch_means.index}

    return {
        "num_classes": num_classes,
        "pct": pct,
        "epoch_means": epoch_means_dict,
    }


def analyze_tta_record(tsv_path: str) -> None:
    """TSV 파일을 분석하여 target_class 수와 epoch별 backbone_acc 평균을 출력합니다."""
    path = Path(tsv_path)
    stats = get_tta_stats(tsv_path)
    if stats is None:
        if not path.exists():
            print(f"Error: 파일을 찾을 수 없습니다: {tsv_path}")
        else:
            print("Error: TSV 파일에 target_class 또는 backbone_acc 컬럼이 없습니다.")
        sys.exit(1)

    num_classes = stats["num_classes"]
    pct = stats["pct"]
    epoch_means = stats["epoch_means"]
    expected_epochs = list(range(-1, 20))

    print("=" * 60)
    print(f"파일: {path.name}")
    print("=" * 60)
    print(f"\n[Target Class 현황]")
    print(f"  등록된 target_class 수: {num_classes:,} / 1,000 ({pct:.2f}%)")

    print(f"\n[Epoch별 Backbone Acc 평균] (총 {len(expected_epochs)}개 epoch)")
    print("-" * 40)
    print(f"{'Epoch':>6} | {'backbone_acc (평균)':>20}")
    print("-" * 40)

    for epoch in expected_epochs:
        if epoch in epoch_means:
            print(f"{epoch:>6} | {epoch_means[epoch]:>20.6f}")
        else:
            print(f"{epoch:>6} | {'(데이터 없음)':>20}")

    print("-" * 40)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TTA record TSV 파일 분석 - target_class 수 및 epoch별 backbone_acc 평균"
    )
    parser.add_argument(
        "tsv_file",
        nargs="?",
        type=str,
        metavar="TSV_FILE",
        help="분석할 TSV 파일 경로",
    )
    parser.add_argument(
        "--tsv_file",
        type=str,
        dest="tsv_file_opt",
        help="분석할 TSV 파일 경로 (위치 인자와 동일)",
    )
    args = parser.parse_args()
    tsv_path = args.tsv_file_opt or args.tsv_file
    if not tsv_path:
        parser.error("tsv_file 또는 --tsv_file 인자가 필요합니다.")
    analyze_tta_record(tsv_path)


if __name__ == "__main__":
    main()
