#!/usr/bin/env python3
"""
TTA record TSV 파일들을 이미지 첨부 corruption 순서대로 하나의 테이블로 출력합니다.

Corruption 순서 (이미지 기준):
- Noise: Gaussian, Shot, Impulse
- Blur: Defocus, Glass, Motion, Zoom
- Weather: Snow, Frost, Fog, Brightness
- Digital: Contrast, Elastic, Pixelate, JPEG
"""

import argparse
from pathlib import Path

from analyze_tta_record import get_tta_stats

# 이미지 첨부 corruption 순서 (카테고리별)
CORRUPTION_ORDER = [
    # Noise
    ("Noise", "Gaussian", "gaussian_noise"),
    ("Noise", "Shot", "shot_noise"),
    ("Noise", "Impulse", "impulse_noise"),
    # Blur
    ("Blur", "Defocus", "defocus_blur"),
    ("Blur", "Glass", "glass_blur"),
    ("Blur", "Motion", "motion_blur"),
    ("Blur", "Zoom", "zoom_blur"),
    # Weather
    ("Weather", "Snow", "snow"),
    ("Weather", "Frost", "frost"),
    ("Weather", "Fog", "fog"),
    ("Weather", "Brightness", "brightness"),
    # Digital
    ("Digital", "Contrast", "contrast"),
    ("Digital", "Elastic", "elastic_transform"),
    ("Digital", "Pixelate", "pixelate"),
    ("Digital", "JPEG", "jpeg_compression"),
]

EXPECTED_EPOCHS = list(range(-1, 20))
TSV_PATTERN = "{prefix}_{corruption}_5_l2_ep20.tsv"


def build_table(record_dir: Path, tsv_prefix: str = "tta_record") -> list:
    """모든 corruption에 대해 분석 결과를 수집합니다."""
    rows = []
    for category, display_name, corruption_key in CORRUPTION_ORDER:
        tsv_name = TSV_PATTERN.format(prefix=tsv_prefix, corruption=corruption_key)
        tsv_path = record_dir / tsv_name

        stats = get_tta_stats(str(tsv_path))
        if stats is None:
            row = {
                "category": category,
                "corruption": display_name,
                "num_classes": "-",
                "pct": "-",
                **{f"E{e}": "-" for e in EXPECTED_EPOCHS},
            }
        else:
            epoch_vals = {
                f"E{e}": f"{stats['epoch_means'].get(e, 0):.4f}"
                for e in EXPECTED_EPOCHS
            }
            row = {
                "category": category,
                "corruption": display_name,
                "num_classes": stats["num_classes"],
                "pct": f"{stats['pct']:.2f}%",
                **epoch_vals,
            }
        rows.append(row)
    return rows


def print_table(rows: list[dict]) -> None:
    """테이블을 콘솔에 출력합니다."""
    # 헤더
    epoch_headers = [f"E{e}" for e in EXPECTED_EPOCHS]
    headers = ["Category", "Corruption", "Target", "%"] + epoch_headers

    # 컬럼 너비 계산
    col_widths = [max(8, len(h)) for h in headers[:4]]
    for i, e in enumerate(EXPECTED_EPOCHS):
        col_widths.append(max(7, len(f"E{e}")))

    def fmt_row(values: list, widths: list[int]) -> str:
        return " | ".join(str(v).rjust(w) for v, w in zip(values, widths))

    # 구분선
    total_width = sum(col_widths) + 3 * (len(col_widths) - 1)
    sep = "=" * total_width
    thin_sep = "-" * total_width

    print(sep)
    print("TTA Record 분석 - Corruption별 통합 테이블")
    print(sep)
    print(fmt_row(headers, col_widths))
    print(thin_sep)

    for row in rows:
        values = [
            row["category"],
            row["corruption"],
            row["num_classes"],
            row["pct"],
        ] + [row[f"E{e}"] for e in EXPECTED_EPOCHS]
        print(fmt_row(values, col_widths))

    print(sep)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TTA record TSV 파일들을 corruption 순서대로 통합 테이블로 출력"
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
        help="TTA record TSV 파일이 있는 디렉토리 (기본: output_tta_record)",
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

    rows = build_table(record_dir, tsv_prefix=args.tsv_prefix)
    print_table(rows)


if __name__ == "__main__":
    main()
