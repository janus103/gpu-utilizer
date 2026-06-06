from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List


SUMMARY_COLUMNS = [
    "status",
    "failure_reason",
    "model",
    "algorithm",
    "measurement",
    "batch_size",
    "requested_batch_size",
    "num_samples",
    "num_batches",
    "latency_ms_per_iter",
    "latency_ms_per_sample",
    "estimated_sm_cycles_per_iter",
    "estimated_sm_cycles_per_sample",
    "energy_j_per_iter",
    "energy_j_per_sample",
    "energy_j_per_iter_est_from_power",
    "energy_j_per_sample_est_from_power",
    "avg_power_w_from_energy",
    "avg_power_w_sampled",
    "estimated_gops",
    "estimated_gops_per_w",
    "energy_source",
    "gops_count_source",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize TTA profile CSV files.")
    parser.add_argument("csv", nargs="+", help="Input TTA result CSV files.")
    parser.add_argument("--output", default="", help="Optional output summary CSV.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = list(load_rows(args.csv))
    rows.sort(key=lambda row: (row.get("model", ""), int(float(row.get("batch_size", "0") or 0)), row.get("algorithm", "")))
    if args.output:
        write_rows(Path(args.output), rows)
        print(f"wrote summary: {args.output}")
    else:
        print_summary(rows)


def load_rows(paths: Iterable[str]) -> Iterable[Dict[str, str]]:
    for path_value in paths:
        path = Path(path_value)
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                normalized = {column: row.get(column, "") for column in SUMMARY_COLUMNS}
                if not normalized["batch_size"]:
                    normalized["batch_size"] = row.get("requested_batch_size", "")
                yield normalized


def write_rows(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: List[Dict[str, str]]) -> None:
    for row in rows:
        latency = row["latency_ms_per_sample"] or row["latency_ms_per_iter"]
        cycles = row["estimated_sm_cycles_per_sample"] or row["estimated_sm_cycles_per_iter"]
        energy = (
            row["energy_j_per_sample"]
            or row["energy_j_per_iter"]
            or row["energy_j_per_sample_est_from_power"]
            or row["energy_j_per_iter_est_from_power"]
        )
        print(
            f"{row['model']:24s} bs={row['batch_size']:>3s} "
            f"{row['algorithm']:>5s} status={row['status'] or 'NA':>5s} "
            f"reason={row['failure_reason'] or '-':>4s} samples={row['num_samples'] or '-':>6s} "
            f"latency_ms_per_sample={fmt(latency)} "
            f"cycles_per_sample={fmt(cycles)} "
            f"energy_j_per_sample={fmt(energy)} "
            f"gops={fmt(row['estimated_gops'])} gops_per_w={fmt(row['estimated_gops_per_w'])}"
        )


def fmt(value: str) -> str:
    if value == "":
        return "NA"
    try:
        return f"{float(value):.4g}"
    except ValueError:
        return value


if __name__ == "__main__":
    main()
