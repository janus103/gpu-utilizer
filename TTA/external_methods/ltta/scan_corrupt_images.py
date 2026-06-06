#!/usr/bin/env python3
"""
Scan a dataset folder and list corrupted / unreadable images.

This script is intentionally standalone: it does NOT modify any training code or dataset files.

It detects issues that commonly crash PIL/Pillow during training, e.g.:
  - "image file is truncated (... bytes not processed)"
  - "cannot identify image file"
  - decompression errors, etc.

Examples
--------
# Scan only ImageNet val split and write a list of bad images:
python3 scan_corrupt_images.py --root /home/oem/jin/datasets/imagenet --splits val --output bad_val.txt

# Scan train+val with 8 workers:
python3 scan_corrupt_images.py --root /home/oem/jin/datasets/imagenet --splits train val -j 8 --output bad_all.txt

# Quick smoke run (first 200 files only):
python3 scan_corrupt_images.py --root /home/oem/jin/datasets/imagenet --splits val --limit 200
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple


def _default_exts() -> List[str]:
    # Common ImageNet-style extensions
    return [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


def _iter_image_files(root: str, exts: Sequence[str], followlinks: bool) -> Iterator[str]:
    exts_lc = {e.lower() for e in exts} if exts else set()
    for dirpath, _, filenames in os.walk(root, followlinks=followlinks):
        for fn in filenames:
            ext = os.path.splitext(fn)[1].lower()
            if exts_lc and ext not in exts_lc:
                continue
            yield os.path.join(dirpath, fn)


def _fmt_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds/60:.1f}m"
    return f"{seconds/3600:.2f}h"


@dataclass(frozen=True)
class CheckResult:
    ok: bool
    error: str = ""


def _check_one_image(path: str, convert_mode: str, strict: bool) -> CheckResult:
    """
    Returns ok=True if image can be opened and fully decoded.
    If strict=True, forces a full decode + convert() to match typical training pipelines.
    """
    try:
        # Import PIL in worker process (keeps main process lightweight).
        from PIL import Image, ImageFile

        # We want to DETECT truncated images (same behavior that crashes training),
        # so do NOT allow truncated images to load silently.
        ImageFile.LOAD_TRUNCATED_IMAGES = False

        # Fast fail on empty files.
        try:
            if os.path.getsize(path) == 0:
                return CheckResult(False, "empty file (0 bytes)")
        except OSError as e:
            return CheckResult(False, f"oserror: {e}")

        # Pass 1: verify structure quickly (doesn't always decode full image).
        with Image.open(path) as im:
            im.verify()

        if strict:
            # Pass 2: reopen and force full decode (catches truncated streams).
            with Image.open(path) as im:
                if convert_mode:
                    im = im.convert(convert_mode)
                im.load()

        return CheckResult(True, "")
    except Exception as e:
        return CheckResult(False, f"{type(e).__name__}: {e}")


def _expand_scan_roots(root: str, splits: Optional[Sequence[str]]) -> List[str]:
    if not splits:
        return [root]
    roots: List[str] = []
    for s in splits:
        p = s if os.path.isabs(s) else os.path.join(root, s)
        roots.append(p)
    return roots


def _validate_roots(roots: Sequence[str]) -> None:
    bad = [p for p in roots if not os.path.isdir(p)]
    if bad:
        joined = "\n".join(f"- {p}" for p in bad)
        raise SystemExit(f"Invalid scan root(s) (not a directory):\n{joined}")


def _write_list(path: str, bad_paths: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for p in bad_paths:
            f.write(p)
            f.write("\n")


def scan_dataset(
    roots: Sequence[str],
    exts: Sequence[str],
    workers: int,
    max_inflight: int,
    limit: int,
    convert_mode: str,
    strict: bool,
    followlinks: bool,
    log_every: int,
) -> Tuple[List[str], int]:
    """
    Returns (bad_paths, processed_count).
    """
    file_iter: Iterator[str]
    # Chain roots without building a giant list (ImageNet train can be ~1.3M files).
    def chained() -> Iterator[str]:
        for r in roots:
            yield from _iter_image_files(r, exts=exts, followlinks=followlinks)

    file_iter = chained()

    bad_paths: List[str] = []
    processed = 0
    started = time.time()

    inflight = set()
    future_to_path = {}

    with ProcessPoolExecutor(max_workers=workers) as ex:
        try:
            for path in file_iter:
                if limit and processed >= limit:
                    break
                fut = ex.submit(_check_one_image, path, convert_mode, strict)
                inflight.add(fut)
                future_to_path[fut] = path

                # Throttle in-flight futures to keep memory stable.
                if len(inflight) >= max_inflight:
                    done, not_done = wait(inflight, return_when=FIRST_COMPLETED)
                    inflight = not_done
                    for f in done:
                        processed += 1
                        res = f.result()
                        if not res.ok:
                            bad_paths.append(future_to_path.get(f, "<unknown>"))
                        future_to_path.pop(f, None)

                        if log_every and processed % log_every == 0:
                            elapsed = time.time() - started
                            rate = processed / max(elapsed, 1e-6)
                            print(
                                f"[scan] processed={processed} bad={len(bad_paths)} "
                                f"elapsed={_fmt_seconds(elapsed)} rate={rate:.1f} img/s",
                                flush=True,
                            )

            # Drain remaining futures.
            while inflight:
                done, not_done = wait(inflight, return_when=FIRST_COMPLETED)
                inflight = not_done
                for f in done:
                    processed += 1
                    res = f.result()
                    if not res.ok:
                        bad_paths.append(future_to_path.get(f, "<unknown>"))
                    future_to_path.pop(f, None)

                    if log_every and processed % log_every == 0:
                        elapsed = time.time() - started
                        rate = processed / max(elapsed, 1e-6)
                        print(
                            f"[scan] processed={processed} bad={len(bad_paths)} "
                            f"elapsed={_fmt_seconds(elapsed)} rate={rate:.1f} img/s",
                            flush=True,
                        )
        except KeyboardInterrupt:
            print("\n[scan] interrupted by user (KeyboardInterrupt). Draining in-flight work...", file=sys.stderr)
            # Best-effort drain quickly (still returns partial results).
            for f in inflight:
                f.cancel()

    return bad_paths, processed


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scan dataset for corrupted images (standalone).")
    p.add_argument("--root", required=True, type=str, help="Dataset root directory.")
    p.add_argument(
        "--splits",
        nargs="*",
        default=None,
        help='Optional subfolders under --root to scan (e.g. "train val"). If omitted, scans --root recursively.',
    )
    p.add_argument(
        "--ext",
        nargs="*",
        default=_default_exts(),
        help='File extensions to include (default: common image extensions). Example: --ext .jpg .jpeg',
    )
    p.add_argument("-j", "--workers", type=int, default=max(1, (os.cpu_count() or 8) // 2), help="Process workers.")
    p.add_argument(
        "--max-inflight",
        type=int,
        default=512,
        help="Max number of in-flight tasks (limits memory). Increase for more throughput.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Stop after N processed files (0 = no limit). Useful for quick smoke tests.",
    )
    p.add_argument(
        "--mode",
        type=str,
        default="RGB",
        help='PIL convert mode used during strict check (default: "RGB"). Use "" to skip convert.',
    )
    p.add_argument(
        "--no-strict",
        action="store_true",
        default=False,
        help="Disable strict full decode + convert check (faster but may miss some truncated cases).",
    )
    p.add_argument("--followlinks", action="store_true", default=False, help="Follow symlinks in os.walk.")
    p.add_argument("--log-every", type=int, default=2000, help="Print progress every N files (0 disables).")
    p.add_argument(
        "--output",
        type=str,
        default="corrupt_images.txt",
        help="Output file path (one bad image path per line).",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    roots = _expand_scan_roots(args.root, args.splits)
    _validate_roots(roots)

    strict = not args.no_strict
    started = time.time()
    print(
        f"[scan] roots={roots} workers={args.workers} strict={strict} mode={args.mode!r} "
        f"ext={args.ext} limit={args.limit or 'none'}",
        flush=True,
    )

    bad_paths, processed = scan_dataset(
        roots=roots,
        exts=args.ext,
        workers=args.workers,
        max_inflight=max(1, args.max_inflight),
        limit=max(0, args.limit),
        convert_mode=args.mode,
        strict=strict,
        followlinks=args.followlinks,
        log_every=max(0, args.log_every),
    )

    _write_list(args.output, bad_paths)
    elapsed = time.time() - started

    print(
        f"[scan] done. processed={processed} bad={len(bad_paths)} "
        f"elapsed={_fmt_seconds(elapsed)} output={os.path.abspath(args.output)}",
        flush=True,
    )
    if bad_paths:
        print("[scan] first few bad files:", flush=True)
        for p in bad_paths[:10]:
            print(f" - {p}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


