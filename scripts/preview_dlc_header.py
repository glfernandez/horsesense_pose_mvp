#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Save a preview of the newest DLC CSV header")
    p.add_argument("--video_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_lines", type=int, default=40)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    video_path = Path(args.video_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_path = output_dir / "dlc_header_preview.txt"

    candidates = find_csv_candidates(video_path, output_dir)
    if not candidates:
        print("[WARN] No CSV candidates found for header preview")
        return 1

    newest = candidates[0]
    print(f"[INFO] Header preview source: {newest}")
    lines = []
    with newest.open("r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if i >= args.max_lines:
                break
            lines.append(line.rstrip("\n"))

    content = [f"# source: {newest}", f"# generated_at_epoch: {time.time():.0f}", ""] + lines
    preview_path.write_text("\n".join(content) + "\n", encoding="utf-8")
    print(f"[OK] Wrote header preview: {preview_path}")
    return 0


def find_csv_candidates(video_path: Path, output_dir: Path) -> list[Path]:
    stem = video_path.stem
    search_roots = [output_dir, video_path.parent]
    seen: set[Path] = set()
    found: list[Path] = []
    patterns = [f"{stem}*DLC*.csv", f"{stem}*.csv", "*.csv"]
    for root in search_roots:
        if not root.exists():
            continue
        for pattern in patterns:
            for p in root.glob(pattern):
                rp = p.resolve()
                if rp in seen or not p.is_file():
                    continue
                seen.add(rp)
                found.append(rp)
        for p in root.rglob(f"{stem}*.csv"):
            rp = p.resolve()
            if rp in seen or not p.is_file():
                continue
            seen.add(rp)
            found.append(rp)
    filtered = [p for p in found if not _is_pipeline_metrics_csv(p)]
    return sorted(filtered, key=_candidate_sort_key, reverse=True)


def _is_pipeline_metrics_csv(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith("_metrics.csv") or name == "metrics.csv"


def _candidate_sort_key(path: Path) -> tuple[int, float]:
    name = path.name.lower()
    priority = 0
    if "dlc" in name:
        priority = 3
    elif "keypoints" in name:
        priority = 2
    elif name.endswith(".csv"):
        priority = 1
    return (priority, path.stat().st_mtime)


if __name__ == "__main__":
    raise SystemExit(main())
