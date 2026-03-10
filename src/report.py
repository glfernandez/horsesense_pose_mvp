from __future__ import annotations

from pathlib import Path

import pandas as pd

from .features import summarize_metrics
from .utils import ensure_dir, stem_name


def qc_status(metrics_df: pd.DataFrame) -> str:
    pct_good = float((metrics_df["mean_likelihood"] > 0.6).mean())
    if pct_good >= 0.7:
        return "Green"
    if pct_good >= 0.4:
        return "Yellow"
    return "Red"


def generate_report(
    input_video: str,
    metrics_df: pd.DataFrame,
    output_dir: str,
    thresholds: tuple[float, float],
    fps: float | None = None,
    notes: list[str] | None = None,
) -> Path:
    out_dir = ensure_dir(output_dir)
    out_path = out_dir / f"{stem_name(input_video)}_report.md"
    summary = summarize_metrics(metrics_df)
    status = qc_status(metrics_df)
    t1, t2 = thresholds
    worst_window = worst_likelihood_window(metrics_df, fps=fps)

    notes = notes or []
    if not notes:
        notes = [
            "Metrics are movement/posture proxies and not indicators of stress, pain, or medical condition.",
            "Tracking quality is sensitive to angle, lighting, occlusion, and crop quality.",
        ]

    lines = [
        f"# Clip Report: {Path(input_video).name}",
        "",
        "## Summary",
        f"- QC status: **{status}**",
        f"- Frames: {summary['n_frames']}",
        f"- Mean move score: {summary['mean_move_score']:.3f}",
        f"- Mean likelihood: {summary['mean_likelihood']:.3f}",
        f"- % frames with mean_likelihood > 0.6: {summary['pct_frames_likelihood_gt_0_6']:.1f}%",
        f"- % frames with mean_likelihood < 0.2: {summary['pct_frames_likelihood_lt_0_2']:.1f}%",
        f"- Mean jitter proxy: {summary['mean_jitter']:.3f}",
        (
            f"- Worst 10s likelihood window: {worst_window['start_sec']:.1f}s-{worst_window['end_sec']:.1f}s "
            f"(avg={worst_window['mean_likelihood']:.3f})"
            if worst_window is not None
            else "- Worst 10s likelihood window: n/a"
        ),
        "",
        "## Activity State Share",
    ]
    for state, pct in summary["state_share_pct"].items():
        lines.append(f"- {state}: {pct:.1f}%")

    lines.extend(
        [
            "",
            "## Thresholds Used",
            f"- Standing: move_score < {t1}",
            f"- Walking: {t1} <= move_score < {t2}",
            f"- Active: move_score >= {t2}",
            "",
            "## Notes / Limitations",
        ]
    )
    lines.extend([f"- {n}" for n in notes])
    if summary["pct_frames_likelihood_lt_0_2"] > 50.0:
        lines.append("- Warning: Most frames have very low confidence (<0.2); movement/activity labels may be unreliable.")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def worst_likelihood_window(metrics_df: pd.DataFrame, fps: float | None = None, window_sec: float = 10.0) -> dict | None:
    if metrics_df.empty or "mean_likelihood" not in metrics_df.columns:
        return None
    if fps is None or fps <= 0:
        if "t_sec" in metrics_df.columns and len(metrics_df) >= 2:
            dt = float(metrics_df["t_sec"].iloc[1] - metrics_df["t_sec"].iloc[0])
            fps = (1.0 / dt) if dt > 0 else 30.0
        else:
            fps = 30.0
    window_n = max(1, int(round(window_sec * fps)))
    roll = metrics_df["mean_likelihood"].rolling(window=window_n, min_periods=1).mean()
    idx = int(roll.idxmin())
    start_idx = max(0, idx - window_n + 1)
    start_sec = float(metrics_df["t_sec"].iloc[start_idx]) if "t_sec" in metrics_df.columns else start_idx / fps
    end_sec = float(metrics_df["t_sec"].iloc[idx]) if "t_sec" in metrics_df.columns else idx / fps
    return {
        "start_sec": start_sec,
        "end_sec": end_sec,
        "mean_likelihood": float(roll.iloc[idx]),
    }
