from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


LIKELIHOOD_SUFFIX = "_likelihood"
X_SUFFIX = "_x"
Y_SUFFIX = "_y"


def _keypoint_bases(columns: Iterable[str]) -> list[str]:
    cols = list(columns)
    bases = []
    for col in cols:
        if col.endswith(X_SUFFIX):
            base = col[: -len(X_SUFFIX)]
            if f"{base}{Y_SUFFIX}" in cols and f"{base}{LIKELIHOOD_SUFFIX}" in cols:
                bases.append(base)
    return sorted(set(bases))


def load_keypoints_csv(keypoints_csv: str | Path) -> pd.DataFrame:
    path = Path(keypoints_csv)
    df = _read_flat_or_dlc_multilevel_csv(path)
    if "frame" not in df.columns:
        df.insert(0, "frame", np.arange(len(df), dtype=int))
    return df


def compute_metrics(
    keypoints_df: pd.DataFrame,
    fps: float,
    t1: float = 0.8,
    t2: float = 2.0,
    min_confidence_for_state: float = 0.2,
) -> pd.DataFrame:
    if fps <= 0:
        raise ValueError("fps must be > 0")

    df = keypoints_df.copy()
    bases = _keypoint_bases(df.columns)
    if not bases:
        raise ValueError(
            "No keypoint columns found. Expected flat columns like head_x, head_y, head_likelihood"
        )

    x_cols = [f"{b}{X_SUFFIX}" for b in bases]
    y_cols = [f"{b}{Y_SUFFIX}" for b in bases]
    l_cols = [f"{b}{LIKELIHOOD_SUFFIX}" for b in bases]

    x = df[x_cols].astype(float)
    y = df[y_cols].astype(float)
    likelihood = df[l_cols].astype(float).clip(lower=0.0, upper=1.0)
    valid_mask = likelihood >= 0.2
    x_masked = x.where(valid_mask)
    y_masked = y.where(valid_mask)

    centroid_x = x_masked.mean(axis=1).fillna(x.mean(axis=1))
    centroid_y = y_masked.mean(axis=1).fillna(y.mean(axis=1))
    centroid_dx = centroid_x.diff().fillna(0.0)
    centroid_dy = centroid_y.diff().fillna(0.0)
    move_score = np.sqrt(centroid_dx**2 + centroid_dy**2)

    head_candidates = [b for b in bases if "head" in b.lower() or "nose" in b.lower() or "neck" in b.lower()]
    head_col = f"{(head_candidates[0] if head_candidates else bases[0])}{Y_SUFFIX}"
    frame_height_proxy = y.max(axis=1).replace(0, np.nan)
    head_height = 1.0 - (df[head_col].astype(float) / frame_height_proxy).fillna(0.0)
    head_height = head_height.clip(lower=-2.0, upper=2.0)

    leg_candidates = [b for b in bases if any(token in b.lower() for token in ("hoof", "paw", "ankle", "hock", "fetlock", "leg"))]
    stride_source = y[f"{leg_candidates[0]}{Y_SUFFIX}"] if leg_candidates else centroid_y
    stride_rhythm = _rolling_periodicity_proxy(pd.Series(stride_source, dtype=float), window=30)

    mean_likelihood = likelihood.mean(axis=1)
    jitter = _mean_keypoint_displacement(x_masked, y_masked)

    metrics = pd.DataFrame(
        {
            "frame": df["frame"].astype(int),
            "t_sec": df["frame"].astype(float) / float(fps),
            "move_score": move_score.astype(float),
            "head_height": head_height.astype(float),
            "stride_rhythm": stride_rhythm.astype(float),
            "mean_likelihood": mean_likelihood.astype(float),
            "jitter": jitter.astype(float),
        }
    )
    metrics["state"] = [
        classify_state(move_score=ms, mean_likelihood=ml, t1=t1, t2=t2, min_confidence_for_state=min_confidence_for_state)
        for ms, ml in zip(metrics["move_score"], metrics["mean_likelihood"])
    ]
    return metrics


def classify_state(
    move_score: float,
    mean_likelihood: float,
    t1: float,
    t2: float,
    min_confidence_for_state: float = 0.2,
) -> str:
    if mean_likelihood < min_confidence_for_state:
        return "Unknown"
    if move_score < t1:
        return "Standing"
    if move_score < t2:
        return "Walking"
    return "Active"


def summarize_metrics(metrics_df: pd.DataFrame) -> dict:
    state_share = (
        metrics_df["state"].value_counts(normalize=True).rename_axis("state").mul(100).round(1).to_dict()
    )
    return {
        "n_frames": int(len(metrics_df)),
        "mean_move_score": float(metrics_df["move_score"].mean()),
        "mean_likelihood": float(metrics_df["mean_likelihood"].mean()),
        "pct_frames_likelihood_gt_0_6": float((metrics_df["mean_likelihood"] > 0.6).mean() * 100.0),
        "pct_frames_likelihood_lt_0_2": float((metrics_df["mean_likelihood"] < 0.2).mean() * 100.0),
        "mean_jitter": float(metrics_df["jitter"].mean()),
        "state_share_pct": state_share,
    }


def _mean_keypoint_displacement(x: pd.DataFrame, y: pd.DataFrame) -> pd.Series:
    dx = x.diff().fillna(0.0)
    dy = y.diff().fillna(0.0)
    dist = np.sqrt(dx.pow(2) + dy.pow(2))
    return dist.mean(axis=1)


def _rolling_periodicity_proxy(signal: pd.Series, window: int = 30) -> pd.Series:
    arr = signal.ffill().bfill().to_numpy(dtype=float)
    if len(arr) == 0:
        return pd.Series(dtype=float)
    out = np.zeros_like(arr, dtype=float)
    for i in range(len(arr)):
        start = max(0, i - window + 1)
        segment = arr[start : i + 1]
        if len(segment) < 4:
            out[i] = 0.0
            continue
        centered = segment - float(np.mean(segment))
        if np.allclose(centered, 0):
            out[i] = 0.0
            continue
        ac = np.correlate(centered, centered, mode="full")
        ac = ac[len(ac) // 2 :]
        if len(ac) < 3 or ac[0] == 0:
            out[i] = 0.0
            continue
        out[i] = float(np.max(ac[1:]) / ac[0])
    return pd.Series(out)


def _read_flat_or_dlc_multilevel_csv(path: Path) -> pd.DataFrame:
    # Try common DLC export first: 3-row header (scorer, bodypart, coord)
    try:
        multi = pd.read_csv(path, header=[0, 1, 2])
        flattened = _flatten_dlc_multiindex_columns(multi)
        if _keypoint_bases(flattened.columns):
            return flattened
    except Exception:
        pass

    # Fallback to ordinary CSV (already flat fixture or custom export)
    flat = pd.read_csv(path)
    if isinstance(flat.columns, pd.MultiIndex):
        flat.columns = ["_".join(str(x) for x in tup if str(x) != "nan") for tup in flat.columns]
    return flat


def _flatten_dlc_multiindex_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    unnamed_cols = []
    for col in df.columns:
        if not isinstance(col, tuple) or len(col) < 3:
            unnamed_cols.append(col)
            continue
        level0, level1, level2 = [str(c) for c in col[:3]]
        if level1.startswith("Unnamed"):
            unnamed_cols.append(col)
            continue
        coord = level2.strip().lower()
        if coord not in {"x", "y", "likelihood"}:
            continue
        out[f"{level1}_{coord}"] = pd.to_numeric(df[col], errors="coerce")

    if unnamed_cols:
        first = unnamed_cols[0]
        try:
            out.insert(0, "frame", pd.to_numeric(df[first], errors="coerce").ffill().fillna(0).astype(int))
        except Exception:
            pass
    return out
