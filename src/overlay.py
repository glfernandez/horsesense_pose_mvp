from __future__ import annotations

from pathlib import Path

import cv2
import pandas as pd

from .utils import ensure_dir, stem_name


def render_overlay_video(
    input_video: str,
    keypoints_df: pd.DataFrame,
    output_dir: str,
    metrics_df: pd.DataFrame | None = None,
    point_radius: int = 3,
) -> Path:
    """Render a simple keypoint overlay if flat keypoint columns are present.

    Expected columns: <keypoint>_x, <keypoint>_y, <keypoint>_likelihood
    """
    video_path = Path(input_video)
    out_dir = ensure_dir(output_dir)
    out_path = out_dir / f"{stem_name(video_path)}_overlay.mp4"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )

    x_cols = [c for c in keypoints_df.columns if c.endswith("_x")]
    skeleton_edges = _build_skeleton_edges(x_cols)
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx < len(keypoints_df):
            row = keypoints_df.iloc[frame_idx]
            # Draw skeleton lines first.
            for a_base, b_base in skeleton_edges:
                ax_col = f"{a_base}_x"
                ay_col = f"{a_base}_y"
                al_col = f"{a_base}_likelihood"
                bx_col = f"{b_base}_x"
                by_col = f"{b_base}_y"
                bl_col = f"{b_base}_likelihood"
                if (
                    ax_col not in keypoints_df.columns
                    or ay_col not in keypoints_df.columns
                    or bx_col not in keypoints_df.columns
                    or by_col not in keypoints_df.columns
                ):
                    continue
                ax = row.get(ax_col)
                ay = row.get(ay_col)
                bx = row.get(bx_col)
                by = row.get(by_col)
                al = float(row.get(al_col, 1.0))
                bl = float(row.get(bl_col, 1.0))
                if (
                    pd.isna(ax)
                    or pd.isna(ay)
                    or pd.isna(bx)
                    or pd.isna(by)
                    or al < 0.2
                    or bl < 0.2
                ):
                    continue
                cv2.line(
                    frame,
                    (int(ax), int(ay)),
                    (int(bx), int(by)),
                    (0, 200, 255),
                    2,
                )

            for x_col in x_cols:
                base = x_col[:-2]
                y_col = f"{base}_y"
                l_col = f"{base}_likelihood"
                if y_col not in keypoints_df.columns:
                    continue
                x = row.get(x_col)
                y = row.get(y_col)
                likelihood = float(row.get(l_col, 1.0))
                if pd.isna(x) or pd.isna(y) or likelihood < 0.2:
                    continue
                cv2.circle(frame, (int(x), int(y)), point_radius, (0, 255, 0), -1)
            cv2.putText(
                frame,
                "H1",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                f"frame {frame_idx}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            if metrics_df is not None and frame_idx < len(metrics_df):
                mrow = metrics_df.iloc[frame_idx]
                state = str(mrow.get("state", ""))
                t_sec = float(mrow.get("t_sec", frame_idx / fps))
                cv2.putText(
                    frame,
                    f"t={t_sec:0.2f}s  state={state}",
                    (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )
                if state == "Unknown":
                    cv2.rectangle(frame, (8, 88), (width - 8, 120), (0, 0, 180), -1)
                    cv2.putText(
                        frame,
                        "LOW CONFIDENCE - RECALIBRATE CAMERA",
                        (14, 111),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (255, 255, 255),
                        2,
                    )
        writer.write(frame)
        frame_idx += 1

    writer.release()
    cap.release()
    return out_path


def _build_skeleton_edges(x_cols: list[str]) -> list[tuple[str, str]]:
    bases = [c[:-2] for c in x_cols]
    base_set = set(bases)
    nmap = {b.lower().replace("-", "_"): b for b in bases}

    def pick(*aliases: str) -> str | None:
        for a in aliases:
            key = a.lower().replace("-", "_")
            if key in nmap:
                return nmap[key]
        return None

    named_pairs = [
        (pick("nose"), pick("head", "forehead", "poll")),
        (pick("head", "forehead", "poll"), pick("neck")),
        (pick("neck"), pick("withers", "spine1")),
        (pick("withers", "spine1"), pick("mid_back", "spine2", "back")),
        (pick("mid_back", "spine2", "back"), pick("croup", "hip", "pelvis")),
        (pick("croup", "hip", "pelvis"), pick("tail_base", "tail")),
        (pick("shoulder", "left_shoulder", "l_shoulder"), pick("elbow", "left_elbow", "l_elbow")),
        (pick("elbow", "left_elbow", "l_elbow"), pick("knee", "left_knee", "l_knee", "carpus")),
        (pick("knee", "left_knee", "l_knee", "carpus"), pick("hoof_front", "left_hoof_front", "l_front_hoof")),
        (pick("hip", "left_hip", "l_hip"), pick("stifle", "left_stifle", "l_stifle")),
        (pick("stifle", "left_stifle", "l_stifle"), pick("hock", "left_hock", "l_hock")),
        (pick("hock", "left_hock", "l_hock"), pick("hoof_hind", "left_hoof_hind", "l_hind_hoof")),
    ]
    edges = [(a, b) for a, b in named_pairs if a and b and a in base_set and b in base_set]
    if edges:
        return edges

    # Fallback: connect consecutive keypoints in column order.
    if len(bases) >= 2:
        return [(bases[i], bases[i + 1]) for i in range(len(bases) - 1)]
    return []
