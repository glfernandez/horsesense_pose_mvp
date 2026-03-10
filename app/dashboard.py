from __future__ import annotations

from pathlib import Path
import base64
from datetime import datetime
import re
import subprocess

import cv2
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = ROOT / "outputs"
MAX_VIDEO_WIDTH_PX = 1400

STATE_COLORS = {
    "Green": "#2e7d32",
    "Yellow": "#f9a825",
    "Red": "#c62828",
    "Standing": "#4caf50",
    "Walking": "#fb8c00",
    "Active": "#e53935",
    "Unknown": "#757575",
}


def main() -> None:
    st.set_page_config(page_title="HorseSense Dashboard", layout="wide")
    inject_video_css()
    st.title("HorseSense Pose MVP Dashboard")
    st.caption("Review overlay video, QC status, metrics, and report outputs for sample and Jack clips.")

    st.sidebar.markdown("### Display")
    video_width_px = st.sidebar.slider("Video width (px)", min_value=640, max_value=1800, value=1400, step=40)
    st.session_state["video_width_px"] = video_width_px
    show_demo_runs = st.sidebar.checkbox("Show demo runs", value=False)

    render_upload_panel()

    runs = discover_runs(OUTPUTS_DIR, show_demo_runs=show_demo_runs)
    if not runs:
        st.warning(f"No runs found under {OUTPUTS_DIR}")
        st.info("Generate outputs first with `run_pose_pipeline.py`.")
        return

    labels = [r["label"] for r in runs]
    selected_label = st.sidebar.selectbox("Select run", labels, index=0)
    run = next(r for r in runs if r["label"] == selected_label)

    st.sidebar.markdown("### Files")
    st.sidebar.code("\n".join([str(p) for p in run.values() if isinstance(p, Path)]), language="text")

    metrics_df = pd.read_csv(run["metrics_csv"])
    qc = qc_status(metrics_df)
    pct_good = float((metrics_df["mean_likelihood"] >= 0.6).mean() * 100.0) if "mean_likelihood" in metrics_df else 0.0
    pct_low = float((metrics_df["mean_likelihood"] < 0.2).mean() * 100.0) if "mean_likelihood" in metrics_df else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("QC Status", qc)
    c2.metric("Frames", f"{len(metrics_df)}")
    c3.metric("High-Confidence Frames", f"{pct_good:.1f}%")
    c4.metric("Low-Confidence Frames", f"{pct_low:.1f}%")

    st.subheader("Before / After Video")
    original_tab, overlay_tab = st.tabs(["Before (Original)", "After (Overlay)"])
    with original_tab:
        if run.get("source_video") and run["source_video"].exists():
            render_video(run["source_video"])
        else:
            st.info("Original source video not found automatically for this run.")
    with overlay_tab:
        if run.get("overlay_mp4") and run["overlay_mp4"].exists():
            render_video(run["overlay_mp4"])
        else:
            st.info("No overlay video found for this run.")

    st.subheader("Report")
    if run.get("report_md") and run["report_md"].exists():
        st.markdown(run["report_md"].read_text(encoding="utf-8", errors="replace"))
    else:
        st.info("No report markdown found.")

    st.subheader("Metrics")
    metric_options = [c for c in ["move_score", "head_height", "stride_rhythm", "mean_likelihood", "jitter"] if c in metrics_df.columns]
    if metric_options:
        selected_metrics = st.multiselect("Series", metric_options, default=[c for c in ["move_score", "mean_likelihood"] if c in metric_options])
        if selected_metrics:
            plot_df = metrics_df[["t_sec", *selected_metrics]].copy() if "t_sec" in metrics_df else metrics_df[selected_metrics].copy()
            if "t_sec" in plot_df.columns:
                plot_df = plot_df.set_index("t_sec")
            st.line_chart(plot_df)

    if "state" in metrics_df.columns:
        st.subheader("State Timeline")
        timeline_df = build_state_segments(metrics_df)
        if not timeline_df.empty:
            st.dataframe(timeline_df, use_container_width=True, hide_index=True)
            st.bar_chart(
                metrics_df["state"].value_counts().reindex(["Standing", "Walking", "Active", "Unknown"]).fillna(0),
                color="#4e79a7",
            )

    with st.expander("Raw Metrics Preview"):
        st.dataframe(metrics_df.head(200), use_container_width=True)

    with st.expander("Header Preview (if available)"):
        header_preview = run.get("header_preview_txt")
        if header_preview and header_preview.exists():
            st.code(header_preview.read_text(encoding="utf-8", errors="replace"), language="text")
        else:
            st.info("No `dlc_header_preview.txt` found.")


def discover_runs(outputs_dir: Path, show_demo_runs: bool = False) -> list[dict]:
    runs: list[dict] = []
    for metrics_csv in sorted(outputs_dir.rglob("*_metrics.csv")):
        stem = metrics_csv.name[: -len("_metrics.csv")]
        parent = metrics_csv.parent
        rel_parent = str(parent.relative_to(outputs_dir))
        if not show_demo_runs and rel_parent.startswith("demo"):
            continue
        overlay_mp4 = parent / f"{stem}_overlay.mp4"
        source_video = find_source_video(stem)
        duration_sec = infer_run_duration_sec(metrics_csv=metrics_csv, overlay_mp4=overlay_mp4, source_video=source_video)
        folder_label = rel_parent
        duration_label = f"{duration_sec:.1f}s" if duration_sec is not None else "n/a"
        run = {
            "label": f"{folder_label} / {stem} ({duration_label})",
            "metrics_csv": metrics_csv,
            "overlay_mp4": overlay_mp4,
            "report_md": parent / f"{stem}_report.md",
            "header_preview_txt": parent / "dlc_header_preview.txt",
            "source_video": source_video,
            "duration_sec": duration_sec,
        }
        runs.append(run)
    return sorted(runs, key=lambda r: r["metrics_csv"].stat().st_mtime, reverse=True)


def qc_status(metrics_df: pd.DataFrame) -> str:
    if metrics_df.empty or "mean_likelihood" not in metrics_df.columns:
        return "Red"
    pct_good = float((metrics_df["mean_likelihood"] >= 0.6).mean())
    if pct_good >= 0.7:
        return "Green"
    if pct_good >= 0.4:
        return "Yellow"
    return "Red"


def build_state_segments(metrics_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_df.empty or "state" not in metrics_df.columns:
        return pd.DataFrame()
    times = metrics_df["t_sec"] if "t_sec" in metrics_df.columns else pd.Series(range(len(metrics_df)))
    rows = []
    current_state = None
    start_idx = 0
    for i, state in enumerate(metrics_df["state"].astype(str)):
        if current_state is None:
            current_state = state
            start_idx = i
            continue
        if state != current_state:
            rows.append(_segment_row(metrics_df, times, start_idx, i - 1, current_state))
            current_state = state
            start_idx = i
    if current_state is not None:
        rows.append(_segment_row(metrics_df, times, start_idx, len(metrics_df) - 1, current_state))
    return pd.DataFrame(rows)


def _segment_row(metrics_df: pd.DataFrame, times: pd.Series, start_idx: int, end_idx: int, state: str) -> dict:
    return {
        "state": state,
        "start_frame": int(metrics_df.iloc[start_idx]["frame"]) if "frame" in metrics_df.columns else start_idx,
        "end_frame": int(metrics_df.iloc[end_idx]["frame"]) if "frame" in metrics_df.columns else end_idx,
        "start_sec": float(times.iloc[start_idx]),
        "end_sec": float(times.iloc[end_idx]),
        "duration_sec": float(times.iloc[end_idx] - times.iloc[start_idx]) if end_idx > start_idx else 0.0,
    }


def find_source_video(stem: str) -> Path | None:
    search_roots = [ROOT / "data" / "samples", ROOT / "data" / "jack_processed", ROOT / "data" / "jack_raw"]
    exts = [".mp4", ".mov", ".m4v", ".avi"]
    for root in search_roots:
        if not root.exists():
            continue
        for ext in exts:
            candidate = root / f"{stem}{ext}"
            if candidate.exists():
                return candidate
        for ext in exts:
            matches = list(root.rglob(f"{stem}{ext}"))
            if matches:
                return matches[0]
    return None


def inject_video_css() -> None:
    st.markdown(
        """
        <style>
        div[data-testid="stVideo"] video {
            height: auto !important;
            max-height: 70vh !important;
            object-fit: contain !important;
            background: #000;
        }
        .horsesense-video-wrap {
            max-width: 960px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_video(video_path: Path) -> None:
    # Include source mtime in cache key so refreshed overlays are picked up.
    video_path = ensure_web_playable_video(video_path, int(video_path.stat().st_mtime))
    # Use a native HTML video tag to avoid Streamlit's wrapper sizing bug that can clip height.
    video_bytes = video_path.read_bytes()
    b64 = base64.b64encode(video_bytes).decode("ascii")
    ext = video_path.suffix.lower().lstrip(".") or "mp4"
    mime = "video/mp4" if ext in {"mp4", "m4v"} else f"video/{ext}"
    intrinsic_w, intrinsic_h = get_video_dimensions(video_path)
    # Upscale small clips for visibility and honor the UI width setting.
    configured_width = int(st.session_state.get("video_width_px", MAX_VIDEO_WIDTH_PX))
    target_w = max(640, configured_width)
    target_h = int(round(target_w * (intrinsic_h / intrinsic_w))) if intrinsic_w > 0 else 360
    html = f"""
    <div style="display:flex; justify-content:center; width:100%;">
      <video controls preload="metadata"
             width="{target_w}" height="{target_h}"
             style="width:min(100%, {target_w}px); height:auto; display:block; background:#000; border-radius:8px;">
        <source src="data:{mime};base64,{b64}" type="{mime}">
        Your browser does not support the video tag.
      </video>
    </div>
    """
    # Reserve iframe height based on aspect ratio + controls/margins.
    iframe_h = max(220, target_h + 70)
    components.html(html, height=iframe_h, scrolling=False)


def render_upload_panel() -> None:
    st.sidebar.markdown("### Upload Clip")
    uploaded = st.sidebar.file_uploader(
        "Upload horse clip",
        type=["mp4", "mov", "m4v", "avi"],
        accept_multiple_files=False,
        help="Uploads to data/jack_raw and can run SuperAnimal pose inference.",
    )
    if not uploaded:
        return

    safe_name = _safe_filename(uploaded.name)
    save_path = ROOT / "data" / "jack_raw" / safe_name
    if st.sidebar.button("Save Upload", use_container_width=True):
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_bytes(uploaded.getbuffer())
        st.sidebar.success(f"Saved: {save_path.name}")

    if st.sidebar.button("Run Detection (SuperAnimal)", use_container_width=True):
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if not save_path.exists():
            save_path.write_bytes(uploaded.getbuffer())
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = ROOT / "outputs" / "jack_tests" / f"{save_path.stem}_{run_id}"
        cmd = [
            str(ROOT / ".venv_dlc_clean" / "bin" / "python"),
            str(ROOT / "run_pose_pipeline.py"),
            "--input_video",
            str(save_path),
            "--output_dir",
            str(out_dir),
            "--mode",
            "both",
            "--model_source",
            "hf_superanimal_quadruped",
        ]
        with st.sidebar.status("Running inference...", expanded=True) as status:
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.stdout:
                st.sidebar.code(proc.stdout[-6000:], language="text")
            if proc.returncode == 0:
                status.update(label="Inference complete. Refresh run list.", state="complete")
                st.sidebar.success(f"Run saved to {out_dir.relative_to(ROOT)}")
            else:
                if proc.stderr:
                    st.sidebar.code(proc.stderr[-6000:], language="text")
                status.update(label="Inference failed", state="error")


def _safe_filename(name: str) -> str:
    p = Path(name)
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", p.stem).strip("._") or "clip"
    suffix = p.suffix.lower()
    if suffix not in {".mp4", ".mov", ".m4v", ".avi"}:
        suffix = ".mp4"
    return f"{stem}{suffix}"


def get_video_dimensions(video_path: Path) -> tuple[int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return (640, 360)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    if w <= 0 or h <= 0:
        return (640, 360)
    return (w, h)


@st.cache_data(show_spinner=False)
def ensure_web_playable_video(video_path: Path, _source_mtime: int) -> Path:
    codec = probe_codec(video_path)
    if codec in {"h264", "avc1"}:
        return video_path

    previews_dir = ROOT / "outputs" / ".web_previews"
    previews_dir.mkdir(parents=True, exist_ok=True)
    out_path = previews_dir / f"{video_path.stem}_web.mp4"

    if out_path.exists() and out_path.stat().st_mtime >= video_path.stat().st_mtime:
        return out_path

    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(out_path),
    ]
    try:
        subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return out_path
    except Exception:
        # Fall back to original file if ffmpeg is unavailable or transcode fails.
        return video_path


def probe_codec(video_path: Path) -> str | None:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() or None
    except Exception:
        return None


def infer_run_duration_sec(metrics_csv: Path, overlay_mp4: Path | None, source_video: Path | None) -> float | None:
    # Prefer overlay video duration, then source video, then metrics t_sec range.
    for p in [overlay_mp4, source_video]:
        if p is None or not p.exists():
            continue
        d = probe_duration(p)
        if d is not None and d > 0:
            return d
    try:
        df = pd.read_csv(metrics_csv, usecols=["t_sec"])
        if "t_sec" in df.columns and not df.empty:
            return float(df["t_sec"].max())
    except Exception:
        pass
    return None


def probe_duration(video_path: Path) -> float | None:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        out = result.stdout.strip()
        return float(out) if out else None
    except Exception:
        return None


if __name__ == "__main__":
    main()
