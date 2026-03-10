#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

VIDEO_PATH="${VIDEO_PATH:-data/samples/horse30_sample1.mp4}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/gate_c_smoke}"
DLC_PROJECT="${DLC_PROJECT:-}"
MODE_INFER_ONLY="${MODE_INFER_ONLY:-false}"
FPS_OVERRIDE="${FPS_OVERRIDE:-}"
KEEP_DLC_OUTPUTS_IN_VIDEO_DIR="${KEEP_DLC_OUTPUTS_IN_VIDEO_DIR:-false}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ -z "$DLC_PROJECT" ]]; then
  echo "[ERROR] Set DLC_PROJECT=/absolute/path/to/dlc_project before running."
  exit 2
fi

echo "[INFO] Root: $ROOT_DIR"
echo "[INFO] Python: $PYTHON_BIN"
echo "[INFO] Video: $VIDEO_PATH"
echo "[INFO] Output: $OUTPUT_DIR"
echo "[INFO] DLC project: $DLC_PROJECT"

echo "[CHECK] Python version"
"$PYTHON_BIN" -V

echo "[CHECK] DeepLabCut import"
"$PYTHON_BIN" - <<'PY'
try:
    import deeplabcut
    print('[OK] deeplabcut import succeeded')
except Exception as e:
    print('[ERROR] deeplabcut import failed:', type(e).__name__, str(e))
    raise
PY

echo "[CHECK] ffmpeg"
if command -v ffmpeg >/dev/null 2>&1; then
  ffmpeg -version | head -n 1
else
  echo "[ERROR] ffmpeg not found in PATH"
  exit 2
fi

echo "[CHECK] Video metadata"
if command -v ffprobe >/dev/null 2>&1; then
  ffprobe -v error \
    -select_streams v:0 \
    -show_entries stream=codec_name,width,height,avg_frame_rate,r_frame_rate:format=duration \
    -of default=noprint_wrappers=1 "$VIDEO_PATH" || true
else
  "$PYTHON_BIN" - <<PY
import cv2
cap = cv2.VideoCapture(r"""$VIDEO_PATH""")
if not cap.isOpened():
    print("[WARN] cv2 could not open video for metadata check")
else:
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    dur = (frames / fps) if fps else 0
    print(f"fps={fps}")
    print(f"width={w}")
    print(f"height={h}")
    print(f"frame_count={frames}")
    print(f"duration={dur}")
cap.release()
PY
fi

mkdir -p "$OUTPUT_DIR"

EXTRA_ARGS=()
if [[ -n "$FPS_OVERRIDE" ]]; then
  EXTRA_ARGS+=(--fps_override "$FPS_OVERRIDE")
fi

if [[ "$MODE_INFER_ONLY" == "true" ]]; then
  echo "[RUN] DLC inference only"
  "$PYTHON_BIN" run_pose_pipeline.py \
    --input_video "$VIDEO_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --dlc_project "$DLC_PROJECT" \
    --mode infer_only \
    --keep_dlc_outputs_in_video_dir "$KEEP_DLC_OUTPUTS_IN_VIDEO_DIR" \
    "${EXTRA_ARGS[@]}"
  "$PYTHON_BIN" scripts/preview_dlc_header.py --video_path "$VIDEO_PATH" --output_dir "$OUTPUT_DIR" || true
  echo "[OK] Inference-only smoke test completed"
  exit 0
fi

echo "[RUN] DLC inference only (pre-check)"
"$PYTHON_BIN" run_pose_pipeline.py \
  --input_video "$VIDEO_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --dlc_project "$DLC_PROJECT" \
  --mode infer_only \
  --keep_dlc_outputs_in_video_dir "$KEEP_DLC_OUTPUTS_IN_VIDEO_DIR"
"$PYTHON_BIN" scripts/preview_dlc_header.py --video_path "$VIDEO_PATH" --output_dir "$OUTPUT_DIR" || true

echo "[RUN] Full pipeline"
"$PYTHON_BIN" run_pose_pipeline.py \
  --input_video "$VIDEO_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --dlc_project "$DLC_PROJECT" \
  --mode both \
  --keep_dlc_outputs_in_video_dir "$KEEP_DLC_OUTPUTS_IN_VIDEO_DIR" \
  "${EXTRA_ARGS[@]}"

echo "[OK] Gate C smoke test completed"
echo "[INFO] Outputs in: $OUTPUT_DIR"
ls -1 "$OUTPUT_DIR" || true
