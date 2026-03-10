# HorseSense Pose MVP

Pose-first MVP for horse posture/keypoint tracking using DeepLabCut (SuperAnimal Quadruped), with simple movement/activity metrics and offline evaluation on stable phone footage.

## Goal
Demonstrate horse pose tracking on a known-good sample video first, then run the same pipeline on Jack's stable footage offline.

## MVP Outputs
- Overlay video showing horse keypoints/skeleton
- Per-frame metrics CSV (movement, head-height proxy, stride-rhythm proxy, confidence)
- Basic report summarizing activity states and timestamps

## Non-Goals (MVP)
- No stress/emotion claims
- No medical diagnosis
- No realtime on-phone inference

## Project Layout
See `docs/REPO_STRUCTURE.md` for the required structure and file responsibilities.

## Setup
1. Create a Python 3.10 environment (3.9-3.11 supported)
2. Install `ffmpeg`
3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run Demo on Sample Video
1. Put a sample horse video in `data/samples/`
2. Run:

```bash
python run_pose_pipeline.py \
  --input_video data/samples/sample.mp4 \
  --output_dir outputs/demo \
  --dlc_project /path/to/dlc_project \
  --mode infer_only
```

Then run full pipeline:

```bash
python run_pose_pipeline.py \
  --input_video data/samples/sample.mp4 \
  --output_dir outputs/demo \
  --dlc_project /path/to/dlc_project \
  --mode both
```

## Run on Jack Clip
1. Place clip in `data/jack_raw/` (optionally trim/stabilize into `data/jack_processed/`)
2. Run:

```bash
python run_pose_pipeline.py \
  --input_video data/jack_processed/jack_clip_01.mp4 \
  --output_dir outputs/jack_tests/jack_clip_01 \
  --dlc_project /path/to/dlc_project \
  --mode both
```

## CLI (MVP)
Required:
- `--input_video path`
- `--output_dir path`
- `--mode infer_only|overlay|metrics|both`

Optional:
- `--dlc_project /path/to/dlc_project` (required unless `--keypoints_csv` is provided)
- `--shuffle 1`
- `--trainingsetindex 0`
- `--gputouse 0`
- `--model_source hf_superanimal_quadruped` (reserved metadata / future model-zoo convenience)
- `--use_detector_crop true|false`
- `--t1 0.8`
- `--t2 2.0`
- `--fps_override 30`
- `--min_confidence_for_state 0.2`
- `--save_keypoints true|false`
- `--keep_dlc_outputs_in_video_dir true|false`
- `--keypoints_csv path` (optional shortcut during development)

## Fixture Mode (No DLC Required)
For overlay/metrics/report development, you can skip DeepLabCut inference and provide an existing keypoints CSV:

```bash
python run_pose_pipeline.py \
  --input_video data/samples/sample.mp4 \
  --output_dir outputs/demo \
  --keypoints_csv data/samples/sample_keypoints.csv \
  --mode both
```

## Gate C Smoke Test (Target Machine)
Use the helper script to validate Python, DeepLabCut import, `ffmpeg`, DLC inference, and the full pipeline in one run:

```bash
cd "/Users/gl_fernandez/Library/CloudStorage/GoogleDrive-gary@dvaco.io/My Drive/HorseSense/horsesense_pose_mvp"
DLC_PROJECT=/absolute/path/to/dlc_project ./scripts/gate_c_smoke_test.sh
```

Optional env vars:
- `MODE_INFER_ONLY=true` (run only DLC inference)
- `FPS_OVERRIDE=30`
- `KEEP_DLC_OUTPUTS_IN_VIDEO_DIR=true`
- `VIDEO_PATH=data/jack_processed/jack_clip_01.mp4`

## Simple Dashboard (Local Review)
Launch a lightweight Streamlit dashboard to inspect overlay videos, QC status, metrics, and reports:

```bash
cd "/Users/gl_fernandez/Library/CloudStorage/GoogleDrive-gary@dvaco.io/My Drive/HorseSense/horsesense_pose_mvp"
./scripts/run_dashboard.sh
```

Or directly:

```bash
python -m streamlit run app/dashboard.py
```

## Outputs
- `*_overlay.mp4`: video with keypoints overlay
- `*_metrics.csv`: per-frame metrics and state
- `*_report.md`: summary, state distribution, caveats, and recommendations
- `*_keypoints.csv`: raw x/y/likelihood per keypoint per frame (when saved)

## Limitations
- Performance depends heavily on angle, occlusion, lighting, and crop quality
- Metrics are movement/posture proxies only, not welfare/emotion indicators
- Stride rhythm proxy is heuristic and may be unstable on poor clips

## Capture Tips (Best Results)
- Side view preferred, full horse visible
- Stable phone (tripod/railing)
- Good lighting, avoid backlight
- One horse per frame for first tests
- Minimal handler occlusion

## Core References
- DeepLabCut: https://github.com/DeepLabCut/DeepLabCut
- DLC Model Zoo (SuperAnimal Quadruped): https://huggingface.co/mwmathis/DeepLabCutModelZoo-SuperAnimal-Quadruped
- Horse-30 dataset: https://huggingface.co/datasets/mwmathis/Horse-30
