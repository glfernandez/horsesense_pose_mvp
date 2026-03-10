# Technical Spec (MVP)

## Architecture Overview
1. `src/dlc_infer.py`
- Adapter for DeepLabCut inference (SuperAnimal Quadruped model by default)
- Writes/returns keypoint CSV path

2. `src/features.py`
- Parses keypoints
- Computes movement, head-height proxy, stride rhythm proxy, confidence
- Applies rule-based state thresholds

3. `src/overlay.py`
- Renders keypoints onto video frames
- Saves overlay MP4

4. `src/report.py`
- Builds Markdown summary from metrics and QC stats

5. `run_pose_pipeline.py`
- CLI orchestration
- Validates inputs/arguments
- Controls outputs by mode

## Data Contracts
### Keypoints DataFrame (normalized internal format)
Columns:
- `frame` (int)
- `keypoint` (str)
- `x` (float)
- `y` (float)
- `likelihood` (float)

### Metrics CSV
Columns (minimum):
- `frame`
- `t_sec`
- `move_score`
- `head_height`
- `stride_rhythm`
- `mean_likelihood`
- `state`

## Threshold Defaults
- `t1=0.8` (Standing upper bound)
- `t2=2.0` (Walking upper bound)
These are placeholders and must be calibrated/documented per sample resolution/crop.

## Quality Evaluation
Per clip compute:
- `% frames with mean_likelihood > 0.6`
- keypoint jitter proxy (avg frame-to-frame displacement)
- visual QC status: Green / Yellow / Red

## Optional Extension (Not Required for MVP)
Detector-based cropping (YOLO horse box -> crop -> DLC inference) behind `--use_detector_crop`.
