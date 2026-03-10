# Product Spec (MVP)

## Project Summary
HorseSense Pose MVP demonstrates horse pose tracking on a known-good sample video (Horse-30 / DLC model-zoo compatible) and on real stable footage recorded by phone, offline.

## User/Stakeholder
- Primary: Internal demo operator / developer
- Secondary: Jack (stable owner/operator) reviewing feasibility on real footage

## Deliverables
### D1: Known-Good Demo
- `outputs/demo_horse30_overlay.mp4`
- `outputs/demo_horse30_metrics.csv`
- `outputs/demo_horse30_report.md`

### D2: Jack Stable Offline Test
- `outputs/jack_clip_01_overlay.mp4` (up to 5 clips)
- `outputs/jack_clip_01_metrics.csv`
- `outputs/jack_summary_report.md`

### D3: One-Command Runner
- `run_pose_pipeline.py`

## Functional Requirements
1. Run pose inference on input video with DLC SuperAnimal Quadruped (or compatible DLC keypoint CSV for development).
2. Save or ingest per-frame keypoints (x, y, likelihood by keypoint).
3. Render overlay video showing visible tracked keypoints.
4. Compute per-frame metrics:
- `move_score`
- `head_height`
- `stride_rhythm`
- `mean_likelihood`
5. Classify activity state using rule-based thresholds (`Standing`, `Walking`, `Active`).
6. Generate per-clip Markdown report with summary and limitations.

## Non-Functional Requirements
- Reproducible local setup (requirements or environment file)
- CLI-driven execution
- Outputs are deterministic given same inputs/thresholds
- Graceful failure with actionable errors if DLC/model unavailable

## Constraints
- Offline office execution acceptable
- GPU optional
- Initial demo should prioritize a known-good sample before Jack footage
