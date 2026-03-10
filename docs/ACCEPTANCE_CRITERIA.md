# Acceptance Criteria

## A. Known-Good Inference + Overlay
- A sample horse video is placed in `data/samples/`
- DLC inference produces keypoint outputs (or compatible CSV for dev testing)
- Overlay video is generated and playable
- Keypoints visually remain attached to horse for most frames
- Raw keypoints are saved as CSV when `--save_keypoints true`

## B. Feature Extraction
- Metrics CSV includes: `frame,t_sec,move_score,head_height,stride_rhythm,mean_likelihood`
- Values are non-trivial (not all zeros / NaN / exploding)
- Confidence values stay within expected range (0-1 for likelihood-derived fields)

## C. Activity Classifier
- Metrics CSV includes `state`
- States limited to `Standing`, `Walking`, `Active`
- Report includes % time in each state and threshold values used

## D. Jack Stable Offline Test
- At least 1 Jack clip runs end-to-end and produces overlay + metrics + report
- Report documents failures/limitations when tracking quality is poor
- Report includes camera-angle recommendations and fine-tune recommendation (yes/no)

## E. CLI / Reproducibility
- `python run_pose_pipeline.py --help` works
- README documents setup and commands for sample and Jack clips
- Dependency file exists (`requirements.txt` or `environment.yml`)
