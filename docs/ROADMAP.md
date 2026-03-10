# Roadmap

## Phase 0 - Workspace & Spec Foundation (Now)
- Create repo skeleton
- Add spec-driven documents (north star, objective, product/tech spec, tasks, acceptance criteria)
- Add CLI/module skeleton

## Phase 1 - Known-Good Demo (Pose First)
- Fetch sample horse video (Horse-30 or DLC example)
- Integrate DLC SuperAnimal Quadruped inference
- Produce first overlay video and keypoints CSV
- Validate visual tracking quality

## Phase 2 - Metrics & Reporting
- Implement movement/head-height/stride-rhythm/confidence metrics
- Add rule-based activity states
- Generate demo report with limitations and threshold calibration notes

## Phase 3 - Jack Stable Offline Test
- Ingest phone clips
- Preprocess (trim/resize/stabilize as needed)
- Run pipeline on 1-5 clips
- Produce summary report + camera recommendations

## Phase 4 - Fine-Tune Decision Gate
- Evaluate Green/Yellow/Red status per clip
- If Yellow/Red dominates, define minimal fine-tune labeling set (100-200 frames)
- Compare before/after (if fine-tuning is executed)
