# Tasks

## Legend
- `TODO`
- `IN_PROGRESS`
- `DONE`
- `BLOCKED`

## Foundation
- `DONE` Create project scaffold and directories
- `DONE` Create spec-driven docs (north star/objective/spec/roadmap/tasks/acceptance)
- `DONE` Add agent instructions (`AGENTS.md`)
- `DONE` Add CLI and module skeletons
- `DONE` Add fixture-mode keypoints CSV (`data/samples/sample_keypoints.csv`)
- `DONE` Add Gate C smoke test script (`scripts/gate_c_smoke_test.sh`)
- `DONE` Add simple local dashboard for reviewing outputs (`app/dashboard.py`)
- `TODO` Add environment lockfile or conda environment for target machine

## Known-Good Demo (D1)
- `TODO` Download known-good sample horse video (Horse-30 or DLC example)
- `DONE` Integrate DLC inference via local DLC project path in `src/dlc_infer.py`
- `TODO` Save raw keypoints CSV in canonical format
- `DONE` Add DLC output discovery logging and fallback candidate search
- `TODO` Generate `demo_horse30_overlay.mp4`
- `TODO` Generate `demo_horse30_metrics.csv`
- `TODO` Generate `demo_horse30_report.md`
- `TODO` Calibrate movement thresholds (`t1`, `t2`) for demo resolution/crop

## Jack Stable Offline Test (D2)
- `TODO` Capture 5-10 short clips per protocol
- `TODO` Place clips in `data/jack_raw/`
- `TODO` Preprocess clips into `data/jack_processed/`
- `TODO` Run pipeline on at least 1 Jack clip
- `TODO` Produce `jack_summary_report.md`
- `TODO` Assign Green/Yellow/Red status per clip

## Fine-Tuning (Conditional)
- `TODO` Decide if fine-tuning is needed based on Jack results
- `TODO` Select 3-5 clips and sample 100-200 frames
- `TODO` Label frames in DLC GUI
- `TODO` Fine-tune and compare likelihood/jitter improvements

## Documentation / Demo Readiness
- `DONE` Add capture protocol document
- `DONE` Add demo script for Jack presentation
- `TODO` Add example outputs/screenshots once demo is generated

## Workspace Organization
- `DONE` Create separate root-level business analysis folder for owner investment planning so non-MVP commercial work stays isolated from pose MVP deliverables
- `DONE` Add business developer interview pack, data request checklist, and assumptions register for safer commercial modeling
