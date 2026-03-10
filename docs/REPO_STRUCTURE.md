# Repo Structure

```text
horsesense_pose_mvp/
  README.md
  requirements.txt
  run_pose_pipeline.py
  src/
    dlc_infer.py
    features.py
    overlay.py
    report.py
    utils.py
  data/
    samples/
    jack_raw/
    jack_processed/
  outputs/
    demo/
    jack_tests/
  notebooks/
  docs/
```

## Notes
- `data/jack_raw/` contains original phone clips and should not be committed.
- `outputs/` is generated content (video overlays, metrics, reports).
- `docs/` holds the spec-driven planning and execution artifacts.
