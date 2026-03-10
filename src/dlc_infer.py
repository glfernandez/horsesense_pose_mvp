from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Optional

from .utils import ensure_dir, stem_name


class DLCInferenceError(RuntimeError):
    pass


def run_dlc_inference(
    input_video: str,
    output_dir: str,
    dlc_project: str | None = None,
    shuffle: int = 1,
    trainingsetindex: int = 0,
    gputouse: Optional[int] = None,
    save_keypoints: bool = True,
    videotype: str = "mp4",
    keep_dlc_outputs_in_video_dir: bool = False,
    model_source: str = "hf_superanimal_quadruped",
) -> Path:
    """Run DeepLabCut inference and return CSV path.

    Supports:
    - local DLC project (`dlc_project`, snapshot-based inference)
    - SuperAnimal quadruped model zoo (`model_source=hf_superanimal_quadruped`)
    """
    video_path = Path(input_video).expanduser().resolve()
    if not video_path.exists():
        raise DLCInferenceError(f"Input video not found: {video_path}")

    out_dir = ensure_dir(output_dir)
    project_path: Path | None = None
    config_path: Path | None = None
    if dlc_project:
        project_path = Path(dlc_project).expanduser().resolve()
        if not project_path.exists():
            raise DLCInferenceError(f"DLC project not found: {project_path}")
        config_path = project_path / "config.yaml" if project_path.is_dir() else project_path
        if not config_path.exists():
            raise DLCInferenceError(f"DLC config.yaml not found: {config_path}")

    _inject_keras_legacy_tf_layers_alias()
    _inject_pandas_hdf_csv_fallback()

    try:
        import deeplabcut
    except Exception as exc:  # pragma: no cover - environment dependent
        raise DLCInferenceError(
            "DeepLabCut is not available. Install deeplabcut and compatible torch dependencies."
        ) from exc

    if gputouse is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gputouse)

    started_at = time.time()
    try:
        if config_path is not None:
            deeplabcut.analyze_videos(
                str(config_path),
                [str(video_path)],
                shuffle=shuffle,
                trainingsetindex=trainingsetindex,
                save_as_csv=save_keypoints,
                videotype=videotype,
            )
        elif model_source == "hf_superanimal_quadruped":
            deeplabcut.video_inference_superanimal(
                [str(video_path)],
                "superanimal_quadruped",
                videotype=f".{videotype.lstrip('.')}",
                plot_trajectories=False,
            )
        else:
            raise DLCInferenceError(
                "No valid inference source provided. Pass --dlc_project with snapshots or "
                "--model_source hf_superanimal_quadruped."
            )
    except Exception as exc:  # pragma: no cover - environment dependent
        import traceback

        traceback.print_exc()
        raise DLCInferenceError(f"DeepLabCut analyze_videos failed: {exc}") from exc

    h5_path, csv_path = _find_dlc_outputs(video_path, started_at=started_at)
    if h5_path is None and csv_path is None:
        raise DLCInferenceError(
            f"No DLC outputs found next to video after inference: {video_path.parent}"
        )

    moved_h5 = h5_path if keep_dlc_outputs_in_video_dir else _move_if_present(h5_path, out_dir)
    moved_csv = csv_path if keep_dlc_outputs_in_video_dir else _move_if_present(csv_path, out_dir)
    if save_keypoints and moved_csv is None:
        raise DLCInferenceError(
            "DLC inference finished but no CSV output was found. Re-run with save_keypoints=true "
            "or inspect the generated .h5 output."
        )
    if moved_csv is not None:
        return moved_csv
    if moved_h5 is not None:
        # Pipeline expects CSV. Keep explicit failure until H5 parsing is implemented.
        raise DLCInferenceError(
            f"DLC produced H5 only ({moved_h5}); CSV output is required for this MVP pipeline."
        )
    raise DLCInferenceError("Unexpected DLC output state.")


def _find_dlc_outputs(video_path: Path, started_at: float | None = None) -> tuple[Optional[Path], Optional[Path]]:
    stem = stem_name(video_path)
    parent = video_path.parent
    h5_pattern = f"{stem}*DLC*.h5"
    csv_pattern = f"{stem}*DLC*.csv"
    h5_candidates = sorted(
        parent.glob(h5_pattern),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    csv_candidates = sorted(
        parent.glob(csv_pattern),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    print(f"[DLC] Scanning output dir: {parent}")
    print(f"[DLC] Patterns: h5='{h5_pattern}', csv='{csv_pattern}'")
    print(f"[DLC] H5 candidates: {[p.name for p in h5_candidates]}")
    print(f"[DLC] CSV candidates: {[p.name for p in csv_candidates]}")

    if not h5_candidates and not csv_candidates and started_at is not None:
        recent_h5 = _recent_files(parent, "*.h5", started_at)
        recent_csv = _recent_files(parent, "*.csv", started_at)
        print("[DLC] Fallback (recent files by mtime in video dir)")
        print(f"[DLC] Recent H5 candidates: {[p.name for p in recent_h5]}")
        print(f"[DLC] Recent CSV candidates: {[p.name for p in recent_csv]}")
        h5_candidates = recent_h5 or h5_candidates
        csv_candidates = recent_csv or csv_candidates

    if not h5_candidates and not csv_candidates:
        recursive_h5 = sorted(parent.rglob(f"{stem}*.h5"), key=lambda p: p.stat().st_mtime, reverse=True)
        recursive_csv = sorted(parent.rglob(f"{stem}*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        print("[DLC] Fallback (recursive search)")
        print(f"[DLC] Recursive H5 candidates: {[str(p.relative_to(parent)) for p in recursive_h5[:10]]}")
        print(f"[DLC] Recursive CSV candidates: {[str(p.relative_to(parent)) for p in recursive_csv[:10]]}")
        h5_candidates = recursive_h5 or h5_candidates
        csv_candidates = recursive_csv or csv_candidates

    return (h5_candidates[0] if h5_candidates else None, csv_candidates[0] if csv_candidates else None)


def _move_if_present(path: Optional[Path], out_dir: Path) -> Optional[Path]:
    if path is None:
        return None
    dest = out_dir / path.name
    if dest.exists():
        dest.unlink()
    path.replace(dest)
    return dest


def _recent_files(parent: Path, pattern: str, started_at: float, slack_seconds: float = 3.0) -> list[Path]:
    threshold = started_at - slack_seconds
    files = []
    for p in parent.glob(pattern):
        try:
            if p.stat().st_mtime >= threshold:
                files.append(p)
        except FileNotFoundError:
            continue
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)


def _inject_keras_legacy_tf_layers_alias() -> None:
    """Patch missing `keras.legacy_tf_layers` imports for some TF/Keras setups.

    DLC's TensorFlow path imports `keras.legacy_tf_layers`, but newer packaging layouts
    may expose this only under `tensorflow.python.keras.legacy_tf_layers`.
    """
    try:
        import importlib
        import types

        if "keras.legacy_tf_layers" in sys.modules:
            return
        try:
            importlib.import_module("keras.legacy_tf_layers")
            return
        except Exception:
            pass

        base = importlib.import_module("tensorflow.python.keras.legacy_tf_layers")
        sys.modules["keras.legacy_tf_layers"] = base

        for sub in ("base", "core", "convolutional", "normalization", "pooling"):
            tf_name = f"tensorflow.python.keras.legacy_tf_layers.{sub}"
            try:
                mod = importlib.import_module(tf_name)
            except Exception:
                if sub == "normalization":
                    try:
                        # Preferred: use Keras' real legacy TF layers module if available.
                        mod = importlib.import_module("keras.src.legacy_tf_layers.normalization")
                    except Exception:
                        try:
                            import tensorflow as tf

                            mod = types.ModuleType("keras.legacy_tf_layers.normalization")

                            class LegacyBatchNormalization(tf.keras.layers.BatchNormalization):
                                def __init__(self, *args, **kwargs):
                                    # tf_slim passes TF1-only kwargs that modern keras BN rejects.
                                    for k in (
                                        "_scope",
                                        "_reuse",
                                        "renorm",
                                        "renorm_clipping",
                                        "renorm_momentum",
                                        "virtual_batch_size",
                                        "adjustment",
                                        "fused",
                                    ):
                                        kwargs.pop(k, None)
                                    super().__init__(*args, **kwargs)

                                def apply(self, inputs, *args, **kwargs):
                                    return self(inputs, *args, **kwargs)

                            def batch_normalization(inputs, *args, **kwargs):
                                layer = LegacyBatchNormalization(*args, **kwargs)
                                return layer(inputs)

                            mod.BatchNormalization = LegacyBatchNormalization
                            mod.BatchNorm = LegacyBatchNormalization
                            mod.batch_normalization = batch_normalization
                            mod.batch_norm = batch_normalization
                        except Exception:
                            continue
                else:
                    continue
            sys.modules[f"keras.legacy_tf_layers.{sub}"] = mod
    except Exception:
        # Best effort only; if this fails DLC import will raise a clearer error next.
        return


def _inject_pandas_hdf_csv_fallback() -> None:
    """Fallback for environments without `tables` (PyTables).

    DLC SuperAnimal writes results via `DataFrame.to_hdf`. On macOS/Python 3.11, building
    `tables` can fail. For MVP inference, write a sibling CSV instead so downstream parsing
    can proceed.
    """
    try:
        import importlib
        import pandas as pd

        try:
            importlib.import_module("tables")
            return
        except Exception:
            pass

        if getattr(pd.DataFrame.to_hdf, "_horsesense_patched", False):
            return

        orig_to_hdf = pd.DataFrame.to_hdf

        def _to_hdf_or_csv(self, path_or_buf, *args, **kwargs):
            try:
                return orig_to_hdf(self, path_or_buf, *args, **kwargs)
            except ImportError as exc:
                msg = str(exc).lower()
                if "pytables" not in msg and "tables" not in msg:
                    raise
                h5_path = Path(str(path_or_buf))
                csv_path = h5_path.with_suffix(".csv")
                print(
                    f"[DLC] PyTables unavailable; writing CSV fallback instead: {csv_path}"
                )
                self.to_csv(csv_path)
                return None

        _to_hdf_or_csv._horsesense_patched = True  # type: ignore[attr-defined]
        pd.DataFrame.to_hdf = _to_hdf_or_csv  # type: ignore[assignment]
    except Exception:
        return
