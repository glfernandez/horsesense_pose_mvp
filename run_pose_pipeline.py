from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from src.dlc_infer import DLCInferenceError, run_dlc_inference
from src.features import compute_metrics, load_keypoints_csv
from src.overlay import render_overlay_video
from src.report import generate_report
from src.utils import ensure_dir, str2bool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HorseSense Pose MVP pipeline runner")
    parser.add_argument("--input_video", required=True, help="Path to input video")
    parser.add_argument("--output_dir", required=True, help="Directory for outputs")
    parser.add_argument(
        "--mode",
        required=True,
        choices=["infer_only", "overlay", "metrics", "both"],
        help="Which outputs to generate",
    )
    parser.add_argument(
        "--dlc_project",
        default=None,
        help="Path to a DeepLabCut project directory (required unless --keypoints_csv is provided)",
    )
    parser.add_argument("--shuffle", type=int, default=1, help="DLC shuffle index")
    parser.add_argument("--trainingsetindex", type=int, default=0, help="DLC training set index")
    parser.add_argument("--gputouse", type=int, default=None, help="CUDA device index (optional)")
    parser.add_argument("--use_detector_crop", default="false", help="Reserved for future YOLO crop stage, true|false")
    parser.add_argument(
        "--model_source",
        default="hf_superanimal_quadruped",
        help="Reserved metadata only (model-zoo convenience path not implemented in CLI yet)",
    )
    parser.add_argument("--t1", type=float, default=0.8, help="Standing threshold")
    parser.add_argument("--t2", type=float, default=2.0, help="Walking threshold")
    parser.add_argument("--fps_override", type=float, default=None, help="Override video FPS if metadata is wrong")
    parser.add_argument(
        "--min_confidence_for_state",
        type=float,
        default=0.2,
        help="Frames below this mean likelihood are labeled Unknown",
    )
    parser.add_argument("--save_keypoints", default="true", help="true|false")
    parser.add_argument(
        "--keep_dlc_outputs_in_video_dir",
        default="false",
        help="true|false; keep DLC-generated CSV/H5 next to the input video instead of moving to output_dir",
    )
    parser.add_argument(
        "--keypoints_csv",
        default=None,
        help="Optional existing keypoints CSV path (development shortcut)",
    )
    return parser


def get_video_fps(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return float(fps) if fps and fps > 0 else 30.0


def main() -> int:
    args = build_parser().parse_args()
    output_dir = ensure_dir(args.output_dir)
    input_video = str(Path(args.input_video))

    if args.keypoints_csv:
        keypoints_csv = Path(args.keypoints_csv)
        if not keypoints_csv.exists():
            raise FileNotFoundError(f"Keypoints CSV not found: {keypoints_csv}")
        if args.mode == "infer_only":
            print(f"[OK] Inference skipped (fixture mode). Keypoints CSV: {keypoints_csv}")
            return 0
    else:
        if not args.dlc_project and args.model_source != "hf_superanimal_quadruped":
            print("[ERROR] Provide --keypoints_csv, --dlc_project, or --model_source hf_superanimal_quadruped.")
            return 2
        try:
            keypoints_csv = run_dlc_inference(
                input_video=input_video,
                output_dir=str(output_dir),
                dlc_project=str(args.dlc_project) if args.dlc_project else None,
                shuffle=args.shuffle,
                trainingsetindex=args.trainingsetindex,
                gputouse=args.gputouse,
                save_keypoints=str2bool(args.save_keypoints),
                keep_dlc_outputs_in_video_dir=str2bool(args.keep_dlc_outputs_in_video_dir),
                model_source=args.model_source,
            )
        except DLCInferenceError as exc:
            print(f"[ERROR] {exc}")
            print("Tip: use --keypoints_csv during development to test overlay/metrics/report modules.")
            return 2
        if args.mode == "infer_only":
            print(f"[OK] DLC inference completed. Keypoints CSV: {keypoints_csv}")
            return 0

    keypoints_df = load_keypoints_csv(keypoints_csv)
    fps = float(args.fps_override) if args.fps_override and args.fps_override > 0 else get_video_fps(input_video)
    metrics_df = compute_metrics(
        keypoints_df=keypoints_df,
        fps=fps,
        t1=args.t1,
        t2=args.t2,
        min_confidence_for_state=args.min_confidence_for_state,
    )

    metrics_path = output_dir / f"{Path(input_video).stem}_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"[OK] Metrics saved: {metrics_path}")

    if args.mode in {"overlay", "both"}:
        overlay_path = render_overlay_video(
            input_video=input_video,
            keypoints_df=keypoints_df,
            output_dir=str(output_dir),
            metrics_df=metrics_df,
        )
        print(f"[OK] Overlay saved: {overlay_path}")

    report_path = generate_report(
        input_video=input_video,
        metrics_df=metrics_df,
        output_dir=str(output_dir),
        thresholds=(args.t1, args.t2),
        fps=fps,
    )
    print(f"[OK] Report saved: {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
