from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from depth_guard.dashboard import save_dashboard
from depth_guard.engine import analyze_frame
from depth_guard.io import load_rules, write_report_json
from depth_guard.perception import (
    capture_webcam_frame,
    estimate_depth_map_midas,
    run_yolo_on_array,
    run_yolo_on_image,
)
from depth_guard.reporting import format_text_report
from depth_guard.occupancy_grid import build_occupancy_grid
from depth_guard.visualization import (
    save_depth_heatmap,
    save_depth_map_plot,
    save_occupancy_grid_plot,
    save_overlay_plot,
    save_original_image,
    save_scene_plot,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Optional YOLO-based Depth-Guard demo for real images or one webcam frame.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", help="Path to an RGB image for YOLO detection.")
    source.add_argument("--webcam", type=int, help="Capture one frame from the selected webcam index.")
    parser.add_argument("--depth-map", help="Optional .npy or image depth map aligned with --image.")
    parser.add_argument("--rules", default="data/logic_constraints.json")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model name or local model path.")
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--default-depth", type=float, default=2.0)
    parser.add_argument("--midas", action="store_true", help="Estimate a dense depth map with MiDaS when no depth map file is provided.")
    parser.add_argument("--midas-model", default="DPT_Hybrid", help="MiDaS model name for depth estimation.")
    parser.add_argument("--out-dir", default="reports/live_ai")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rules = load_rules(PROJECT_ROOT / args.rules)
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        image = None
        depth_map = None
        if args.image:
            image_path = PROJECT_ROOT / args.image
            if args.depth_map:
                try:
                    import cv2
                except ImportError as exc:
                    raise RuntimeError("OpenCV is required to load depth-map image files.") from exc
                depth_path = PROJECT_ROOT / args.depth_map
                if depth_path.suffix.lower() == ".npy":
                    depth_map = np.load(depth_path).astype(float)
                else:
                    loaded = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
                    if loaded is None:
                        raise ValueError(f"Could not read depth map: {depth_path}")
                    depth_map = loaded.astype(float)
                    if depth_map.ndim == 3:
                        depth_map = depth_map[:, :, 0]
                    if depth_map.max(initial=0) > 50.0:
                        depth_map = depth_map / 1000.0
            elif args.midas:
                try:
                    import cv2
                except ImportError as exc:
                    raise RuntimeError("OpenCV is required for MiDaS depth estimation.") from exc
                image_for_depth = cv2.imread(str(image_path))
                if image_for_depth is None:
                    raise ValueError(f"Could not read image for depth estimation: {image_path}")
                depth_map = estimate_depth_map_midas(image_for_depth, model_name=args.midas_model)

            frame, image = run_yolo_on_image(
                image_path,
                frame_id=image_path.stem,
                model_name=args.model,
                confidence=args.confidence,
                depth_map=depth_map,
                depth_map_path=None,
                default_depth_m=args.default_depth,
            )
        else:
            image, width, height = capture_webcam_frame(args.webcam)
            if args.midas:
                depth_map = estimate_depth_map_midas(image, model_name=args.midas_model)
            frame = run_yolo_on_array(
                image,
                image_width_px=width,
                image_height_px=height,
                frame_id=f"webcam_{args.webcam}",
                model_name=args.model,
                confidence=args.confidence,
                depth_map=depth_map,
                default_depth_m=args.default_depth,
            )
    except RuntimeError as exc:
        print(f"Depth-Guard AI demo could not start: {exc}", file=sys.stderr)
        return 1

    analysis = analyze_frame(frame, rules)
    write_report_json(out_dir / "status_report.json", analysis.to_dict())
    save_original_image(image, out_dir / "frame_original.png")
    save_overlay_plot(frame, analysis, out_dir / "frame_overlay.png", image=image)
    if depth_map is not None:
        save_depth_map_plot(depth_map, out_dir / "estimated_depth_map.png", title="Estimated Dense Depth Map")
    save_scene_plot(analysis, out_dir / "scene_3d.png")
    save_depth_heatmap(frame, analysis, out_dir / "depth_heatmap.png")
    save_occupancy_grid_plot(build_occupancy_grid(analysis), out_dir / "occupancy_grid.png")
    save_dashboard(
        frame,
        analysis,
        out_dir / "depth_guard_dashboard.html",
        original_image="frame_original.png",
        overlay_image="frame_overlay.png",
        depth_map_image="estimated_depth_map.png" if depth_map is not None else None,
        scene_image="scene_3d.png",
        heatmap_image="depth_heatmap.png",
        occupancy_grid_image="occupancy_grid.png",
    )

    print(format_text_report(analysis))
    print()
    print(f"Saved AI perception dashboard: {out_dir / 'depth_guard_dashboard.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
