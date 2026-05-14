from __future__ import annotations

import argparse
import json
from pathlib import Path

from .engine import analyze_frame
from .io import load_rules, load_vision_feed, write_report_json
from .reporting import format_text_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="depth-guard",
        description="Analyze a warehouse camera frame for 3D loading-zone occupancy.",
    )
    parser.add_argument(
        "--vision",
        default="data/vision_feed_sample.json",
        help="Path to the simulated vision feed JSON.",
    )
    parser.add_argument(
        "--rules",
        default="data/logic_constraints.json",
        help="Path to the occupancy rules JSON.",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Console output format.",
    )
    parser.add_argument("--output", help="Optional path to save the structured JSON report.")
    parser.add_argument("--plot", help="Optional path to save a 3D occupancy plot.")
    parser.add_argument("--overlay", help="Optional path to save a 2D detection overlay.")
    parser.add_argument("--heatmap", help="Optional path to save a simulated depth heatmap.")
    parser.add_argument("--occupancy-grid", help="Optional path to save an occupancy grid risk map.")
    parser.add_argument("--dashboard", help="Optional path to save a browser-ready HTML dashboard.")
    parser.add_argument(
        "--fail-on-blocked",
        action="store_true",
        help="Return exit code 2 when the loading zone is blocked. Useful for automation.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    frame = load_vision_feed(Path(args.vision))
    rules = load_rules(Path(args.rules))
    analysis = analyze_frame(frame, rules)
    payload = analysis.to_dict()

    if args.format == "json":
        print(json.dumps(payload, indent=2))
    else:
        print(format_text_report(analysis))

    if args.output:
        write_report_json(args.output, payload)
    if args.plot:
        from .visualization import save_scene_plot

        save_scene_plot(analysis, args.plot)
    if args.overlay:
        from .visualization import save_overlay_plot

        save_overlay_plot(frame, analysis, args.overlay)
    if args.heatmap:
        from .visualization import save_depth_heatmap

        save_depth_heatmap(frame, analysis, args.heatmap)
    if args.occupancy_grid:
        from .occupancy_grid import build_occupancy_grid
        from .visualization import save_occupancy_grid_plot

        save_occupancy_grid_plot(build_occupancy_grid(analysis), args.occupancy_grid)
    if args.dashboard:
        from .dashboard import save_dashboard

        dashboard_path = Path(args.dashboard)
        save_dashboard(
            frame,
            analysis,
            dashboard_path,
            overlay_image=Path(args.overlay or "frame_overlay.png").name,
            scene_image=Path(args.plot or "scene_3d.png").name,
            heatmap_image=Path(args.heatmap or "depth_heatmap.png").name,
            occupancy_grid_image=Path(args.occupancy_grid or "occupancy_grid.png").name,
        )

    if args.fail_on_blocked and analysis.is_blocked:
        return 2
    return 0
