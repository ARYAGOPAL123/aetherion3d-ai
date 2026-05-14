from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from depth_guard.engine import analyze_frame
from depth_guard.evaluation import (
    evaluate_against_expectations,
    format_scorecard_markdown,
    load_expectations,
)
from depth_guard.io import load_rules, load_vision_feed, write_report_json
from depth_guard.reporting import format_text_report
from depth_guard.safety import build_safety_events, write_safety_events_csv, write_safety_events_json
from depth_guard.dashboard import save_dashboard
from depth_guard.batch import analyze_scenarios, scenario_summary_rows, write_scenario_csv
from depth_guard.occupancy_grid import build_occupancy_grid
from depth_guard.tracking import TemporalOccupancyMonitor, write_temporal_csv
from depth_guard.visualization import (
    save_depth_heatmap,
    save_occupancy_grid_plot,
    save_overlay_plot,
    save_scene_plot,
    save_temporal_timeline_plot,
)


def main() -> int:
    frame = load_vision_feed(PROJECT_ROOT / "data" / "vision_feed_sample.json")
    rules = load_rules(PROJECT_ROOT / "data" / "logic_constraints.json")
    analysis = analyze_frame(frame, rules)

    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    write_report_json(reports_dir / "status_report.json", analysis.to_dict())
    (reports_dir / "terminal_report.txt").write_text(format_text_report(analysis) + "\n", encoding="utf-8")
    safety_events = build_safety_events(analysis)
    write_safety_events_json(reports_dir / "safety_events.json", safety_events)
    write_safety_events_csv(reports_dir / "safety_events.csv", safety_events)

    expectations = load_expectations(PROJECT_ROOT / "data" / "evaluation_expectations.json")
    scorecard = evaluate_against_expectations(analysis, expectations)
    (reports_dir / "evaluation_scorecard.json").write_text(
        json.dumps(scorecard, indent=2) + "\n",
        encoding="utf-8",
    )
    (reports_dir / "evaluation_scorecard.md").write_text(
        format_scorecard_markdown(scorecard),
        encoding="utf-8",
    )

    save_overlay_plot(frame, analysis, reports_dir / "frame_overlay.png")
    save_scene_plot(analysis, reports_dir / "scene_3d.png")
    save_depth_heatmap(frame, analysis, reports_dir / "depth_heatmap.png")
    save_occupancy_grid_plot(build_occupancy_grid(analysis), reports_dir / "occupancy_grid.png")
    scenario_paths = list((PROJECT_ROOT / "data" / "scenarios").glob("*.json"))
    scenario_rows = scenario_summary_rows(analyze_scenarios(scenario_paths, rules))
    (reports_dir / "scenario_summary.json").write_text(
        json.dumps(scenario_rows, indent=2) + "\n",
        encoding="utf-8",
    )
    write_scenario_csv(reports_dir / "scenario_summary.csv", scenario_rows)

    timeline_monitor = TemporalOccupancyMonitor(hazard_frames_to_block=2, clear_frames_to_release=2)
    temporal_decisions = []
    for frame_path in sorted((PROJECT_ROOT / "data" / "timeline_frames").glob("*.json")):
        temporal_frame = load_vision_feed(frame_path)
        temporal_analysis = analyze_frame(temporal_frame, rules)
        temporal_decisions.append(timeline_monitor.update(temporal_analysis))
    (reports_dir / "temporal_summary.json").write_text(
        json.dumps([decision.to_dict() for decision in temporal_decisions], indent=2) + "\n",
        encoding="utf-8",
    )
    write_temporal_csv(reports_dir / "temporal_summary.csv", temporal_decisions)
    save_temporal_timeline_plot(temporal_decisions, reports_dir / "temporal_timeline.png")

    save_dashboard(
        frame,
        analysis,
        reports_dir / "depth_guard_dashboard.html",
        overlay_image="frame_overlay.png",
        scene_image="scene_3d.png",
        heatmap_image="depth_heatmap.png",
        occupancy_grid_image="occupancy_grid.png",
        temporal_timeline_image="temporal_timeline.png",
    )

    print("Generated Depth-Guard submission artifacts:")
    print(f"- {reports_dir / 'depth_guard_dashboard.html'}")
    print(f"- {reports_dir / 'status_report.json'}")
    print(f"- {reports_dir / 'evaluation_scorecard.json'}")
    print(f"- {reports_dir / 'evaluation_scorecard.md'}")
    print(f"- {reports_dir / 'safety_events.json'}")
    print(f"- {reports_dir / 'safety_events.csv'}")
    print(f"- {reports_dir / 'terminal_report.txt'}")
    print(f"- {reports_dir / 'frame_overlay.png'}")
    print(f"- {reports_dir / 'scene_3d.png'}")
    print(f"- {reports_dir / 'depth_heatmap.png'}")
    print(f"- {reports_dir / 'occupancy_grid.png'}")
    print(f"- {reports_dir / 'scenario_summary.json'}")
    print(f"- {reports_dir / 'scenario_summary.csv'}")
    print(f"- {reports_dir / 'temporal_summary.json'}")
    print(f"- {reports_dir / 'temporal_summary.csv'}")
    print(f"- {reports_dir / 'temporal_timeline.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
