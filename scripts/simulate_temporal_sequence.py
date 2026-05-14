from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from depth_guard.engine import analyze_frame
from depth_guard.io import load_rules, load_vision_feed
from depth_guard.tracking import TemporalOccupancyMonitor, write_temporal_csv
from depth_guard.visualization import save_temporal_timeline_plot


def main() -> int:
    rules = load_rules(PROJECT_ROOT / "data" / "logic_constraints.json")
    frame_paths = sorted((PROJECT_ROOT / "data" / "timeline_frames").glob("*.json"))
    monitor = TemporalOccupancyMonitor(hazard_frames_to_block=2, clear_frames_to_release=2)

    decisions = []
    for frame_path in frame_paths:
        frame = load_vision_feed(frame_path)
        analysis = analyze_frame(frame, rules)
        decisions.append(monitor.update(analysis))

    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / "temporal_summary.json"
    csv_path = reports_dir / "temporal_summary.csv"
    plot_path = reports_dir / "temporal_timeline.png"

    json_path.write_text(
        json.dumps([decision.to_dict() for decision in decisions], indent=2) + "\n",
        encoding="utf-8",
    )
    write_temporal_csv(csv_path, decisions)
    save_temporal_timeline_plot(decisions, plot_path)

    print("Depth-Guard Temporal Sequence")
    print(f"{'Frame':<34} {'Instant':<8} {'Latched':<8} {'Hazard':>6} {'Clear':>5} Reason")
    for decision in decisions:
        print(
            f"{decision.frame_id:<34} "
            f"{str(decision.instantaneous_blocked):<8} "
            f"{str(decision.latched_blocked):<8} "
            f"{decision.hazard_streak:>6} "
            f"{decision.clear_streak:>5} "
            f"{decision.reason}"
        )
    print(f"\nSaved {json_path}")
    print(f"Saved {csv_path}")
    print(f"Saved {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
