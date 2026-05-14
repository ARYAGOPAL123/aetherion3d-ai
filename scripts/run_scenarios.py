from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from depth_guard.batch import analyze_scenarios, scenario_summary_rows, write_scenario_csv
from depth_guard.io import load_rules


def main() -> int:
    rules = load_rules(PROJECT_ROOT / "data" / "logic_constraints.json")
    scenario_paths = list((PROJECT_ROOT / "data" / "scenarios").glob("*.json"))
    analyses = analyze_scenarios(scenario_paths, rules)
    rows = scenario_summary_rows(analyses)

    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary_json = reports_dir / "scenario_summary.json"
    summary_csv = reports_dir / "scenario_summary.csv"

    summary_json.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    write_scenario_csv(summary_csv, rows)

    print("Depth-Guard Scenario Evaluation")
    print(f"{'Frame':<32} {'Decision':<9} {'Hazards':>7} {'Max Risk':>8} {'Zone A cm3':>11}")
    for row in rows:
        print(
            f"{str(row['frame_id']):<32} "
            f"{str(row['decision']):<9} "
            f"{int(row['hazards']):>7} "
            f"{float(row['max_risk']):>8.2f} "
            f"{float(row['zone_a_volume_cm3']):>11.1f}"
        )
    print(f"\nSaved {summary_json}")
    print(f"Saved {summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
