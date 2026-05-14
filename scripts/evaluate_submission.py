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
from depth_guard.io import load_rules, load_vision_feed


def main() -> int:
    frame = load_vision_feed(PROJECT_ROOT / "data" / "vision_feed_sample.json")
    rules = load_rules(PROJECT_ROOT / "data" / "logic_constraints.json")
    expectations = load_expectations(PROJECT_ROOT / "data" / "evaluation_expectations.json")
    analysis = analyze_frame(frame, rules)
    scorecard = evaluate_against_expectations(analysis, expectations)

    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / "evaluation_scorecard.json"
    md_path = reports_dir / "evaluation_scorecard.md"
    json_path.write_text(json.dumps(scorecard, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(format_scorecard_markdown(scorecard), encoding="utf-8")

    print("Depth-Guard Evaluation Scorecard")
    print(f"Decision match: {scorecard['final_decision_match']}")
    print(f"Status accuracy: {scorecard['status_accuracy']:.3f}")
    print(f"Zone accuracy: {scorecard['zone_accuracy']:.3f}")
    print(f"Total score: {scorecard['total_score_percent']:.2f}%")
    print(f"Passed: {scorecard['passed']}")
    print(f"Saved {json_path}")
    print(f"Saved {md_path}")
    return 0 if scorecard["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
