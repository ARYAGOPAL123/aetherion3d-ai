from __future__ import annotations

import csv
from pathlib import Path

from .engine import analyze_frame
from .io import load_vision_feed
from .models import DepthGuardRules, FrameAnalysis


def analyze_scenarios(paths: list[Path], rules: DepthGuardRules) -> list[FrameAnalysis]:
    analyses: list[FrameAnalysis] = []
    for path in sorted(paths):
        frame = load_vision_feed(path)
        analyses.append(analyze_frame(frame, rules))
    return analyses


def scenario_summary_rows(analyses: list[FrameAnalysis]) -> list[dict[str, str | int | float]]:
    rows: list[dict[str, str | int | float]] = []
    for analysis in analyses:
        rows.append(
            {
                "frame_id": analysis.frame_id,
                "decision": "BLOCKED" if analysis.is_blocked else "CLEAR",
                "objects": len(analysis.objects),
                "hazards": analysis.hazard_count,
                "safe": analysis.safe_count,
                "ignored": analysis.ignored_count,
                "max_risk": round(analysis.max_risk_score, 3),
                "zone_a_volume_cm3": round(analysis.occupied_volume_cm3, 2),
                "alerts": " | ".join(analysis.alerts),
            }
        )
    return rows


def write_scenario_csv(path: str | Path, rows: list[dict[str, str | int | float]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
