from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import FrameAnalysis


def load_expectations(path: str | Path) -> dict[str, Any]:
    expectation_path = Path(path)
    with expectation_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("evaluation expectations must be a JSON object")
    return data


def evaluate_against_expectations(
    analysis: FrameAnalysis,
    expectations: dict[str, Any],
) -> dict[str, Any]:
    expected_objects = expectations.get("objects", {})
    actual_by_id = {item.object_id: item for item in analysis.objects}
    object_results: list[dict[str, Any]] = []
    status_matches = 0
    zone_matches = 0

    for object_id, expected in sorted(expected_objects.items()):
        actual = actual_by_id.get(object_id)
        if actual is None:
            object_results.append(
                {
                    "object_id": object_id,
                    "found": False,
                    "status_match": False,
                    "zone_match": False,
                    "expected_status": expected.get("expected_status"),
                    "actual_status": None,
                    "expected_zone": expected.get("expected_zone"),
                    "actual_zone": None,
                }
            )
            continue

        expected_status = expected.get("expected_status")
        expected_zone = expected.get("expected_zone")
        status_match = actual.status.value == expected_status
        zone_match = actual.zone == expected_zone
        status_matches += int(status_match)
        zone_matches += int(zone_match)
        object_results.append(
            {
                "object_id": object_id,
                "found": True,
                "status_match": status_match,
                "zone_match": zone_match,
                "expected_status": expected_status,
                "actual_status": actual.status.value,
                "expected_zone": expected_zone,
                "actual_zone": actual.zone,
                "depth_m": round(actual.depth_m, 3),
                "volume_cm3": round(actual.volume_cm3, 2),
            }
        )

    expected_decision = expectations.get("expected_final_decision")
    actual_decision = "BLOCKED" if analysis.is_blocked else "CLEAR"
    final_decision_match = actual_decision == expected_decision
    total_objects = max(len(expected_objects), 1)
    status_accuracy = status_matches / total_objects
    zone_accuracy = zone_matches / total_objects
    final_decision_score = 1.0 if final_decision_match else 0.0
    total_score = round(
        100.0 * (0.60 * status_accuracy + 0.20 * zone_accuracy + 0.20 * final_decision_score),
        2,
    )

    return {
        "frame_id": analysis.frame_id,
        "expected_frame_id": expectations.get("frame_id"),
        "actual_final_decision": actual_decision,
        "expected_final_decision": expected_decision,
        "final_decision_match": final_decision_match,
        "status_accuracy": round(status_accuracy, 3),
        "zone_accuracy": round(zone_accuracy, 3),
        "total_score_percent": total_score,
        "passed": total_score >= 95.0 and final_decision_match,
        "object_results": object_results,
    }


def format_scorecard_markdown(scorecard: dict[str, Any]) -> str:
    lines = [
        "# Depth-Guard Evaluation Scorecard",
        "",
        f"- Frame: `{scorecard['frame_id']}`",
        f"- Expected decision: `{scorecard['expected_final_decision']}`",
        f"- Actual decision: `{scorecard['actual_final_decision']}`",
        f"- Status accuracy: `{scorecard['status_accuracy']:.3f}`",
        f"- Zone accuracy: `{scorecard['zone_accuracy']:.3f}`",
        f"- Total score: `{scorecard['total_score_percent']:.2f}%`",
        f"- Passed: `{scorecard['passed']}`",
        "",
        "| Object | Expected Status | Actual Status | Expected Zone | Actual Zone | Result |",
        "|---|---|---|---|---|---|",
    ]
    for item in scorecard["object_results"]:
        result = "PASS" if item["status_match"] and item["zone_match"] else "CHECK"
        lines.append(
            "| "
            f"{item['object_id']} | "
            f"{item['expected_status']} | "
            f"{item['actual_status']} | "
            f"{item['expected_zone']} | "
            f"{item['actual_zone']} | "
            f"{result} |"
        )
    lines.append("")
    return "\n".join(lines)
