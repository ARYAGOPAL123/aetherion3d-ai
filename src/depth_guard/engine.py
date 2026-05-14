from __future__ import annotations

from collections import defaultdict

from .geometry import estimate_object_dimensions, estimate_object_position
from .models import DepthGuardRules, FrameAnalysis, ObjectAnalysis, ObjectStatus, VisionFrame


def analyze_frame(frame: VisionFrame, rules: DepthGuardRules) -> FrameAnalysis:
    object_results: list[ObjectAnalysis] = []

    for detection in frame.detections:
        width_m, height_m, thickness_m, volume_cm3 = estimate_object_dimensions(detection, frame, rules)
        x_m, y_m, z_m = estimate_object_position(detection, frame, rules)
        zone = rules.zone_for_depth(detection.avg_depth_m)
        status, reason = classify_object(detection.avg_depth_m, volume_cm3, rules)
        clearance_margin_m = detection.avg_depth_m - rules.critical_distance_m
        risk_score = calculate_risk_score(detection.avg_depth_m, volume_cm3, status, rules)

        object_results.append(
            ObjectAnalysis(
                object_id=detection.object_id,
                label=detection.label,
                bbox_xywh=detection.bbox.to_xywh(),
                depth_m=detection.avg_depth_m,
                confidence=getattr(detection, "confidence", 1.0),
                width_m=width_m,
                height_m=height_m,
                thickness_m=thickness_m,
                volume_cm3=volume_cm3,
                zone=zone,
                status=status,
                risk_score=risk_score,
                clearance_margin_m=clearance_margin_m,
                reason=reason,
                x_m=x_m,
                y_m=y_m,
                z_m=z_m,
            )
        )

    hazards = [item for item in object_results if item.status == ObjectStatus.HAZARD]
    occupied_volume_cm3 = sum(
        item.volume_cm3
        for item in object_results
        if item.status != ObjectStatus.IGNORE and item.zone == "Zone A"
    )
    alerts = build_alerts(hazards, occupied_volume_cm3, rules)
    zone_summary = summarize_zones(object_results)
    safe_count = sum(1 for item in object_results if item.status == ObjectStatus.SAFE)
    ignored_count = sum(1 for item in object_results if item.status == ObjectStatus.IGNORE)
    max_risk_score = max((item.risk_score for item in object_results), default=0.0)

    return FrameAnalysis(
        frame_id=frame.frame_id,
        objects=tuple(object_results),
        is_blocked=bool(hazards),
        alerts=tuple(alerts),
        occupied_volume_cm3=occupied_volume_cm3,
        crowded_volume_cm3=rules.crowded_volume_cm3,
        hazard_count=len(hazards),
        ignored_count=ignored_count,
        safe_count=safe_count,
        max_risk_score=max_risk_score,
        zone_summary=zone_summary,
    )


def classify_object(
    depth_m: float,
    volume_cm3: float,
    rules: DepthGuardRules,
) -> tuple[ObjectStatus, str]:
    if volume_cm3 < rules.min_volume_cm3:
        return (
            ObjectStatus.IGNORE,
            f"estimated volume is below the {rules.min_volume_cm3:.1f} cm3 minimum",
        )
    if depth_m <= rules.critical_distance_m:
        return (
            ObjectStatus.HAZARD,
            f"object is inside the {rules.critical_distance_m:.2f}m critical distance",
        )
    return ObjectStatus.SAFE, "object is outside the critical distance"


def calculate_risk_score(
    depth_m: float,
    volume_cm3: float,
    status: ObjectStatus,
    rules: DepthGuardRules,
) -> float:
    if status == ObjectStatus.IGNORE:
        return 0.0
    distance_pressure = max(0.0, (rules.critical_distance_m - depth_m) / rules.critical_distance_m)
    proximity_signal = max(0.0, 1.0 - min(depth_m / max(rules.critical_distance_m * 3.0, 0.001), 1.0))
    volume_signal = min(volume_cm3 / max(rules.crowded_volume_cm3, 1.0), 1.0)
    if status == ObjectStatus.HAZARD:
        score = 0.70 + 0.20 * distance_pressure + 0.10 * volume_signal
    else:
        score = 0.55 * proximity_signal + 0.45 * volume_signal
    return min(round(score, 4), 1.0)


def summarize_zones(object_results: list[ObjectAnalysis]) -> dict[str, dict[str, float | int]]:
    summary: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {"objects": 0, "hazards": 0, "ignored": 0, "volume_cm3": 0.0}
    )
    for item in object_results:
        zone = summary[item.zone]
        zone["objects"] += 1
        zone["volume_cm3"] = float(zone["volume_cm3"]) + item.volume_cm3
        if item.status == ObjectStatus.HAZARD:
            zone["hazards"] += 1
        if item.status == ObjectStatus.IGNORE:
            zone["ignored"] += 1

    return {
        zone: {
            "objects": int(values["objects"]),
            "hazards": int(values["hazards"]),
            "ignored": int(values["ignored"]),
            "volume_cm3": round(float(values["volume_cm3"]), 2),
        }
        for zone, values in sorted(summary.items())
    }


def build_alerts(
    hazards: list[ObjectAnalysis],
    occupied_volume_cm3: float,
    rules: DepthGuardRules,
) -> list[str]:
    alerts: list[str] = []
    if hazards:
        hazard_names = ", ".join(f"{item.object_id} ({item.label})" for item in hazards)
        alerts.append(f"Loading zone blocked by {hazard_names}.")
    if occupied_volume_cm3 >= rules.crowded_volume_cm3:
        alerts.append(
            "Crowding warning: Zone A occupied volume "
            f"{occupied_volume_cm3:.1f} cm3 exceeds {rules.crowded_volume_cm3:.1f} cm3."
        )
    if not alerts:
        alerts.append("No blocking or crowding alert triggered.")
    return alerts
