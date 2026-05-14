from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import FrameAnalysis, ObjectStatus


@dataclass(frozen=True)
class SafetyEvent:
    event_id: str
    frame_id: str
    severity: str
    category: str
    message: str
    recommended_action: str
    related_objects: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "frame_id": self.frame_id,
            "severity": self.severity,
            "category": self.category,
            "message": self.message,
            "recommended_action": self.recommended_action,
            "related_objects": list(self.related_objects),
        }


def build_safety_events(analysis: FrameAnalysis) -> list[SafetyEvent]:
    events: list[SafetyEvent] = []
    hazards = tuple(item.object_id for item in analysis.objects if item.status == ObjectStatus.HAZARD)
    ignored = tuple(item.object_id for item in analysis.objects if item.status == ObjectStatus.IGNORE)

    if hazards:
        events.append(
            SafetyEvent(
                event_id=f"{analysis.frame_id}-critical-blocked",
                frame_id=analysis.frame_id,
                severity="CRITICAL",
                category="ZONE_BLOCKED",
                message=f"Loading zone blocked by {', '.join(hazards)}.",
                recommended_action="Stop robot entry, notify operator, and wait for zone-clear confirmation.",
                related_objects=hazards,
            )
        )

    if analysis.occupied_volume_cm3 >= analysis.crowded_volume_cm3:
        events.append(
            SafetyEvent(
                event_id=f"{analysis.frame_id}-warning-crowded",
                frame_id=analysis.frame_id,
                severity="WARNING",
                category="CROWDING",
                message=(
                    "Relevant Zone A occupied volume "
                    f"{analysis.occupied_volume_cm3:.1f} cm3 exceeds "
                    f"{analysis.crowded_volume_cm3:.1f} cm3."
                ),
                recommended_action="Reduce loading-zone clutter before autonomous movement.",
                related_objects=tuple(
                    item.object_id for item in analysis.objects if item.status != ObjectStatus.IGNORE
                ),
            )
        )

    if ignored:
        events.append(
            SafetyEvent(
                event_id=f"{analysis.frame_id}-info-ignored-small",
                frame_id=analysis.frame_id,
                severity="INFO",
                category="SMALL_OBJECT_FILTER",
                message=f"Ignored small object(s): {', '.join(ignored)}.",
                recommended_action="No stop action required; keep threshold calibrated for site policy.",
                related_objects=ignored,
            )
        )

    if not events:
        events.append(
            SafetyEvent(
                event_id=f"{analysis.frame_id}-info-clear",
                frame_id=analysis.frame_id,
                severity="INFO",
                category="ZONE_CLEAR",
                message="No blocking or crowding safety event detected.",
                recommended_action="Zone may remain available for robot operation.",
                related_objects=(),
            )
        )

    return events


def write_safety_events_json(path: str | Path, events: list[SafetyEvent]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps([event.to_dict() for event in events], indent=2) + "\n",
        encoding="utf-8",
    )


def write_safety_events_csv(path: str | Path, events: list[SafetyEvent]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [event.to_dict() for event in events]
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
