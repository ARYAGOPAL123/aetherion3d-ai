from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import FrameAnalysis, ObjectStatus


@dataclass(frozen=True)
class TemporalDecision:
    frame_id: str
    instantaneous_blocked: bool
    latched_blocked: bool
    hazard_streak: int
    clear_streak: int
    active_hazards: tuple[str, ...]
    max_risk_score: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "instantaneous_blocked": self.instantaneous_blocked,
            "latched_blocked": self.latched_blocked,
            "hazard_streak": self.hazard_streak,
            "clear_streak": self.clear_streak,
            "active_hazards": list(self.active_hazards),
            "max_risk_score": round(self.max_risk_score, 3),
            "reason": self.reason,
        }


class TemporalOccupancyMonitor:
    """Debounce zone state across frames to reduce flicker from noisy perception."""

    def __init__(
        self,
        hazard_frames_to_block: int = 2,
        clear_frames_to_release: int = 2,
    ) -> None:
        if hazard_frames_to_block < 1:
            raise ValueError("hazard_frames_to_block must be >= 1")
        if clear_frames_to_release < 1:
            raise ValueError("clear_frames_to_release must be >= 1")
        self.hazard_frames_to_block = hazard_frames_to_block
        self.clear_frames_to_release = clear_frames_to_release
        self.hazard_streak = 0
        self.clear_streak = 0
        self.latched_blocked = False

    def update(self, analysis: FrameAnalysis) -> TemporalDecision:
        active_hazards = tuple(
            item.object_id for item in analysis.objects if item.status == ObjectStatus.HAZARD
        )

        if active_hazards:
            self.hazard_streak += 1
            self.clear_streak = 0
        else:
            self.clear_streak += 1
            self.hazard_streak = 0

        if self.hazard_streak >= self.hazard_frames_to_block:
            self.latched_blocked = True
        elif self.clear_streak >= self.clear_frames_to_release:
            self.latched_blocked = False

        if self.latched_blocked:
            reason = (
                "blocked state latched after stable hazard evidence"
                if active_hazards
                else "blocked state held until clear evidence is stable"
            )
        else:
            reason = (
                "hazard observed but debounce threshold not reached"
                if active_hazards
                else "clear state confirmed"
            )

        return TemporalDecision(
            frame_id=analysis.frame_id,
            instantaneous_blocked=analysis.is_blocked,
            latched_blocked=self.latched_blocked,
            hazard_streak=self.hazard_streak,
            clear_streak=self.clear_streak,
            active_hazards=active_hazards,
            max_risk_score=analysis.max_risk_score,
            reason=reason,
        )


def write_temporal_csv(path: str | Path, decisions: list[TemporalDecision]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [decision.to_dict() for decision in decisions]
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
