from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class ObjectStatus(str, Enum):
    HAZARD = "HAZARD"
    SAFE = "SAFE"
    IGNORE = "IGNORE"


@dataclass(frozen=True)
class BoundingBox:
    x: float
    y: float
    width: float
    height: float

    @classmethod
    def from_xywh(cls, values: list[float] | tuple[float, float, float, float]) -> "BoundingBox":
        if len(values) != 4:
            raise ValueError(f"bbox_xywh must contain exactly four numbers, got {values!r}")
        x, y, width, height = values
        if width <= 0 or height <= 0:
            raise ValueError(f"bounding box width and height must be positive, got {values!r}")
        return cls(float(x), float(y), float(width), float(height))

    @property
    def center_x(self) -> float:
        return self.x + self.width / 2.0

    @property
    def center_y(self) -> float:
        return self.y + self.height / 2.0

    def to_xywh(self) -> list[float]:
        return [self.x, self.y, self.width, self.height]


@dataclass(frozen=True)
class Detection:
    object_id: str
    label: str
    bbox: BoundingBox
    avg_depth_m: float
    confidence: float = 1.0

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "Detection":
        required = {"object_id", "label", "bbox_xywh", "avg_depth_m"}
        missing = required.difference(data)
        if missing:
            raise ValueError(f"detection is missing required fields: {sorted(missing)}")
        depth = float(data["avg_depth_m"])
        if depth <= 0:
            raise ValueError(f"avg_depth_m must be positive for {data['object_id']!r}")
        return cls(
            object_id=str(data["object_id"]),
            label=str(data["label"]),
            bbox=BoundingBox.from_xywh(data["bbox_xywh"]),
            avg_depth_m=depth,
            confidence=float(data.get("confidence", 1.0)),
        )


@dataclass(frozen=True)
class VisionFrame:
    frame_id: str
    image_width_px: int
    image_height_px: int
    detections: tuple[Detection, ...]
    timestamp: str | None = None


@dataclass(frozen=True)
class CameraModel:
    image_width_px: int
    image_height_px: int
    horizontal_fov_degrees: float
    vertical_fov_degrees: float

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "CameraModel":
        return cls(
            image_width_px=int(data.get("image_width_px", 800)),
            image_height_px=int(data.get("image_height_px", 600)),
            horizontal_fov_degrees=float(data.get("horizontal_fov_degrees", 69.0)),
            vertical_fov_degrees=float(data.get("vertical_fov_degrees", 55.0)),
        )


@dataclass(frozen=True)
class ZoneRule:
    name: str
    min_depth_m: float
    max_depth_m: float | None

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "ZoneRule":
        max_depth = data.get("max_depth_m")
        return cls(
            name=str(data["name"]),
            min_depth_m=float(data.get("min_depth_m", 0.0)),
            max_depth_m=None if max_depth is None else float(max_depth),
        )

    def contains(self, depth_m: float) -> bool:
        if depth_m < self.min_depth_m:
            return False
        if self.max_depth_m is None:
            return True
        return depth_m < self.max_depth_m


@dataclass(frozen=True)
class DepthGuardRules:
    critical_distance_m: float
    min_volume_cm3: float
    crowded_volume_cm3: float
    camera: CameraModel
    zones: tuple[ZoneRule, ...]
    class_depth_factors: dict[str, float]

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "DepthGuardRules":
        zones = tuple(ZoneRule.from_mapping(item) for item in data.get("zones", []))
        if not zones:
            zones = (
                ZoneRule("Zone A", 0.0, 2.0),
                ZoneRule("Zone B", 2.0, 5.0),
                ZoneRule("Zone C", 5.0, None),
            )
        return cls(
            critical_distance_m=float(data.get("critical_distance_m", 1.0)),
            min_volume_cm3=float(data.get("min_volume_cm3", 20.0)),
            crowded_volume_cm3=float(data.get("crowded_volume_cm3", 5000.0)),
            camera=CameraModel.from_mapping(data.get("camera", {})),
            zones=zones,
            class_depth_factors={
                str(key).lower(): float(value)
                for key, value in data.get("class_depth_factors", {"default": 0.5}).items()
            },
        )

    def depth_factor_for_label(self, label: str) -> float:
        normalized = label.lower().strip()
        if normalized in self.class_depth_factors:
            return self.class_depth_factors[normalized]
        for key, value in self.class_depth_factors.items():
            if key != "default" and key in normalized:
                return value
        return self.class_depth_factors.get("default", 0.5)

    def zone_for_depth(self, depth_m: float) -> str:
        for zone in self.zones:
            if zone.contains(depth_m):
                return zone.name
        return "Out of Range"


@dataclass(frozen=True)
class ObjectAnalysis:
    object_id: str
    label: str
    bbox_xywh: list[float]
    depth_m: float
    confidence: float
    width_m: float
    height_m: float
    thickness_m: float
    volume_cm3: float
    zone: str
    status: ObjectStatus
    risk_score: float
    clearance_margin_m: float
    reason: str
    x_m: float
    y_m: float
    z_m: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "label": self.label,
            "bbox_xywh": self.bbox_xywh,
            "depth_m": round(self.depth_m, 4),
            "confidence": round(self.confidence, 3),
            "estimated_size_m": {
                "width": round(self.width_m, 4),
                "height": round(self.height_m, 4),
                "thickness": round(self.thickness_m, 4),
            },
            "estimated_volume_cm3": round(self.volume_cm3, 2),
            "zone": self.zone,
            "status": self.status.value,
            "risk_score": round(self.risk_score, 3),
            "clearance_margin_m": round(self.clearance_margin_m, 4),
            "reason": self.reason,
            "position_m": {
                "x": round(self.x_m, 4),
                "y": round(self.y_m, 4),
                "z": round(self.z_m, 4),
            },
        }


@dataclass(frozen=True)
class FrameAnalysis:
    frame_id: str
    objects: tuple[ObjectAnalysis, ...]
    is_blocked: bool
    alerts: tuple[str, ...]
    occupied_volume_cm3: float
    crowded_volume_cm3: float
    hazard_count: int
    ignored_count: int
    safe_count: int
    max_risk_score: float
    zone_summary: dict[str, dict[str, float | int]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "final_decision": "BLOCKED" if self.is_blocked else "CLEAR",
            "is_blocked": self.is_blocked,
            "hazard_count": self.hazard_count,
            "safe_count": self.safe_count,
            "ignored_count": self.ignored_count,
            "max_risk_score": round(self.max_risk_score, 3),
            "occupied_volume_cm3": round(self.occupied_volume_cm3, 2),
            "crowded_volume_cm3": round(self.crowded_volume_cm3, 2),
            "zone_summary": self.zone_summary,
            "alerts": list(self.alerts),
            "objects": [item.to_dict() for item in self.objects],
        }
