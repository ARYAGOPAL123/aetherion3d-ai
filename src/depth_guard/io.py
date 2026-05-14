from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import DepthGuardRules, Detection, VisionFrame


def load_json(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{file_path} must contain a JSON object")
    return data


def load_vision_feed(path: str | Path) -> VisionFrame:
    data = load_json(path)
    image_size = data.get("image_size", [800, 600])
    if len(image_size) != 2:
        raise ValueError("image_size must contain [width, height]")
    detections = tuple(Detection.from_mapping(item) for item in data.get("objects", []))
    return VisionFrame(
        frame_id=str(data.get("frame_id", Path(path).stem)),
        timestamp=data.get("timestamp"),
        image_width_px=int(image_size[0]),
        image_height_px=int(image_size[1]),
        detections=detections,
    )


def load_rules(path: str | Path) -> DepthGuardRules:
    return DepthGuardRules.from_mapping(load_json(path))


def write_report_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
