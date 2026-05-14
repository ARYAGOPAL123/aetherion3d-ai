"""Real-world dataset loaders for production evaluation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .models import BoundingBox, Detection, VisionFrame


class NuScenesLoader:
    """Load nuScenes dataset and convert to VisionFrame format."""

    def __init__(self, dataroot: str | Path):
        """
        Initialize nuScenes loader.
        
        Args:
            dataroot: Path to nuScenes root directory (contains v1.0-{mini,trainval})
        """
        self.dataroot = Path(dataroot)
        self.version_dir = None
        self._load_metadata()

    def _load_metadata(self) -> None:
        """Load nuScenes JSON metadata files."""
        # Check for mini or full version
        mini_dir = self.dataroot / "v1.0-mini"
        trainval_dir = self.dataroot / "v1.0-trainval"
        
        if mini_dir.exists():
            self.version_dir = mini_dir
            self.version = "v1.0-mini"
        elif trainval_dir.exists():
            self.version_dir = trainval_dir
            self.version = "v1.0-trainval"
        else:
            raise ValueError(
                f"nuScenes data not found in {self.dataroot}. "
                "Expected v1.0-mini or v1.0-trainval subdirectory."
            )

        # Load metadata JSON files
        self.scenes = self._load_json("scenes.json")
        self.samples = self._load_json("sample.json")
        self.sample_data = self._load_json("sample_data.json")
        self.calibrated_sensor = self._load_json("calibrated_sensor.json")
        self.ego_pose = self._load_json("ego_pose.json")
        self.annotations_3d = self._load_json("annotation.json")
        self.categories = self._load_json("category.json")

    def _load_json(self, filename: str) -> list[dict[str, Any]]:
        """Load a nuScenes JSON metadata file."""
        path = self.version_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"nuScenes metadata not found: {path}")
        with open(path, "r") as f:
            return json.load(f)

    def get_scenes(self) -> list[dict[str, Any]]:
        """Return all scene records."""
        return self.scenes

    def get_scene_sample_tokens(self, scene_token: str) -> list[str]:
        """Get all sample tokens for a scene in order."""
        for scene in self.scenes:
            if scene["token"] == scene_token:
                sample_token = scene["first_sample_token"]
                tokens = []
                while sample_token:
                    tokens.append(sample_token)
                    for sample in self.samples:
                        if sample["token"] == sample_token:
                            sample_token = sample["next"]
                            break
                    else:
                        sample_token = ""
                return tokens
        return []

    def sample_token_to_frame(
        self,
        sample_token: str,
        camera_channel: str = "CAM_FRONT",
    ) -> VisionFrame:
        """
        Convert a nuScenes sample to VisionFrame.

        Args:
            sample_token: nuScenes sample token
            camera_channel: Which camera to use (CAM_FRONT, CAM_LEFT, CAM_RIGHT, etc.)

        Returns:
            VisionFrame with detections from the sample
        """
        sample = None
        for s in self.samples:
            if s["token"] == sample_token:
                sample = s
                break
        
        if not sample:
            raise ValueError(f"Sample token not found: {sample_token}")

        # Get camera sample_data
        sample_data_token = sample["data"][camera_channel]
        sample_data = None
        for sd in self.sample_data:
            if sd["token"] == sample_data_token:
                sample_data = sd
                break
        
        if not sample_data:
            raise ValueError(f"Sample data not found: {sample_data_token}")

        # Get image dimensions
        image_path = self.version_dir / sample_data["filename"]
        image_width_px, image_height_px = self._get_image_dimensions(image_path)

        # Get calibration
        calibration = None
        for c in self.calibrated_sensor:
            if c["token"] == sample_data["calibrated_sensor_token"]:
                calibration = c
                break
        
        if not calibration:
            raise ValueError(f"Calibration not found")

        # Get ego pose
        ego = None
        for e in self.ego_pose:
            if e["token"] == sample_data["ego_pose_token"]:
                ego = e
                break
        
        if not ego:
            raise ValueError(f"Ego pose not found")

        # Get annotations
        ann_tokens = sample.get("anns", [])
        detections = []

        for idx, ann_token in enumerate(ann_tokens):
            ann = None
            for a in self.annotations_3d:
                if a["token"] == ann_token:
                    ann = a
                    break
            
            if not ann:
                continue

            # Get category name
            category = None
            for c in self.categories:
                if c["token"] == ann["category_token"]:
                    category = c
                    break
            
            label = category["name"] if category else "unknown"

            # Project 3D to 2D (simplified)
            center_3d = ann["translation"]
            size_3d = ann["size"]

            try:
                depth = float(center_3d[2])  # z is depth
                if depth <= 0.5 or depth > 100.0:
                    continue

                # Simplified 2D projection
                bbox_2d = BoundingBox(
                    x=400.0 + (center_3d[0] * 100.0 / depth),
                    y=300.0 + (center_3d[1] * 100.0 / depth),
                    width=max(10.0, size_3d[0] * 100.0 / depth),
                    height=max(10.0, size_3d[2] * 100.0 / depth),
                )

                detection = Detection(
                    object_id=f"{camera_channel}_{idx:04d}",
                    label=label,
                    bbox=bbox_2d,
                    avg_depth_m=depth,
                )
                detections.append(detection)
            except Exception:
                continue

        return VisionFrame(
            frame_id=f"{sample_token}_{camera_channel}",
            image_width_px=image_width_px,
            image_height_px=image_height_px,
            detections=tuple(detections),
            timestamp=sample_data.get("timestamp"),
        )

    def _get_image_dimensions(self, image_path: Path) -> tuple[int, int]:
        """Get image dimensions (defaults to nuScenes standard)."""
        try:
            from PIL import Image
            img = Image.open(image_path)
            return img.width, img.height
        except Exception:
            return 1600, 900