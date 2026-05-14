from __future__ import annotations

import math

from .models import CameraModel, Detection, DepthGuardRules, VisionFrame


def estimate_object_dimensions(
    detection: Detection,
    frame: VisionFrame,
    rules: DepthGuardRules,
) -> tuple[float, float, float, float]:
    """Estimate width, height, thickness, and volume from 2D bbox plus depth."""
    camera = CameraModel(
        image_width_px=frame.image_width_px or rules.camera.image_width_px,
        image_height_px=frame.image_height_px or rules.camera.image_height_px,
        horizontal_fov_degrees=rules.camera.horizontal_fov_degrees,
        vertical_fov_degrees=rules.camera.vertical_fov_degrees,
    )

    depth_m = detection.avg_depth_m
    hfov_rad = math.radians(camera.horizontal_fov_degrees)
    vfov_rad = math.radians(camera.vertical_fov_degrees)

    visible_width_at_depth_m = 2.0 * depth_m * math.tan(hfov_rad / 2.0)
    visible_height_at_depth_m = 2.0 * depth_m * math.tan(vfov_rad / 2.0)

    width_m = visible_width_at_depth_m * detection.bbox.width / camera.image_width_px
    height_m = visible_height_at_depth_m * detection.bbox.height / camera.image_height_px
    thickness_factor = rules.depth_factor_for_label(detection.label)
    thickness_m = max(min(width_m, height_m) * thickness_factor, 0.001)
    volume_cm3 = width_m * height_m * thickness_m * 1_000_000.0
    return width_m, height_m, thickness_m, volume_cm3


def estimate_object_position(
    detection: Detection,
    frame: VisionFrame,
    rules: DepthGuardRules,
) -> tuple[float, float, float]:
    """Project the bbox center into a simple camera-centric 3D coordinate."""
    camera = CameraModel(
        image_width_px=frame.image_width_px or rules.camera.image_width_px,
        image_height_px=frame.image_height_px or rules.camera.image_height_px,
        horizontal_fov_degrees=rules.camera.horizontal_fov_degrees,
        vertical_fov_degrees=rules.camera.vertical_fov_degrees,
    )
    depth_m = detection.avg_depth_m
    normalized_x = (detection.bbox.center_x / camera.image_width_px) - 0.5
    normalized_y = 0.5 - (detection.bbox.center_y / camera.image_height_px)
    half_width_m = depth_m * math.tan(math.radians(camera.horizontal_fov_degrees) / 2.0)
    half_height_m = depth_m * math.tan(math.radians(camera.vertical_fov_degrees) / 2.0)
    x_m = normalized_x * 2.0 * half_width_m
    y_m = normalized_y * 2.0 * half_height_m
    z_m = depth_m
    return x_m, y_m, z_m
