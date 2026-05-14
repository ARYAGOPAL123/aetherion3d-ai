from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .models import BoundingBox, Detection, VisionFrame


DEFAULT_DEPTH_PRIORS_M = {
    "person": 2.4,
    "human": 2.4,
    "cardboard box": 1.4,
    "box": 1.4,
    "pallet": 1.8,
    "tool": 1.2,
}


def average_depth_from_map(
    depth_map: np.ndarray,
    bbox: BoundingBox,
    default_depth_m: float,
) -> float:
    """Sample average metric depth inside a bbox from a dense depth map."""
    if depth_map.ndim != 2:
        raise ValueError("depth_map must be a 2D array of metric depth values")

    height, width = depth_map.shape
    x0 = max(0, int(round(bbox.x)))
    y0 = max(0, int(round(bbox.y)))
    x1 = min(width, int(round(bbox.x + bbox.width)))
    y1 = min(height, int(round(bbox.y + bbox.height)))
    if x0 >= x1 or y0 >= y1:
        return float(default_depth_m)

    crop = depth_map[y0:y1, x0:x1]
    valid = crop[np.isfinite(crop) & (crop > 0)]
    if valid.size == 0:
        return float(default_depth_m)
    return float(np.median(valid))


def depth_from_size_prior(
    label: str,
    bbox: BoundingBox,
    image_height_px: int,
    default_depth_m: float,
) -> float:
    """Fallback depth estimate for demo-only webcam use when no depth map exists."""
    normalized = label.lower().strip()
    base_depth = DEFAULT_DEPTH_PRIORS_M.get(normalized)
    if base_depth is None:
        base_depth = next(
            (value for key, value in DEFAULT_DEPTH_PRIORS_M.items() if key in normalized),
            default_depth_m,
        )

    height_ratio = bbox.height / max(float(image_height_px), 1.0)
    if height_ratio <= 0:
        return float(base_depth)
    scale = 0.22 / max(height_ratio, 0.03)
    return float(np.clip(base_depth * scale, 0.35, 6.0))


def load_depth_map(path: str | Path) -> np.ndarray:
    depth_path = Path(path)
    if depth_path.suffix.lower() == ".npy":
        return np.load(depth_path).astype(float)

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required to load image depth maps. Install opencv-python.") from exc

    image = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not read depth map: {depth_path}")
    depth = image.astype(float)
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    if depth.max(initial=0) > 50.0:
        depth = depth / 1000.0
    return depth


def estimate_depth_map_midas(
    image: np.ndarray,
    model_name: str = "DPT_Large",
) -> np.ndarray:
    """Estimate a dense relative depth map for a color image using MiDaS."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "MiDaS depth estimation requires torch. Install it with: python -m pip install torch torchvision"
        ) from exc

    try:
        model = torch.hub.load("intel-isl/MiDaS", model_name, pretrained=True)
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    except Exception as exc:
        raise RuntimeError(
            "Failed to load MiDaS model. Ensure internet access or use a local model cache."
        ) from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    transform = transforms.small_transform if model_name.endswith("_small") else transforms.default_transform
    input_tensor = transform(image).to(device)
    with torch.no_grad():
        prediction = model(input_tensor)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_map = prediction.cpu().numpy()
    depth_map = depth_map.astype(float)
    if np.isnan(depth_map).all():
        raise RuntimeError("MiDaS returned an invalid depth map")
    return depth_map


def normalize_depth_map(depth_map: np.ndarray) -> np.ndarray:
    values = depth_map.astype(float)
    finite_mask = np.isfinite(values)
    if not finite_mask.any():
        return np.zeros_like(values)
    minimum = float(np.nanmin(values))
    maximum = float(np.nanmax(values))
    if maximum <= minimum:
        return np.zeros_like(values)
    normalized = (values - minimum) / (maximum - minimum)
    normalized[~finite_mask] = 0.0
    return normalized


def build_frame_from_yolo_result(
    result: Any,
    image_width_px: int,
    image_height_px: int,
    frame_id: str,
    depth_map: np.ndarray | None = None,
    default_depth_m: float = 2.0,
) -> VisionFrame:
    detections: list[Detection] = []
    names = getattr(result, "names", None) or {}

    for index, box in enumerate(result.boxes):
        xyxy = box.xyxy[0].detach().cpu().numpy().astype(float)
        class_id = int(box.cls[0].detach().cpu().item())
        confidence = float(box.conf[0].detach().cpu().item()) if hasattr(box, "conf") else 1.0
        label = str(names.get(class_id, f"class_{class_id}"))
        bbox = BoundingBox(
            x=float(xyxy[0]),
            y=float(xyxy[1]),
            width=float(xyxy[2] - xyxy[0]),
            height=float(xyxy[3] - xyxy[1]),
        )
        if depth_map is not None:
            depth_m = average_depth_from_map(depth_map, bbox, default_depth_m)
        else:
            depth_m = depth_from_size_prior(label, bbox, image_height_px, default_depth_m)

        detections.append(
            Detection(
                object_id=f"YOLO_{index + 1:03d}",
                label=label,
                bbox=bbox,
                avg_depth_m=depth_m,
                confidence=confidence,
            )
        )

    return VisionFrame(
        frame_id=frame_id,
        image_width_px=image_width_px,
        image_height_px=image_height_px,
        detections=tuple(detections),
    )


def run_yolo_on_image(
    image_path: str | Path,
    frame_id: str,
    model_name: str = "yolov8n.pt",
    confidence: float = 0.35,
    depth_map_path: str | Path | None = None,
    depth_map: np.ndarray | None = None,
    default_depth_m: float = 2.0,
) -> tuple[VisionFrame, Any]:
    try:
        import cv2
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Live AI perception requires optional dependencies. "
            "Install them with: python -m pip install opencv-python ultralytics"
        ) from exc

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    model = YOLO(model_name)
    result = model.predict(image, conf=confidence, verbose=False)[0]
    if depth_map is None and depth_map_path is not None:
        depth_map = load_depth_map(depth_map_path)
    height, width = image.shape[:2]
    frame = build_frame_from_yolo_result(
        result,
        image_width_px=width,
        image_height_px=height,
        frame_id=frame_id,
        depth_map=depth_map,
        default_depth_m=default_depth_m,
    )
    return frame, image


def capture_webcam_frame(camera_index: int = 0) -> tuple[Any, int, int]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for webcam capture. Install opencv-python.") from exc

    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open webcam index {camera_index}")
    ok, frame = capture.read()
    capture.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not capture frame from webcam index {camera_index}")
    height, width = frame.shape[:2]
    return frame, width, height


def run_yolo_on_array(
    image: Any,
    image_width_px: int,
    image_height_px: int,
    frame_id: str,
    model_name: str = "yolov8n.pt",
    confidence: float = 0.35,
    depth_map: np.ndarray | None = None,
    default_depth_m: float = 2.0,
) -> VisionFrame:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Live AI perception requires ultralytics. Install it with: python -m pip install ultralytics"
        ) from exc

    model = YOLO(model_name)
    result = model.predict(image, conf=confidence, verbose=False)[0]
    return build_frame_from_yolo_result(
        result,
        image_width_px=image_width_px,
        image_height_px=image_height_px,
        frame_id=frame_id,
        depth_map=depth_map,
        default_depth_m=default_depth_m,
    )
