from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from .models import FrameAnalysis, ObjectStatus, VisionFrame
from .occupancy_grid import OccupancyGrid
from .tracking import TemporalDecision


STATUS_COLORS = {
    ObjectStatus.HAZARD: "#d62828",
    ObjectStatus.SAFE: "#2a9d8f",
    ObjectStatus.IGNORE: "#6c757d",
}


def save_scene_plot(analysis: FrameAnalysis, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(9, 6), dpi=140)
    axis = fig.add_subplot(111, projection="3d")

    for item in analysis.objects:
        color = STATUS_COLORS[item.status]
        marker_size = float(np.clip(item.volume_cm3 / 8.0, 25.0, 550.0))
        axis.scatter(item.x_m, item.z_m, item.y_m, s=marker_size, color=color, alpha=0.88)
        axis.text(item.x_m, item.z_m, item.y_m + 0.03, item.object_id, fontsize=8)

    axis.set_title("Depth-Guard 3D Occupancy Map")
    axis.set_xlabel("Horizontal offset (m)")
    axis.set_ylabel("Depth from camera (m)")
    axis.set_zlabel("Vertical offset (m)")
    axis.view_init(elev=22, azim=-58)
    axis.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_original_image(
    image: np.ndarray,
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, axis = plt.subplots(figsize=(10, 7.5), dpi=120)
    axis.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axis.set_title("Original Input Frame")
    axis.axis("off")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_depth_map_plot(
    depth_map: np.ndarray,
    output_path: str | Path,
    title: str = "Estimated Depth Map",
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    depth = np.array(depth_map, dtype=float)
    valid = np.isfinite(depth)
    if valid.any():
        min_val = float(np.nanmin(depth))
        max_val = float(np.nanmax(depth))
        if max_val > min_val:
            depth = (depth - min_val) / (max_val - min_val)
        else:
            depth = np.zeros_like(depth)
    else:
        depth = np.zeros_like(depth)

    depth[np.logical_not(valid)] = 0.0

    fig, axis = plt.subplots(figsize=(10, 7.5), dpi=120)
    image = axis.imshow(depth, cmap="gray")
    axis.set_title(title)
    axis.axis("off")
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Relative depth")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_overlay_plot(
    frame: VisionFrame,
    analysis: FrameAnalysis,
    output_path: str | Path,
    image: np.ndarray | None = None,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, axis = plt.subplots(figsize=(10, 7.5), dpi=120)
    if image is None:
        axis.imshow(np.full((frame.image_height_px, frame.image_width_px, 3), 242, dtype=np.uint8))
    else:
        axis.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axis.set_xlim(0, frame.image_width_px)
    axis.set_ylim(frame.image_height_px, 0)
    axis.set_title("Depth-Guard 2D Detection Overlay")
    axis.set_xlabel("Image x (px)")
    axis.set_ylabel("Image y (px)")

    for item in analysis.objects:
        color = STATUS_COLORS[item.status]
        x, y, width, height = item.bbox_xywh
        axis.add_patch(
            Rectangle(
                (x, y),
                width,
                height,
                linewidth=2.5,
                edgecolor=color,
                facecolor="none",
            )
        )
        label = f"{item.object_id} | {item.label} | {item.confidence:.2f}"
        axis.text(
            x,
            max(y - 8, 8),
            label,
            fontsize=8,
            color="white",
            bbox={"facecolor": color, "edgecolor": color, "pad": 2.5},
        )

    axis.axis("off")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_depth_heatmap(
    frame: VisionFrame,
    analysis: FrameAnalysis,
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    heatmap = np.full((frame.image_height_px, frame.image_width_px), np.nan, dtype=float)
    for item in analysis.objects:
        x, y, width, height = [int(round(value)) for value in item.bbox_xywh]
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(frame.image_width_px, x + max(1, width))
        y1 = min(frame.image_height_px, y + max(1, height))
        heatmap[y0:y1, x0:x1] = item.depth_m

    fig, axis = plt.subplots(figsize=(10, 7.5), dpi=120)
    image = axis.imshow(
        heatmap,
        cmap="viridis_r",
        vmin=0.0,
        vmax=max(3.0, max((item.depth_m for item in analysis.objects), default=3.0)),
    )
    axis.imshow(np.isnan(heatmap), cmap="gray", alpha=0.08)
    axis.set_title("Depth-Guard Simulated Depth Heatmap")
    axis.set_xlabel("Image x (px)")
    axis.set_ylabel("Image y (px)")
    colorbar = fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Depth (m)")

    for item in analysis.objects:
        x, y, width, height = item.bbox_xywh
        axis.text(
            x + width / 2.0,
            y + height / 2.0,
            f"{item.depth_m:.2f}m",
            ha="center",
            va="center",
            fontsize=8,
            color="white",
            bbox={"facecolor": "#172026", "edgecolor": "none", "pad": 2.5, "alpha": 0.78},
        )

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_occupancy_grid_plot(grid: OccupancyGrid, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, axis = plt.subplots(figsize=(9, 6), dpi=130)
    image = axis.imshow(
        grid.grid,
        cmap="RdYlGn_r",
        vmin=0.0,
        vmax=1.0,
        origin="lower",
        extent=[grid.x_min_m, grid.x_max_m, grid.z_min_m, grid.z_max_m],
        aspect="auto",
    )
    axis.axhline(1.0, color="#c81e1e", linewidth=2, linestyle="--", label="1.0m critical line")
    axis.set_title("Depth-Guard Occupancy Grid Risk Map")
    axis.set_xlabel("Horizontal offset from camera (m)")
    axis.set_ylabel("Depth from camera (m)")
    axis.legend(loc="upper right")
    colorbar = fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Cell risk")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_temporal_timeline_plot(decisions: list[TemporalDecision], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    x_values = np.arange(1, len(decisions) + 1)
    instant = [1 if decision.instantaneous_blocked else 0 for decision in decisions]
    latched = [1 if decision.latched_blocked else 0 for decision in decisions]
    risk = [decision.max_risk_score for decision in decisions]

    fig, axis = plt.subplots(figsize=(10, 5), dpi=130)
    axis.step(x_values, instant, where="mid", linewidth=2, color="#f4a261", label="Instant blocked")
    axis.step(x_values, latched, where="mid", linewidth=2.5, color="#c81e1e", label="Latched blocked")
    axis.plot(x_values, risk, marker="o", color="#2454a6", linewidth=2, label="Max risk")
    axis.set_ylim(-0.05, 1.05)
    axis.set_xticks(x_values)
    axis.set_xlabel("Timeline frame")
    axis.set_ylabel("State / risk")
    axis.set_title("Depth-Guard Temporal Safety Debounce")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="upper right")

    for x_value, decision in zip(x_values, decisions):
        label = "BLOCKED" if decision.latched_blocked else "CLEAR"
        axis.text(x_value, 0.05, label, ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
