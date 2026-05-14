from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .models import FrameAnalysis, ObjectStatus


@dataclass(frozen=True)
class OccupancyGrid:
    grid: np.ndarray
    x_min_m: float
    x_max_m: float
    z_min_m: float
    z_max_m: float
    cell_size_m: float
    labels: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        occupied_cells = int(np.count_nonzero(self.grid > 0))
        hazard_cells = int(np.count_nonzero(self.grid >= 0.75))
        return {
            "x_range_m": [self.x_min_m, self.x_max_m],
            "z_range_m": [self.z_min_m, self.z_max_m],
            "cell_size_m": self.cell_size_m,
            "shape": list(self.grid.shape),
            "occupied_cells": occupied_cells,
            "hazard_cells": hazard_cells,
            "max_cell_risk": round(float(np.max(self.grid)) if self.grid.size else 0.0, 3),
            "labels": list(self.labels),
        }


def build_occupancy_grid(
    analysis: FrameAnalysis,
    x_range_m: tuple[float, float] = (-1.2, 1.2),
    z_range_m: tuple[float, float] = (0.0, 4.0),
    cell_size_m: float = 0.2,
) -> OccupancyGrid:
    if cell_size_m <= 0:
        raise ValueError("cell_size_m must be positive")

    x_min, x_max = x_range_m
    z_min, z_max = z_range_m
    cols = int(np.ceil((x_max - x_min) / cell_size_m))
    rows = int(np.ceil((z_max - z_min) / cell_size_m))
    grid = np.zeros((rows, cols), dtype=float)
    labels: list[str] = []

    for item in analysis.objects:
        if item.status == ObjectStatus.IGNORE:
            continue
        col = int((item.x_m - x_min) / cell_size_m)
        row = int((item.z_m - z_min) / cell_size_m)
        if 0 <= row < rows and 0 <= col < cols:
            risk = max(item.risk_score, 0.25)
            if item.status == ObjectStatus.HAZARD:
                risk = max(risk, 0.85)
            grid[row, col] = max(grid[row, col], risk)
            labels.append(f"{item.object_id}@r{row}c{col}")

    return OccupancyGrid(
        grid=grid,
        x_min_m=x_min,
        x_max_m=x_max,
        z_min_m=z_min,
        z_max_m=z_max,
        cell_size_m=cell_size_m,
        labels=tuple(labels),
    )
