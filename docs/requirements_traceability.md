# Depth-Guard Requirements Traceability

This document maps the original challenge requirements to the implemented project files.

## Challenge Requirement Checklist

| Requirement | Status | Where Implemented |
|---|---:|---|
| Python-based system | Done | `src/depth_guard/`, `scripts/depth_guard_demo.py`, `scripts/generate_submission_artifacts.py` |
| Process Dataset A vision feed | Done | `data/vision_feed_sample.json`, `src/depth_guard/io.py` |
| Process Dataset B logic constraints | Done | `data/logic_constraints.json`, `src/depth_guard/io.py` |
| Object detection boxes | Done | Simulated boxes in `data/vision_feed_sample.json`; optional YOLO adapter in `src/depth_guard/perception.py` |
| Depth mapping | Done | Simulated average depth in dataset; optional aligned depth-map support in `src/depth_guard/perception.py` |
| Estimate 3D object size / volume | Done | `src/depth_guard/geometry.py` |
| Calculate distance from camera | Done | `avg_depth_m` from dataset and projected position in `src/depth_guard/geometry.py` |
| Object closer than 1m becomes hazard | Done | `src/depth_guard/engine.py` |
| Ignore objects below 20 cm3 | Done | `src/depth_guard/engine.py` |
| Categorize by zones | Done | `src/depth_guard/models.py`, `data/logic_constraints.json` |
| Print warning / alert | Done | `src/depth_guard/reporting.py`, terminal output, dashboard alert panel |
| Final blocked decision | Done | `src/depth_guard/engine.py`, `reports/status_report.json`, dashboard |
| 3D scatter plot | Done | `src/depth_guard/visualization.py`, `reports/scene_3d.png` |
| Professional visual output | Done | `reports/depth_guard_dashboard.html`, overlay, heatmap, 3D plot |
| Unit tests | Done | `tests/test_depth_guard.py` |
| Scenario evaluation | Done | `data/scenarios/`, `scripts/run_scenarios.py` |
| Temporal safety monitoring | Done | `src/depth_guard/tracking.py`, `scripts/simulate_temporal_sequence.py` |
| Occupancy grid output | Done | `src/depth_guard/occupancy_grid.py`, `reports/occupancy_grid.png` |
| Formal scoring against expected output | Done | `data/evaluation_expectations.json`, `scripts/evaluate_submission.py` |
| Safety/audit event logging | Done | `src/depth_guard/safety.py`, `reports/safety_events.json` |

## Simple Tech Recommendation Mapping

| Recommended Tech | Status | Notes |
|---|---:|---|
| Python | Done | Entire project is Python. |
| NumPy | Done | Used for depth math, sampling, clipping, heatmap arrays, and geometry support. |
| Matplotlib | Done | Used for 2D overlay, 3D scatter plot, and depth heatmap artifacts. |
| OpenCV | Done as optional production adapter | Used by `src/depth_guard/perception.py` for real image loading, webcam capture, and depth-map image loading. The core challenge path does not require OpenCV so the project runs without heavy dependencies. |
| YOLOv8 | Done as optional extension | `scripts/ai_perception_demo.py` supports YOLOv8 through `ultralytics`. |
| MiDaS / stereo / simulated depth | Done via simulated depth | The challenge-provided simulated depth feed is implemented. The optional adapter can accept aligned depth maps from MiDaS, stereo depth, RealSense, Kinect, or LiDAR. |

## Expected Challenge Output

The implemented sample produces the required decisions:

| Object | Expected | Implemented |
|---|---|---|
| `Obj_101` Cardboard Box at `0.8m` | `HAZARD` | `HAZARD` |
| `Obj_104` Small Tool at `0.5m` | `IGNORE` | `IGNORE` |
| `Obj_102` Human at `2.5m` | `SAFE` / Zone B | `SAFE` / Zone B |
| Final Decision | Loading Zone Blocked | Loading Zone Blocked |

## Important Professional Note

The core challenge path is intentionally deterministic and lightweight because the interviewer supplied simulated JSON data. The optional AI path is separated cleanly:

- Required evaluation: run `python -B scripts/depth_guard_demo.py`.
- Professional dashboard: run `python -B scripts/generate_submission_artifacts.py`.
- Optional real AI demo: install `requirements-vision.txt` and run `scripts/ai_perception_demo.py`.

This separation is a good engineering decision: the safety/occupancy logic remains testable even when the AI model or camera hardware changes.
