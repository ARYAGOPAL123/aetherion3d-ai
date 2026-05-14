# Depth-Guard - 3D Spatial Occupancy Monitor

Depth-Guard is a Python challenge submission for the AI / 3D perception problem: decide whether a smart warehouse loading zone is clear or blocked using object detections, estimated depth, and approximate 3D occupancy.

The project is built as a production-style mini system instead of a disposable script. It has a configurable rules engine, documented geometry assumptions, terminal and JSON reports, a browser dashboard, generated visualizations, scenario evaluation, and tests.

## What It Solves

A 2D detector can draw a box around an object, but it cannot reliably tell whether the object has real physical presence in the loading zone. Depth-Guard combines:

- 2D object detections: object id, label, bounding box.
- Depth perception: average depth in meters.
- Camera geometry: converts pixels plus depth into approximate physical size.
- Occupancy rules: classifies objects as `HAZARD`, `SAFE`, or `IGNORE`.
- Final loading-zone decision: `BLOCKED` when a relevant object is too close.

## Quick Start

From the project root:

```powershell
python scripts/depth_guard_demo.py
```

Generate the professional demo package:

```powershell
python -B scripts/generate_submission_artifacts.py
```

Then open:

```text
reports/depth_guard_dashboard.html
```

Run tests:

```powershell
python -B -m unittest discover -s tests
```

Install as a package, optional:

```powershell
python -m pip install -e .
depth-guard --vision data/vision_feed_sample.json --rules data/logic_constraints.json
```

Optional YOLO/OpenCV image or webcam demo:

```powershell
python -m pip install -r requirements-vision.txt
python -B scripts/ai_perception_demo.py --image samples/frame.jpg --depth-map samples/depth.npy
python -B scripts/ai_perception_demo.py --webcam 0
```

## Expected Console Output

```text
Depth-Guard Frame Analysis
Frame: warehouse_frame_001

Object   Label          Depth   Volume      Zone   Risk Status   Reason
Obj_101  Cardboard Box  0.80m     327.9 cm3 Zone A 0.75 HAZARD   object is inside the 1.00m critical distance
Obj_102  Human          2.50m   18153.8 cm3 Zone B 0.54 SAFE     object is outside the critical distance
Obj_103  Pallet         1.20m     144.8 cm3 Zone A 0.34 SAFE     object is outside the critical distance
Obj_104  Small Tool     0.50m       0.1 cm3 Zone A 0.00 IGNORE   estimated volume is below the 20.0 cm3 minimum

Scene Summary: 1 hazard(s), 2 safe object(s), 1 ignored object(s), max risk 0.75
Final Decision: Loading Zone Blocked
Alert: Loading zone blocked by Obj_101 (Cardboard Box).
```

Exact volume values can vary if camera calibration or class thickness factors are changed in `data/logic_constraints.json`.

## Project Structure

```text
.
|-- data/
|   |-- evaluation_expectations.json
|   |-- scenarios/
|   |-- logic_constraints.json
|   |-- timeline_frames/
|   `-- vision_feed_sample.json
|-- docs/
|   |-- architecture.md
|   |-- failure_modes_and_safety_case.md
|   |-- project_walkthrough.md
|   |-- reviewer_guide.md
|   |-- requirements_traceability.md
|   `-- submission_brief.md
|-- reports/
|   |-- depth_guard_dashboard.html
|   |-- evaluation_scorecard.json
|   |-- evaluation_scorecard.md
|   |-- frame_overlay.png
|   |-- scene_3d.png
|   |-- depth_heatmap.png
|   |-- occupancy_grid.png
|   |-- safety_events.csv
|   |-- safety_events.json
|   |-- scenario_summary.csv
|   |-- scenario_summary.json
|   |-- temporal_summary.csv
|   |-- temporal_summary.json
|   |-- temporal_timeline.png
|   `-- status_report.json
|-- scripts/
|   |-- ai_perception_demo.py
|   |-- depth_guard_demo.py
|   |-- evaluate_submission.py
|   |-- generate_submission_artifacts.py
|   |-- package_submission.py
|   |-- run_scenarios.py
|   `-- simulate_temporal_sequence.py
|-- src/
|   `-- depth_guard/
|       |-- batch.py
|       |-- cli.py
|       |-- dashboard.py
|       |-- engine.py
|       |-- evaluation.py
|       |-- geometry.py
|       |-- io.py
|       |-- models.py
|       |-- occupancy_grid.py
|       |-- perception.py
|       |-- reporting.py
|       |-- safety.py
|       |-- tracking.py
|       `-- visualization.py
|-- tests/
|   `-- test_depth_guard.py
|-- Dockerfile
|-- pyproject.toml
|-- requirements.txt
|-- requirements-vision.txt
`-- README.md
```

## System Architecture Pipeline

Depth-Guard is built as a layered pipeline that separates perception, geometry, safety rules, and visualization.

1. Perception input
   - `scripts/depth_guard_demo.py` loads challenge-provided simulated JSON frames.
   - `scripts/ai_perception_demo.py` loads real RGB input from `--image` or `--webcam`.
   - YOLOv8 via `ultralytics` provides 2D detections (label, bbox, confidence).
   - Optional MiDaS depth estimation is available via `--midas` for dense depth maps.

2. Frame representation
   - `src/depth_guard/perception.py` converts detections into internal `VisionFrame` objects.
   - Each detection becomes a `Detection` with `label`, `bbox`, `confidence`, and `avg_depth_m`.
   - Dense depth maps are sampled per bounding box; otherwise fallback priors estimate depth from object size.

3. Geometry and size approximation
   - `src/depth_guard/geometry.py` converts pixel bbox size and depth into approximate physical width and height.
   - A configurable object thickness factor is applied from `data/logic_constraints.json`.
   - Estimated volume is computed in cubic centimeters.

4. Safety rules engine
   - `src/depth_guard/engine.py` applies rules to each object using the configured `logic_constraints`.
   - Objects below the minimum volume are marked `IGNORE`.
   - Objects within the critical distance are marked `HAZARD` unless deemed safe by class/zone rules.
   - The frame is marked `BLOCKED` when a hazard is present.

5. Visualization and reporting
   - `src/depth_guard/visualization.py` saves original frame, overlay, scene plot, depth map, heatmap, and occupancy grid.
   - `src/depth_guard/dashboard.py` generates an HTML dashboard showing original RGB, detection overlay, depth map, 3D scene, and risk maps.
   - `src/depth_guard/reporting.py` formats terminal summaries and structured JSON outputs.

This architecture makes it easy to swap perception sources without changing the core safety logic, and it supports both the challenge dataset and real image + depth workflows.

## Visual Demo Outputs

The main review artifact is `reports/depth_guard_dashboard.html`. It includes:

- Executive decision cards: blocked or clear, hazard count, max risk, occupied volume.
- 2D detection overlay with color-coded statuses.
- 3D occupancy scatter plot.
- Simulated depth heatmap.
- Occupancy grid risk map.
- Object-level explainability table with risk, volume, distance, and decision reason.
- Embedded structured JSON output for review.

The project also includes a scenario runner:

```powershell
python -B scripts/run_scenarios.py
```

It produces `reports/scenario_summary.json` and `reports/scenario_summary.csv` across blocked, clear, and crowded examples.

The project also includes temporal safety monitoring:

```powershell
python -B scripts/simulate_temporal_sequence.py
```

It produces `reports/temporal_summary.json`, `reports/temporal_summary.csv`, and `reports/temporal_timeline.png`. This demonstrates a professional debounce strategy: a loading zone is latched as blocked only after stable hazard evidence and released only after stable clear evidence.

The project includes a formal scorecard:

```powershell
python -B scripts/evaluate_submission.py
```

It writes `reports/evaluation_scorecard.json` and `reports/evaluation_scorecard.md`.

Package everything for upload:

```powershell
python -B scripts/package_submission.py
```

It creates `DepthGuard_Submission.zip`.

## Real AI Perception Extension

The required challenge data is simulated, but the project also includes an optional YOLO/OpenCV adapter in `scripts/ai_perception_demo.py`.

Use it with a real image and aligned metric depth map:

```powershell
python -B scripts/ai_perception_demo.py --image samples/frame.jpg --depth-map samples/depth.npy
```

Use it with one webcam frame:

```powershell
python -B scripts/ai_perception_demo.py --webcam 0
```

The webcam mode uses YOLO for object detection and a size-prior fallback for depth if no depth map is available. For production, replace that fallback with MiDaS, Depth Anything, RealSense, Kinect, stereo depth, or LiDAR.

## Core Decision Logic

Depth-Guard applies the rules in this order:

1. Estimate physical width and height from bounding box size, image resolution, depth, and field of view.
2. Estimate physical thickness using a configurable class factor.
3. Convert estimated volume to cubic centimeters.
4. Ignore tiny objects below `MIN_VOLUME_SIZE`.
5. Mark relevant objects within `CRITICAL_DISTANCE` as `HAZARD`.
6. Calculate object-level risk scores for explainability.
7. Mark the loading zone `BLOCKED` if at least one hazard exists.

This means the sample `Small Tool` at `0.5m` is close, but still ignored because it is below the configured minimum volume. The `Cardboard Box` at `0.8m` is both close and large enough, so it blocks the loading zone.

## Geometry Assumption

The provided challenge dataset has average depth and 2D bounding boxes, but no real RGB-D image, point cloud, segmentation mask, or camera calibration. Depth-Guard therefore uses a transparent pinhole-camera approximation:

```text
physical_width_m  = 2 * depth_m * tan(horizontal_fov / 2) * bbox_width_px / image_width_px
physical_height_m = 2 * depth_m * tan(vertical_fov / 2) * bbox_height_px / image_height_px
thickness_m       = min(width_m, height_m) * class_depth_factor
volume_cm3        = width_m * height_m * thickness_m * 1,000,000
```

The `class_depth_factor` values are intentionally stored in the rules JSON so the system can be calibrated when a real camera or object measurements become available.

## Optional Real AI Extension

The production path for live data is:

- YOLOv8 or another detector for object labels and bounding boxes.
- MiDaS, Depth Anything, stereo depth, Intel RealSense, Azure Kinect, or LiDAR for depth.
- Replace the simulated JSON loader with a frame adapter that emits the same internal `Detection` objects.

The occupancy engine does not depend on a specific detector. That separation is deliberate: model choice can change without rewriting the safety logic.

## Project Highlights

- Clear separation of perception input, geometry, rules, reporting, and visualization.
- Configurable safety constraints instead of hard-coded magic numbers.
- Deterministic unit tests for the sample challenge outcome.
- Browser dashboard suitable for live review and operational demos.
- Scenario runner for blocked, clear, and crowded cases.
- Temporal monitoring to avoid flicker from noisy one-frame detections.
- Occupancy grid risk map for robot-style spatial reasoning.
- Formal evaluation scorecard against the challenge's expected output.
- Machine-readable safety event logs for auditability.
- Submission packager for clean delivery.
- Dockerfile and GitHub Actions workflow for reproducibility.
- Professional CLI suitable for demos and automated evaluation.
- Honest handling of assumptions where the challenge data is simulated.

For a requirement-by-requirement checklist, see `docs/requirements_traceability.md`.
