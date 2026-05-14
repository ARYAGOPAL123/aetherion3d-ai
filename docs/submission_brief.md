# Depth-Guard Submission Brief

## Project

Depth-Guard is a Python-based 3D spatial occupancy monitor for a smart warehouse loading zone. It combines object detections, depth values, approximate metric volume estimation, and deterministic safety rules to decide whether a zone is clear, blocked, or crowded.

## Primary Demo Artifact

Open this file in a browser:

```text
reports/depth_guard_dashboard.html
```

The dashboard shows:

- Final blocked or clear decision.
- Hazard, safe, and ignored object counts.
- 2D detection overlay.
- 3D occupancy map.
- Simulated depth heatmap.
- Occupancy grid risk map.
- Per-object risk, depth, clearance, volume, and decision explanation.
- Embedded JSON output.

Additional generated audit artifacts:

- `reports/evaluation_scorecard.json`
- `reports/evaluation_scorecard.md`
- `reports/safety_events.json`
- `reports/safety_events.csv`

## Commands Used For Evaluation

Generate all visual and structured outputs:

```powershell
python -B scripts/generate_submission_artifacts.py
```

Run the core challenge example:

```powershell
python -B scripts/depth_guard_demo.py
```

Run scenario evaluation:

```powershell
python -B scripts/run_scenarios.py
```

Run formal challenge scorecard:

```powershell
python -B scripts/evaluate_submission.py
```

Create final upload package:

```powershell
python -B scripts/package_submission.py
```

Run temporal safety simulation:

```powershell
python -B scripts/simulate_temporal_sequence.py
```

Optional YOLO/OpenCV demo:

```powershell
python -m pip install -r requirements-vision.txt
python -B scripts/ai_perception_demo.py --image samples/frame.jpg --depth-map samples/depth.npy
```

Run tests:

```powershell
python -B -m unittest discover -s tests
```

## Engineering Highlights

- Modular architecture: input parsing, geometry, occupancy rules, reporting, visualization, dashboard, and batch scenario evaluation are separated.
- Configurable safety rules: critical distance, minimum object volume, camera field of view, zones, and class thickness factors are stored in JSON.
- Explainable decisions: each object includes status, risk score, volume, clearance margin, zone, and reason.
- Browser-ready output: reviewers can inspect the result visually without installing a web framework.
- Optional AI perception adapter: YOLO image/webcam input can feed the same occupancy engine.
- Temporal safety debounce: avoids overreacting to one noisy frame.
- Occupancy grid map: provides robot-style spatial reasoning output.
- Formal evaluation scorecard: verifies expected statuses and final decision.
- Safety event log: machine-readable critical/warning/info events.
- Extensible AI path: YOLOv8, MiDaS, stereo depth, RealSense, Kinect, or LiDAR can replace the simulated JSON feed while reusing the same occupancy engine.

## Current Sample Decision

The sample frame is `BLOCKED` because `Obj_101` is a cardboard box at `0.8m`, larger than the minimum volume threshold and inside the `1.0m` critical distance. `Obj_104` is closer at `0.5m`, but it is ignored because its estimated volume is below `20 cm3`.
