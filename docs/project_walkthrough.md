# Depth-Guard Project Walkthrough

Use this as a concise demo script for project review or technical discussion.

## 30-Second Summary

Depth-Guard takes object detections and depth values, estimates each object's approximate 3D footprint, and applies configurable warehouse safety rules. The key output is whether the loading zone is blocked, plus a per-object explanation.

## What To Demo

1. Generate the full demo package:

   ```powershell
   python -B scripts/generate_submission_artifacts.py
   ```

2. Open the browser dashboard:

   ```text
   reports/depth_guard_dashboard.html
   ```

3. Run the sample analysis in the terminal:

   ```powershell
   python -B scripts/depth_guard_demo.py
   ```

4. Point out the three important sample decisions:

   - `Obj_101` is a cardboard box at `0.8m`, large enough to matter, so it is a `HAZARD`.
   - `Obj_104` is only `0.5m` away, but its estimated volume is below `20 cm3`, so it is `IGNORE`.
   - `Obj_102` is a human at `2.5m`, so it is outside the critical loading-zone distance and is `SAFE`.

5. Run scenario evaluation:

   ```powershell
   python -B scripts/run_scenarios.py
   ```

6. Run temporal safety simulation:

   ```powershell
   python -B scripts/simulate_temporal_sequence.py
   ```

7. Mention that the geometry is intentionally configurable because the challenge data is simulated. With a real RGB-D camera, the same engine can consume measured object dimensions or point-cloud volumes.

## Architecture Talking Points

- `perception.py` contains optional adapters for YOLO/image/webcam inputs.
- `io.py` loads simulated camera frames and rule files.
- `geometry.py` converts pixel boxes plus depth into approximate metric dimensions.
- `engine.py` contains the deterministic safety logic.
- `dashboard.py` generates the browser-ready executive dashboard.
- `batch.py` evaluates multiple warehouse scenarios.
- `tracking.py` debounces decisions across frames.
- `occupancy_grid.py` creates robot-style spatial risk maps.
- `reporting.py` and `visualization.py` handle presentation.
- Tests validate the challenge's expected decisions.

## Extension Plan For Real Deployment

- Replace the JSON vision feed with YOLOv8 detections.
- Replace simulated average depth with MiDaS, stereo depth, RealSense, Kinect, or LiDAR values.
- Calibrate field of view and class depth factors using measured warehouse objects.
- Add temporal smoothing so one noisy frame does not trigger false alarms.
- Stream alerts to a robot controller, PLC, dashboard, or warehouse management system.

## Optional Real AI Demo

If optional dependencies are available:

```powershell
python -m pip install -r requirements-vision.txt
python -B scripts/ai_perception_demo.py --image samples/frame.jpg --depth-map samples/depth.npy
```

This demonstrates how the same occupancy engine can consume YOLO detections. If no metric depth map is available, the script can still run with a webcam using a clearly labeled depth fallback:

```powershell
python -B scripts/ai_perception_demo.py --webcam 0
```
