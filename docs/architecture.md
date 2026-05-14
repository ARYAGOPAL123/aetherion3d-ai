# Depth-Guard Architecture

## System Pipeline

```text
Camera / Simulated Feed
        |
        v
Perception Adapter
  - challenge JSON
  - optional YOLO image/webcam input
  - optional aligned depth map
        |
        v
Frame Model
  - object id
  - label
  - bbox x/y/w/h
  - average depth in meters
        |
        v
3D Geometry
  - metric width estimate
  - metric height estimate
  - class-aware thickness estimate
  - approximate volume in cm3
        |
        v
Occupancy Engine
  - ignore tiny objects
  - classify hazards inside 1 meter
  - compute risk score
  - summarize zones and alerts
        |
        v
Temporal Safety Monitor
  - debounce hazard evidence
  - latch blocked state
  - release after stable clear frames
        |
        v
Reports
  - terminal report
  - JSON output
  - 2D overlay
  - 3D scatter plot
  - depth heatmap
  - occupancy grid
  - temporal timeline
  - browser dashboard
```

## Why The Design Is Professional

The challenge gives simulated detections and depths, so the reliable path is deterministic and testable. The production path is added as an adapter instead of being mixed into the safety logic. That means YOLOv8, MiDaS, stereo depth, RealSense, or LiDAR can be changed without rewriting the occupancy rules.

## Required Challenge Path

The required path uses:

- `data/vision_feed_sample.json` for object detections and depth.
- `data/logic_constraints.json` for critical distance, minimum volume, zones, camera field of view, and class thickness factors.
- `depth_guard.engine.analyze_frame` for occupancy classification.

This path has no heavy model dependency and is covered by unit tests.

## Temporal Safety Layer

Real perception systems can be noisy. A single frame may briefly miss a box or misread depth. `src/depth_guard/tracking.py` adds a debounce monitor:

- `hazard_frames_to_block=2`: latch blocked only after stable hazard evidence.
- `clear_frames_to_release=2`: release blocked only after stable clear evidence.

Run:

```powershell
python -B scripts/simulate_temporal_sequence.py
```

This produces `reports/temporal_timeline.png`, `reports/temporal_summary.json`, and `reports/temporal_summary.csv`.

## Occupancy Grid

`src/depth_guard/occupancy_grid.py` converts object positions into a robot-style x/depth grid. Each occupied cell stores the strongest risk value in that area. This gives a compact spatial map that a robot planner or warehouse dashboard could consume.

## Optional AI Path

The optional script `scripts/ai_perception_demo.py` can run YOLO detection on:

- A still image: `--image path/to/frame.jpg`
- One webcam frame: `--webcam 0`

If an aligned metric depth map is available, pass:

```powershell
python -B scripts/ai_perception_demo.py --image samples/frame.jpg --depth-map samples/depth.npy
```

If no depth map is passed, the script uses a size-prior fallback for demo purposes. This fallback is clearly separated from the core challenge logic and should be replaced with MiDaS, stereo depth, RealSense, Kinect, or LiDAR for production.

## Volume Estimation

The sample data does not include camera intrinsics or point clouds. Depth-Guard therefore uses a transparent pinhole approximation:

```text
width_m  = visible_width_at_depth * bbox_width_px / image_width_px
height_m = visible_height_at_depth * bbox_height_px / image_height_px
depth_m  = min(width_m, height_m) * class_depth_factor
volume   = width_m * height_m * depth_m
```

The class depth factor is configurable because a real warehouse deployment should calibrate it with measured objects.

## Production Upgrade Path

For a real robot or warehouse camera, the next steps would be:

- Use YOLOv8 or RT-DETR for robust object detection.
- Use RealSense, Azure Kinect, stereo depth, LiDAR, or MiDaS/Depth Anything for dense depth.
- Replace approximate thickness with segmentation masks or point-cloud volume.
- Add temporal smoothing across frames to reduce false positives.
- Send alerts to a robot controller, PLC, WMS dashboard, or MQTT/Kafka topic.
