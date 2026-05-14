# Reviewer Guide

This is the fastest way to evaluate the Depth-Guard submission.

## 1. Verify Core Challenge Output

```powershell
python -B scripts/depth_guard_demo.py
```

Expected result:

- `Obj_101`: `HAZARD`
- `Obj_104`: `IGNORE`
- `Obj_102`: `SAFE`
- Final decision: `Loading Zone Blocked`

## 2. Run Automated Tests

```powershell
python -B -m unittest discover -s tests
```

## 3. Run The Scorecard

```powershell
python -B scripts/evaluate_submission.py
```

Expected result:

```text
Total score: 100.00%
Passed: True
```

## 4. Generate Full Demo Package

```powershell
python -B scripts/generate_submission_artifacts.py
```

Open:

```text
reports/depth_guard_dashboard.html
```

## 5. Create Upload Package

```powershell
python -B scripts/package_submission.py
```

This creates:

```text
DepthGuard_Submission.zip
```

## 6. Inspect Advanced Features

- `reports/occupancy_grid.png`: robot-style spatial risk map.
- `reports/temporal_timeline.png`: temporal debounce behaviour.
- `reports/scenario_summary.csv`: blocked, clear, and crowded scenarios.
- `reports/safety_events.json`: machine-readable safety event log.
- `reports/evaluation_scorecard.md`: challenge pass/fail scorecard.

## 7. Optional AI Extension

Install optional vision dependencies only if testing real images or webcam input:

```powershell
python -m pip install -r requirements-vision.txt
python -B scripts/ai_perception_demo.py --webcam 0
```

The core challenge does not require these heavy dependencies because the supplied data is simulated.
