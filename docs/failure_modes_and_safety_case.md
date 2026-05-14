# Failure Modes And Safety Case

Depth-Guard is a perception decision-support system. In a real warehouse it should be integrated with robot safety controls, not treated as the only safety layer.

## Key Failure Modes

| Failure Mode | Impact | Mitigation In This Project |
|---|---|---|
| Depth noise around the 1m boundary | Flickering blocked/clear decisions | Temporal debounce monitor in `tracking.py` |
| Tiny objects close to camera | False blocked events | Minimum volume filter |
| Large objects just outside 1m | Crowding risk even when not blocked | Crowding alert and risk score |
| Detector misses an object | False clear | Multi-frame monitoring and production recommendation for better sensors |
| Bad camera calibration | Incorrect volume estimate | Camera FOV and class thickness factors are externalized in JSON |
| Unknown object class | Poor thickness estimate | Default class factor and calibration path |
| Single-frame glitch | Unstable robot behavior | Latched blocked state released only after stable clear frames |

## Safety Event Levels

- `CRITICAL`: loading zone is blocked and robot entry should stop.
- `WARNING`: zone is crowded or approaching unsafe conditions.
- `INFO`: non-blocking explanatory event, such as small-object filtering.

Generated safety logs:

```text
reports/safety_events.json
reports/safety_events.csv
```

## Production Safety Recommendations

- Use a hardware-rated safety sensor or interlock for final motion authorization.
- Add temporal tracking with object IDs across frames.
- Calibrate depth and object dimensions with site-specific measurements.
- Validate thresholds against warehouse operating procedures.
- Log all blocked/clear transitions for audit review.
- Monitor model drift after camera moves, lighting changes, or warehouse layout changes.
