from __future__ import annotations

from .models import FrameAnalysis


def format_text_report(analysis: FrameAnalysis) -> str:
    lines = [
        "Depth-Guard Frame Analysis",
        f"Frame: {analysis.frame_id}",
        "",
        (
            f"{'Object':<8} {'Label':<14} {'Depth':>7} {'Volume':>12} "
            f"{'Zone':<7} {'Risk':>5} {'Status':<8} Reason"
        ),
    ]
    for item in analysis.objects:
        lines.append(
            f"{item.object_id:<8} "
            f"{item.label:<14.14} "
            f"{item.depth_m:>6.2f}m "
            f"{item.volume_cm3:>10.1f} cm3 "
            f"{item.zone:<7} "
            f"{item.risk_score:>5.2f} "
            f"{item.status.value:<8} "
            f"{item.reason}"
        )

    lines.extend(
        [
            "",
            (
                "Scene Summary: "
                f"{analysis.hazard_count} hazard(s), "
                f"{analysis.safe_count} safe object(s), "
                f"{analysis.ignored_count} ignored object(s), "
                f"max risk {analysis.max_risk_score:.2f}"
            ),
            f"Final Decision: {'Loading Zone Blocked' if analysis.is_blocked else 'Loading Zone Clear'}",
        ]
    )
    lines.extend(f"Alert: {alert}" for alert in analysis.alerts)
    return "\n".join(lines)
