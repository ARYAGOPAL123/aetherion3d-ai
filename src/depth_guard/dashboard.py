from __future__ import annotations

import html
import json
from pathlib import Path

from .models import FrameAnalysis, ObjectStatus, VisionFrame


STATUS_CLASS = {
    ObjectStatus.HAZARD: "hazard",
    ObjectStatus.SAFE: "safe",
    ObjectStatus.IGNORE: "ignore",
}


def save_dashboard(
    frame: VisionFrame,
    analysis: FrameAnalysis,
    output_path: str | Path,
    original_image: str | None = None,
    overlay_image: str = "frame_overlay.png",
    depth_map_image: str | None = None,
    scene_image: str = "scene_3d.png",
    heatmap_image: str = "depth_heatmap.png",
    occupancy_grid_image: str = "occupancy_grid.png",
    temporal_timeline_image: str | None = None,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        render_dashboard_html(
            frame=frame,
            analysis=analysis,
            original_image=original_image,
            overlay_image=overlay_image,
            depth_map_image=depth_map_image,
            scene_image=scene_image,
            heatmap_image=heatmap_image,
            occupancy_grid_image=occupancy_grid_image,
            temporal_timeline_image=temporal_timeline_image,
        ),
        encoding="utf-8",
    )


def render_dashboard_html(
    frame: VisionFrame,
    analysis: FrameAnalysis,
    original_image: str | None,
    overlay_image: str,
    depth_map_image: str | None,
    scene_image: str,
    heatmap_image: str,
    occupancy_grid_image: str,
    temporal_timeline_image: str | None = None,
) -> str:
    payload_json = json.dumps(analysis.to_dict(), indent=2)
    rows = "\n".join(render_object_row(item) for item in analysis.objects)
    zone_cards = "\n".join(render_zone_card(zone, values) for zone, values in analysis.zone_summary.items())
    alert_items = "\n".join(f"<li>{html.escape(alert)}</li>" for alert in analysis.alerts)
    decision_class = "blocked" if analysis.is_blocked else "clear"
    decision_label = "BLOCKED" if analysis.is_blocked else "CLEAR"
    temporal_panel = ""
    if temporal_timeline_image:
        temporal_panel = f"""
    <section class="panel" style="margin-bottom: 18px;">
      <h2>Temporal Safety Debounce</h2>
      <img src="{html.escape(temporal_timeline_image)}" alt="temporal safety debounce timeline">
    </section>
"""

    original_panel = (
        f'<img src="{html.escape(original_image)}" alt="original input frame">'
        if original_image
        else '<div class="placeholder">Original input frame unavailable.</div>'
    )
    depth_panel = (
        f'<img src="{html.escape(depth_map_image)}" alt="estimated depth map">'
        if depth_map_image
        else '<div class="placeholder">Estimated depth map unavailable.</div>'
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Depth-Guard Spatial Occupancy Dashboard</title>
  <style>
    :root {{
      --ink: #172026;
      --muted: #607080;
      --line: #d8e0e8;
      --panel: #ffffff;
      --page: #f4f7fa;
      --danger: #c81e1e;
      --safe: #147d64;
      --ignore: #596674;
      --accent: #2454a6;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, Segoe UI, Arial, sans-serif;
      color: var(--ink);
      background: var(--page);
      line-height: 1.45;
    }}
    header {{
      background: #101820;
      color: white;
      padding: 28px 32px;
      border-bottom: 5px solid #2a9d8f;
    }}
    header h1 {{
      margin: 0;
      font-size: 32px;
      letter-spacing: 0;
    }}
    header p {{
      max-width: 980px;
      margin: 8px 0 0;
      color: #c9d5df;
      font-size: 15px;
    }}
    main {{
      max-width: 1280px;
      margin: 0 auto;
      padding: 24px;
    }}
    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(5, minmax(160px, 1fr));
      gap: 14px;
      margin-bottom: 20px;
    }}
    .metric, .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 10px 24px rgba(16, 24, 32, 0.06);
    }}
    .metric {{
      padding: 16px;
      min-height: 112px;
    }}
    .metric span {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      font-weight: 700;
      letter-spacing: 0.04em;
    }}
    .metric strong {{
      display: block;
      margin-top: 10px;
      font-size: 28px;
    }}
    .metric small {{ color: var(--muted); }}
    .decision.blocked strong {{ color: var(--danger); }}
    .decision.clear strong {{ color: var(--safe); }}
    .grid-2 {{
      display: grid;
      grid-template-columns: minmax(0, 1.2fr) minmax(360px, 0.8fr);
      gap: 18px;
      margin-bottom: 18px;
    }}
    .grid-3 {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 18px;
      margin-bottom: 18px;
    }}
    .panel {{ padding: 18px; }}
    .panel h2 {{
      margin: 0 0 14px;
      font-size: 18px;
    }}
    img {{
      display: block;
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #f8fafc;
    }}
    .alerts {{
      margin: 0;
      padding-left: 18px;
    }}
    .zone-card {{
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 12px;
      margin-bottom: 10px;
      background: #f9fbfd;
    }}
    .zone-card strong {{
      display: block;
      margin-bottom: 6px;
    }}
    .zone-card span {{
      display: inline-block;
      margin-right: 12px;
      color: var(--muted);
      font-size: 13px;
    }}
    .toolbar {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 12px;
    }}
    button {{
      border: 1px solid var(--line);
      background: #f8fafc;
      color: var(--ink);
      padding: 8px 10px;
      border-radius: 6px;
      cursor: pointer;
      font-weight: 700;
      font-size: 13px;
    }}
    button:hover {{ border-color: var(--accent); }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 10px 8px;
      text-align: left;
      vertical-align: middle;
    }}
    th {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.03em;
    }}
    .pill {{
      display: inline-block;
      min-width: 72px;
      padding: 4px 8px;
      border-radius: 999px;
      color: white;
      text-align: center;
      font-weight: 800;
      font-size: 11px;
    }}
    .pill.hazard {{ background: var(--danger); }}
    .pill.safe {{ background: var(--safe); }}
    .pill.ignore {{ background: var(--ignore); }}
    .risk {{
      width: 110px;
      height: 10px;
      border-radius: 999px;
      background: #e4ebf2;
      overflow: hidden;
    }}
    .risk div {{
      height: 100%;
      background: linear-gradient(90deg, #2a9d8f, #f4a261, #c81e1e);
    }}
    pre {{
      margin: 0;
      max-height: 420px;
      overflow: auto;
      background: #101820;
      color: #eaf2f8;
      padding: 16px;
      border-radius: 6px;
      font-size: 12px;
    }}
    .rule-list {{
      margin: 0;
      padding-left: 18px;
      color: #31404d;
      font-size: 14px;
    }}
    footer {{
      color: var(--muted);
      font-size: 12px;
      padding: 8px 2px 24px;
    }}
    @media (max-width: 980px) {{
      .summary-grid, .grid-2, .grid-3 {{
        grid-template-columns: 1fr;
      }}
      header, main {{ padding-left: 16px; padding-right: 16px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Depth-Guard Spatial Occupancy Dashboard</h1>
    <p>Frame {html.escape(frame.frame_id)} analyzed with 2D detections, depth estimates, metric size approximation, and deterministic warehouse occupancy rules.</p>
  </header>
  <main>
    <section class="summary-grid">
      <article class="metric decision {decision_class}">
        <span>Final Decision</span>
        <strong>{decision_label}</strong>
        <small>{analysis.hazard_count} blocking hazard(s)</small>
      </article>
      <article class="metric">
        <span>Objects Tracked</span>
        <strong>{len(analysis.objects)}</strong>
        <small>{analysis.ignored_count} ignored as too small</small>
      </article>
      <article class="metric">
        <span>Max Risk</span>
        <strong>{analysis.max_risk_score:.2f}</strong>
        <small>0.00 low, 1.00 critical</small>
      </article>
      <article class="metric">
        <span>Zone A Volume</span>
        <strong>{analysis.occupied_volume_cm3:.1f}</strong>
        <small>cm3 occupied by relevant objects</small>
      </article>
      <article class="metric">
        <span>Camera Frame</span>
        <strong>{frame.image_width_px}x{frame.image_height_px}</strong>
        <small>simulated RGB-D feed</small>
      </article>
    </section>

    <section class="grid-3">
      <article class="panel">
        <h2>Original Input Frame</h2>
        {original_panel}
      </article>
      <article class="panel">
        <h2>Detection Overlay</h2>
        <img src="{html.escape(overlay_image)}" alt="2D detection overlay">
      </article>
      <article class="panel">
        <h2>Estimated Depth Map</h2>
        {depth_panel}
      </article>
    </section>

    <section class="grid-2">
      <article class="panel">
        <h2>Operational Alerts</h2>
        <ul class="alerts">{alert_items}</ul>
        <h2 style="margin-top: 22px;">Zone Summary</h2>
        {zone_cards}
      </article>
      <article class="panel">
        <h2>3D Spatial Map</h2>
        <img src="{html.escape(scene_image)}" alt="3D occupancy map">
      </article>
    </section>

    <section class="grid-2">
      <article class="panel">
        <h2>Depth Heatmap</h2>
        <img src="{html.escape(heatmap_image)}" alt="depth heatmap">
      </article>
      <article class="panel">
        <h2>Occupancy Grid Risk Map</h2>
        <img src="{html.escape(occupancy_grid_image)}" alt="occupancy grid risk map">
      </article>
    </section>

    <section class="panel" style="margin-bottom: 18px;">
      <h2>Occupancy Grid Risk Map</h2>
      <img src="{html.escape(occupancy_grid_image)}" alt="occupancy grid risk map">
    </section>

    {temporal_panel}

    <section class="panel" style="margin-bottom: 18px;">
      <h2>Object-Level Explainability</h2>
      <div class="toolbar">
        <button type="button" onclick="filterRows('all')">All</button>
        <button type="button" onclick="filterRows('HAZARD')">Hazards</button>
        <button type="button" onclick="filterRows('SAFE')">Safe</button>
        <button type="button" onclick="filterRows('IGNORE')">Ignored</button>
      </div>
      <table>
        <thead>
          <tr>
            <th>Object</th>
            <th>Label</th>
            <th>Status</th>
            <th>Depth</th>
            <th>Clearance</th>
            <th>Volume</th>
            <th>Risk</th>
            <th>Reason</th>
          </tr>
        </thead>
        <tbody>{rows}</tbody>
      </table>
    </section>

    <section class="grid-2">
      <article class="panel">
        <h2>Rules Applied</h2>
        <ul class="rule-list">
          <li>Ignore objects below the minimum physical volume threshold.</li>
          <li>Classify relevant objects inside the critical distance as hazards.</li>
          <li>Summarize occupied volume in Zone A for crowding awareness.</li>
          <li>Keep geometry assumptions configurable for camera calibration.</li>
        </ul>
      </article>
      <article class="panel">
        <h2>Structured JSON Output</h2>
        <pre>{html.escape(payload_json)}</pre>
      </article>
    </section>

    <section class="panel" style="margin-bottom: 18px;">
      <h2>Review And Audit Artifacts</h2>
      <ul class="rule-list">
        <li><a href="evaluation_scorecard.md">Evaluation scorecard</a></li>
        <li><a href="safety_events.json">Safety event log</a></li>
        <li><a href="scenario_summary.csv">Scenario summary</a></li>
        <li><a href="temporal_summary.csv">Temporal safety summary</a></li>
        <li><a href="../docs/reviewer_guide.md">Reviewer guide</a></li>
        <li><a href="../docs/failure_modes_and_safety_case.md">Failure modes and safety case</a></li>
      </ul>
    </section>

    <footer>
      Depth-Guard challenge submission. Generated from deterministic Python analysis, ready for demo or review.
    </footer>
  </main>
  <script>
    function filterRows(status) {{
      document.querySelectorAll("tbody tr").forEach(function(row) {{
        row.style.display = status === "all" || row.dataset.status === status ? "" : "none";
      }});
    }}
  </script>
</body>
</html>
"""


def render_object_row(item) -> str:
    status_class = STATUS_CLASS[item.status]
    clearance = f"{item.clearance_margin_m:+.2f}m"
    risk_width = max(0, min(100, int(round(item.risk_score * 100))))
    return f"""
          <tr data-status="{html.escape(item.status.value)}">
            <td>{html.escape(item.object_id)}</td>
            <td>{html.escape(item.label)}</td>
            <td><span class="pill {status_class}">{html.escape(item.status.value)}</span></td>
            <td>{item.depth_m:.2f}m</td>
            <td>{clearance}</td>
            <td>{item.volume_cm3:.1f} cm3</td>
            <td><div class="risk" title="{item.risk_score:.2f}"><div style="width: {risk_width}%"></div></div></td>
            <td>{html.escape(item.reason)}</td>
          </tr>"""


def render_zone_card(zone: str, values: dict[str, float | int]) -> str:
    return f"""
        <div class="zone-card">
          <strong>{html.escape(zone)}</strong>
          <span>{int(values["objects"])} object(s)</span>
          <span>{int(values["hazards"])} hazard(s)</span>
          <span>{float(values["volume_cm3"]):.1f} cm3</span>
        </div>"""
