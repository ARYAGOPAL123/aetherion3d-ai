from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from depth_guard.engine import analyze_frame
from depth_guard.evaluation import evaluate_against_expectations, load_expectations
from depth_guard.io import load_rules, load_vision_feed
from depth_guard.models import ObjectStatus
from depth_guard.models import BoundingBox
from depth_guard.occupancy_grid import build_occupancy_grid
from depth_guard.perception import average_depth_from_map, depth_from_size_prior
from depth_guard.reporting import format_text_report
from depth_guard.safety import build_safety_events
from depth_guard.dashboard import render_dashboard_html
from depth_guard.tracking import TemporalOccupancyMonitor


class DepthGuardEngineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = load_vision_feed(PROJECT_ROOT / "data" / "vision_feed_sample.json")
        self.rules = load_rules(PROJECT_ROOT / "data" / "logic_constraints.json")
        self.analysis = analyze_frame(self.frame, self.rules)
        self.by_id = {item.object_id: item for item in self.analysis.objects}

    def test_sample_challenge_statuses(self) -> None:
        self.assertEqual(self.by_id["Obj_101"].status, ObjectStatus.HAZARD)
        self.assertEqual(self.by_id["Obj_104"].status, ObjectStatus.IGNORE)
        self.assertEqual(self.by_id["Obj_102"].status, ObjectStatus.SAFE)
        self.assertTrue(self.analysis.is_blocked)

    def test_small_tool_is_ignored_even_when_close(self) -> None:
        tool = self.by_id["Obj_104"]
        self.assertLess(tool.depth_m, self.rules.critical_distance_m)
        self.assertLess(tool.volume_cm3, self.rules.min_volume_cm3)
        self.assertEqual(tool.status, ObjectStatus.IGNORE)

    def test_human_is_classified_into_zone_b(self) -> None:
        human = self.by_id["Obj_102"]
        self.assertEqual(human.zone, "Zone B")
        self.assertEqual(human.status, ObjectStatus.SAFE)

    def test_report_contains_final_decision(self) -> None:
        report = format_text_report(self.analysis)
        self.assertIn("Final Decision: Loading Zone Blocked", report)
        self.assertIn("Obj_101", report)

    def test_risk_and_zone_summary_are_exposed(self) -> None:
        self.assertGreater(self.analysis.max_risk_score, 0.7)
        self.assertIn("Zone A", self.analysis.zone_summary)
        self.assertEqual(self.analysis.hazard_count, 1)

    def test_dashboard_contains_visual_sections(self) -> None:
        html = render_dashboard_html(
            self.frame,
            self.analysis,
            original_image="frame_original.png",
            overlay_image="frame_overlay.png",
            depth_map_image="estimated_depth_map.png",
            scene_image="scene_3d.png",
            heatmap_image="depth_heatmap.png",
            occupancy_grid_image="occupancy_grid.png",
            temporal_timeline_image="temporal_timeline.png",
        )
        self.assertIn("Depth-Guard Spatial Occupancy Dashboard", html)
        self.assertIn("Object-Level Explainability", html)
        self.assertIn("Occupancy Grid Risk Map", html)
        self.assertIn("Temporal Safety Debounce", html)
        self.assertIn("Review And Audit Artifacts", html)
        self.assertIn("Obj_101", html)

    def test_depth_map_sampling_uses_bbox_median(self) -> None:
        depth_map = [[1.0, 1.0, 5.0], [2.0, 2.0, 6.0], [3.0, 3.0, 7.0]]
        sampled = average_depth_from_map(
            np.array(depth_map),
            BoundingBox(0, 0, 2, 2),
            default_depth_m=9.0,
        )
        self.assertEqual(sampled, 1.5)

    def test_size_prior_depth_is_bounded(self) -> None:
        depth = depth_from_size_prior(
            "Cardboard Box",
            BoundingBox(0, 0, 80, 120),
            image_height_px=600,
            default_depth_m=2.0,
        )
        self.assertGreaterEqual(depth, 0.35)
        self.assertLessEqual(depth, 6.0)

    def test_occupancy_grid_marks_hazard_cells(self) -> None:
        grid = build_occupancy_grid(self.analysis)
        payload = grid.to_dict()
        self.assertGreater(payload["occupied_cells"], 0)
        self.assertGreater(payload["hazard_cells"], 0)

    def test_temporal_monitor_debounces_blocked_state(self) -> None:
        monitor = TemporalOccupancyMonitor(hazard_frames_to_block=2, clear_frames_to_release=2)
        first = monitor.update(self.analysis)
        second = monitor.update(self.analysis)
        self.assertTrue(first.instantaneous_blocked)
        self.assertFalse(first.latched_blocked)
        self.assertTrue(second.latched_blocked)

    def test_evaluation_scorecard_passes_challenge_expectations(self) -> None:
        expectations = load_expectations(PROJECT_ROOT / "data" / "evaluation_expectations.json")
        scorecard = evaluate_against_expectations(self.analysis, expectations)
        self.assertTrue(scorecard["passed"])
        self.assertEqual(scorecard["total_score_percent"], 100.0)

    def test_safety_events_include_critical_blocked_event(self) -> None:
        events = build_safety_events(self.analysis)
        severities = {event.severity for event in events}
        categories = {event.category for event in events}
        self.assertIn("CRITICAL", severities)
        self.assertIn("ZONE_BLOCKED", categories)


if __name__ == "__main__":
    unittest.main()
