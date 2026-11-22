"""Unit tests for the Scenario Evaluator Workflow."""

import unittest
from realtime_citynav.scenario_evaluator import ScenarioEvaluator, ScenarioResult


class TestScenarioEvaluator(unittest.TestCase):
    def setUp(self):
        self.evaluator = ScenarioEvaluator(verbose=False)

    def test_altitude_above_landmark(self):
        """Scenario 1: 'go 50m above fountain' targets Z=-50m directly over fountain."""
        res = self.evaluator.run_scenario_altitude_above_fountain()
        self.assertTrue(res.passed, f"Scenario failed: {res.error_message}")
        self.assertAlmostEqual(res.details["target_position"][0], 32.0, places=1)
        self.assertAlmostEqual(res.details["target_position"][1], 0.0, places=1)
        self.assertAlmostEqual(res.details["target_position"][2], -50.0, places=1)
        self.assertEqual(res.details["observation_altitude"], 50.0)

    def test_spatial_prepositions_suite(self):
        """Scenario 2: All spatial prepositions (in front, left, right, behind, over) map to proper coordinates."""
        res = self.evaluator.run_scenario_spatial_prepositions()
        self.assertTrue(res.passed, f"Scenario failed: {res.error_message}")
        self.assertEqual(len(res.details["cases"]), 5)

    def test_midflight_interruption_redirection(self):
        """Scenario 3: Mid-flight interruption redirects destination active target."""
        res = self.evaluator.run_scenario_midflight_interruption()
        self.assertTrue(res.passed, f"Scenario failed: {res.error_message}")
        self.assertIn("blue building", res.details["active_target"].lower())

    def test_spatial_memory_deduplication(self):
        """Scenario 4: Noisy multi-observation clustering eliminates duplicate landmarks."""
        res = self.evaluator.run_scenario_spatial_memory_deduplication()
        self.assertTrue(res.passed, f"Scenario failed: {res.error_message}")
        self.assertEqual(res.details["landmark_count"], 3)

    def test_reactive_obstacle_avoidance(self):
        """Scenario 5: Reactive APF triggers lateral repulsion when close barrier appears."""
        res = self.evaluator.run_scenario_obstacle_avoidance()
        self.assertTrue(res.passed, f"Scenario failed: {res.error_message}")
        self.assertEqual(res.details["avoidance_direction"], "RIGHT")

    def test_spatial_recall_query(self):
        """Scenario 6: Memory digest query responds with registered landmarks."""
        res = self.evaluator.run_scenario_recall_query()
        self.assertTrue(res.passed, f"Scenario failed: {res.error_message}")
        self.assertIn("fountain", res.details["reply"].lower())

    def test_stress_cases(self):
        """Scenario 7: Rapid command barrage does not cause deadlocks or corrupt vehicle state."""
        res = self.evaluator.run_scenario_stress_cases()
        self.assertTrue(res.passed, f"Scenario failed: {res.error_message}")

    def test_compound_spatial_prepositions(self):
        """Scenario 8: Compound prepositions apply vertical and horizontal offsets simultaneously."""
        res = self.evaluator.run_scenario_compound_spatial_prepositions()
        self.assertTrue(res.passed, f"Scenario failed: {res.error_message}")
        self.assertEqual(len(res.details["cases"]), 4)

    def test_elevated_landmark_altitude(self):
        """Scenario 9: Altitude above elevated landmark calculates Z relative to landmark elevation."""
        res = self.evaluator.run_scenario_elevated_landmark_altitude()
        self.assertTrue(res.passed, f"Scenario failed: {res.error_message}")
        self.assertAlmostEqual(res.details["target_pos"][2], -52.0, places=1)
        self.assertEqual(res.details["target_alt"], 52.0)

    def test_continuous_memory_pruning(self):
        """Scenario 10: Automatic memory maintenance prunes transient noise and consolidates duplicates."""
        res = self.evaluator.run_scenario_continuous_memory_pruning()
        self.assertTrue(res.passed, f"Scenario failed: {res.error_message}")
        self.assertEqual(len(res.details["remaining_landmarks"]), 1)

    def test_malformed_and_punctuation_robustness(self):
        """Scenario 11: Malformed and punctuation inputs do not divert to invalid targets."""
        res = self.evaluator.run_scenario_malformed_and_punctuation_robustness()
        self.assertTrue(res.passed, f"Scenario failed: {res.error_message}")

    def test_multileg_route_and_shorthand_resolution(self):
        """Scenario 12: Multi-leg from-to routes and adjective shorthands resolve and transition properly."""
        res = self.evaluator.run_scenario_multileg_route_and_shorthand_resolution()
        self.assertTrue(res.passed, f"Scenario failed: {res.error_message}")
        self.assertEqual(res.details["subgoals"], ["fountain", "red building", "blue building"])
        self.assertEqual(res.details["subgoals2"], ["red building", "blue building"])
        self.assertEqual(res.details["initial_target"], "fountain")
        self.assertEqual(res.details["second_target"], "red building")
        self.assertEqual(res.details["third_target"], "blue building")
        self.assertEqual(res.details["interruption_target"], "red building")

    def test_single_landmark_orbit_kinematics(self):
        """Scenario 13: Single landmark orbit engages ORBIT mode and points inward."""
        res = self.evaluator.run_scenario_orbit_blue_building()
        self.assertTrue(res.passed, f"Scenario failed: {res.error_message}")

    def test_tangent_orbit_speed_and_radius(self):
        """Scenario 14: Tangent orbit with custom speed and radius."""
        res = self.evaluator.run_scenario_tangent_orbit_speed_radius()
        self.assertTrue(res.passed, f"Scenario failed: {res.error_message}")

    def test_clockwise_orbit_parameters(self):
        """Scenario 15: Clockwise orbit direction and altitude parameters."""
        res = self.evaluator.run_scenario_clockwise_orbit_parameters()
        self.assertTrue(res.passed, f"Scenario failed: {res.error_message}")

    def test_compound_multi_altitude_orbit(self):
        """Scenario 16: Compound multi-altitude orbit preserves parameters and transitions."""
        res = self.evaluator.run_scenario_compound_multi_altitude_orbit()
        self.assertTrue(res.passed, f"Scenario failed: {res.error_message}")

    def test_explicit_3d_waypoint_path(self):
        """Scenario 17: Explicit 3D continuous waypoint path followed."""
        res = self.evaluator.run_scenario_explicit_3d_waypoint_path()
        self.assertTrue(res.passed, f"Scenario failed: {res.error_message}")

    def test_collective_quantifier_city_tour(self):
        """Scenario 18: Universal collective quantifier expands to full city tour."""
        res = self.evaluator.run_scenario_collective_quantifier_city_tour()
        self.assertTrue(res.passed, f"Scenario failed: {res.error_message}")

    def test_stop_and_hover_interruption(self):
        """Scenario 19: Stop and hover interruption halts navigation cleanly."""
        res = self.evaluator.run_scenario_stop_and_hover_interruption()
        self.assertTrue(res.passed, f"Scenario failed: {res.error_message}")

    def test_midflight_orbit_interruption(self):
        """Scenario 20: Mid-flight linear navigation diverted to orbit."""
        res = self.evaluator.run_scenario_midflight_orbit_interruption()
        self.assertTrue(res.passed, f"Scenario failed: {res.error_message}")

    def test_pure_altitude_change(self):
        """Scenario 21: Pure altitude change without landmark updates vertical target."""
        res = self.evaluator.run_scenario_pure_altitude_change()
        self.assertTrue(res.passed, f"Scenario failed: {res.error_message}")

    def test_geofence_boundary_safety(self):
        """Scenario 22: Geofence boundary restrains drone safely."""
        res = self.evaluator.run_scenario_geofence_boundary_safety()
        self.assertTrue(res.passed, f"Scenario failed: {res.error_message}")

    def test_mid_orbit_spatial_target_interruption(self):
        """Scenario 23: Mid-orbit interruption with spatial target diverts cleanly."""
        res = self.evaluator.run_scenario_mid_orbit_spatial_target_interruption()
        self.assertTrue(res.passed, f"Scenario failed: {res.error_message}")

    def test_return_to_past_landmark(self):
        """Scenario 24: Return to past landmark routes to prior memory location."""
        res = self.evaluator.run_scenario_return_to_past_landmark()
        self.assertTrue(res.passed, f"Scenario failed: {res.error_message}")

    def test_result_serialization(self):
        """ScenarioResult serializes cleanly to dict for automated reporting and CI."""
        res = ScenarioResult(
            name="Test Serialization",
            description="Sample description",
            passed=True,
            duration_sec=0.1234,
            details={"key": "value"},
            logs=["step 1", "step 2"]
        )
        d = res.to_dict()
        self.assertEqual(d["name"], "Test Serialization")
        self.assertTrue(d["passed"])
        self.assertEqual(d["duration_sec"], 0.123)
        self.assertEqual(d["details"]["key"], "value")


if __name__ == "__main__":
    unittest.main()
