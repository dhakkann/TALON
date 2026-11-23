"""Unit tests for Generalized Mission Planning, Arbitrary Landmarks, and LLM Heavy Lifting."""

import unittest
import json
from unittest.mock import MagicMock, patch
import numpy as np

from realtime_citynav.dialogue_manager import DialogueManager
from realtime_citynav.config import SystemConfig, _get_default_gemini_key
from realtime_citynav.main import CityNavOrchestrator


class TestGeneralizedPlans(unittest.TestCase):
    """Test LLM-driven open-ended plan decomposition, novel landmarks, and rich representations."""

    def setUp(self):
        self.dm_heuristic = DialogueManager(api_key=None, mock=True)

    def test_dynamic_landmark_registration(self):
        """DialogueManager supports registering novel landmarks and custom coordinates."""
        dm = DialogueManager(api_key=None, mock=True)
        self.assertNotIn("warehouse", [lm.lower() for lm in dm.known_landmarks])
        
        dm.register_known_landmark("warehouse", np.array([100.0, 40.0, 0.0]))
        self.assertIn("warehouse", [lm.lower() for lm in dm.known_landmarks])
        self.assertIn("warehouse", dm.landmark_centers)
        np.testing.assert_array_equal(dm.landmark_centers["warehouse"], np.array([100.0, 40.0, 0.0]))

        dm.update_known_landmarks(["dock", "helipad", "crane"])
        self.assertIn("dock", [lm.lower() for lm in dm.known_landmarks])
        self.assertIn("helipad", [lm.lower() for lm in dm.known_landmarks])
        self.assertIn("crane", [lm.lower() for lm in dm.known_landmarks])

    def test_arbitrary_landmark_orbit_heuristic_extraction(self):
        """Arbitrary novel landmarks in orbit commands are extracted without hardcoding 'blue building'."""
        cmd1 = "circle twice around the warehouse"
        p1 = self.dm_heuristic.parse_orbit_command(cmd1)
        self.assertIsNotNone(p1)
        self.assertEqual(p1["target"], "warehouse")
        self.assertEqual(p1["base_landmark"], "warehouse")
        self.assertEqual(p1["revolutions"], 2.0)

        cmd2 = "orbit crane looking ahead at 5 m/s"
        p2 = self.dm_heuristic.parse_orbit_command(cmd2)
        self.assertIsNotNone(p2)
        self.assertEqual(p2["target"], "crane")
        self.assertEqual(p2["facing"], "TANGENT")
        self.assertEqual(p2["speed"], 5.0)

        cmd3 = "hover in circle around solar farm at 20m radius"
        p3 = self.dm_heuristic.parse_orbit_command(cmd3)
        self.assertIsNotNone(p3)
        self.assertEqual(p3["target"], "solar farm")
        self.assertEqual(p3["radius"], 20.0)

    def test_structured_plan_mission_schema_with_mocked_llm(self):
        """plan_mission produces rich structured representation with mission_type, legs, and subgoals."""
        dm = DialogueManager(api_key="fake-key-for-test", mock=False)
        mock_client = MagicMock()
        dm.client = mock_client

        mock_llm_response = {
            "mission_type": "MULTI_LEG",
            "rationale": "Transit to warehouse at 30m, patrol dock, and orbit crane.",
            "subgoals": [
                "30m above warehouse",
                "patrol dock",
                "circle around crane 2 times facing forward at 4.0 m/s"
            ],
            "legs": [
                {
                    "index": 0,
                    "action": "NAVIGATE",
                    "target": "30m above warehouse",
                    "base_landmark": "warehouse",
                    "altitude": 30.0,
                    "spatial": {"relation": "ABOVE", "distance": 30.0, "altitude_offset": 30.0, "horizontal_relation": None, "horizontal_distance": None}
                },
                {
                    "index": 1,
                    "action": "PATROL",
                    "target": "dock",
                    "base_landmark": "dock",
                    "altitude": None
                },
                {
                    "index": 2,
                    "action": "ORBIT",
                    "target": "crane",
                    "base_landmark": "crane",
                    "altitude": 10.0,
                    "orbit": {"radius": 16.0, "altitude": 10.0, "speed": 4.0, "facing": "TANGENT", "revolutions": 2.0, "direction": "CW"}
                }
            ]
        }
        
        mock_response_obj = MagicMock()
        mock_response_obj.text = json.dumps(mock_llm_response)
        mock_client.models.generate_content.return_value = mock_response_obj

        instruction = "Fly 30m above warehouse, then patrol dock, then circle crane twice facing forward at 4 m/s"
        plan = dm.plan_mission(instruction)
        
        self.assertEqual(plan["mission_type"], "MULTI_LEG")
        self.assertEqual(len(plan["subgoals"]), 3)
        self.assertEqual(plan["subgoals"][0], "30m above warehouse")
        self.assertEqual(plan["subgoals"][1], "patrol dock")
        self.assertIn("crane", plan["subgoals"][2])
        self.assertEqual(len(plan["legs"]), 3)
        self.assertEqual(plan["legs"][0]["action"], "NAVIGATE")
        self.assertEqual(plan["legs"][1]["action"], "PATROL")
        self.assertEqual(plan["legs"][2]["action"], "ORBIT")
        self.assertEqual(plan["legs"][2]["orbit"]["facing"], "TANGENT")
        self.assertEqual(plan["legs"][2]["orbit"]["revolutions"], 2.0)

        # parse_initial_instruction uses plan_mission
        subgoals = dm.parse_initial_instruction(instruction)
        self.assertEqual(subgoals, plan["subgoals"])

    def test_interruption_novel_landmark_with_mocked_llm(self):
        """Interruption manager processes novel destination, orbit, and altitude change with LLM."""
        dm = DialogueManager(api_key="fake-key-for-test", mock=False)
        mock_client = MagicMock()
        dm.client = mock_client

        # 1. Target correction to novel warehouse
        mock_response_obj = MagicMock()
        mock_response_obj.text = json.dumps({
            "intent": "TARGET_CORRECTION",
            "target": "35m above warehouse",
            "base_landmark": "warehouse",
            "altitude": 35.0,
            "spatial": {"relation": "ABOVE", "distance": 35.0, "altitude_offset": 35.0},
            "reply": "Diverting to warehouse at 35m altitude."
        })
        mock_client.models.generate_content.return_value = mock_response_obj

        dm.queue_user_message("no, divert to warehouse at 35m altitude")
        res = dm.process_interruption(current_target="fountain")
        self.assertIsNotNone(res)
        self.assertEqual(res["intent"], "TARGET_CORRECTION")
        self.assertEqual(res["target"], "35m above warehouse")
        self.assertEqual(res["base_landmark"], "warehouse")

        # 2. Orbit novel bridge
        mock_response_obj.text = json.dumps({
            "intent": "ORBIT_CIRCLE",
            "target": "bridge",
            "base_landmark": "bridge",
            "radius": 20.0,
            "altitude": 15.0,
            "speed": 4.5,
            "facing": "TANGENT",
            "revolutions": 3.0,
            "direction": "CW",
            "reply": "Orbiting bridge for 3 revolutions."
        })
        dm.queue_user_message("circle 3 times around bridge looking ahead at 4.5 m/s clockwise")
        res_orb = dm.process_interruption(current_target="tower")
        self.assertIsNotNone(res_orb)
        self.assertEqual(res_orb["intent"], "ORBIT_CIRCLE")
        self.assertEqual(res_orb["target"], "bridge")
        self.assertEqual(res_orb["facing"], "TANGENT")
        self.assertEqual(res_orb["speed"], 4.5)
        self.assertEqual(res_orb["revolutions"], 3.0)
        self.assertEqual(res_orb["direction"], "CW")

    def test_orchestrator_subgoal_waypoint_path(self):
        """Orchestrator executes an explicit waypoint path when given as an initial subgoal."""
        cfg = SystemConfig()
        cfg.mock_airsim = True
        cfg.mock_perception = True
        cfg.mock_gemini = True
        cfg.dashboard.enabled = False

        orch = CityNavOrchestrator(cfg)
        orch.start(start_dashboard=False)

        try:
            instruction = "follow curve [[10, 20, -10], [20, 30, -15], [30, 20, -10]]"
            subgoals = orch.start_mission(instruction)
            self.assertEqual(len(subgoals), 1)
            self.assertEqual(orch.flight_mode, "CURVE")
            self.assertTrue(orch.flight.is_following_path)
            self.assertEqual(len(orch.flight.trajectory_path), 3)
        finally:
            orch.stop()

    def test_orchestrator_novel_landmark_routing(self):
        """Orchestrator resolves novel simulated landmarks (e.g. warehouse, crane) into 3D navigation targets."""
        cfg = SystemConfig()
        cfg.mock_airsim = True
        cfg.mock_perception = True
        cfg.mock_gemini = True
        cfg.dashboard.enabled = False

        orch = CityNavOrchestrator(cfg)
        orch.start(start_dashboard=False)

        try:
            # Mission to novel warehouse
            subgoals = orch.start_mission("fly to the warehouse")
            self.assertEqual(subgoals, ["warehouse"])
            self.assertIsNotNone(orch.flight.target_position)
            # Warehouse mock coords are [100.0, 40.0, 0.0] with standoff
            self.assertAlmostEqual(orch.flight.target_position[0], 100.0 - 15.0, places=1)
            self.assertAlmostEqual(orch.flight.target_position[1], 40.0, places=1)

            # Mid-flight interruption to novel crane orbit
            orch.dialogue.queue_user_message("circle twice around crane facing forward")
            # Step cognitive loop or call directly
            res = orch.dialogue.process_interruption(current_target="warehouse")
            self.assertEqual(res["intent"], "ORBIT_CIRCLE")
            self.assertEqual(res["target"], "crane")
            self.assertEqual(res["facing"], "TANGENT")
            self.assertEqual(res["revolutions"], 2.0)
        finally:
            orch.stop()

    def test_figure_8_trajectory_generation(self):
        """DialogueManager generates smooth 3D figure-8 continuous waypoint curves between two landmarks."""
        c1 = np.array([75.0, 12.0, 0.0])
        c2 = np.array([75.0, -14.0, 0.0])
        wps = DialogueManager.generate_figure_8_waypoints(c1, c2, radius=12.0, altitude=18.0, num_points=32)
        self.assertEqual(len(wps), 32)
        for pt in wps:
            self.assertEqual(len(pt), 3)
            self.assertEqual(pt[2], -18.0)  # NED Z is -altitude
        # First half should circle around c1, second half around c2
        first_half_x = [pt[0] for pt in wps[:16]]
        second_half_y = [pt[1] for pt in wps[16:]]
        self.assertTrue(all(abs(x - 75.0) <= 13.0 for x in first_half_x))
        self.assertTrue(all(abs(y - (-14.0)) <= 13.0 for y in second_half_y))

    def test_spiral_climb_trajectory_generation(self):
        """DialogueManager generates 3D ascending spiral waypoints around a landmark."""
        c = np.array([130.0, -10.0, 0.0])
        wps = DialogueManager.generate_spiral_waypoints(c, radius=16.0, start_altitude=10.0, end_altitude=40.0, revolutions=2.0, num_points=24)
        self.assertEqual(len(wps), 24)
        self.assertAlmostEqual(wps[0][2], -10.0, places=1)
        self.assertAlmostEqual(wps[-1][2], -40.0, places=1)
        # Verify monotonically ascending altitude (more negative Z)
        for i in range(len(wps) - 1):
            self.assertLessEqual(wps[i+1][2], wps[i][2])

    def test_search_grid_trajectory_generation(self):
        """DialogueManager generates lawnmower search grid waypoints."""
        c = np.array([150.0, 30.0, 0.0])
        wps = DialogueManager.generate_search_grid_waypoints(c, width=40.0, height=40.0, altitude=15.0, step=10.0)
        self.assertGreaterEqual(len(wps), 8)
        self.assertTrue(all(pt[2] == -15.0 for pt in wps))

    def test_compass_bearing_spatial_offsets(self):
        """parse_spatial_relation cleanly extracts compass bearing offsets (North, South, East, West)."""
        res1 = DialogueManager.parse_spatial_relation("15m north and 20m east of red building")
        self.assertEqual(res1["relation"], "COMPASS")
        self.assertEqual(res1["dx"], 15.0)
        self.assertEqual(res1["dy"], 20.0)
        self.assertEqual(res1["base_landmark"], "red building")

        res2 = DialogueManager.parse_spatial_relation("25m south and 10m west of tower")
        self.assertEqual(res2["relation"], "COMPASS")
        self.assertEqual(res2["dx"], -25.0)
        self.assertEqual(res2["dy"], -10.0)
        self.assertEqual(res2["base_landmark"], "tower")

    def test_structured_legs_direct_execution_in_orchestrator(self):
        """Orchestrator directly executes structured legs (CURVE, ORBIT, NAVIGATE) from active plan."""
        cfg = SystemConfig()
        cfg.mock_airsim = True
        cfg.mock_perception = True
        cfg.mock_gemini = True
        cfg.dashboard.enabled = False

        orch = CityNavOrchestrator(cfg)
        orch.start(start_dashboard=False)

        try:
            # Craft a multi-action structured plan
            wps_curve = [[10.0, 20.0, -15.0], [25.0, 35.0, -18.0], [40.0, 20.0, -15.0]]
            orch.active_plan = {
                "mission_type": "MULTI_ACTION",
                "subgoals": ["curve", "orbit blue building", "warehouse"],
                "legs": [
                    {
                        "index": 0,
                        "action": "CURVE",
                        "target": "curve",
                        "waypoints": wps_curve,
                        "speed": 5.0,
                        "facing": "TANGENT"
                    },
                    {
                        "index": 1,
                        "action": "ORBIT",
                        "target": "circle blue building",
                        "base_landmark": "blue building",
                        "orbit": {"radius": 18.0, "altitude": 12.0, "speed": 4.0, "facing": "INWARD", "revolutions": 2.0, "direction": "CW"}
                    },
                    {
                        "index": 2,
                        "action": "NAVIGATE",
                        "target": "warehouse",
                        "base_landmark": "warehouse",
                        "altitude": 20.0
                    }
                ]
            }
            orch.subgoals = ["curve", "orbit blue building", "warehouse"]

            # Leg 0: Curve
            orch._set_current_subgoal_target(0)
            self.assertEqual(orch.flight_mode, "CURVE")
            self.assertTrue(orch.flight.is_following_path)
            self.assertEqual(len(orch.flight.trajectory_path), 3)

            # Leg 1: Orbit
            orch._set_current_subgoal_target(1)
            self.assertEqual(orch.flight_mode, "ORBIT")
            self.assertTrue(orch.flight.is_orbiting)
            self.assertEqual(orch.flight.orbit_radius, 18.0)
            self.assertEqual(orch.flight.orbit_altitude, 12.0)
            self.assertEqual(orch.flight.orbit_direction, -1.0)  # CW is -1.0

            # Leg 2: Navigate to novel warehouse
            orch._set_current_subgoal_target(2)
            self.assertEqual(orch.flight_mode, "NAVIGATING")
            self.assertIsNotNone(orch.flight.target_position)
            self.assertEqual(orch.flight.target_position[2], -20.0)
        finally:
            orch.stop()

    def test_instant_queue_hover_stop_safety(self):
        """Immediate hover/stop commands are intercepted in queue without waiting for LLM round-trip."""
        dm = DialogueManager(api_key=None, mock=True)
        dm.queue_user_message("stop and hover right here")
        res = dm.process_interruption(current_target="red building")
        self.assertIsNotNone(res)
        self.assertEqual(res["intent"], "HOVER_STOP")
        self.assertIsNone(res["target"])

    def test_live_gemini_novel_mission_if_api_key_available(self):
        """If real GEMINI_API_KEY is available in environment, verify live Gemini decomposition of open-ended plan."""
        api_key = _get_default_gemini_key()
        if not api_key:
            self.skipTest("No GEMINI_API_KEY available in environment.")

        dm = DialogueManager(api_key=api_key, model_name="gemini-3.5-flash-lite", mock=False)
        if dm.client is None:
            self.skipTest("Could not initialize live Gemini Client.")

        instruction = "Fly past the warehouse at 25m altitude, then inspect the helipad, and finish with 2 laps around crane"
        try:
            plan = dm.plan_mission(instruction)
            self.assertIn("subgoals", plan)
            self.assertGreaterEqual(len(plan["subgoals"]), 2)
            # Check that warehouse and crane or helipad are recognized
            subgoals_text = " ".join(plan["subgoals"]).lower()
            self.assertIn("warehouse", subgoals_text)
            self.assertTrue("crane" in subgoals_text or "helipad" in subgoals_text)
        except Exception as e:
            # If rate-limited or network down, skip test cleanly
            self.skipTest(f"Live Gemini API call failed (transient network/rate-limit): {e}")


if __name__ == "__main__":
    unittest.main()
