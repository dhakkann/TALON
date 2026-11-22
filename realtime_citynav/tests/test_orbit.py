"""Unit and integration tests for Orbit Circle and Continuous Waypoint Trajectory Curve."""

import unittest
import math
import time
import numpy as np

from realtime_citynav.dialogue_manager import DialogueManager
from realtime_citynav.flight_controller import RealtimeFlightController
from realtime_citynav.config import SystemConfig
from realtime_citynav.main import CityNavOrchestrator


class TestOrbitDialogueParsing(unittest.TestCase):
    """Test natural language parsing of orbit and curve trajectory commands."""

    def setUp(self):
        self.dm = DialogueManager(api_key=None, mock=True)

    def test_is_orbit_command_detection(self):
        """Orbit commands are identified across diverse phrasing variations."""
        valid_orbit_phrases = [
            "hover in circle around blue building",
            "hover in a circle around the blue building",
            "circle around blue building",
            "circle blue building",
            "circle the blue building",
            "orbit around blue building",
            "orbit blue building",
            "circle twice around blue building",
            "circle 3 times around red building",
            "orbit tower facing forward",
            "circle looking ahead around blue building",
            "hover in circle around blue building at 18m radius and 12m altitude",
            "continuous curve around blue building",
            "make a circle around tower",
            "fly in circles around fountain",
        ]
        for phrase in valid_orbit_phrases:
            self.assertTrue(self.dm.is_orbit_command(phrase), f"Failed to detect orbit in: '{phrase}'")

        invalid_phrases = [
            "hover 100m above red building",
            "stop and hover right here",
            "fly to the fountain, then pass red building",
            "go from red to blue building",
            "what have you seen so far?",
            "climb to 15m altitude",
        ]
        for phrase in invalid_phrases:
            self.assertFalse(self.dm.is_orbit_command(phrase), f"False positive orbit detected in: '{phrase}'")

    def test_parse_orbit_command_default_parameters(self):
        """Orbit command parses sensible defaults (radius=16m, alt=10m, speed=3m/s, facing=INWARD, revolutions=None)."""
        cmd = "hover in circle around blue building"
        params = self.dm.parse_orbit_command(cmd)
        self.assertIsNotNone(params)
        self.assertEqual(params["target"], "blue building")
        self.assertEqual(params["base_landmark"], "blue building")
        self.assertEqual(params["radius"], 16.0)
        self.assertEqual(params["altitude"], 10.0)
        self.assertEqual(params["speed"], 3.0)
        self.assertEqual(params["facing"], "INWARD")
        self.assertIsNone(params["revolutions"])
        self.assertEqual(params["direction"], "CCW")
        self.assertIsNotNone(params["waypoints"])
        self.assertEqual(len(params["waypoints"]), 16)

    def test_parse_orbit_command_revolutions(self):
        """Revolutions counts (twice, thrice, N times, laps) are parsed accurately."""
        # 'twice' -> 2.0
        p1 = self.dm.parse_orbit_command("circle twice around blue building")
        self.assertEqual(p1["revolutions"], 2.0)

        # 'thrice' -> 3.0
        p2 = self.dm.parse_orbit_command("circle thrice around red building")
        self.assertEqual(p2["revolutions"], 3.0)

        # 'once' -> 1.0
        p3 = self.dm.parse_orbit_command("circle once around tower")
        self.assertEqual(p3["revolutions"], 1.0)

        # '4 times' -> 4.0
        p4 = self.dm.parse_orbit_command("orbit 4 times around hospital")
        self.assertEqual(p4["revolutions"], 4.0)

        # '2 laps' -> 2.0
        p5 = self.dm.parse_orbit_command("circle 2 laps around blue building")
        self.assertEqual(p5["revolutions"], 2.0)

    def test_parse_orbit_command_facing(self):
        """Facing mode correctly distinguishes default INWARD inspection vs TANGENT forward flight."""
        # Default is INWARD
        p_inward = self.dm.parse_orbit_command("hover in circle around blue building")
        self.assertEqual(p_inward["facing"], "INWARD")

        # Explicit tangent / forward / looking ahead
        p_fwd = self.dm.parse_orbit_command("circle blue building facing forward")
        self.assertEqual(p_fwd["facing"], "TANGENT")

        p_ahead = self.dm.parse_orbit_command("orbit tower looking ahead")
        self.assertEqual(p_ahead["facing"], "TANGENT")

        p_tangent = self.dm.parse_orbit_command("circle facing tangent around fountain")
        self.assertEqual(p_tangent["facing"], "TANGENT")

    def test_parse_orbit_command_custom_geometry(self):
        """Custom radius, altitude, speed, and direction are extracted accurately."""
        cmd = "hover in circle around blue building at 18m radius and 12m altitude at 4 m/s clockwise"
        p = self.dm.parse_orbit_command(cmd)
        self.assertEqual(p["radius"], 18.0)
        self.assertEqual(p["altitude"], 12.0)
        self.assertEqual(p["speed"], 4.0)
        self.assertEqual(p["direction"], "CW")

    def test_shorthand_landmark_resolution_in_orbit(self):
        """Shorthand adjectives ('blue', 'red') resolve to full landmark names in orbit commands."""
        p_blue = self.dm.parse_orbit_command("hover in circle around blue")
        self.assertEqual(p_blue["target"], "blue building")

        p_red = self.dm.parse_orbit_command("circle twice around red")
        self.assertEqual(p_red["target"], "red building")

    def test_generate_orbit_waypoints_geometry(self):
        """Waypoint coordinate generator produces points at exact radius and altitude."""
        center = np.array([75.0, -14.0, 0.0])
        wps = DialogueManager.generate_orbit_waypoints(center, radius=16.0, altitude=10.0, num_points=8)
        self.assertEqual(len(wps), 8)

        for pt in wps:
            # Altitude check: NED z = -10.0
            self.assertAlmostEqual(pt[2], -10.0, places=1)
            # Radius check: hypot(x - 75, y - (-14)) == 16.0
            r = math.hypot(pt[0] - center[0], pt[1] - center[1])
            self.assertAlmostEqual(r, 16.0, places=1)

    def test_interruption_orbit_classification(self):
        """Interruption manager classifies orbit commands into ORBIT_CIRCLE intent."""
        self.dm.queue_user_message("hover in circle around blue building")
        res = self.dm.process_interruption(current_target="fountain")
        self.assertIsNotNone(res)
        self.assertEqual(res["intent"], "ORBIT_CIRCLE")
        self.assertEqual(res["target"], "blue building")
        self.assertEqual(res["radius"], 16.0)
        self.assertEqual(res["altitude"], 10.0)
        self.assertEqual(res["facing"], "INWARD")
        self.assertIn("Orbiting", res["reply"])

    def test_interruption_coordinate_path_request(self):
        """When operator asks for coordinates/path curve, DialogueManager attaches 3D coordinates."""
        self.dm.queue_user_message("give a path coordinates to circle blue building")
        res = self.dm.process_interruption(current_target="tower")
        self.assertIsNotNone(res)
        self.assertEqual(res["intent"], "WAYPOINT_PATH")
        self.assertIsNotNone(res["waypoints"])
        self.assertGreaterEqual(len(res["waypoints"]), 8)

    def test_interruption_explicit_waypoint_path(self):
        """Explicit waypoint array [[x, y, z], ...] is classified as WAYPOINT_PATH."""
        self.dm.queue_user_message("follow curve [[10, 20, -5], [15, 25, -6], [20, 30, -5]]")
        res = self.dm.process_interruption()
        self.assertIsNotNone(res)
        self.assertEqual(res["intent"], "WAYPOINT_PATH")
        self.assertEqual(len(res["waypoints"]), 3)
        self.assertEqual(res["waypoints"][0], [10.0, 20.0, -5.0])

    def test_parse_initial_instruction_orbit_subgoal(self):
        """Initial instruction containing orbit decomposes cleanly into orbit subgoal."""
        subgoals = self.dm.parse_initial_instruction("hover in circle around blue building")
        self.assertEqual(len(subgoals), 1)
        self.assertTrue(self.dm.is_orbit_command(subgoals[0]))

    def test_parse_initial_instruction_compound_multi_orbit(self):
        """Compound multi-orbit instructions retain all orbit legs and exact revolution counts."""
        instruction = "hover in a circle around blue building 3 times, and then around red building 2 times"
        subgoals = self.dm.parse_initial_instruction(instruction)
        self.assertEqual(len(subgoals), 2)
        self.assertEqual(subgoals[0], "circle around blue building 3 times")
        self.assertEqual(subgoals[1], "circle around red building 2 times")
        p0 = self.dm.parse_orbit_command(subgoals[0])
        p1 = self.dm.parse_orbit_command(subgoals[1])
        self.assertEqual(p0["target"], "blue building")
        self.assertEqual(p0["revolutions"], 3.0)
        self.assertEqual(p1["target"], "red building")
        self.assertEqual(p1["revolutions"], 2.0)


class TestFlightControllerOrbitKinematics(unittest.TestCase):
    """Test 20-30 Hz continuous orbit mathematical curve and yaw tracking."""

    def setUp(self):
        self.controller = RealtimeFlightController(
            cruise_speed=4.0,
            max_speed=6.0,
            proportional_gain=1.2,
            target_altitude=10.0,
            mock=True
        )
        self.controller.takeoff(altitude=10.0)

    def test_set_orbit_target_initialization(self):
        """Configuring orbit target sets state and flags correctly."""
        center = np.array([75.0, -14.0, 0.0])
        self.controller.set_orbit_target(
            center_xyz=center,
            radius=16.0,
            altitude=10.0,
            speed=3.0,
            facing="INWARD",
            revolutions=2.0,
            direction="CCW",
            target_name="blue building"
        )
        self.assertTrue(self.controller.is_orbiting)
        self.assertFalse(self.controller.is_hovering)
        self.assertEqual(self.controller.orbit_radius, 16.0)
        self.assertEqual(self.controller.orbit_facing, "INWARD")
        self.assertEqual(self.controller.orbit_revolutions, 2.0)
        self.assertEqual(self.controller.orbit_direction, 1)

    def test_inward_facing_yaw_tracking(self):
        """In inward-facing orbit, drone yaw actively aligns toward landmark center."""
        center = np.array([75.0, -14.0, 0.0])
        # Place drone directly South of center: (75, -30) -> center is North (+Y)
        self.controller._sim_pos = np.array([75.0, -30.0, -10.0], dtype=float)
        self.controller._sim_yaw = 0.0

        self.controller.set_orbit_target(
            center_xyz=center,
            radius=16.0,
            altitude=10.0,
            speed=3.0,
            facing="INWARD",
            target_name="blue building"
        )

        # Step controller
        state = self.controller.step_control()
        # Vector to center: (0, +16) -> atan2(16, 0) = +pi/2 (+90 deg)
        expected_yaw = math.atan2(center[1] - (-30.0), center[0] - 75.0)  # +pi/2
        self.assertAlmostEqual(expected_yaw, math.pi / 2.0, places=2)

        # Integrate several steps: yaw should turn toward +pi/2
        for _ in range(25):
            self.controller.step_control()

        curr_yaw = self.controller.get_pose()[3]
        self.assertGreater(curr_yaw, 0.5, "Drone yaw should rotate toward landmark center")

    def test_tangent_facing_yaw_tracking(self):
        """In tangent-facing orbit, drone yaw aligns along velocity vector."""
        center = np.array([75.0, -14.0, 0.0])
        # Place drone at (75 + 16, -14) = (91, -14) (East of center)
        self.controller._sim_pos = np.array([91.0, -14.0, -10.0], dtype=float)

        self.controller.set_orbit_target(
            center_xyz=center,
            radius=16.0,
            altitude=10.0,
            speed=3.0,
            facing="TANGENT",
            direction="CCW",
            target_name="blue building"
        )

        state = self.controller.step_control()
        cmd_v = state["cmd_velocity"]

        # At theta=0, CCW tangent is [-sin(0), cos(0)] = [0, +1] (North/East)
        self.assertAlmostEqual(cmd_v[0], 0.0, places=1)
        self.assertGreater(cmd_v[1], 2.0, "Tangent velocity should point along +Y")

    def test_continuous_orbit_kinematic_integration(self):
        """Drone stays locked on orbital radius during continuous stepping."""
        center = np.array([75.0, -14.0, 0.0])
        # Start exactly on the circle perimeter at radius 16m
        self.controller._sim_pos = np.array([75.0 + 16.0, -14.0, -10.0], dtype=float)

        self.controller.set_orbit_target(
            center_xyz=center,
            radius=16.0,
            altitude=10.0,
            speed=3.0,
            facing="INWARD",
            revolutions=None,  # continuous
            direction="CCW",
            target_name="blue building"
        )

        # Simulate 100 steps (~2.5-3 seconds of orbit)
        for _ in range(100):
            state = self.controller.step_control()
            curr_pos = self.controller.get_pose()[:3]
            r = math.hypot(curr_pos[0] - center[0], curr_pos[1] - center[1])
            # Radius must stay tightly maintained within 1.0m of 16m
            self.assertAlmostEqual(r, 16.0, delta=1.0)
            self.assertAlmostEqual(curr_pos[2], -10.0, delta=0.5)

        self.assertTrue(self.controller.is_orbiting)
        self.assertGreater(self.controller.orbit_accumulated_angle, 0.15)

    def test_radial_error_correction_outside_circle(self):
        """When drone starts outside radius (e.g. 25m away), controller commands inward velocity."""
        center = np.array([75.0, -14.0, 0.0])
        # Place drone 30m East of center (radius error: 16 - 30 = -14m)
        self.controller._sim_pos = np.array([75.0 + 30.0, -14.0, -10.0], dtype=float)

        self.controller.set_orbit_target(
            center_xyz=center,
            radius=16.0,
            altitude=10.0,
            speed=3.0,
            target_name="blue building"
        )

        state = self.controller.step_control()
        cmd_v = state["cmd_velocity"]

        # Horizontal velocity should command inward radial component (-X)
        self.assertLess(cmd_v[0], -1.0, "Velocity should steer inward toward orbit radius")

    def test_revolution_termination_hover(self):
        """Counted orbit terminates after N complete revolutions and holds stationary hover."""
        center = np.array([75.0, -14.0, 0.0])
        self.controller._sim_pos = np.array([75.0 + 16.0, -14.0, -10.0], dtype=float)

        # Command exactly 1 revolution
        self.controller.set_orbit_target(
            center_xyz=center,
            radius=16.0,
            altitude=10.0,
            speed=4.0,
            revolutions=1.0,
            target_name="blue building"
        )

        # Simulate angular progress artificially exceeding 2*pi (1 full revolution)
        self.controller.orbit_accumulated_angle = 2.0 * math.pi + 0.1

        state = self.controller.step_control()
        self.assertTrue(state["has_arrived"])
        self.assertFalse(self.controller.is_orbiting)
        self.assertTrue(self.controller.is_hovering)
        self.assertAlmostEqual(state["cmd_velocity"][0], 0.0, places=2)
        self.assertAlmostEqual(state["cmd_velocity"][1], 0.0, places=2)

    def test_waypoint_path_following(self):
        """Controller steps through discrete continuous waypoint sequence."""
        wps = [
            [10.0, 0.0, -5.0],
            [20.0, 5.0, -5.0],
            [30.0, 10.0, -5.0]
        ]
        self.controller.set_trajectory_path(wps, speed=3.0)
        self.assertTrue(self.controller.is_following_path)
        self.assertEqual(self.controller.trajectory_index, 0)

        # Place drone near waypoint 0 -> advances to waypoint 1
        self.controller._sim_pos = np.array([10.5, 0.2, -5.0], dtype=float)
        state = self.controller.step_control()
        self.assertEqual(self.controller.trajectory_index, 1)

        # Place drone near waypoint 1 -> advances to waypoint 2
        self.controller._sim_pos = np.array([20.2, 5.1, -5.0], dtype=float)
        state = self.controller.step_control()
        self.assertEqual(self.controller.trajectory_index, 2)

        # Place drone near final waypoint -> completes path and hovers
        self.controller._sim_pos = np.array([30.2, 10.1, -5.0], dtype=float)
        state = self.controller.step_control()
        self.assertTrue(state["has_arrived"])
        self.assertFalse(self.controller.is_following_path)
        self.assertTrue(self.controller.is_hovering)

    def test_flight_controller_compound_multi_orbit_transit(self):
        """Simulate sequential compound orbit: blue building 3x revs then red building 2x revs."""
        self.controller.takeoff(altitude=10.0)
        blue_center = np.array([75.0, -14.0, 0.0])
        red_center = np.array([75.0, 12.0, 0.0])

        # Leg 1: blue building 3 revs
        self.controller.set_orbit_target(
            center_xyz=blue_center,
            radius=16.0,
            altitude=10.0,
            speed=6.0,
            revolutions=3.0,
            target_name="blue building"
        )
        for _ in range(15000):
            state = self.controller.step_control()
            if state.get("has_arrived"):
                break

        self.assertTrue(self.controller.orbit_completed)
        self.assertGreaterEqual(self.controller.orbit_accumulated_angle / (2.0 * math.pi), 3.0)

        # Leg 2: red building 2 revs
        self.controller.set_orbit_target(
            center_xyz=red_center,
            radius=16.0,
            altitude=10.0,
            speed=6.0,
            revolutions=2.0,
            target_name="red building"
        )
        for _ in range(15000):
            state = self.controller.step_control()
            if state.get("has_arrived"):
                break

        self.assertTrue(self.controller.orbit_completed)
        self.assertGreaterEqual(self.controller.orbit_accumulated_angle / (2.0 * math.pi), 2.0)
        p = self.controller.get_pose()
        dist_to_red = math.hypot(p[0] - red_center[0], p[1] - red_center[1])
        self.assertAlmostEqual(dist_to_red, 16.0, delta=1.5)


class TestOrchestratorOrbitIntegration(unittest.TestCase):
    """End-to-end integration tests for orbit mission launch and operator interruptions."""

    @classmethod
    def setUpClass(cls):
        cls.config = SystemConfig()
        cls.config.mock_airsim = True
        cls.config.mock_perception = True
        cls.config.mock_gemini = True
        cls.config.dashboard.port = 5056

        cls.orchestrator = CityNavOrchestrator(cls.config)
        cls.orchestrator.start()
        time.sleep(0.3)

    @classmethod
    def tearDownClass(cls):
        cls.orchestrator.stop()

    def test_start_mission_orbit_blue_building(self):
        """Starting mission with 'hover in circle around blue building' engages orbit mode."""
        self.orchestrator.start_mission("hover in circle around blue building")
        time.sleep(0.4)

        telem = self.orchestrator.get_telemetry()
        self.assertEqual(telem["mode"], "ORBIT")
        self.assertTrue(telem["is_orbiting"])
        self.assertEqual(telem["orbit_target"], "blue building")
        self.assertEqual(telem["orbit_radius"], 16.0)
        self.assertIsNotNone(telem["orbit_center"])
        self.assertAlmostEqual(telem["orbit_center"][0], 75.0, places=1)
        self.assertAlmostEqual(telem["orbit_center"][1], -14.0, places=1)

    def test_mid_flight_interruption_orbit(self):
        """Mid-flight operator command 'circle twice around blue building' diverts into counted orbit."""
        # Operator commands counted orbit
        self.orchestrator.dialogue.queue_user_message("circle twice facing forward around blue building")
        time.sleep(0.4)

        telem = self.orchestrator.get_telemetry()
        self.assertEqual(telem["mode"], "ORBIT")
        self.assertTrue(telem["is_orbiting"])
        self.assertEqual(telem["orbit_facing"], "TANGENT")

    def test_mid_flight_interruption_stop_hover_from_orbit(self):
        """Operator 'stop and hover' halts active orbit cleanly."""
        self.orchestrator.dialogue.queue_user_message("stop and hover")
        time.sleep(0.4)

        telem = self.orchestrator.get_telemetry()
        self.assertEqual(telem["mode"], "HOVERING")
        self.assertFalse(telem["is_orbiting"])
        self.assertTrue(self.orchestrator.flight.is_hovering)


if __name__ == "__main__":
    unittest.main()
