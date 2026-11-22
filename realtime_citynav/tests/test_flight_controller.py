"""Unit tests for Continuous Flight Controller and APF Obstacle Avoidance."""

import unittest
import numpy as np
from realtime_citynav.flight_controller import RealtimeFlightController


class TestFlightController(unittest.TestCase):
    def setUp(self):
        self.controller = RealtimeFlightController(
            cruise_speed=4.0,
            max_speed=6.0,
            proportional_gain=1.2,
            arrival_radius=1.8,
            obstacle_dist_threshold=4.0,
            target_altitude=5.0,
            mock=True
        )

    def test_takeoff_altitude(self):
        """Takeoff transitions vehicle to target altitude in NED coordinates."""
        success = self.controller.takeoff(altitude=5.0)
        self.assertTrue(success)
        self.assertTrue(self.controller.is_in_air)
        pose = self.controller.get_pose()
        self.assertAlmostEqual(pose[2], -5.0, places=1)

    def test_proportional_guidance_forward(self):
        """Target directly ahead produces forward velocity vector."""
        self.controller.takeoff(altitude=5.0)
        # Place target 20 meters North (+X)
        self.controller.set_target(np.array([20.0, 0.0, -5.0]))

        state = self.controller.step_control(depth_img=None)
        cmd_v = state["cmd_velocity"]

        # Forward velocity (vx) should be positive and close to cruise speed (4.0 m/s)
        self.assertGreater(cmd_v[0], 2.0)
        self.assertLessEqual(state["speed"], 6.0)
        self.assertAlmostEqual(cmd_v[1], 0.0, places=1)
        self.assertFalse(state["has_arrived"])

    def test_waypoint_arrival_hover(self):
        """When drone is within arrival radius (1.8m), controller hovers."""
        self.controller.takeoff(altitude=5.0)
        curr_x, curr_y, curr_z, _ = self.controller.get_pose()
        # Set target 1.0m away (inside 1.8m arrival radius)
        self.controller.set_target(np.array([curr_x + 1.0, curr_y, curr_z]))

        state = self.controller.step_control(depth_img=None)
        self.assertTrue(state["has_arrived"])
        # Horizontal command velocity should be zero (hovering)
        self.assertAlmostEqual(state["cmd_velocity"][0], 0.0, places=2)
        self.assertAlmostEqual(state["cmd_velocity"][1], 0.0, places=2)

    def test_reactive_apf_avoidance_steers_right(self):
        """Obstacle concentrated on left field-of-view triggers RIGHT deflection."""
        depth_img = np.ones((480, 640), dtype=np.float32) * 20.0
        # Place dense close obstacle (<4m) on left half of central ROI
        h, w = 480, 640
        depth_img[h // 4: 3 * h // 4, w // 4: w // 2] = 2.0  # 2m close on left

        rep_vector, direction = self.controller.compute_obstacle_repulsion(depth_img, drone_yaw=0.0)
        self.assertEqual(direction, "RIGHT")
        self.assertGreater(rep_vector[1], 0.0)  # +Y is East / Right
        self.assertLess(rep_vector[0], 0.0)     # Braking / back off (-X)

    def test_reactive_apf_avoidance_steers_left(self):
        """Obstacle concentrated on right field-of-view triggers LEFT deflection."""
        depth_img = np.ones((480, 640), dtype=np.float32) * 20.0
        # Place dense close obstacle on right half of central ROI
        h, w = 480, 640
        depth_img[h // 4: 3 * h // 4, w // 2: 3 * w // 4] = 2.0  # 2m close on right

        rep_vector, direction = self.controller.compute_obstacle_repulsion(depth_img, drone_yaw=0.0)
        self.assertEqual(direction, "LEFT")
        self.assertLess(rep_vector[1], 0.0)  # -Y is West / Left
        self.assertLess(rep_vector[0], 0.0)  # Braking / back off (-X)

    def test_kinematic_continuous_integration(self):
        """Iterative stepping updates vehicle position smoothly."""
        self.controller.takeoff(altitude=5.0)
        self.controller.set_target(np.array([50.0, 0.0, -5.0]))

        start_x = self.controller.get_pose()[0]
        # Simulate 20 steps (~1 second of flight at 20 Hz)
        for _ in range(20):
            self.controller.step_control(depth_img=None)

        end_x = self.controller.get_pose()[0]
        self.assertGreater(end_x, start_x + 0.5, "Drone should advance along +X")

    def test_ground_stationary_idle(self):
        """When vehicle is on ground (not airborne), step_control must not command movement."""
        self.assertFalse(self.controller.is_in_air)
        initial_pose = self.controller.get_pose()
        for _ in range(10):
            state = self.controller.step_control(depth_img=None)
            self.assertEqual(state["speed"], 0.0)
            self.assertEqual(state["cmd_velocity"], [0.0, 0.0, 0.0])
        final_pose = self.controller.get_pose()
        self.assertEqual(initial_pose, final_pose)

    def test_explicit_hover_mode(self):
        """Explicit hover mode commands zero horizontal velocity while holding altitude."""
        self.controller.takeoff(altitude=5.0)
        self.controller.hover()
        self.assertTrue(self.controller.is_hovering)
        self.assertIsNone(self.controller.target_position)

        state = self.controller.step_control(depth_img=None)
        cmd_v = state["cmd_velocity"]
        self.assertAlmostEqual(cmd_v[0], 0.0, places=2)
        self.assertAlmostEqual(cmd_v[1], 0.0, places=2)

    def test_obstacle_repulsion_nan_robustness(self):
        """APF obstacle repulsion handles degenerate depth maps containing NaNs and Infs."""
        depth_img = np.full((480, 640), np.nan, dtype=np.float32)
        rep, direction = self.controller.compute_obstacle_repulsion(depth_img, drone_yaw=0.0)
        self.assertIsNone(direction)
        np.testing.assert_allclose(rep, [0.0, 0.0, 0.0])

        # Mixed valid obstacles with NaNs
        depth_img[120:360, 160:320] = 2.0  # Obstacle on left
        rep, direction = self.controller.compute_obstacle_repulsion(depth_img, drone_yaw=0.0)
        self.assertEqual(direction, "RIGHT")
        self.assertGreater(rep[1], 0.0)

    def test_vehicle_name_support(self):
        """Flight controller correctly stores and handles custom vehicle_name."""
        custom_fc = RealtimeFlightController(vehicle_name="Drone_1", mock=True)
        self.assertEqual(custom_fc.vehicle_name, "Drone_1")

    def test_geofence_perimeter_enforcement(self):
        """Outside geofence radius (>160m), open search steers back towards city center."""
        self.controller.takeoff(altitude=5.0)
        self.controller._sim_pos = np.array([250.0, 50.0, -5.0], dtype=float)
        self.controller.is_hovering = False
        self.controller.set_target(None)  # Open search sweep mode

        state = self.controller.step_control(depth_img=None)
        cmd_v = state["cmd_velocity"]

        # Velocity must point back towards city center (negative X and negative Y from [250, 50])
        self.assertLess(cmd_v[0], -1.0)
        self.assertLess(cmd_v[1], -0.2)


if __name__ == "__main__":
    unittest.main()
