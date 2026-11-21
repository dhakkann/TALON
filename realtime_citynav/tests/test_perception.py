"""Unit tests for Real-Time Perception and 3D Back-Projection Math."""

import unittest
import time
import numpy as np
from realtime_citynav.perception import RealtimePerception


class TestPerceptionMath(unittest.TestCase):
    def setUp(self):
        # 640x480 resolution, 90 deg FOV
        self.perception = RealtimePerception(
            fov_degrees=90.0,
            width=640,
            height=480,
            mock=True
        )

    def test_intrinsics_math(self):
        """Verify focal length and optical center precomputations."""
        # For 90 degree FOV: tan(45 deg) = 1.0 => fx = 640 / (2 * 1) = 320.0
        self.assertAlmostEqual(self.perception.fx, 320.0, places=3)
        self.assertAlmostEqual(self.perception.cx, 320.0, places=3)
        self.assertAlmostEqual(self.perception.cy, 240.0, places=3)

    def test_direct_forward_projection(self):
        """A point at optical center (u=cx, v=cy) with 10m depth should be directly 10m ahead."""
        depth_img = np.ones((480, 640), dtype=np.float32) * 10.0
        # Drone at world origin (0, 0, -5), facing North (yaw = 0)
        drone_pose = (0.0, 0.0, -5.0, 0.0)

        world_pos = self.perception.project_centroid_to_world(
            u_c=320.0,
            v_c=240.0,
            depth_img=depth_img,
            drone_pose=drone_pose
        )
        self.assertIsNotNone(world_pos)
        # World NED: X is forward (North), Y is right (East), Z is down
        self.assertAlmostEqual(world_pos[0], 10.0, places=2)  # 10m North
        self.assertAlmostEqual(world_pos[1], 0.0, places=2)   # 0m East
        self.assertAlmostEqual(world_pos[2], -5.0, places=2)  # -5m Alt

    def test_yaw_rotation_projection(self):
        """When drone yaw is 90 deg (facing East, +Y), optical center point should project East."""
        depth_img = np.ones((480, 640), dtype=np.float32) * 10.0
        # Drone at origin facing East (yaw = +pi/2)
        yaw_east = np.pi / 2.0
        drone_pose = (0.0, 0.0, -5.0, yaw_east)

        world_pos = self.perception.project_centroid_to_world(
            u_c=320.0,
            v_c=240.0,
            depth_img=depth_img,
            drone_pose=drone_pose
        )
        self.assertIsNotNone(world_pos)
        # Point is directly ahead of drone (in camera frame), so in world it should be at +Y (East)
        self.assertAlmostEqual(world_pos[0], 0.0, places=2)   # 0m North
        self.assertAlmostEqual(world_pos[1], 10.0, places=2)  # 10m East
        self.assertAlmostEqual(world_pos[2], -5.0, places=2)

    def test_lateral_offset_projection(self):
        """A point to the right of center (u > cx) should have positive Y_body."""
        depth_img = np.ones((480, 640), dtype=np.float32) * 10.0
        drone_pose = (0.0, 0.0, 0.0, 0.0)

        # u = cx + fx = 320 + 320 = 640 (at right edge of image)
        # x_cam = (640 - 320) * 10 / 320 = 10.0m right
        world_pos = self.perception.project_centroid_to_world(
            u_c=640.0,
            v_c=240.0,
            depth_img=depth_img,
            drone_pose=drone_pose
        )
        self.assertIsNotNone(world_pos)
        self.assertAlmostEqual(world_pos[0], 10.0, places=2)  # Forward
        self.assertAlmostEqual(world_pos[1], 10.0, places=2)  # Right

    def test_invalid_depth_filtering(self):
        """Depth beyond max_depth (e.g. sky = 100m) or below min_depth (0.1m) should be filtered."""
        depth_img = np.ones((480, 640), dtype=np.float32) * 150.0  # Sky / horizon
        drone_pose = (0.0, 0.0, -5.0, 0.0)

        world_pos = self.perception.project_centroid_to_world(
            u_c=320.0, v_c=240.0, depth_img=depth_img, drone_pose=drone_pose
        )
        self.assertIsNone(world_pos)

    def test_projection_latency_benchmark(self):
        """Projection calculation must complete in sub-millisecond (<1.0 ms) time."""
        depth_img = np.random.uniform(5.0, 30.0, (480, 640)).astype(np.float32)
        drone_pose = (10.0, 20.0, -5.0, 0.5)

        # Warmup
        for _ in range(10):
            self.perception.project_centroid_to_world(320.0, 240.0, depth_img, drone_pose)

        t0 = time.perf_counter()
        iters = 500
        for _ in range(iters):
            self.perception.project_centroid_to_world(320.0, 240.0, depth_img, drone_pose)
        total_time_ms = (time.perf_counter() - t0) * 1000.0
        avg_ms = total_time_ms / iters

        print(f"\n[Benchmark] Centroid projection latency: {avg_ms:.4f} ms per call ({iters/total_time_ms*1000:.0f} Hz)")
        self.assertLess(avg_ms, 1.0, f"Projection took {avg_ms:.4f} ms, exceeding 1ms constraint")

    def test_dynamic_vocabulary_update(self):
        """Vocabulary updates should modify class list."""
        success = self.perception.update_target_vocabulary(["fountain", "blue car", "tower"])
        self.assertTrue(success)
        self.assertIn("fountain", self.perception.current_classes)
        self.assertIn("blue car", self.perception.current_classes)

    def test_out_of_bounds_and_nan_inputs(self):
        """Projection safely rejects NaNs, Infs, and out-of-bounds pixel coordinates."""
        depth_img = np.ones((480, 640), dtype=np.float32) * 10.0
        drone_pose = (0.0, 0.0, -5.0, 0.0)

        # NaN / Inf coordinates
        self.assertIsNone(self.perception.project_centroid_to_world(np.nan, 240.0, depth_img, drone_pose))
        self.assertIsNone(self.perception.project_centroid_to_world(320.0, np.inf, depth_img, drone_pose))

        # Negative and out-of-bounds coordinates
        self.assertIsNone(self.perception.project_centroid_to_world(-10.0, 240.0, depth_img, drone_pose))
        self.assertIsNone(self.perception.project_centroid_to_world(320.0, -5.0, depth_img, drone_pose))
        self.assertIsNone(self.perception.project_centroid_to_world(700.0, 240.0, depth_img, drone_pose))
        self.assertIsNone(self.perception.project_centroid_to_world(320.0, 500.0, depth_img, drone_pose))

    def test_pose_with_6_elements(self):
        """Drone pose passed as (x, y, z, roll, pitch, yaw) unpacks and projects accurately."""
        depth_img = np.ones((480, 640), dtype=np.float32) * 10.0
        drone_pose_6d = (0.0, 0.0, -5.0, 0.0, 0.0, 0.0)
        pos = self.perception.project_centroid_to_world(320.0, 240.0, depth_img, drone_pose_6d)
        self.assertIsNotNone(pos)
        self.assertAlmostEqual(pos[0], 10.0, places=2)

    def test_3d_depth_image_squeeze(self):
        """Depth image with shape (H, W, 1) is automatically handled."""
        depth_3d = np.ones((480, 640, 1), dtype=np.float32) * 10.0
        drone_pose = (0.0, 0.0, -5.0, 0.0)
        pos = self.perception.project_centroid_to_world(320.0, 240.0, depth_3d, drone_pose)
        self.assertIsNotNone(pos)
        self.assertAlmostEqual(pos[0], 10.0, places=2)

    def test_annotate_none_or_empty_frame(self):
        """Annotating None or empty image does not raise error and returns valid black frame."""
        frame = self.perception.annotate_frame(None, [])
        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape, (480, 640, 3))


if __name__ == "__main__":
    unittest.main()
