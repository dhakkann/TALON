"""Unit tests for Multi-Environment Configuration and Perception Engines (COCO, Color Geometric, Segmentation)."""

import os
import unittest
import numpy as np
import cv2
from realtime_citynav.config import SystemConfig, apply_environment_preset, EnvironmentConfig
from realtime_citynav.perception import RealtimePerception


class TestMultiEnvironment(unittest.TestCase):
    def test_environment_presets(self):
        """Verify that apply_environment_preset sets correct parameters for each environment."""
        cfg = SystemConfig()

        # Neighborhood preset
        apply_environment_preset(cfg, "neighborhood")
        self.assertEqual(cfg.env.name, "neighborhood")
        self.assertEqual(cfg.env.takeoff_altitude, 8.0)
        self.assertEqual(cfg.flight.takeoff_altitude, 8.0)
        self.assertEqual(cfg.env.perception_engine, "coco")
        self.assertIn("car", cfg.env.default_classes)
        self.assertIn("white car", cfg.env.default_landmarks)

        # Blocks preset
        apply_environment_preset(cfg, "blocks")
        self.assertEqual(cfg.env.name, "blocks")
        self.assertEqual(cfg.env.takeoff_altitude, 5.0)
        self.assertEqual(cfg.flight.takeoff_altitude, 5.0)
        self.assertEqual(cfg.env.perception_engine, "color")
        self.assertIn("orange block", cfg.env.default_landmarks)

        # City preset
        apply_environment_preset(cfg, "city")
        self.assertEqual(cfg.env.name, "city")
        self.assertEqual(cfg.env.takeoff_altitude, 12.0)
        self.assertEqual(cfg.flight.takeoff_altitude, 12.0)
        self.assertEqual(cfg.env.perception_engine, "yolo_world")
        self.assertIn("fountain", cfg.env.default_landmarks)

    def test_color_geometric_detector(self):
        """Verify the HSV color & contour detector finds colored shapes without neural network."""
        perception = RealtimePerception(mock=True, engine="color")
        self.assertEqual(perception.engine, "color")
        perception.update_target_vocabulary(["orange block", "blue cylinder"])

        # Create 640x480 black image (BGR format for OpenCV)
        img = np.zeros((480, 640, 3), dtype=np.uint8)

        # In BGR: Orange is approximately (B=0, G=140, R=255) -> In HSV: H~16, S~255, V~255
        cv2.rectangle(img, (270, 190), (370, 290), (0, 140, 255), -1)

        # In BGR: Blue is (B=255, G=0, R=0) -> In HSV: H~120, S~255, V~255
        cv2.rectangle(img, (50, 150), (150, 250), (255, 0, 0), -1)

        depth_img = np.ones((480, 640), dtype=np.float32) * 10.0
        drone_pose = (0.0, 0.0, -5.0, 0.0)

        detections = perception._detect_color_geometric(img, depth_img, drone_pose)
        self.assertGreaterEqual(len(detections), 2)

        names = [d["name"].lower() for d in detections]
        self.assertTrue(any("orange" in n for n in names))
        self.assertTrue(any("blue" in n for n in names))

        orange_det = next(d for d in detections if "orange" in d["name"].lower())
        bbox = orange_det["bbox"]
        self.assertAlmostEqual(bbox[0], 270, delta=15)
        self.assertAlmostEqual(bbox[1], 190, delta=15)
        self.assertAlmostEqual(bbox[2], 370, delta=15)
        self.assertAlmostEqual(bbox[3], 290, delta=15)
        self.assertIsNotNone(orange_det["world_pos"])

    def test_segmentation_detector(self):
        """Verify the ground-truth segmentation mask decoder detects labeled actors."""
        perception = RealtimePerception(mock=True, engine="segmentation")
        self.assertEqual(perception.engine, "segmentation")

        # Create an image with an unmapped actor ID (85) and mapped actor ID (1 -> fountain)
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(img, (200, 150), (350, 300), (85, 85, 85), -1)
        cv2.rectangle(img, (400, 150), (550, 300), (1, 1, 1), -1)

        depth_img = np.ones((480, 640), dtype=np.float32) * 8.0
        drone_pose = (0.0, 0.0, -5.0, 0.0)

        detections = perception._detect_segmentation(img, depth_img, drone_pose)
        self.assertGreaterEqual(len(detections), 2)

        names = [d["name"] for d in detections]
        self.assertIn("actor_object_85", names)
        self.assertIn("fountain", names)

    def test_target_vocabulary_update_without_set_classes(self):
        """Verify that changing target vocabulary on COCO/Color updates current_classes safely."""
        perception = RealtimePerception(mock=True, engine="color")
        perception.update_target_vocabulary(["red pillar", "blue cylinder"])
        self.assertEqual(perception.current_classes, ["red pillar", "blue cylinder"])

        coco_perception = RealtimePerception(mock=True, engine="coco")
        coco_perception.update_target_vocabulary(["car", "truck", "stop sign"])
        self.assertEqual(coco_perception.current_classes, ["car", "truck", "stop sign"])

    def test_coco_detector_with_weights(self):
        """Verify COCO engine loads yolov8n.pt if available or handles mock gracefully."""
        if os.path.exists("yolov8n.pt"):
            perception = RealtimePerception(mock=False, engine="coco", model_path="yolov8n.pt")
            self.assertEqual(perception.engine, "coco")
            self.assertIsNotNone(perception.model)


if __name__ == "__main__":
    unittest.main()
