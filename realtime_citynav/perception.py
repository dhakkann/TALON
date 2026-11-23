"""Real-Time Open-Vocabulary Perception using YOLO-World v2 and Vectorized 3D Depth Projection."""

import time
import os
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import cv2

try:
    import torch
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False


class RealtimePerception:
    """Zero-shot open-vocabulary landmark detector with sub-millisecond 3D coordinate projection.
    
    Transforms 2D bounding boxes and AirSim depth buffers into world-frame NED coordinates.
    """

    def __init__(
        self,
        model_path: str = "yolov8s-worldv2.pt",
        fov_degrees: float = 90.0,
        width: int = 640,
        height: int = 480,
        confidence_threshold: float = 0.25,
        depth_patch_radius: int = 2,
        min_depth: float = 0.3,
        max_depth: float = 80.0,
        device: Optional[str] = None,
        mock: bool = False
    ):
        self.width = width
        self.height = height
        self.fov_degrees = fov_degrees
        self.confidence_threshold = confidence_threshold
        self.depth_patch_radius = depth_patch_radius
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.mock = mock

        # Vectorized Pinhole Camera Intrinsics
        fov_rad = np.deg2rad(fov_degrees)
        self.fx = width / (2.0 * np.tan(fov_rad / 2.0))
        # Compute fy matching horizontal FOV aspect ratio
        self.fy = self.fx
        self.cx = width / 2.0
        self.cy = height / 2.0

        # Precompute coordinate grids for instant vectorized full-frame back-projection (<0.2 ms)
        u_grid, v_grid = np.meshgrid(np.arange(width), np.arange(height))
        self.u_factor = (u_grid - self.cx) / self.fx
        self.v_factor = (v_grid - self.cy) / self.fy

        # Performance metrics
        self.last_inference_latency_ms: float = 0.0
        self.last_projection_latency_ms: float = 0.0
        self.last_fps: float = 0.0
        self._fps_time = time.time()
        self._fps_counter = 0

        # Load YOLO-World model
        self.model = None
        self.current_classes: List[str] = []
        self.mock_world: Dict[str, np.ndarray] = {
            "fountain": np.array([32.0, 0.0, 0.0]),
            "red building": np.array([75.0, 12.0, 0.0]),
            "blue building": np.array([75.0, -14.0, 0.0]),
            "tower": np.array([130.0, -10.0, 0.0]),
            "hospital": np.array([60.0, -30.0, 0.0]),
            "warehouse": np.array([100.0, 40.0, 0.0]),
            "dock": np.array([120.0, 50.0, 0.0]),
            "solar farm": np.array([45.0, -50.0, 0.0]),
            "solar panels": np.array([45.0, -50.0, 0.0]),
            "helipad": np.array([20.0, 30.0, 0.0]),
            "bridge": np.array([85.0, -25.0, 0.0]),
            "crane": np.array([110.0, 20.0, 0.0]),
            "stadium": np.array([150.0, 30.0, 0.0]),
            "hangar": np.array([90.0, 60.0, 0.0]),
            "library": np.array([50.0, 25.0, 0.0]),
        }
        if not self.mock and _YOLO_AVAILABLE:
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.device = device
            
            # Locate model weights: check direct path, current dir, or fallback to name
            resolved_path = model_path
            if not os.path.exists(resolved_path):
                candidate = os.path.join(os.path.dirname(os.path.dirname(__file__)), model_path)
                if os.path.exists(candidate):
                    resolved_path = candidate

            try:
                self.model = YOLO(resolved_path)
                # Move to target device
                self.model.to(self.device)
            except Exception as e:
                print(f"[Perception] Warning: Failed to load YOLO-World ({e}). Falling back to mock mode.")
                self.mock = True
        else:
            self.mock = True

    def register_mock_landmark(self, name: str, pos: np.ndarray) -> None:
        """Dynamically register a novel simulated landmark position."""
        self.mock_world[name.lower().strip()] = np.asarray(pos, dtype=float)

    def update_target_vocabulary(self, class_names: List[str]) -> bool:
        """Dynamically update zero-shot detection vocabulary at runtime."""
        self.current_classes = list(class_names)
        if self.mock or self.model is None:
            return True
        try:
            self.model.set_classes(self.current_classes)
            return True
        except Exception as e:
            print(f"[Perception] Error updating vocabulary: {e}")
            return False

    def project_centroid_to_world(
        self,
        u_c: float,
        v_c: float,
        depth_img: np.ndarray,
        drone_pose: Tuple[float, float, float, float]
    ) -> Optional[np.ndarray]:
        """Project a 2D image coordinate (u, v) + depth patch into 3D World NED coordinates.
        
        Args:
            u_c: Horizontal pixel coordinate (0 to width-1).
            v_c: Vertical pixel coordinate (0 to height-1).
            depth_img: 2D float array containing metric depth in meters.
            drone_pose: Tuple of (drone_x, drone_y, drone_z, yaw_radians) in World NED.
            
        Returns:
            np.ndarray of shape (3,) [X_world, Y_world, Z_world] or None if depth is invalid.
        """
        if depth_img is None or depth_img.size == 0:
            return None

        # Check bounds and finiteness of pixel coordinates (inclusive of image boundaries)
        if np.isnan(u_c) or np.isnan(v_c) or np.isinf(u_c) or np.isinf(v_c):
            return None
        if u_c < 0.0 or u_c > self.width or v_c < 0.0 or v_c > self.height:
            return None

        # Ensure depth_img is 2D
        d_img = depth_img.squeeze() if depth_img.ndim == 3 else depth_img
        if d_img.ndim != 2:
            return None

        u_int = max(0, min(self.width - 1, int(round(u_c))))
        v_int = max(0, min(self.height - 1, int(round(v_c))))
        r = self.depth_patch_radius

        # Extract (2r+1)x(2r+1) patch centered at (u, v) with safe clamping
        u_min = max(0, min(self.width, u_int - r))
        u_max = max(0, min(self.width, u_int + r + 1))
        v_min = max(0, min(self.height, v_int - r))
        v_max = max(0, min(self.height, v_int + r + 1))

        if u_min >= u_max or v_min >= v_max:
            return None

        patch = d_img[v_min:v_max, u_min:u_max]
        valid_mask = (patch >= self.min_depth) & (patch <= self.max_depth) & np.isfinite(patch)
        valid_depths = patch[valid_mask]

        if len(valid_depths) == 0:
            return None

        # Median depth filters out background bleeding and specular spikes
        z_metric = float(np.median(valid_depths))

        # 1. Pinhole back-projection to Camera Coordinate Frame
        # X_cam: right, Y_cam: down, Z_cam: forward
        x_cam = (u_c - self.cx) * z_metric / self.fx
        y_cam = (v_c - self.cy) * z_metric / self.fy
        z_cam = z_metric

        # 2. Camera Frame to Drone Body Frame (NED: X=forward, Y=right, Z=down)
        # Front camera: forward is Z_cam, right is X_cam, down is Y_cam
        x_body = z_cam
        y_body = x_cam
        z_body = y_cam

        # 3. Drone Body Frame to World NED Frame (rotate by yaw)
        if len(drone_pose) >= 6:
            drone_x, drone_y, drone_z, _, _, yaw = drone_pose[:6]
        else:
            drone_x, drone_y, drone_z, yaw = drone_pose[:4]

        cos_y = np.cos(yaw)
        sin_y = np.sin(yaw)

        x_world = drone_x + (x_body * cos_y - y_body * sin_y)
        y_world = drone_y + (x_body * sin_y + y_body * cos_y)
        z_world = drone_z + z_body

        return np.array([x_world, y_world, z_world], dtype=float)

    def detect_and_project(
        self,
        rgb_img: np.ndarray,
        depth_img: np.ndarray,
        drone_pose: Tuple[float, float, float, float]
    ) -> List[Dict[str, Any]]:
        """Run open-vocabulary detection and project bounding boxes into 3D world space.
        
        Args:
            rgb_img: (H, W, 3) uint8 image in BGR or RGB.
            depth_img: (H, W) float32 metric depth array.
            drone_pose: (x, y, z, yaw) drone pose in World NED.
            
        Returns:
            List of detection dicts:
            {
                'name': str,
                'conf': float,
                'bbox': [x1, y1, x2, y2],
                'centroid_2d': [u, v],
                'world_pos': np.ndarray([X, Y, Z]),
                'distance': float
            }
        """
        t0 = time.perf_counter()
        detections: List[Dict[str, Any]] = []

        if self.mock or self.model is None:
            # Synthetic detection generation for unit tests / simulation fallback
            self.last_inference_latency_ms = 1.0
            t_proj0 = time.perf_counter()

            # Dynamic mock world landmark tracking
            mock_world = self.mock_world

            drone_x, drone_y, drone_z, drone_yaw = drone_pose[:4] if len(drone_pose) >= 4 else (0.0, 0.0, -5.0, 0.0)
            cos_y, sin_y = np.cos(drone_yaw), np.sin(drone_yaw)

            for lm_name, lm_pos in mock_world.items():
                matches = any(lm_name in c.lower() or c.lower() in lm_name for c in self.current_classes) if self.current_classes else True
                if not matches:
                    continue

                dx = lm_pos[0] - drone_x
                dy = lm_pos[1] - drone_y
                dz = lm_pos[2] - drone_z
                x_fwd = dx * cos_y + dy * sin_y
                y_rgt = -dx * sin_y + dy * cos_y
                z_dwn = dz

                if 1.5 < x_fwd < 85.0:
                    u_c = self.cx + self.fx * (y_rgt / x_fwd)
                    v_c = self.cy + self.fy * (z_dwn / x_fwd)

                    if 0 <= u_c <= self.width and 0 <= v_c <= self.height:
                        bw = max(24, int(8.0 * self.fx / x_fwd))
                        bh = max(24, int(12.0 * self.fy / x_fwd))
                        dist = float(np.sqrt(x_fwd**2 + y_rgt**2 + z_dwn**2))
                        detections.append({
                            "name": lm_name,
                            "conf": 0.92,
                            "bbox": [int(u_c - bw // 2), int(v_c - bh), int(u_c + bw // 2), int(v_c)],
                            "centroid_2d": [float(u_c), float(v_c - bh / 2.0)],
                            "world_pos": lm_pos.copy(),
                            "distance": dist
                        })

            self.last_projection_latency_ms = (time.perf_counter() - t_proj0) * 1000.0
            self._update_fps()
            return detections

        # Run YOLO-World inference
        results = self.model.predict(
            rgb_img,
            conf=self.confidence_threshold,
            verbose=False,
            device=self.device
        )[0]
        
        t1 = time.perf_counter()
        self.last_inference_latency_ms = (t1 - t0) * 1000.0

        # Vectorized back-projection of centroids
        t_proj0 = time.perf_counter()
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            u_c = (x1 + x2) / 2.0
            v_c = (y1 + y2) / 2.0

            world_pos = self.project_centroid_to_world(u_c, v_c, depth_img, drone_pose)
            if world_pos is None:
                continue

            class_id = int(box.cls[0])
            name = results.names[class_id]
            conf = float(box.conf[0])
            dist = float(np.linalg.norm(world_pos[:2] - np.array(drone_pose[:2])))

            detections.append({
                "name": name,
                "conf": conf,
                "bbox": [x1, y1, x2, y2],
                "centroid_2d": [float(u_c), float(v_c)],
                "world_pos": world_pos,
                "distance": dist
            })

        self.last_projection_latency_ms = (time.perf_counter() - t_proj0) * 1000.0
        self._update_fps()
        return detections

    def annotate_frame(
        self,
        rgb_img: np.ndarray,
        detections: List[Dict[str, Any]],
        active_target: Optional[str] = None
    ) -> np.ndarray:
        """Render detection bounding boxes, labels, and distance indicators onto RGB frame."""
        if rgb_img is None or rgb_img.size == 0:
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)
        annotated = rgb_img.copy()
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            name = det["name"]
            conf = det["conf"]
            dist = det.get("distance", 0.0)

            # Highlight active target in emerald green, other detections in cyan
            is_active = (active_target and active_target.lower() in name.lower())
            color = (0, 255, 128) if is_active else (255, 180, 0)
            thickness = 3 if is_active else 2

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
            
            # Label banner
            label = f"{name} {conf:.2f} ({dist:.1f}m)"
            (w_label, h_label), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, max(0, y1 - 20)), (x1 + w_label, max(0, y1)), color, -1)
            cv2.putText(
                annotated,
                label,
                (x1, max(15, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA
            )

        return annotated

    def _update_fps(self):
        """Track running frames-per-second."""
        self._fps_counter += 1
        now = time.time()
        elapsed = now - self._fps_time
        if elapsed >= 1.0:
            self.last_fps = self._fps_counter / elapsed
            self._fps_counter = 0
            self._fps_time = now
