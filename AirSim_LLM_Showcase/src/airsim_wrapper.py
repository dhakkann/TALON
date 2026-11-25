import math
import threading
import time

import airsim
import cv2
import numpy as np

objects_dict = {
    "turbine1": "BP_Wind_Turbines_C_1",
    "turbine2": "StaticMeshActor_2",
    "solarpanels": "StaticMeshActor_146",
    "crowd": "StaticMeshActor_6",
    "car": "StaticMeshActor_10",
    "tower1": "SM_Electric_trellis_179",
    "tower2": "SM_Electric_trellis_7",
    "tower3": "SM_Electric_trellis_8",
}

TARGET_ALIASES = {
    "turbine 1": "turbine1",
    "turbine 2": "turbine2",
    "solar panel": "solarpanels",
    "solar panels": "solarpanels",
    "solarpanel": "solarpanels",
    "people": "crowd",
    "person": "crowd",
    "vehicle": "car",
    "tower 1": "tower1",
    "tower 2": "tower2",
    "tower 3": "tower3",
}


def compute_front_distance(depth_img: np.ndarray, max_range: float = 100.0) -> float:
    """Compute front obstacle clearance (meters) from a 2D DepthPlanar array."""
    if depth_img is None or depth_img.size == 0:
        return max_range
    h, w = depth_img.shape[:2]
    if h < 4 or w < 4:
        roi = depth_img
    else:
        # Central forward corridor avoiding top sky/rotor and bottom ground skew
        roi = depth_img[h // 3 : (2 * h) // 3, w // 3 : (2 * w) // 3]
    valid = roi[np.isfinite(roi) & (roi > 0.25) & (roi < max_range * 2.0)]
    if valid.size == 0:
        return max_range
    dist = float(np.percentile(valid, 10))
    return min(dist, max_range)


def colorize_depth(depth_img: np.ndarray, max_vis_range: float = 40.0) -> np.ndarray:
    """Convert a 2D DepthPlanar float array to a BGR colormap image."""
    if depth_img is None or depth_img.size == 0:
        return np.zeros((240, 320, 3), dtype=np.uint8)
    clean = np.nan_to_num(depth_img, nan=max_vis_range, posinf=max_vis_range, neginf=0.0)
    clipped = np.clip(clean, 0.0, max_vis_range)
    norm = (255.0 * (1.0 - clipped / max_vis_range)).astype(np.uint8)
    return cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)


class AirSimWrapper:
    def __init__(self, ip: str = "127.0.0.1", port: int = 41451, connect: bool = True):
        self._lock = threading.RLock()
        self._abort_event = threading.Event()
        self.client = None
        self._step_callback = None
        self._step_dt = None
        self._latest_rgb_jpg = None
        self._latest_depth_jpg = None
        self._latest_telemetry = {
            "position": [0.0, 0.0, 0.0],
            "yaw": 0.0,
            "front_distance": 100.0,
            "status": "IDLE",
            "avoiding": False,
            "timestamp": time.time(),
        }
        if connect:
            self.client = airsim.MultirotorClient(ip=ip, port=port)
            self.client.confirmConnection()
            self.client.enableApiControl(True)
            self.client.armDisarm(True)
            self.refresh_snapshot()

    def set_step_recorder(self, callback=None, dt: float = 1.0 / 24.0):
        """Enable/disable deterministic stepped simulation capture during async waits."""
        self._step_callback = callback
        self._step_dt = float(dt) if callback is not None else None

    def _wait_future(self, future, max_wait_sec: float = 25.0):
        """Wait for an AirSim async future, stepping sim clock smoothly if recording."""
        if self._step_callback is None:
            future.join()
            return
        dt = self._step_dt or (1.0 / 24.0)
        max_steps = max(1, int(max_wait_sec / dt))
        for _ in range(max_steps):
            if getattr(future, "_set_flag", False) or self._abort_event.is_set():
                break
            self._step_callback()
            self.client.simContinueForTime(dt)
        if not getattr(future, "_set_flag", False) and not self._abort_event.is_set():
            self.client.simPause(False)
            future.join()
            self.client.simPause(True)

    def request_abort(self):
        self._abort_event.set()

    def clear_abort(self):
        self._abort_event.clear()

    @staticmethod
    def _to_ned_z(z_val: float) -> float:
        """Accept positive-up altitude (preferred) or legacy negative-up NED Z."""
        return -float(z_val) if float(z_val) > 0 else float(z_val)

    def takeoff(self):
        with self._lock:
            self._abort_event.clear()
            self._latest_telemetry["status"] = "TAKING OFF"
            self.client.enableApiControl(True)
            self.client.armDisarm(True)
            self._wait_future(self.client.takeoffAsync(timeout_sec=8.0), max_wait_sec=10.0)
            pos = self.get_drone_position()
            if pos[2] < 1.5:
                self._wait_future(
                    self.client.moveToPositionAsync(
                        pos[0], pos[1], -2.5, 3.0, timeout_sec=5.0
                    ),
                    max_wait_sec=7.0,
                )
            self._refresh_snapshot_unlocked()
            self._latest_telemetry["status"] = "HOVERING"

    def land(self):
        with self._lock:
            self._latest_telemetry["status"] = "LANDING"
            pos = self.get_drone_position()
            if pos[2] > 0.5:
                descend_timeout = max(6.0, pos[2] / 2.0 + 3.0)
                self._wait_future(
                    self.client.moveToPositionAsync(
                        pos[0], pos[1], -0.5, 4.0, timeout_sec=descend_timeout
                    ),
                    max_wait_sec=descend_timeout + 2.0,
                )
            self._wait_future(self.client.landAsync(timeout_sec=20.0), max_wait_sec=22.0)
            self._refresh_snapshot_unlocked()
            self._latest_telemetry["status"] = "LANDED"

    def get_drone_position(self):
        """Return drone [x, y, z] with Z positive-up in meters."""
        with self._lock:
            pose = self.client.simGetVehiclePose()
            return [
                float(pose.position.x_val),
                float(pose.position.y_val),
                float(-pose.position.z_val),
            ]

    def get_gps_data(self):
        with self._lock:
            gps_data = self.client.getGpsData().gnss.geo_point
            return [gps_data.altitude, gps_data.latitude, gps_data.longitude]

    def fly_to(self, point, speed: float = 8.0):
        """Fly directly to [x, y, z] (Z positive-up in meters)."""
        with self._lock:
            z_ned = self._to_ned_z(point[2])
            pose = self.client.simGetVehiclePose().position
            dist = math.sqrt(
                (float(point[0]) - pose.x_val) ** 2
                + (float(point[1]) - pose.y_val) ** 2
                + (z_ned - pose.z_val) ** 2
            )
            timeout = max(4.0, dist / max(float(speed) * 0.55, 1.0) + 2.5)
            # Cap lookahead for short steps so SimpleFlight does not skip targets <6m away
            lookahead = max(0.5, min(2.0, dist * 0.35))
            self._latest_telemetry["status"] = "FLYING"
            self._wait_future(
                self.client.moveToPositionAsync(
                    float(point[0]),
                    float(point[1]),
                    z_ned,
                    float(speed),
                    timeout_sec=timeout,
                    lookahead=lookahead,
                    adaptive_lookahead=0,
                ),
                max_wait_sec=timeout + 2.0,
            )
            self._refresh_snapshot_unlocked()
            self._latest_telemetry["status"] = "HOVERING"

    def fly_path(self, points, speed: float = 5.0):
        with self._lock:
            airsim_points = [
                airsim.Vector3r(float(p[0]), float(p[1]), self._to_ned_z(p[2]))
                for p in points
            ]
            self._latest_telemetry["status"] = "FLYING PATH"
            self._wait_future(
                self.client.moveOnPathAsync(
                    airsim_points,
                    float(speed),
                    120,
                    airsim.DrivetrainType.ForwardOnly,
                    airsim.YawMode(False, 0),
                    20,
                    1,
                ),
                max_wait_sec=120.0,
            )
            self._refresh_snapshot_unlocked()
            self._latest_telemetry["status"] = "HOVERING"

    def set_yaw(self, yaw: float):
        with self._lock:
            self._wait_future(self.client.rotateToYawAsync(float(yaw), 5), max_wait_sec=7.0)

    def get_yaw(self) -> float:
        """Return current yaw in degrees."""
        with self._lock:
            orientation_quat = self.client.simGetVehiclePose().orientation
            yaw_rad = airsim.to_eularian_angles(orientation_quat)[2]
            return float(math.degrees(yaw_rad))

    def get_position(self, object_name: str):
        """Return scene actor [x, y, z] with Z positive-up in meters."""
        key = object_name.strip().lower()
        key = TARGET_ALIASES.get(key, key)
        if key not in objects_dict:
            raise ValueError(
                f"Unknown target '{object_name}'. Valid targets: {list(objects_dict.keys())}"
            )
        query_string = objects_dict[key] + ".*"
        with self._lock:
            object_names_ue = []
            for _ in range(10):
                object_names_ue = self.client.simListSceneObjects(query_string)
                if object_names_ue:
                    break
                time.sleep(0.1)
            if not object_names_ue:
                raise RuntimeError(f"Scene actor not found for '{key}' ({query_string})")
            pose = self.client.simGetObjectPose(object_names_ue[0])
            return [
                float(pose.position.x_val),
                float(pose.position.y_val),
                float(-pose.position.z_val),
            ]

    def get_distance(self) -> float:
        """Return front obstacle clearance in meters using DepthPlanar."""
        with self._lock:
            responses = self.client.simGetImages(
                [airsim.ImageRequest("0", airsim.ImageType.DepthPlanar, True, False)]
            )
            if not responses or responses[0].height == 0 or responses[0].width == 0:
                return 100.0
            r = responses[0]
            depth_img = np.array(r.image_data_float, dtype=np.float32).reshape(
                r.height, r.width
            )
            return compute_front_distance(depth_img)

    def fly_safe(
        self,
        point,
        step_size: float = 10.0,
        min_clearance: float = 7.0,
        max_steps: int = 35,
        speed: float = 9.0,
        on_step=None,
    ) -> bool:
        """Step-by-step navigation with DepthPlanar obstacle detection and sidestep/climb."""
        tx, ty, tz_ned = float(point[0]), float(point[1]), self._to_ned_z(point[2])
        for _ in range(max_steps):
            if self._abort_event.is_set():
                return False
            px, py, pz_up = self.get_drone_position()
            pz_ned = -pz_up
            dx, dy, dz = tx - px, ty - py, tz_ned - pz_ned
            horiz_dist = math.hypot(dx, dy)
            total_dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            if total_dist <= 0.5:
                return True
            if horiz_dist > 1.5:
                heading_deg = math.degrees(math.atan2(dy, dx))
                yaw_err = abs((heading_deg - self.get_yaw() + 180.0) % 360.0 - 180.0)
                if yaw_err > 8.0:
                    self.set_yaw(heading_deg)
            clearance = self.get_distance()
            blocking = clearance < total_dist + 1.0 and (
                clearance < min_clearance or clearance < min_clearance + 3.0
            )
            if total_dist > 1.5 and blocking:
                with self._lock:
                    self._latest_telemetry["avoiding"] = True
                    self._latest_telemetry["status"] = "AVOIDING OBSTACLE"
                yaw_rad = math.radians(self.get_yaw())
                right_x, right_y = -math.sin(yaw_rad), math.cos(yaw_rad)
                if on_step is not None:
                    on_step(True, clearance)
                self.fly_to([px + right_x * 6.0, py + right_y * 6.0, pz_up + 3.0], speed=speed)
                continue
            with self._lock:
                self._latest_telemetry["avoiding"] = False
            if on_step is not None:
                on_step(False, clearance)
            max_safe_step = (
                step_size
                if clearance >= total_dist
                else max(3.0, clearance - min_clearance)
            )
            step = min(step_size, total_dist, max_safe_step)
            if step >= total_dist - 0.5:
                self.fly_to([tx, ty, -tz_ned], speed=speed)
                return True
            scale = step / total_dist
            self.fly_to([px + dx * scale, py + dy * scale, -(pz_ned + dz * scale)], speed=speed)
        return False

    def _refresh_snapshot_unlocked(self):
        """Fetch RGB + DepthPlanar + pose and cache JPEG previews (caller holds _lock)."""
        if self.client is None:
            return
        try:
            pose = self.client.simGetVehiclePose()
            yaw_rad = airsim.to_eularian_angles(pose.orientation)[2]
            responses = self.client.simGetImages(
                [
                    airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
                    airsim.ImageRequest("0", airsim.ImageType.DepthPlanar, True, False),
                ]
            )
            front_dist = 100.0
            if len(responses) >= 1 and responses[0].width > 0 and responses[0].height > 0:
                r0 = responses[0]
                rgb = np.frombuffer(r0.image_data_uint8, dtype=np.uint8).reshape(
                    r0.height, r0.width, 3
                )
                ok_rgb, buf_rgb = cv2.imencode(
                    ".jpg", rgb, [int(cv2.IMWRITE_JPEG_QUALITY), 80]
                )
                if ok_rgb:
                    self._latest_rgb_jpg = buf_rgb.tobytes()
            if len(responses) >= 2 and responses[1].width > 0 and responses[1].height > 0:
                r1 = responses[1]
                depth_img = np.array(r1.image_data_float, dtype=np.float32).reshape(
                    r1.height, r1.width
                )
                front_dist = compute_front_distance(depth_img)
                depth_bgr = colorize_depth(depth_img)
                ok_d, buf_d = cv2.imencode(
                    ".jpg", depth_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80]
                )
                if ok_d:
                    self._latest_depth_jpg = buf_d.tobytes()

            self._latest_telemetry.update(
                {
                    "position": [
                        round(float(pose.position.x_val), 2),
                        round(float(pose.position.y_val), 2),
                        round(float(-pose.position.z_val), 2),
                    ],
                    "yaw": round(float(math.degrees(yaw_rad)), 1),
                    "front_distance": round(float(front_dist), 2),
                    "timestamp": time.time(),
                }
            )
        except Exception:
            pass

    def refresh_snapshot(self, non_blocking: bool = False) -> dict:
        """Refresh FPV + telemetry if lock is available, else return cached telemetry."""
        acquired = self._lock.acquire(blocking=not non_blocking)
        if not acquired:
            return dict(self._latest_telemetry)
        try:
            self._refresh_snapshot_unlocked()
            return dict(self._latest_telemetry)
        finally:
            self._lock.release()

    def get_cached_frames(self):
        return self._latest_rgb_jpg, self._latest_depth_jpg, dict(self._latest_telemetry)
