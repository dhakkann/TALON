"""Continuous 20-30 Hz Flight Controller with Proportional Guidance and APF Obstacle Avoidance."""

import time
import math
import threading
from typing import Dict, Optional, Tuple, Any
import numpy as np

try:
    import airsim
    _AIRSIM_AVAILABLE = True
except ImportError:
    _AIRSIM_AVAILABLE = False


class RealtimeFlightController:
    """Continuous velocity flight controller for multirotor drones.
    
    Operates at 20-30 Hz using non-blocking velocity vectors.
    Includes proportional waypoint tracking, reactive depth-based APF obstacle avoidance,
    and a kinematic simulation fallback when AirSim is not actively running.
    """

    def __init__(
        self,
        ip: str = "127.0.0.1",
        port: int = 41451,
        vehicle_name: str = "",
        cruise_speed: float = 4.0,
        max_speed: float = 6.0,
        proportional_gain: float = 1.2,
        arrival_radius: float = 1.8,
        obstacle_dist_threshold: float = 4.0,
        obstacle_critical_ratio: float = 0.15,
        repulsion_lateral_speed: float = 2.0,
        repulsion_reverse_speed: float = 1.5,
        target_altitude: float = 5.0,
        mock: bool = False
    ):
        self.ip = ip
        self.port = port
        self.vehicle_name = vehicle_name
        self.cruise_speed = cruise_speed
        self.max_speed = max_speed
        self.kp = proportional_gain
        self.arrival_radius = arrival_radius
        self.obstacle_dist_thresh = obstacle_dist_threshold
        self.obstacle_critical_ratio = obstacle_critical_ratio
        self.repulsion_lateral = repulsion_lateral_speed
        self.repulsion_reverse = repulsion_reverse_speed
        self.target_altitude = target_altitude  # NED z = -target_altitude

        self.mock = mock
        self.client = None
        self.target_position: Optional[np.ndarray] = None
        self.is_armed = False
        self.is_in_air = False
        self.is_hovering = True
        self.last_cmd_velocity = np.zeros(3, dtype=float)
        self.last_repulsion_vector = np.zeros(3, dtype=float)
        self.last_avoidance_direction: Optional[str] = None
        
        # Kinematic simulator state for mock mode
        # NED convention: +X north, +Y east, +Z down
        self._sim_pos = np.array([0.0, 0.0, 0.0], dtype=float)
        self._sim_yaw = 0.0  # radians
        self._last_step_time = time.time()
        self._lock = threading.Lock()

        self._initialize_connection()

    def _initialize_connection(self) -> None:
        """Attempt AirSim RPC handshake; fall back to mock simulator if unreachable."""
        if not self.mock and _AIRSIM_AVAILABLE:
            try:
                self.client = airsim.MultirotorClient(ip=self.ip, port=self.port, timeout_value=3)
                self.client.confirmConnection()
                self.client.enableApiControl(True, vehicle_name=self.vehicle_name)
                self.client.armDisarm(True, vehicle_name=self.vehicle_name)
                self.is_armed = True
                print(f"[FlightController] Successfully connected to AirSim at {self.ip}:{self.port} (vehicle: '{self.vehicle_name}')")
                return
            except Exception as e:
                print(f"[FlightController] AirSim unreachable ({e}). Switching to mock kinematic simulator.")
                self.mock = True
                self.client = None
        else:
            self.mock = True

        # Initialize mock simulator
        self.is_armed = True
        self._sim_pos = np.array([0.0, 0.0, 0.0], dtype=float)
        self._sim_yaw = 0.0

    def takeoff(self, altitude: Optional[float] = None) -> bool:
        """Arm and take off to cruise altitude (NED coordinate: -altitude)."""
        alt = altitude or self.target_altitude
        with self._lock:
            if not self.mock and self.client is not None:
                try:
                    self.client.takeoffAsync(vehicle_name=self.vehicle_name).join()
                    self.client.moveToZAsync(-alt, 2.0, vehicle_name=self.vehicle_name).join()
                    self.is_in_air = True
                    self.is_hovering = False
                    return True
                except Exception as e:
                    print(f"[FlightController] Takeoff failed in AirSim: {e}")
                    return False
            else:
                self._sim_pos[2] = -alt
                self.is_in_air = True
                self.is_hovering = False
                return True

    def land(self) -> bool:
        """Land vehicle safely."""
        with self._lock:
            if not self.mock and self.client is not None:
                try:
                    self.client.landAsync(vehicle_name=self.vehicle_name).join()
                    self.is_in_air = False
                    self.is_hovering = True
                    return True
                except Exception as e:
                    print(f"[FlightController] Land failed in AirSim: {e}")
                    return False
            else:
                self._sim_pos[2] = 0.0
                self.is_in_air = False
                self.is_hovering = True
                return True

    def hover(self) -> None:
        """Command vehicle to hold current position and zero horizontal velocity."""
        with self._lock:
            self.is_hovering = True
            self.target_position = None
            self.last_avoidance_direction = None
            self.last_cmd_velocity = np.zeros(3, dtype=float)

    def set_altitude(self, altitude_m: float) -> None:
        """Set target cruise altitude in meters."""
        with self._lock:
            self.target_altitude = max(1.0, float(altitude_m))

    def set_target(self, world_xyz: Optional[np.ndarray]) -> None:
        """Set active 3D navigation target in World NED coordinates."""
        with self._lock:
            if world_xyz is not None:
                self.target_position = np.asarray(world_xyz, dtype=float).flatten()
                self.is_hovering = False
            else:
                self.target_position = None

    def get_pose(self) -> Tuple[float, float, float, float]:
        """Get current drone position and yaw in World NED.
        
        Returns:
            Tuple: (x, y, z, yaw_radians)
        """
        with self._lock:
            if not self.mock and self.client is not None:
                try:
                    kinematics = self.client.getMultirotorState(vehicle_name=self.vehicle_name).kinematics_estimated
                    pos = kinematics.position
                    ori = kinematics.orientation
                    # Convert quaternion to yaw
                    yaw = airsim.to_eularian_angles(ori)[2]
                    return float(pos.x_val), float(pos.y_val), float(pos.z_val), float(yaw)
                except Exception:
                    pass
            # Fallback to internal simulation pose
            return float(self._sim_pos[0]), float(self._sim_pos[1]), float(self._sim_pos[2]), float(self._sim_yaw)

    def compute_obstacle_repulsion(self, depth_img: np.ndarray, drone_yaw: float) -> Tuple[np.ndarray, Optional[str]]:
        """Calculate reactive safety repulsion vector from depth map.
        
        Analyzes the central field of view:
        If depth < obstacle_dist_threshold across sufficient pixels, generates a lateral
        and braking deflection vector.
        
        Returns:
            Tuple of (repulsion_world_xyz, avoidance_direction_str)
        """
        if depth_img is None or depth_img.size == 0:
            return np.zeros(3, dtype=float), None

        # Clean NaN/Inf in depth map
        depth_clean = np.nan_to_num(depth_img, nan=100.0, posinf=100.0, neginf=0.0)

        h, w = depth_clean.shape[:2]
        # Central Region of Interest
        r_ymin, r_ymax = h // 4, 3 * h // 4
        r_xmin, r_xmax = w // 4, 3 * w // 4
        roi = depth_clean[r_ymin:r_ymax, r_xmin:r_xmax]

        # Valid close obstacles
        close_mask = (roi > 0.2) & (roi < self.obstacle_dist_thresh)
        ratio = float(np.mean(close_mask))

        if ratio < self.obstacle_critical_ratio:
            return np.zeros(3, dtype=float), None

        # Compare obstacle density between left and right sectors
        roi_w = r_xmax - r_xmin
        left_sector = roi[:, :roi_w // 2]
        right_sector = roi[:, roi_w // 2:]

        left_density = float(np.mean((left_sector > 0.2) & (left_sector < self.obstacle_dist_thresh)))
        right_density = float(np.mean((right_sector > 0.2) & (right_sector < self.obstacle_dist_thresh)))

        # Body frame deflection: X_body=forward, Y_body=right
        # If obstacle is concentrated on left, steer right (+Y); otherwise steer left (-Y)
        if left_density >= right_density:
            rep_y_body = self.repulsion_lateral
            direction = "RIGHT"
        else:
            rep_y_body = -self.repulsion_lateral
            direction = "LEFT"

        rep_x_body = -self.repulsion_reverse * 2.0  # Strong braking / back off
        rep_z_body = -2.5  # Climb vertically (in NED, negative Z is upwards)

        # Rotate body-frame repulsion vector into World NED
        cos_y = math.cos(drone_yaw)
        sin_y = math.sin(drone_yaw)
        rep_x_world = rep_x_body * cos_y - rep_y_body * sin_y
        rep_y_world = rep_x_body * sin_y + rep_y_body * cos_y

        rep_vector = np.array([rep_x_world, rep_y_world, rep_z_body], dtype=float)
        return rep_vector, direction

    def step_control(self, depth_img: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Execute a single continuous flight control step (called at 20-30 Hz).
        
        Calculates proportional velocity toward target, adds reactive obstacle repulsion,
        and dispatches non-blocking velocity command to AirSim.
        
        Returns:
            Dict containing telemetry state, velocity command, and avoidance flags.
        """
        curr_x, curr_y, curr_z, curr_yaw = self.get_pose()
        curr_pos = np.array([curr_x, curr_y, curr_z], dtype=float)

        # If vehicle is not in air, remain stationary on ground
        if not self.is_in_air:
            self.last_cmd_velocity = np.zeros(3, dtype=float)
            return {
                "position": curr_pos.tolist(),
                "yaw": curr_yaw,
                "cmd_velocity": [0.0, 0.0, 0.0],
                "speed": 0.0,
                "distance_to_target": None,
                "has_arrived": False,
                "obstacle_avoidance": False,
                "avoidance_direction": None
            }

        target = self.target_position
        dist_to_target = None
        has_arrived = False

        if self.is_hovering:
            # Explicit hover mode: hold current position and altitude
            v_des_xy = np.zeros(2, dtype=float)
            target_yaw = curr_yaw
            target_z = -self.target_altitude
            delta_z = target_z - curr_z
            v_des_z = float(np.clip(delta_z * self.kp, -1.5, 1.5))
            v_des = np.array([0.0, 0.0, v_des_z], dtype=float)
        elif target is not None:
            delta = target - curr_pos
            # Maintain cruise altitude or target z
            delta_xy = delta[:2]
            dist_xy = float(np.linalg.norm(delta_xy))
            dist_to_target = float(np.linalg.norm(delta))

            if dist_xy > self.arrival_radius:
                # Proportional guidance horizontal velocity
                dir_xy = delta_xy / max(dist_xy, 1e-4)
                speed = min(self.cruise_speed, dist_xy * self.kp)
                v_des_xy = dir_xy * speed
                
                # Desired yaw aligns toward target
                target_yaw = math.atan2(delta_xy[1], delta_xy[0])
            else:
                v_des_xy = np.zeros(2, dtype=float)
                has_arrived = True
                target_yaw = curr_yaw

            # Vertical velocity toward target altitude
            target_z = target[2] if abs(target[2]) > 0.5 else -self.target_altitude
            delta_z = target_z - curr_z
            v_des_z = float(np.clip(delta_z * self.kp, -2.0, 2.0))

            v_des = np.array([v_des_xy[0], v_des_xy[1], v_des_z], dtype=float)
        else:
            # Forward search cruise when navigating without an acquired target lock
            v_des = np.array([
                self.cruise_speed * math.cos(curr_yaw),
                self.cruise_speed * math.sin(curr_yaw),
                float(np.clip((-self.target_altitude - curr_z) * self.kp, -1.5, 1.5))
            ], dtype=float)
            target_yaw = curr_yaw

        # Compute reactive depth APF repulsion
        v_rep, avoid_dir = self.compute_obstacle_repulsion(depth_img, curr_yaw)
        self.last_repulsion_vector = v_rep
        self.last_avoidance_direction = avoid_dir

        # If obstacle is detected ahead, actively brake forward velocity
        if avoid_dir is not None:
            v_des[0] *= 0.1
            v_des[1] *= 0.1

        # Fused velocity command
        v_cmd = v_des + v_rep

        # Enforce maximum horizontal speed constraint
        cmd_speed_xy = float(np.linalg.norm(v_cmd[:2]))
        if cmd_speed_xy > self.max_speed:
            v_cmd[:2] = (v_cmd[:2] / cmd_speed_xy) * self.max_speed

        self.last_cmd_velocity = v_cmd

        # Dispatch command
        now = time.time()
        dt = max(0.01, min(0.1, now - self._last_step_time))
        self._last_step_time = now

        if not self.mock and self.client is not None and _AIRSIM_AVAILABLE:
            try:
                # Non-blocking continuous velocity dispatch with active heading alignment
                yaw_mode = airsim.YawMode(is_rate=False, yaw_or_rate=float(math.degrees(target_yaw)))
                self.client.moveByVelocityAsync(
                    float(v_cmd[0]),
                    float(v_cmd[1]),
                    float(v_cmd[2]),
                    duration=0.1,
                    drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                    yaw_mode=yaw_mode,
                    vehicle_name=self.vehicle_name
                )
            except Exception as e:
                print(f"[FlightController] Error issuing moveByVelocityAsync: {e}")
        else:
            # Kinematic simulation integration step
            with self._lock:
                if self.is_in_air:
                    self._sim_pos += v_cmd * dt
                    # Smoothly turn yaw toward target heading
                    yaw_err = (target_yaw - self._sim_yaw + math.pi) % (2 * math.pi) - math.pi
                    self._sim_yaw += math.copysign(min(abs(yaw_err), 1.5 * dt), yaw_err)

                    # Physical barrier: prevent entering building collision boxes
                    buildings = [
                        {"x": 75.0, "y": 12.0, "w": 12.0, "d": 12.0, "h": 18.0},
                        {"x": 75.0, "y": -14.0, "w": 12.0, "d": 12.0, "h": 18.0},
                        {"x": 130.0, "y": -10.0, "w": 8.0, "d": 8.0, "h": 32.0},
                        {"x": 60.0, "y": -30.0, "w": 16.0, "d": 16.0, "h": 12.0},
                    ]
                    curr_alt = -self._sim_pos[2]
                    for b in buildings:
                        if curr_alt < b["h"]:
                            min_x, max_x = b["x"] - b["w"] / 2.0 - 0.5, b["x"] + b["w"] / 2.0 + 0.5
                            min_y, max_y = b["y"] - b["d"] / 2.0 - 0.5, b["y"] + b["d"] / 2.0 + 0.5
                            if min_x <= self._sim_pos[0] <= max_x and min_y <= self._sim_pos[1] <= max_y:
                                dists = [
                                    (abs(self._sim_pos[0] - min_x), "min_x"),
                                    (abs(self._sim_pos[0] - max_x), "max_x"),
                                    (abs(self._sim_pos[1] - min_y), "min_y"),
                                    (abs(self._sim_pos[1] - max_y), "max_y")
                                ]
                                dists.sort(key=lambda t: t[0])
                                side = dists[0][1]
                                if side == "min_x":
                                    self._sim_pos[0] = min_x - 0.1
                                elif side == "max_x":
                                    self._sim_pos[0] = max_x + 0.1
                                elif side == "min_y":
                                    self._sim_pos[1] = min_y - 0.1
                                else:
                                    self._sim_pos[1] = max_y + 0.1

        return {
            "position": curr_pos.tolist(),
            "yaw": curr_yaw,
            "cmd_velocity": v_cmd.tolist(),
            "speed": float(np.linalg.norm(v_cmd)),
            "distance_to_target": dist_to_target,
            "has_arrived": has_arrived,
            "obstacle_avoidance": avoid_dir is not None,
            "avoidance_direction": avoid_dir
        }
