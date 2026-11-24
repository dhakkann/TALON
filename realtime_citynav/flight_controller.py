"""Continuous 20-30 Hz Flight Controller with Proportional Guidance and APF Obstacle Avoidance."""

import time
import math
import threading
from typing import Dict, List, Optional, Tuple, Any
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

        # Orbit and circular trajectory state
        self.is_orbiting: bool = False
        self.orbit_center: Optional[np.ndarray] = None
        self.orbit_radius: float = 16.0
        self.orbit_speed: float = 3.0
        self.orbit_altitude: float = 10.0
        self.orbit_facing: str = "INWARD"  # "INWARD" or "TANGENT"
        self.orbit_revolutions: Optional[float] = None  # None = continuous until interrupted
        self.orbit_direction: int = 1  # +1 for CCW, -1 for CW
        self.orbit_target_name: Optional[str] = None
        self.orbit_accumulated_angle: float = 0.0
        self.orbit_last_angle: Optional[float] = None
        self.orbit_completed: bool = False

        # Waypoint continuous curve following state
        self.is_following_path: bool = False
        self.trajectory_path: Optional[List[np.ndarray]] = None
        self.trajectory_index: int = 0
        self.trajectory_speed: float = 3.0
        self.trajectory_facing: str = "TANGENT"
        self.trajectory_focal_point: Optional[np.ndarray] = None
        self.trajectory_target_name: Optional[str] = None
        
        # Kinematic simulator state for mock mode
        # NED convention: +X north, +Y east, +Z down
        self._sim_pos = np.array([0.0, 0.0, 0.0], dtype=float)
        self._sim_yaw = 0.0  # radians
        self._last_step_time = time.time()
        self._lock = threading.RLock()

        self._initialize_connection()

    def _initialize_connection(self) -> None:
        """Attempt AirSim RPC handshake; fall back to mock simulator if unreachable."""
        if not self.mock and _AIRSIM_AVAILABLE:
            max_retries = 3
            for attempt in range(1, max_retries + 1):
                try:
                    print(f"[FlightController] Connecting to AirSim at {self.ip}:{self.port} (attempt {attempt}/{max_retries})...")
                    self.client = airsim.MultirotorClient(ip=self.ip, port=self.port, timeout_value=5)
                    self.client.confirmConnection()
                    self.client.enableApiControl(True, vehicle_name=self.vehicle_name)
                    self.client.armDisarm(True, vehicle_name=self.vehicle_name)
                    self.is_armed = True
                    print(f"[FlightController] Successfully connected to AirSim at {self.ip}:{self.port} (vehicle: '{self.vehicle_name}')")
                    return
                except Exception as e:
                    if attempt < max_retries:
                        time.sleep(1.5)
                    else:
                        print(f"[FlightController] AirSim unreachable ({e}). Switching to mock kinematic simulator.")
                        print(f"[FlightController] TIP: Ensure Blocks.exe or AirSimNH.exe is running on {self.ip}:{self.port}")
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
            self.is_orbiting = False
            self.is_following_path = False
            self.orbit_center = None
            self.trajectory_path = None
            self.target_position = None
            self.last_avoidance_direction = None
            self.last_cmd_velocity = np.zeros(3, dtype=float)

    def set_altitude(self, altitude_m: float) -> None:
        """Set target cruise altitude in meters."""
        with self._lock:
            self.target_altitude = max(1.0, float(altitude_m))
            if self.is_orbiting:
                self.orbit_altitude = max(1.0, float(altitude_m))

    def set_target(self, world_xyz: Optional[np.ndarray]) -> None:
        """Set active 3D navigation target in World NED coordinates."""
        with self._lock:
            self.is_orbiting = False
            self.is_following_path = False
            if world_xyz is not None:
                self.target_position = np.asarray(world_xyz, dtype=float).flatten()
                self.is_hovering = False
            else:
                self.target_position = None

    def set_orbit_target(
        self,
        center_xyz: np.ndarray,
        radius: float = 16.0,
        altitude: float = 10.0,
        speed: float = 3.0,
        facing: str = "INWARD",
        revolutions: Optional[float] = None,
        direction: str = "CCW",
        target_name: Optional[str] = None
    ) -> None:
        """Configure continuous mathematical circular orbit around landmark center.
        
        Args:
            center_xyz: [x, y, z] center of target landmark in World NED.
            radius: Horizontal circle radius in meters.
            altitude: Flight altitude in meters (NED z = -altitude).
            speed: Horizontal orbit tangential speed in m/s.
            facing: 'INWARD' (drone yaw faces landmark center) or 'TANGENT' (faces forward along curve).
            revolutions: Number of complete 360-deg revolutions before hover, or None for continuous.
            direction: 'CCW' (counter-clockwise) or 'CW' (clockwise).
            target_name: Human-readable name of the landmark.
        """
        with self._lock:
            curr_x, curr_y, _, _ = self.get_pose()
            c = np.asarray(center_xyz, dtype=float).flatten()
            self.orbit_center = c
            self.orbit_radius = max(2.0, float(radius))
            self.orbit_altitude = max(1.0, float(altitude))
            self.orbit_speed = min(self.max_speed, max(0.5, float(speed)))
            self.cruise_speed = self.orbit_speed
            self.orbit_facing = "TANGENT" if facing.upper() in ("TANGENT", "FORWARD", "AHEAD") else "INWARD"
            self.orbit_revolutions = float(revolutions) if (revolutions is not None and float(revolutions) > 0) else None
            self.orbit_direction = -1 if direction.upper() in ("CW", "CLOCKWISE") else 1
            self.orbit_target_name = target_name
            self.orbit_accumulated_angle = 0.0
            self.orbit_last_angle = math.atan2(curr_y - c[1], curr_x - c[0])
            self.orbit_completed = False

            self.is_orbiting = True
            self.is_hovering = False
            self.is_following_path = False
            self.target_position = None
            self.trajectory_path = None

    def set_trajectory_path(
        self,
        waypoints: List[Any],
        speed: float = 3.0,
        facing: str = "TANGENT",
        focal_point: Optional[np.ndarray] = None,
        target_name: Optional[str] = None
    ) -> None:
        """Configure multi-waypoint continuous trajectory curve following.
        
        Args:
            waypoints: List of [x, y, z] coordinates.
            speed: Flight speed along curve in m/s.
            facing: 'TANGENT' (heading along trajectory tangent) or 'INWARD' (facing focal point).
            focal_point: Optional [x, y, z] center to point yaw toward if facing='INWARD'.
            target_name: Human-readable name of the trajectory.
        """
        with self._lock:
            pts = [np.asarray(wp, dtype=float).flatten() for wp in waypoints if len(wp) >= 3]
            if not pts:
                return
            self.trajectory_path = pts
            self.trajectory_index = 0
            self.trajectory_speed = min(self.max_speed, max(0.5, float(speed)))
            self.trajectory_facing = "INWARD" if facing.upper() == "INWARD" else "TANGENT"
            self.trajectory_focal_point = np.asarray(focal_point, dtype=float).flatten() if focal_point is not None else None
            self.trajectory_target_name = target_name

            self.is_following_path = True
            self.is_orbiting = False
            self.is_hovering = False
            self.target_position = None

    def get_orbit_telemetry(self) -> Dict[str, Any]:
        """Get live telemetry specific to active orbit or waypoint trajectory."""
        with self._lock:
            return {
                "is_orbiting": self.is_orbiting,
                "orbit_center": self.orbit_center.tolist() if self.orbit_center is not None else None,
                "orbit_radius": round(float(self.orbit_radius), 2) if self.is_orbiting else None,
                "orbit_target": self.orbit_target_name if self.is_orbiting else None,
                "orbit_facing": self.orbit_facing if self.is_orbiting else None,
                "revolution_count": round(self.orbit_accumulated_angle / (2.0 * math.pi), 2) if self.is_orbiting else 0.0,
                "is_following_path": self.is_following_path,
            }

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
                "avoidance_direction": None,
                "is_orbiting": False,
                "orbit_center": None,
                "orbit_radius": None,
                "orbit_target": None,
                "orbit_facing": None,
                "revolution_count": 0.0,
                "is_following_path": False
            }

        target = self.target_position
        dist_to_target = None
        has_arrived = False

        if self.orbit_completed:
            has_arrived = True
            v_des_xy = np.zeros(2, dtype=float)
            target_yaw = curr_yaw
            target_z = -self.orbit_altitude
            delta_z = target_z - curr_z
            v_des_z = float(np.clip(delta_z * self.kp, -1.5, 1.5))
            v_des = np.array([0.0, 0.0, v_des_z], dtype=float)
        elif self.is_hovering:
            # Explicit hover mode: hold current position and altitude
            v_des_xy = np.zeros(2, dtype=float)
            target_yaw = curr_yaw
            target_z = -self.target_altitude
            delta_z = target_z - curr_z
            v_des_z = float(np.clip(delta_z * self.kp, -1.5, 1.5))
            v_des = np.array([0.0, 0.0, v_des_z], dtype=float)
        elif self.is_orbiting and self.orbit_center is not None:
            xc = self.orbit_center[0]
            yc = self.orbit_center[1]
            dx = curr_pos[0] - xc
            dy = curr_pos[1] - yc
            r_curr = float(math.hypot(dx, dy))
            theta_curr = math.atan2(dy, dx)

            # Check if drone is in transit approach phase (outside orbit corridor)
            in_orbit_corridor = abs(r_curr - self.orbit_radius) <= 3.0

            if not in_orbit_corridor:
                # Transit mode: drive straight towards nearest point on orbit perimeter at cruise speed
                ur_x = dx / max(r_curr, 1e-4)
                ur_y = dy / max(r_curr, 1e-4)
                # If r_curr > orbit_radius, drive inward (-ur); if r_curr < orbit_radius, drive outward (+ur)
                approach_sign = -1.0 if r_curr > self.orbit_radius else 1.0
                approach_dir_x = approach_sign * ur_x
                approach_dir_y = approach_sign * ur_y
                v_des_xy = np.array([self.cruise_speed * approach_dir_x, self.cruise_speed * approach_dir_y], dtype=float)
                target_yaw = math.atan2(approach_dir_y, approach_dir_x)

                target_z = -self.orbit_altitude
                delta_z = target_z - curr_z
                v_des_z = float(np.clip(delta_z * self.kp, -self.max_speed, self.max_speed))
                v_des = np.array([v_des_xy[0], v_des_xy[1], v_des_z], dtype=float)

                self.orbit_last_angle = theta_curr
                dist_to_target = float(abs(r_curr - self.orbit_radius))
            else:
                # Active orbit flight: track angular progress algebraically to prevent jitter drift
                if self.orbit_last_angle is not None:
                    d_theta = (theta_curr - self.orbit_last_angle + math.pi) % (2.0 * math.pi) - math.pi
                    if abs(d_theta) < 1.0:
                        progress = self.orbit_direction * d_theta
                        self.orbit_accumulated_angle = max(0.0, self.orbit_accumulated_angle + progress)
                self.orbit_last_angle = theta_curr

                # Check revolution completion condition
                revs_done = self.orbit_accumulated_angle / (2.0 * math.pi)
                if self.orbit_revolutions is not None and revs_done >= self.orbit_revolutions:
                    self.is_orbiting = False
                    self.is_hovering = True
                    self.orbit_completed = True
                    has_arrived = True
                    v_des_xy = np.zeros(2, dtype=float)
                    target_yaw = curr_yaw
                    target_z = -self.orbit_altitude
                    delta_z = target_z - curr_z
                    v_des_z = float(np.clip(delta_z * self.kp, -1.5, 1.5))
                    v_des = np.array([0.0, 0.0, v_des_z], dtype=float)
                else:
                    # 1. Tangent unit vector: perpendicular to radial vector
                    # CCW (+1): [-sin(theta), cos(theta)]
                    # CW (-1):  [sin(theta), -cos(theta)]
                    dir_mult = float(self.orbit_direction)
                    t_x = -dir_mult * math.sin(theta_curr)
                    t_y = dir_mult * math.cos(theta_curr)

                    # 2. Radial error correction: drive drone towards target radius
                    # If r_curr > R, r_err < 0 -> steer inward (-radial)
                    # If r_curr < R, r_err > 0 -> steer outward (+radial)
                    r_err = self.orbit_radius - r_curr
                    ur_x = dx / max(r_curr, 1e-4)
                    ur_y = dy / max(r_curr, 1e-4)

                    # Radial velocity proportional to radius error
                    v_rad_speed = float(np.clip(r_err * self.kp, -self.max_speed, self.max_speed))
                    v_rad_x = v_rad_speed * ur_x
                    v_rad_y = v_rad_speed * ur_y

                    # Tangential velocity
                    v_tan_x = self.orbit_speed * t_x
                    v_tan_y = self.orbit_speed * t_y

                    v_des_xy = np.array([v_tan_x + v_rad_x, v_tan_y + v_rad_y], dtype=float)

                    # Altitude velocity toward orbit altitude
                    target_z = -self.orbit_altitude
                    delta_z = target_z - curr_z
                    v_des_z = float(np.clip(delta_z * self.kp, -self.max_speed, self.max_speed))

                    v_des = np.array([v_des_xy[0], v_des_xy[1], v_des_z], dtype=float)

                    # 3. Target Yaw Heading
                    if self.orbit_facing == "INWARD":
                        # Drone camera points directly at landmark center
                        target_yaw = math.atan2(-dy, -dx)
                    else:
                        # Drone camera points forward along velocity tangent
                        target_yaw = math.atan2(v_des_xy[1], v_des_xy[0])

                    dist_to_target = abs(r_curr - self.orbit_radius)
        elif self.is_following_path and self.trajectory_path:
            idx = self.trajectory_index
            if idx < len(self.trajectory_path):
                target_wp = self.trajectory_path[idx]
                delta = target_wp - curr_pos
                dist_xy = float(np.linalg.norm(delta[:2]))
                dist_3d = float(np.linalg.norm(delta))
                dist_to_target = dist_3d

                # Check waypoint arrival
                if dist_3d <= self.arrival_radius:
                    self.trajectory_index += 1
                    if self.trajectory_index >= len(self.trajectory_path):
                        self.is_following_path = False
                        self.is_hovering = True
                        has_arrived = True
                        v_des_xy = np.zeros(2, dtype=float)
                        target_yaw = curr_yaw
                        delta_z = target_wp[2] - curr_z
                        v_des_z = float(np.clip(delta_z * self.kp, -1.5, 1.5))
                        v_des = np.array([0.0, 0.0, v_des_z], dtype=float)
                    else:
                        target_wp = self.trajectory_path[self.trajectory_index]
                        delta = target_wp - curr_pos
                        dist_xy = float(np.linalg.norm(delta[:2]))

                if self.is_following_path:
                    dir_xy = delta[:2] / max(dist_xy, 1e-4)
                    speed = min(self.trajectory_speed, max(dist_xy * self.kp, 1.0))
                    v_des_xy = dir_xy * speed
                    delta_z = target_wp[2] - curr_z
                    v_des_z = float(np.clip(delta_z * self.kp, -self.max_speed, self.max_speed))
                    v_des = np.array([v_des_xy[0], v_des_xy[1], v_des_z], dtype=float)

                    if self.trajectory_facing == "INWARD" and self.trajectory_focal_point is not None:
                        f_dx = self.trajectory_focal_point[0] - curr_pos[0]
                        f_dy = self.trajectory_focal_point[1] - curr_pos[1]
                        target_yaw = math.atan2(f_dy, f_dx)
                    else:
                        target_yaw = math.atan2(v_des_xy[1], v_des_xy[0])
            else:
                self.is_following_path = False
                self.is_hovering = True
                has_arrived = True
                v_des = np.zeros(3, dtype=float)
                target_yaw = curr_yaw
        elif target is not None:
            delta = target - curr_pos
            # Maintain cruise altitude or target z
            delta_xy = delta[:2]
            dist_xy = float(np.linalg.norm(delta_xy))
            dist_to_target = float(np.linalg.norm(delta))

            # Vertical velocity toward target altitude
            target_z = target[2] if abs(target[2]) > 0.5 else -self.target_altitude
            delta_z = target_z - curr_z
            v_des_z = float(np.clip(delta_z * self.kp, -self.max_speed, self.max_speed))

            # 3D arrival check: must be within horizontal arrival radius AND vertical altitude margin
            z_arrival_margin = max(self.arrival_radius, 2.0)
            horiz_arrived = dist_xy <= self.arrival_radius
            vert_arrived = abs(delta_z) <= z_arrival_margin

            if not horiz_arrived:
                # Proportional guidance horizontal velocity
                dir_xy = delta_xy / max(dist_xy, 1e-4)
                speed = min(self.cruise_speed, dist_xy * self.kp)
                v_des_xy = dir_xy * speed
                # Desired yaw aligns toward target
                target_yaw = math.atan2(delta_xy[1], delta_xy[0])
            else:
                v_des_xy = np.zeros(2, dtype=float)
                target_yaw = curr_yaw

            has_arrived = horiz_arrived and vert_arrived

            v_des = np.array([v_des_xy[0], v_des_xy[1], v_des_z], dtype=float)
        else:
            # Forward search cruise when navigating without an acquired target lock;
            # Enforce geofence boundary limit to prevent endless runaway into deep space.
            dist_from_origin = float(np.linalg.norm(curr_pos[:2]))
            geofence_limit = 160.0
            if dist_from_origin > geofence_limit:
                # Outside perimeter: steer horizontal velocity back towards city center (x=70, y=0)
                center_xy = np.array([70.0, 0.0]) - curr_pos[:2]
                norm_center = float(np.linalg.norm(center_xy))
                if norm_center > 1e-3:
                    center_dir = center_xy / norm_center
                else:
                    center_dir = np.array([-math.cos(curr_yaw), -math.sin(curr_yaw)])
                v_des = np.array([
                    self.cruise_speed * center_dir[0],
                    self.cruise_speed * center_dir[1],
                    float(np.clip((-self.target_altitude - curr_z) * self.kp, -1.5, 1.5))
                ], dtype=float)
                target_yaw = math.atan2(center_dir[1], center_dir[0])
            else:
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
            with self._lock:
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
                    self._sim_yaw += math.copysign(min(abs(yaw_err), 3.0 * dt), yaw_err)

                    # Physical barrier: prevent entering building collision boxes
                    buildings = [
                        {"x": 75.0, "y": 12.0, "w": 12.0, "d": 12.0, "h": 18.0},
                        {"x": 75.0, "y": -14.0, "w": 12.0, "d": 12.0, "h": 18.0},
                        {"x": 130.0, "y": -10.0, "w": 8.0, "d": 8.0, "h": 32.0},
                        {"x": 60.0, "y": -30.0, "w": 12.0, "d": 12.0, "h": 8.0},
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
            "avoidance_direction": avoid_dir,
            "is_orbiting": self.is_orbiting,
            "orbit_center": self.orbit_center.tolist() if self.orbit_center is not None else None,
            "orbit_radius": round(float(self.orbit_radius), 2) if self.is_orbiting else None,
            "orbit_target": self.orbit_target_name if self.is_orbiting else None,
            "orbit_facing": self.orbit_facing if self.is_orbiting else None,
            "revolution_count": round(self.orbit_accumulated_angle / (2.0 * math.pi), 2) if self.is_orbiting else 0.0,
            "is_following_path": self.is_following_path
        }
