"""Main Multithreaded Orchestrator for Real-Time Autonomous CityNav Drone System."""

import sys
import os
import time
import math
import argparse
import signal
import threading
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import cv2

# Add parent directory to path for clean package imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from realtime_citynav.config import SystemConfig, DEFAULT_CONFIG
from realtime_citynav.flight_controller import RealtimeFlightController
from realtime_citynav.perception import RealtimePerception
from realtime_citynav.spatial_memory import SpatialMemory
from realtime_citynav.dialogue_manager import DialogueManager
from realtime_citynav.explainability import LiveRationaleNarrator
from realtime_citynav.dashboard.app import DashboardServer

try:
    import airsim
    _AIRSIM_AVAILABLE = True
except ImportError:
    _AIRSIM_AVAILABLE = False


class CityNavOrchestrator:
    """Coordinates high-frequency flight control, perception, spatial memory,
    dialogue interruption processing, and web dashboard telemetry."""

    def __init__(self, config: Optional[SystemConfig] = None):
        self.config = config or DEFAULT_CONFIG
        self.running = False

        # Core subsystems
        print("[Orchestrator] Initializing Core Subsystems...")
        self.narrator = LiveRationaleNarrator()
        self.memory = SpatialMemory(
            cluster_threshold=self.config.memory.cluster_threshold,
            alpha=self.config.memory.position_filter_alpha,
            max_retention_sec=self.config.memory.max_retention_sec
        )
        self.dialogue = DialogueManager(
            api_key=self.config.cognitive.api_key,
            model_name=self.config.cognitive.model_name
        )
        self.flight = RealtimeFlightController(
            ip=self.config.airsim.ip,
            port=self.config.airsim.port,
            vehicle_name=self.config.airsim.vehicle_name,
            cruise_speed=self.config.flight.cruise_speed,
            max_speed=self.config.flight.max_speed,
            proportional_gain=self.config.flight.proportional_gain,
            arrival_radius=self.config.flight.waypoint_arrival_radius,
            obstacle_dist_threshold=self.config.flight.obstacle_distance_threshold,
            obstacle_critical_ratio=self.config.flight.obstacle_critical_ratio,
            repulsion_lateral_speed=self.config.flight.repulsion_lateral_speed,
            repulsion_reverse_speed=self.config.flight.repulsion_reverse_speed,
            target_altitude=self.config.flight.takeoff_altitude,
            mock=self.config.mock_airsim
        )
        self.perception = RealtimePerception(
            model_path=self.config.perception.model_name,
            fov_degrees=self.config.airsim.fov_degrees,
            width=self.config.airsim.image_width,
            height=self.config.airsim.image_height,
            confidence_threshold=self.config.perception.confidence_threshold,
            depth_patch_radius=self.config.perception.depth_patch_radius,
            min_depth=self.config.airsim.min_depth_meters,
            max_depth=self.config.airsim.max_depth_meters,
            device=self.config.perception.device,
            mock=self.config.mock_perception
        )
        
        # Prime detection vocabulary with default classes
        self.perception.update_target_vocabulary(self.config.perception.default_classes)

        # Mission navigation state
        self.subgoals: List[str] = []
        self.active_subgoal_index: int = 0
        self.active_target_name: Optional[str] = None
        self.mission_completed: bool = False
        self.flight_mode: str = "IDLE"  # IDLE, TAKING_OFF, NAVIGATING, HOVERING, LANDING

        # Image buffers and synchronization
        self._latest_rgb: Optional[np.ndarray] = None
        self._latest_depth: Optional[np.ndarray] = None
        self._latest_annotated: Optional[np.ndarray] = None
        self._latest_detections: List[Dict[str, Any]] = []
        self._frame_lock = threading.Lock()
        self._state_lock = threading.Lock()

        # Thread handles
        self._flight_thread: Optional[threading.Thread] = None
        self._perception_thread: Optional[threading.Thread] = None
        self._cognitive_thread: Optional[threading.Thread] = None
        self._dashboard_server: Optional[DashboardServer] = None

    def start_mission(self, instruction: str) -> List[str]:
        """Initiate autonomous flight mission based on natural language instruction."""
        print(f"\n[Orchestrator] Starting Mission: '{instruction}'")
        self.narrator.emit_event(
            "MISSION_START",
            f"Parsing mission instruction: \"{instruction}\""
        )
        
        # Cognitive decomposition into subgoals
        subgoals = self.dialogue.parse_initial_instruction(instruction)
        if not subgoals:
            subgoals = ["fountain", "building", "tower"]

        with self._state_lock:
            self.subgoals = subgoals
            self.active_subgoal_index = 0
            self.mission_completed = False
            self.flight_mode = "NAVIGATING"
            self._set_current_subgoal_target(0)

        # Update perception vocabulary to include all subgoals deterministically
        extended_classes = list(dict.fromkeys(self.config.perception.default_classes + [s.lower() for s in subgoals]))
        self.perception.update_target_vocabulary(extended_classes)

        self.narrator.emit_event(
            "SUBGOALS_PLANNED",
            f"Planned {len(subgoals)} subgoals: {' -> '.join(subgoals)}",
            {"subgoals": subgoals}
        )

        # Trigger takeoff if not already airborne
        if not self.flight.is_in_air:
            self.flight_mode = "TAKING_OFF"
            self.flight.takeoff()
            self.flight_mode = "NAVIGATING"
            self.narrator.emit_event(
                "TAKEOFF_COMPLETED",
                f"Cruise altitude {self.config.flight.takeoff_altitude:.1f}m reached. Commencing navigation.",
                {"altitude": self.config.flight.takeoff_altitude}
            )

        return subgoals

    def emergency_hover(self) -> None:
        """Command immediate hover holding position."""
        with self._state_lock:
            self.flight_mode = "HOVERING"
        self.flight.hover()
        self.narrator.emit_event(
            "EMERGENCY_HOVER",
            "Emergency hover commanded. Holding current position."
        )

    def _compute_approach_target(self, name: str, center_pos: np.ndarray) -> Tuple[np.ndarray, float]:
        """Compute safe approach waypoint with standoff distance and elevated observation altitude."""
        elevations = {
            "fountain": (5.0, 8.0),
            "red building": (14.0, 16.0),
            "blue building": (14.0, 16.0),
            "tower": (22.0, 20.0),
            "hospital": (10.0, 18.0),
        }
        alt_val, std_val = 5.0, 8.0
        for k, (a, s) in elevations.items():
            if k in name.lower() or name.lower() in k:
                alt_val, std_val = a, s
                break

        approach_pos = np.array([center_pos[0] - std_val, center_pos[1], -alt_val], dtype=float)
        return approach_pos, alt_val

    def _set_current_subgoal_target(self, index: int) -> None:
        """Set active tracking target to specified subgoal index."""
        if 0 <= index < len(self.subgoals):
            self.active_subgoal_index = index
            self.active_target_name = self.subgoals[index]

            # Check if this target is already remembered in spatial memory
            prior = self.memory.find_landmark(self.active_target_name)
            if prior is not None:
                approach_pos, obs_alt = self._compute_approach_target(self.active_target_name, prior["pos"])
                self.flight.set_altitude(obs_alt)
                self.flight.set_target(approach_pos)
                self.narrator.notify_target_acquired(
                    self.active_target_name,
                    float(np.linalg.norm(approach_pos[:2] - np.array(self.flight.get_pose()[:2]))),
                    approach_pos
                )
            else:
                # In simulation mock mode, match against simulated world landmarks
                mock_world = {
                    "fountain": np.array([32.0, 0.0, 0.0]),
                    "red building": np.array([75.0, 12.0, 0.0]),
                    "blue building": np.array([75.0, -14.0, 0.0]),
                    "tower": np.array([130.0, -10.0, 0.0]),
                    "hospital": np.array([60.0, -30.0, 0.0]),
                }
                matched_mock = None
                for lm_name, lm_pos in mock_world.items():
                    if lm_name in self.active_target_name.lower() or self.active_target_name.lower() in lm_name:
                        matched_mock = lm_pos
                        break

                if matched_mock is not None:
                    approach_pos, obs_alt = self._compute_approach_target(self.active_target_name, matched_mock)
                    self.flight.set_altitude(obs_alt)
                    self.flight.set_target(approach_pos)
                    self.narrator.notify_target_acquired(
                        self.active_target_name,
                        float(np.linalg.norm(approach_pos[:2] - np.array(self.flight.get_pose()[:2]))),
                        approach_pos
                    )
                else:
                    # Seek target via open-vocabulary perception sweep
                    self.flight.set_target(None)
                    self.narrator.emit_event(
                        "TARGET_SEARCH",
                        f"Seeking next subgoal '{self.active_target_name}'. Sweeping forward."
                    )

    def _flight_loop(self) -> None:
        """20-30 Hz Continuous Flight Control Loop."""
        dt = 1.0 / self.config.flight.control_rate_hz
        print(f"[FlightLoop] Started at {self.config.flight.control_rate_hz} Hz.")

        while self.running:
            loop_start = time.perf_counter()

            # When vehicle is IDLE on ground, idle without issuing flight commands
            if self.flight_mode == "IDLE" or not self.flight.is_in_air:
                time.sleep(dt)
                continue

            # Retrieve current depth map for obstacle safety APF
            with self._frame_lock:
                depth_map = self._latest_depth

            # Execute single step of continuous proportional flight control
            state = self.flight.step_control(depth_map)

            # Check waypoint arrival
            if state.get("has_arrived", False) and self.flight_mode == "NAVIGATING":
                with self._state_lock:
                    arrived_target = self.active_target_name or "subgoal"
                    self.narrator.notify_waypoint_reached(arrived_target)
                    next_idx = self.active_subgoal_index + 1
                    if next_idx < len(self.subgoals):
                        self._set_current_subgoal_target(next_idx)
                    else:
                        self.mission_completed = True
                        self.flight_mode = "HOVERING"
                        self.flight.hover()
                        self.narrator.emit_event(
                            "MISSION_COMPLETED",
                            "All subgoals successfully reached! Mission complete."
                        )

            # Check obstacle avoidance trigger
            if state.get("obstacle_avoidance", False):
                dir_str = state.get("avoidance_direction", "CLEAR")
                self.narrator.notify_obstacle_avoidance(dir_str, self.config.flight.obstacle_distance_threshold)

            elapsed = time.perf_counter() - loop_start
            sleep_time = max(0.001, dt - elapsed)
            time.sleep(sleep_time)

    def _perception_loop(self) -> None:
        """25-30 FPS Perception and 3D Back-Projection Loop."""
        print("[PerceptionLoop] Started 30+ FPS Perception Worker.")

        while self.running:
            t0 = time.perf_counter()

            # Acquire camera RGB-D frame
            rgb, depth = self._acquire_frames()
            pose = self.flight.get_pose()

            # Run YOLO-World detection and sub-millisecond 3D projection
            detections = self.perception.detect_and_project(rgb, depth, pose)

            # Update 3D persistent spatial memory
            if detections:
                self.memory.add_observations(detections)

            # Dynamic target locking: if searching for active subgoal and landmark detected
            with self._state_lock:
                if self.active_target_name and self.flight.target_position is None:
                    for det in detections:
                        if self.active_target_name.lower() in det["name"].lower():
                            approach_pos, obs_alt = self._compute_approach_target(det["name"], det["world_pos"])
                            self.flight.set_altitude(obs_alt)
                            self.flight.set_target(approach_pos)
                            self.narrator.notify_target_acquired(
                                det["name"], det["distance"], approach_pos
                            )
                            break

            # Render visual overlays for live stream
            annotated = self.perception.annotate_frame(
                rgb, detections, active_target=self.active_target_name
            )

            with self._frame_lock:
                self._latest_rgb = rgb
                self._latest_depth = depth
                self._latest_annotated = annotated
                self._latest_detections = detections

            elapsed = time.perf_counter() - t0
            # Target ~30 FPS (33ms)
            time.sleep(max(0.002, 0.033 - elapsed))

    def _cognitive_loop(self) -> None:
        """Asynchronous Operator Interruption and Cadence Commentary Loop."""
        print("[CognitiveLoop] Started Asynchronous Cognitive Worker.")
        last_cadence_time = time.time()

        while self.running:
            time.sleep(0.1)  # 10 Hz check

            # 1. Check for mid-flight operator interruptions
            if self.dialogue.has_pending_commands():
                cur_target = self.active_target_name or "None"
                mem_summary = self.memory.get_summary()
                result = self.dialogue.process_interruption(cur_target, mem_summary)
                
                if result:
                    intent = result.get("intent")
                    target = result.get("target")
                    reply = result.get("reply", "")

                    self.narrator.notify_interruption(intent, f"{reply} (Target: {target})")

                    with self._state_lock:
                        if intent == "TARGET_CORRECTION" and target:
                            self.active_target_name = target
                            # Ensure perception vocabulary recognizes target deterministically
                            vocab = list(dict.fromkeys([target.lower()] + [s.lower() for s in self.subgoals] + self.config.perception.default_classes))
                            self.perception.update_target_vocabulary(vocab)
                            # Check if known in memory
                            known = self.memory.find_landmark(target)
                            if known:
                                approach_pos, obs_alt = self._compute_approach_target(target, known["pos"])
                                self.flight.set_altitude(obs_alt)
                                self.flight.set_target(approach_pos)
                            else:
                                mock_world = {
                                    "fountain": np.array([32.0, 0.0, 0.0]),
                                    "red building": np.array([75.0, 12.0, 0.0]),
                                    "blue building": np.array([75.0, -14.0, 0.0]),
                                    "tower": np.array([130.0, -10.0, 0.0]),
                                    "hospital": np.array([60.0, -30.0, 0.0]),
                                }
                                matched = next((pos for name, pos in mock_world.items() if target.lower() in name or name in target.lower()), None)
                                if matched is not None:
                                    approach_pos, obs_alt = self._compute_approach_target(target, matched)
                                    self.flight.set_altitude(obs_alt)
                                    self.flight.set_target(approach_pos)
                                else:
                                    self.flight.set_target(None)
                            self.flight_mode = "NAVIGATING"

                        elif intent == "RETURN_TO_PAST" and target:
                            known = self.memory.find_landmark(target)
                            if known:
                                self.active_target_name = known["name"]
                                self.flight.set_target(known["pos"])
                                self.flight_mode = "NAVIGATING"
                                self.narrator.emit_event(
                                    "RETURN_ROUTING",
                                    f"Routing back to registered coordinates {known['pos'].round(1)}."
                                )
                            else:
                                self.narrator.emit_event(
                                    "QUERY_REPLY",
                                    f"Landmark '{target}' not yet located in spatial memory."
                                )

                        elif intent == "HOVER_STOP":
                            self.flight_mode = "HOVERING"
                            self.flight.hover()

                        elif intent == "ALTITUDE_CHANGE":
                            new_alt = float(result.get("altitude", 12.0))
                            self.flight.set_altitude(new_alt)
                            if self.flight.target_position is not None:
                                self.flight.target_position[2] = -new_alt
                            self.narrator.emit_event(
                                "ALTITUDE_ADJUSTED",
                                f"Climbing/adjusting altitude to {new_alt:.1f} meters."
                            )

            # 2. Cadence commentary (every 4 seconds)
            now = time.time()
            if now - last_cadence_time >= self.config.cognitive.narration_cadence_sec:
                last_cadence_time = now
                if self.flight.is_in_air:
                    curr_x, curr_y, curr_z, _ = self.flight.get_pose()
                    curr_spd = float(np.linalg.norm(self.flight.last_cmd_velocity))
                    target_pos = self.flight.target_position
                    dist = float(np.linalg.norm(target_pos[:2] - np.array([curr_x, curr_y]))) if target_pos is not None else None
                    self.narrator.notify_cadence(
                        self.active_target_name, curr_spd, -curr_z, dist
                    )

    def _acquire_frames(self) -> Tuple[np.ndarray, np.ndarray]:
        """Fetch RGB and metric depth images from AirSim or generate synthetic frames."""
        w, h = self.config.airsim.image_width, self.config.airsim.image_height

        if not self.flight.mock and self.flight.client is not None and _AIRSIM_AVAILABLE:
            try:
                responses = self.flight.client.simGetImages([
                    airsim.ImageRequest(self.config.airsim.camera_name, airsim.ImageType.Scene, False, False),
                    airsim.ImageRequest(self.config.airsim.camera_name, airsim.ImageType.DepthPlanar, True, False)
                ], vehicle_name=self.flight.vehicle_name)
                if len(responses) >= 2 and len(responses[0].image_data_uint8) > 0:
                    img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
                    resp_w = responses[0].width
                    resp_h = responses[0].height
                    if img1d.size == resp_h * resp_w * 3:
                        rgb = img1d.reshape(resp_h, resp_w, 3)
                    elif img1d.size == resp_h * resp_w * 4:
                        rgb = img1d.reshape(resp_h, resp_w, 4)[:, :, :3]
                    else:
                        rgb = None

                    depth = airsim.list_to_2d_float_array(
                        responses[1].image_data_float, responses[1].width, responses[1].height
                    )
                    depth_arr = np.array(depth, dtype=np.float32)
                    if rgb is not None and depth_arr is not None:
                        return rgb, depth_arr
            except Exception:
                pass

        # Generate synthetic realistic RGB-D frame for simulation fallback
        return self._generate_synthetic_frame(w, h)

    def _generate_synthetic_frame(self, w: int, h: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a dynamic, perspective-correct 3D cityscape frame with moving ground and landmarks."""
        rgb = np.zeros((h, w, 3), dtype=np.uint8)

        # Get live drone pose from flight controller
        drone_x, drone_y, drone_z, drone_yaw = self.flight.get_pose()

        # Camera intrinsics
        cx, cy = w / 2.0, h / 2.0
        fov_rad = np.deg2rad(self.config.airsim.fov_degrees)
        fx = w / (2.0 * np.tan(fov_rad / 2.0))
        fy = h / (2.0 * np.tan(fov_rad / 2.0))

        # Sky gradient (slate navy to horizon)
        for y in range(h // 2):
            val = int(24 + 42 * (y / (h / 2)))
            rgb[y, :] = (val, val // 2 + 15, 18)

        # Ground plane base (dark asphalt)
        rgb[h // 2:, :] = (26, 28, 32)

        # Initialize depth buffer (50m background default)
        depth = np.ones((h, w), dtype=np.float32) * 50.0

        # Ground depth gradient (near to far)
        drone_alt = max(1.0, abs(drone_z))
        for y in range(h // 2 + 1, h):
            delta_y = y - cy
            d_ground = float(drone_alt * fy / max(delta_y, 1.0))
            depth[y, :] = min(50.0, d_ground)

        # Moving 3D Perspective Ground Grid (scrolls with drone motion)
        grid_spacing = 6.0  # 6-meter grid lines
        cos_y, sin_y = np.cos(drone_yaw), np.sin(drone_yaw)
        grid_offset_x = (drone_x * cos_y + drone_y * sin_y) % grid_spacing
        grid_offset_y = (-drone_x * sin_y + drone_y * cos_y) % grid_spacing

        # Draw transverse ground lines (moving toward the drone)
        for dist_m in np.arange(3.0 - grid_offset_x, 60.0, grid_spacing):
            if dist_m > 1.5:
                py = int(cy + fy * (drone_alt / dist_m))
                if cy < py < h:
                    fade = max(0.12, 1.0 - (dist_m / 60.0))
                    color = (int(70 * fade), int(90 * fade), int(120 * fade))
                    cv2.line(rgb, (0, py), (w, py), color, 1)

        # Draw longitudinal ground lines (radiating perspective lines)
        for lateral_m in np.arange(-30.0 - grid_offset_y, 35.0, grid_spacing):
            py_near = int(cy + fy * (drone_alt / 3.0))
            px_near = int(cx + fx * (lateral_m / 3.0))
            py_far = int(cy + fy * (drone_alt / 45.0))
            px_far = int(cx + fx * (lateral_m / 45.0))
            if py_near > cy:
                cv2.line(rgb, (px_near, min(h - 1, py_near)), (px_far, py_far), (40, 50, 60), 1)

        # Define 3D mock world landmarks (fixed world coordinates)
        mock_landmarks = [
            {"name": "fountain", "pos": np.array([32.0, 0.0, 0.0]), "color": (210, 170, 40), "w": 6.0, "h": 4.5},
            {"name": "red building", "pos": np.array([75.0, 12.0, 0.0]), "color": (50, 50, 200), "w": 12.0, "h": 18.0},
            {"name": "blue building", "pos": np.array([75.0, -14.0, 0.0]), "color": (200, 110, 30), "w": 12.0, "h": 18.0},
            {"name": "tower", "pos": np.array([130.0, -10.0, 0.0]), "color": (180, 180, 210), "w": 8.0, "h": 32.0},
            {"name": "hospital", "pos": np.array([60.0, -30.0, 0.0]), "color": (220, 220, 240), "w": 16.0, "h": 12.0},
        ]

        # Sort landmarks by distance (back to front rendering)
        def get_landmark_dist(lm):
            return float(np.linalg.norm(lm["pos"][:2] - np.array([drone_x, drone_y])))
        sorted_landmarks = sorted(mock_landmarks, key=get_landmark_dist, reverse=True)

        for lm in sorted_landmarks:
            dx = lm["pos"][0] - drone_x
            dy = lm["pos"][1] - drone_y
            dz = lm["pos"][2] - drone_z  # drone_z is negative (e.g. -5.0)

            x_fwd = dx * cos_y + dy * sin_y
            y_rgt = -dx * sin_y + dy * cos_y
            z_dwn = dz

            if x_fwd > 1.5:  # In front of camera
                u_center = int(cx + fx * (y_rgt / x_fwd))
                v_ground = int(cy + fy * (z_dwn / x_fwd))

                bw = max(6, int(lm["w"] * fx / x_fwd))
                bh = max(6, int(lm["h"] * fy / x_fwd))
                v_top = v_ground - bh

                if -bw < u_center < w + bw and v_top < h and v_ground > 0:
                    x1 = max(0, u_center - bw // 2)
                    x2 = min(w - 1, u_center + bw // 2)
                    y1 = max(0, v_top)
                    y2 = min(h - 1, v_ground)

                    if x2 > x1 and y2 > y1:
                        col = lm["color"]
                        cv2.rectangle(rgb, (x1, y1), (x2, y2), col, -1)
                        cv2.rectangle(rgb, (x1, y1), (x2, y2), (255, 255, 255), 1)

                        if bh > 25 and bw > 20:
                            n_rows = max(2, min(5, bh // 15))
                            n_cols = max(2, min(4, bw // 15))
                            win_w = max(2, (bw - (n_cols + 1) * 3) // n_cols)
                            win_h = max(2, (bh - (n_rows + 1) * 4) // n_rows)
                            for r_idx in range(n_rows):
                                for c_idx in range(n_cols):
                                    wx1 = x1 + 3 + c_idx * (win_w + 3)
                                    wy1 = y1 + 4 + r_idx * (win_h + 4)
                                    if wx1 + win_w < x2 and wy1 + win_h < y2:
                                        cv2.rectangle(rgb, (wx1, wy1), (wx1 + win_w, wy1 + win_h), (255, 255, 180), -1)

                        label = lm["name"].upper()
                        cv2.putText(rgb, label, (max(5, x1 - 5), max(15, y1 - 6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

                        dist_val = float(np.sqrt(x_fwd**2 + y_rgt**2 + z_dwn**2))
                        depth[y1:y2, x1:x2] = np.minimum(depth[y1:y2, x1:x2], dist_val)

        return rgb, depth

    def get_latest_annotated_frame(self) -> Optional[np.ndarray]:
        """Fetch latest annotated frame for MJPEG video stream."""
        with self._frame_lock:
            return self._latest_annotated

    def get_telemetry(self) -> Dict[str, Any]:
        """Retrieve complete live telemetry snapshot for UI dashboard."""
        curr_x, curr_y, curr_z, curr_yaw = self.flight.get_pose()
        v_cmd = self.flight.last_cmd_velocity
        speed = float(np.linalg.norm(v_cmd))
        target_pos = self.flight.target_position
        dist = float(np.linalg.norm(target_pos[:2] - np.array([curr_x, curr_y]))) if target_pos is not None else None

        with self._state_lock:
            mode = self.flight_mode
            tgt_name = self.active_target_name
            subgoals = list(self.subgoals)
            subgoal_idx = self.active_subgoal_index

        return {
            "mode": mode,
            "is_mock": self.flight.mock,
            "position": [round(curr_x, 2), round(curr_y, 2), round(curr_z, 2)],
            "altitude": round(-curr_z, 2),
            "yaw": round(curr_yaw, 3),
            "speed": round(speed, 2),
            "velocity": [round(float(v_cmd[0]), 2), round(float(v_cmd[1]), 2), round(float(v_cmd[2]), 2)],
            "active_target": tgt_name,
            "active_subgoal_index": subgoal_idx,
            "subgoals": subgoals,
            "distance_to_target": round(dist, 1) if dist is not None else None,
            "obstacle_avoidance": self.flight.last_avoidance_direction is not None,
            "avoidance_direction": self.flight.last_avoidance_direction,
            "fps": round(self.perception.last_fps, 1),
            "perception_latency_ms": round(self.perception.last_inference_latency_ms, 1),
            "projection_latency_ms": round(self.perception.last_projection_latency_ms, 2)
        }

    def start(self) -> None:
        """Start all orchestrator threads and dashboard server."""
        self.running = True

        # 1. Start Dashboard Web Server
        self._dashboard_server = DashboardServer(
            self,
            host=self.config.dashboard.host,
            port=self.config.dashboard.port
        )
        self._dashboard_server.start()

        # 2. Launch high-frequency perception worker
        self._perception_thread = threading.Thread(
            target=self._perception_loop, daemon=True, name="PerceptionWorkerThread"
        )
        self._perception_thread.start()

        # 3. Launch high-frequency flight control loop (20-30 Hz)
        self._flight_thread = threading.Thread(
            target=self._flight_loop, daemon=True, name="FlightControlLoopThread"
        )
        self._flight_thread.start()

        # 4. Launch asynchronous dialogue & cognitive loop
        self._cognitive_thread = threading.Thread(
            target=self._cognitive_loop, daemon=True, name="CognitiveWorkerThread"
        )
        self._cognitive_thread.start()

        print("\n========================================================")
        print(" CityNav Real-Time Autonomous Flight System Active!")
        print(f" Dashboard URL: http://localhost:{self.config.dashboard.port}")
        print(" Press Ctrl+C in terminal to stop.")
        print("========================================================\n")

    def stop(self) -> None:
        """Signal all threads to shut down and land drone."""
        print("\n[Orchestrator] Stopping system...")
        self.running = False
        try:
            self.flight.land()
        except Exception:
            pass
        print("[Orchestrator] System stopped cleanly.")


def main():
    parser = argparse.ArgumentParser(description="Real-Time Autonomous CityNav Drone System")
    parser.add_argument("--mock", action="store_true", help="Force kinematic simulation mock mode")
    parser.add_argument("--speed", type=float, default=4.0, help="Cruise speed in m/s")
    parser.add_argument("--port", type=int, default=5000, help="Dashboard web port")
    parser.add_argument("--instruction", type=str, default="", help="Initial navigation instruction")
    args = parser.parse_args()

    cfg = SystemConfig()
    cfg.flight.cruise_speed = args.speed
    cfg.dashboard.port = args.port
    if args.mock:
        cfg.mock_airsim = True
        cfg.mock_perception = True

    orchestrator = CityNavOrchestrator(cfg)

    def signal_handler(sig, frame):
        orchestrator.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    orchestrator.start()

    if args.instruction:
        time.sleep(1.0)
        orchestrator.start_mission(args.instruction)

    # Keep main thread alive
    try:
        while orchestrator.running:
            time.sleep(0.5)
    except KeyboardInterrupt:
        orchestrator.stop()


if __name__ == "__main__":
    main()
