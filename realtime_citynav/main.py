"""Main Multithreaded Orchestrator for Real-Time Autonomous CityNav Drone System."""

import sys
import os
import re
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

from realtime_citynav.config import SystemConfig, apply_environment_preset, DEFAULT_CONFIG
from realtime_citynav.flight_controller import RealtimeFlightController
from realtime_citynav.perception import RealtimePerception
from realtime_citynav.spatial_memory import SpatialMemory
from realtime_citynav.dialogue_manager import DialogueManager
from realtime_citynav.explainability import LiveRationaleNarrator
from realtime_citynav.dashboard.app import DashboardServer
from realtime_citynav.voice import VoiceTranscriber, VoiceListenerThread

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
        default_lms = list(self.config.env.default_landmarks.keys()) if self.config.env.default_landmarks else None
        self.dialogue = DialogueManager(
            api_key=self.config.cognitive.api_key if not self.config.mock_gemini else "",
            model_name=self.config.cognitive.model_name,
            mock=self.config.mock_gemini,
            environment=self.config.env.name,
            known_landmarks=default_lms,
            landmark_centers=self.config.env.default_landmarks
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
            engine=self.config.env.perception_engine,
            environment=self.config.env.name,
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
        
        # Prime detection vocabulary with environment default classes
        self.perception.update_target_vocabulary(self.config.env.default_classes)

        # Mission navigation state
        self.subgoals: List[str] = []
        self.active_subgoal_index: int = 0
        self.active_target_name: Optional[str] = None
        self.active_plan: Optional[Dict[str, Any]] = None
        self.mission_completed: bool = False
        self.flight_mode: str = "IDLE"  # IDLE, TAKING_OFF, NAVIGATING, HOVERING, LANDING

        # Image buffers and synchronization
        self._latest_rgb: Optional[np.ndarray] = None
        self._latest_depth: Optional[np.ndarray] = None
        self._latest_annotated: Optional[np.ndarray] = None
        self._latest_detections: List[Dict[str, Any]] = []
        self._frame_lock = threading.RLock()
        self._state_lock = threading.RLock()

        # Dedicated AirSim camera client (avoids Msgpack socket buffer collisions with flight controller)
        self._camera_client = None
        self._last_airsim_rgb: Optional[np.ndarray] = None
        self._last_airsim_depth: Optional[np.ndarray] = None
        self._last_airsim_seg: Optional[np.ndarray] = None
        if not self.config.mock_airsim and _AIRSIM_AVAILABLE:
            for attempt in range(1, 4):
                try:
                    self._camera_client = airsim.MultirotorClient(
                        ip=self.config.airsim.ip,
                        port=self.config.airsim.port,
                        timeout_value=5
                    )
                    self._camera_client.confirmConnection()
                    print(f"[Orchestrator] Dedicated AirSim camera client connected to {self.config.airsim.ip}:{self.config.airsim.port}")
                    if getattr(self.perception, "engine", "") == "segmentation":
                        from realtime_citynav.perception import setup_airsim_segmentation
                        setup_airsim_segmentation(self._camera_client)
                    break
                except Exception as e:
                    if attempt < 3:
                        time.sleep(1.0)
                    else:
                        print(f"[Orchestrator] AirSim camera client connection warning: {e}")
                        self._camera_client = None

        # Thread handles
        self._flight_thread: Optional[threading.Thread] = None
        self._perception_thread: Optional[threading.Thread] = None
        self._cognitive_thread: Optional[threading.Thread] = None
        self._dashboard_server: Optional[DashboardServer] = None

        # Voice subsystem
        self.voice_transcriber = VoiceTranscriber(
            api_key=self.config.cognitive.api_key,
            model_name=self.config.voice.model_name,
            mock=self.config.mock_gemini
        )
        self._voice_listener: Optional[VoiceListenerThread] = None

    def start_mission(self, instruction: str) -> List[str]:
        """Initiate autonomous flight mission based on natural language instruction."""
        print(f"\n[Orchestrator] Starting Mission: '{instruction}'")
        self.narrator.emit_event(
            "MISSION_START",
            f"Parsing mission instruction: \"{instruction}\""
        )
        
        # Synchronize spatial memory landmarks to dialogue manager
        if hasattr(self.dialogue, "update_known_landmarks"):
            mem_landmarks = [lm["name"] for lm in self.memory.get_all_landmarks()]
            if mem_landmarks:
                self.dialogue.update_known_landmarks(mem_landmarks)

        # Cognitive decomposition into subgoals & structured plan
        plan = None
        if hasattr(self.dialogue, "plan_mission"):
            try:
                plan = self.dialogue.plan_mission(instruction)
            except Exception as e:
                print(f"[Orchestrator] plan_mission error: {e}")

        if plan and plan.get("subgoals"):
            subgoals = plan["subgoals"]
        else:
            subgoals = self.dialogue.parse_initial_instruction(instruction)

        if not subgoals:
            subgoals = list(self.dialogue.known_landmarks) if self.dialogue.known_landmarks else ["landmark_1", "landmark_2"]

        with self._state_lock:
            self.active_plan = plan
            self.subgoals = subgoals
            self.active_subgoal_index = 0
            self.mission_completed = False
            self.flight_mode = "NAVIGATING"
            self._set_current_subgoal_target(0)

        # Update perception vocabulary to include base landmarks of all subgoals deterministically
        base_classes = [self.dialogue.extract_base_landmark(s).lower() for s in subgoals]
        extended_classes = list(dict.fromkeys(self.config.perception.default_classes + base_classes))
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
            if self.flight.is_orbiting:
                self.flight_mode = "ORBIT"
            elif self.flight.is_following_path:
                self.flight_mode = "CURVE"
            else:
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

    @property
    def command_queue(self):
        """Operator command queue in dialogue manager."""
        return self.dialogue.command_queue

    def queue_command(self, text: str) -> None:
        """Enqueue an operator message without blocking active flight control."""
        self.dialogue.queue_user_message(text)

    def process_pending_commands(self) -> Optional[Dict[str, Any]]:
        """Synchronously process pending operator commands in queue (if any)."""
        if self.dialogue.has_pending_commands():
            cur_target = self.active_target_name or "None"
            mem_summary = self.memory.get_summary()
            result = self.dialogue.process_interruption(cur_target, mem_summary)
            if result:
                intent = result.get("intent")
                if intent == "HOVER_STOP":
                    with self._state_lock:
                        self.flight_mode = "HOVERING"
                    self.flight.hover()
                return result
        return None

    def _compute_approach_target(
        self, name: str, center_pos: np.ndarray, spatial_override: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, float]:
        """Compute 3D target waypoint respecting spatial relations, prepositions, and elevations.
        
        Maps prepositions:
        - 'above' / 'over': drone climbs directly above landmark (adjusting Z in NED, no subtraction from X)
        - 'in front of': drone positions at standoff distance along approach vector (-X)
        - 'behind': drone positions behind landmark (+X)
        - 'to the left of': drone positions to the left (-Y in NED)
        - 'to the right of': drone positions to the right (+Y in NED)
        - default: approach with standard standoff distance and observation altitude
        """
        spatial = spatial_override if spatial_override else self.dialogue.parse_spatial_relation(name)
        relation = spatial.get("relation", "DEFAULT")
        dist_spec = spatial.get("distance")
        base_name = spatial.get("base_landmark") or name

        elevations = {
            "fountain": (5.0, 8.0),
            "red building": (14.0, 16.0),
            "blue building": (14.0, 16.0),
            "tower": (22.0, 20.0),
            "hospital": (14.0, 18.0),
            "warehouse": (12.0, 15.0),
            "dock": (6.0, 10.0),
            "crane": (25.0, 18.0),
            "stadium": (20.0, 25.0),
            "bridge": (10.0, 12.0),
            "hangar": (15.0, 20.0),
            "helipad": (5.0, 8.0),
            "solar farm": (4.0, 10.0),
            "solar panels": (4.0, 10.0),
            "library": (10.0, 12.0),
            # Blocks arena landmarks
            "orange block": (4.0, 6.0),
            "blue cylinder": (5.0, 6.0),
            "green cube": (4.0, 6.0),
            "red pillar": (6.0, 6.0),
            "yellow pyramid": (4.0, 6.0),
            "cylinder": (5.0, 6.0),
            "pillar": (6.0, 6.0),
            "cube": (4.0, 6.0),
            "block": (4.0, 6.0),
            # Neighborhood suburban landmarks
            "white car": (3.0, 5.0),
            "red car": (3.0, 5.0),
            "car": (3.0, 5.0),
            "truck": (3.5, 6.0),
            "stop sign": (3.0, 4.0),
            "tree": (5.0, 6.0),
            "house": (6.0, 8.0),
            "bench": (2.5, 4.0),
        }
        alt_val, std_val = 5.0, 8.0
        for k, (a, s) in elevations.items():
            if k in base_name.lower() or base_name.lower() in k:
                alt_val, std_val = a, s
                break

        # Parse spatial components: compound, vertical offset, and horizontal relation
        h_rel = spatial.get("horizontal_relation")
        h_dist = spatial.get("horizontal_distance")
        alt_offset = spatial.get("altitude_offset")

        # 1. Vertical Target Altitude (Z coordinate in NED, negative is upwards)
        # Landmark center is at center_pos[2]. If landmark is elevated (e.g. tower with Z = -22m),
        # 'Xm above landmark' means Z = center_pos[2] - dist, ensuring drone altitude is -Z.
        if relation in ("ABOVE", "OVER") or alt_offset is not None:
            dist = alt_offset if alt_offset is not None else (float(dist_spec) if dist_spec is not None else alt_val)
            target_z = float(center_pos[2] - dist)
            target_alt = -target_z
        else:
            # Horizontal approach: maintain safe clearance above landmark base/elevation
            target_z = float(min(center_pos[2] - 2.0, -alt_val)) if center_pos[2] < -2.0 else -alt_val
            target_alt = -target_z

        # 2. Horizontal Target Coordinates (X, Y)
        horiz_mode = h_rel if h_rel is not None else relation
        horiz_offset = float(h_dist) if h_dist is not None else (
            float(dist_spec) if (dist_spec is not None and horiz_mode not in ("ABOVE", "OVER")) else std_val
        )

        x_target = float(center_pos[0])
        y_target = float(center_pos[1])

        if spatial.get("dx") is not None or spatial.get("dy") is not None:
            x_target = center_pos[0] + float(spatial.get("dx", 0.0))
            y_target = center_pos[1] + float(spatial.get("dy", 0.0))
        elif horiz_mode == "IN_FRONT":
            x_target = center_pos[0] - horiz_offset
        elif horiz_mode == "BEHIND":
            x_target = center_pos[0] + horiz_offset
        elif horiz_mode == "LEFT":
            y_target = center_pos[1] - horiz_offset
        elif horiz_mode == "RIGHT":
            y_target = center_pos[1] + horiz_offset
        elif horiz_mode in ("ABOVE", "OVER") and h_rel is None:
            # Pure above/over: drone hovers directly over center of landmark
            x_target = center_pos[0]
            y_target = center_pos[1]
        else:
            # Default standoff approach along approach axis (-X)
            x_target = center_pos[0] - std_val

        approach_pos = np.array([x_target, y_target, target_z], dtype=float)
        return approach_pos, target_alt

    def resolve_landmark_position(self, name: str) -> Optional[np.ndarray]:
        """Resolve 3D world position of a landmark from spatial memory or simulation world map."""
        base_name = self.dialogue.extract_base_landmark(name)
        prior = self.memory.find_landmark(base_name) or self.memory.find_landmark(name)
        if prior is not None:
            return np.asarray(prior["pos"], dtype=float)

        # Check dialogue manager registered landmark centers
        for lm_name, lm_pos in getattr(self.dialogue, "landmark_centers", {}).items():
            if lm_name in base_name.lower() or base_name.lower() in lm_name:
                return lm_pos

        # Check active environment configured landmarks first
        for lm_name, lm_pos in getattr(self.config.env, "default_landmarks", {}).items():
            if lm_name in base_name.lower() or base_name.lower() in lm_name:
                return np.asarray(lm_pos, dtype=float)

        mock_world = {
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
        for lm_name, lm_pos in mock_world.items():
            if lm_name in base_name.lower() or base_name.lower() in lm_name:
                return lm_pos
        return None

    def _set_current_subgoal_target(self, index: int) -> None:
        """Set active tracking target to specified subgoal index."""
        if 0 <= index < len(self.subgoals):
            self.active_subgoal_index = index
            self.active_target_name = self.subgoals[index]
            base_name = self.dialogue.extract_base_landmark(self.active_target_name)

            # 1. Structured plan leg execution (from Gemini LLM planning)
            if self.active_plan and self.active_plan.get("legs") and 0 <= index < len(self.active_plan["legs"]):
                leg = self.active_plan["legs"][index]
                act = str(leg.get("action", "")).upper()
                wps = leg.get("waypoints")
                if wps or act in ("CURVE", "WAYPOINT_PATH", "FIGURE_8", "SPIRAL", "SEARCH_GRID"):
                    if not wps:
                        if "figure" in str(leg.get("target", "")).lower() or act == "FIGURE_8":
                            c1 = self.dialogue.landmark_centers.get("red building", np.array([75.0, 12.0, 0.0]))
                            c2 = self.dialogue.landmark_centers.get("blue building", np.array([75.0, -14.0, 0.0]))
                            wps = self.dialogue.generate_figure_8_waypoints(c1, c2, radius=12.0, altitude=float(leg.get("altitude") or 15.0))
                        elif "spiral" in str(leg.get("target", "")).lower() or act == "SPIRAL":
                            c = self.resolve_landmark_position(base_name)
                            if c is None:
                                c = np.array([130.0, -10.0, 0.0])
                            wps = self.dialogue.generate_spiral_waypoints(c, radius=16.0, start_altitude=10.0, end_altitude=40.0)
                    if wps:
                        spd = float(leg.get("speed") or self.flight.cruise_speed)
                        facing = leg.get("facing", "TANGENT")
                        self.flight_mode = "CURVE"
                        self.flight.set_trajectory_path(
                            waypoints=wps,
                            speed=spd,
                            facing=facing,
                            target_name=leg.get("target", "waypoint_curve")
                        )
                        self.narrator.emit_event(
                            "TRAJECTORY_FOLLOWING",
                            f"Executing continuous waypoint curve trajectory with {len(wps)} points.",
                            {"waypoints_count": len(wps)}
                        )
                        return

                if act == "ORBIT" or (not act and bool(leg.get("orbit"))):
                    orbit_info = leg.get("orbit") or {}
                    base_target = leg.get("base_landmark") or self.dialogue.extract_base_landmark(leg.get("target", base_name))
                    pos = self.resolve_landmark_position(base_target)
                    if pos is None:
                        pos = self.resolve_landmark_position(self.active_target_name)
                    if pos is not None:
                        self.flight_mode = "ORBIT"
                        radius = float(orbit_info.get("radius") or 16.0)
                        alt = float(orbit_info.get("altitude") or 10.0)
                        spd = float(orbit_info.get("speed") or self.flight.cruise_speed)
                        facing = orbit_info.get("facing") or leg.get("facing") or "INWARD"
                        revs = orbit_info.get("revolutions")
                        direction = orbit_info.get("direction", "CCW")
                        self.flight.set_orbit_target(
                            center_xyz=pos,
                            radius=radius,
                            altitude=alt,
                            speed=spd,
                            facing=facing,
                            revolutions=revs,
                            direction=direction,
                            target_name=leg.get("target", base_target)
                        )
                        self.narrator.emit_event(
                            "ORBIT_STARTED",
                            f"Orbiting {base_target} at radius {radius:.1f}m, altitude {alt:.1f}m ({facing} facing).",
                            orbit_info
                        )
                        return

                if act == "HOVER":
                    self.flight_mode = "HOVERING"
                    self.flight.hover()
                    return

                if act == "NAVIGATE":
                    base_target = leg.get("base_landmark") or self.dialogue.extract_base_landmark(leg.get("target", base_name))
                    pos = self.resolve_landmark_position(base_target)
                    if pos is None:
                        pos = self.resolve_landmark_position(self.active_target_name)
                    if pos is not None:
                        approach_pos, obs_alt = self._compute_approach_target(
                            leg.get("target", base_target), pos, spatial_override=leg.get("spatial")
                        )
                        if leg.get("altitude") is not None:
                            obs_alt = abs(float(leg["altitude"]))
                            approach_pos[2] = -obs_alt
                        self.flight.set_altitude(obs_alt)
                        self.flight.set_target(approach_pos)
                        self.flight_mode = "NAVIGATING"
                        self.narrator.notify_target_acquired(
                            self.active_target_name,
                            float(np.linalg.norm(approach_pos[:2] - np.array(self.flight.get_pose()[:2]))),
                            approach_pos
                        )
                        return

            # Check if this subgoal is an explicit 3D waypoint curve
            if self.dialogue.is_explicit_waypoint_path(self.active_target_name):
                wp_match = re.findall(r'\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]', self.active_target_name)
                if wp_match:
                    wps = [[float(x), float(y), float(z)] for x, y, z in wp_match]
                    facing = "INWARD" if "inward" in self.active_target_name.lower() else "TANGENT"
                    self.flight_mode = "CURVE"
                    self.flight.set_trajectory_path(
                        waypoints=wps,
                        speed=self.flight.cruise_speed,
                        facing=facing,
                        target_name="waypoint_curve"
                    )
                    self.narrator.emit_event(
                        "TRAJECTORY_FOLLOWING",
                        f"Executing continuous waypoint curve trajectory with {len(wps)} points.",
                        {"waypoints_count": len(wps)}
                    )
                    return

            # Check if this subgoal is an orbit/circling command
            if self.dialogue.is_orbit_command(self.active_target_name):
                orbit_info = self.dialogue.parse_orbit_command(self.active_target_name)
                if orbit_info:
                    base_target = orbit_info["base_landmark"]
                    pos = self.resolve_landmark_position(base_target)
                    if pos is not None:
                        self.flight_mode = "ORBIT"
                        orbit_speed = orbit_info["speed"] if re.search(r'\b(?:speed|m/s|mps)\b', self.active_target_name, re.I) else self.flight.cruise_speed
                        self.flight.set_orbit_target(
                            center_xyz=pos,
                            radius=orbit_info["radius"],
                            altitude=orbit_info["altitude"],
                            speed=orbit_speed,
                            facing=orbit_info["facing"],
                            revolutions=orbit_info["revolutions"],
                            direction=orbit_info["direction"],
                            target_name=orbit_info["target"]
                        )
                        self.narrator.emit_event(
                            "ORBIT_STARTED",
                            f"Orbiting {orbit_info['target']} at radius {orbit_info['radius']}m, altitude {orbit_info['altitude']}m ({orbit_info['facing']} facing).",
                            orbit_info
                        )
                        return

            # Check if this target is already remembered in spatial memory or simulation world
            target_pos = self.resolve_landmark_position(base_name)
            if target_pos is None:
                target_pos = self.resolve_landmark_position(self.active_target_name)
            if target_pos is not None:
                approach_pos, obs_alt = self._compute_approach_target(self.active_target_name, target_pos)
                self.flight.set_altitude(obs_alt)
                self.flight.set_target(approach_pos)
                self.flight_mode = "NAVIGATING"
                self.narrator.notify_target_acquired(
                    self.active_target_name,
                    float(np.linalg.norm(approach_pos[:2] - np.array(self.flight.get_pose()[:2]))),
                    approach_pos
                )
            else:
                # Seek target via open-vocabulary perception sweep
                self.flight_mode = "NAVIGATING"
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

            # Check waypoint arrival or orbit / curve completion
            if state.get("has_arrived", False) and self.flight_mode in ("NAVIGATING", "ORBIT", "CURVE"):
                with self._state_lock:
                    arrived_target = self.active_target_name or "subgoal"
                    if self.flight_mode == "ORBIT":
                        self.narrator.emit_event(
                            "ORBIT_COMPLETED",
                            f"Completed orbit revolutions around {arrived_target}. Holding stationary hover."
                        )
                    elif self.flight_mode == "CURVE":
                        self.narrator.emit_event(
                            "CURVE_COMPLETED",
                            "Completed continuous waypoint curve trajectory. Holding stationary hover."
                        )
                    else:
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

            # Run multi-engine detection (feed segmentation mask if engine is segmentation)
            det_input = self._last_airsim_seg if (getattr(self.perception, "engine", "") == "segmentation" and self._last_airsim_seg is not None) else rgb
            detections = self.perception.detect_and_project(det_input, depth, pose)

            # Update 3D persistent spatial memory
            if detections:
                self.memory.add_observations(detections)

            # Dynamic target locking: only seek and acquire targets when in active NAVIGATING mode
            with self._state_lock:
                if (
                    self.flight_mode == "NAVIGATING"
                    and not self.flight.is_orbiting
                    and not self.flight.is_following_path
                    and self.active_target_name
                    and self.flight.target_position is None
                ):
                    base_target = self.dialogue.extract_base_landmark(self.active_target_name)
                    for det in detections:
                        if base_target.lower() in det["name"].lower() or det["name"].lower() in base_target.lower():
                            approach_pos, obs_alt = self._compute_approach_target(self.active_target_name, det["world_pos"])
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
        last_maintenance_time = time.time()

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
                        if intent in ("MULTI_SUBGOAL_MISSION", "PATROL_ROUTE") or (result.get("subgoals") and len(result.get("subgoals")) > 1):
                            subgoals = result.get("subgoals")
                            instruction = result.get("instruction") or result.get("raw_message")
                            if subgoals and len(subgoals) > 1:
                                if instruction and hasattr(self.dialogue, "plan_mission"):
                                    try:
                                        self.active_plan = self.dialogue.plan_mission(instruction)
                                    except Exception:
                                        self.active_plan = None
                                else:
                                    self.active_plan = None
                                self.subgoals = list(subgoals)
                                self.active_subgoal_index = 0
                                self.mission_completed = False
                                self.flight_mode = "NAVIGATING"
                                self._set_current_subgoal_target(0)
                                # Update vocabulary for all subgoals
                                base_classes = [self.dialogue.extract_base_landmark(s).lower() for s in self.subgoals]
                                extended_classes = list(dict.fromkeys(self.config.perception.default_classes + base_classes))
                                self.perception.update_target_vocabulary(extended_classes)
                                self.narrator.emit_event(
                                    "SUBGOALS_PLANNED",
                                    f"Operator interruption updated {len(self.subgoals)} subgoals: {' -> '.join(self.subgoals)}",
                                    {"subgoals": self.subgoals}
                                )
                            elif instruction:
                                self.start_mission(instruction)
                            continue

                        # Ensure vehicle is airborne if operator sends a flight maneuver while grounded
                        if intent in ("TARGET_CORRECTION", "ORBIT_CIRCLE", "WAYPOINT_PATH", "RETURN_TO_PAST") and not self.flight.is_in_air:
                            takeoff_alt = float(result.get("altitude", self.config.flight.takeoff_altitude))
                            self.flight.takeoff(altitude=takeoff_alt)
                            self.narrator.emit_event(
                                "TAKEOFF_COMPLETED",
                                f"Autonomous takeoff initiated. Climbing to {takeoff_alt:.1f}m.",
                                {"altitude": takeoff_alt}
                            )

                        if intent == "TARGET_CORRECTION" and target:
                            self.active_target_name = target
                            base_target = result.get("base_landmark") or self.dialogue.extract_base_landmark(target)
                            # Ensure perception vocabulary recognizes target deterministically
                            vocab = list(dict.fromkeys(
                                [base_target.lower()] +
                                [self.dialogue.extract_base_landmark(s).lower() for s in self.subgoals] +
                                self.config.perception.default_classes
                            ))
                            self.perception.update_target_vocabulary(vocab)
                            # Check if known in memory
                            matched = self.resolve_landmark_position(base_target)
                            if matched is None:
                                matched = self.resolve_landmark_position(target)
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

                        elif intent == "ORBIT_CIRCLE":
                            self.active_target_name = target or "orbit"
                            base_target = result.get("base_landmark") or self.dialogue.extract_base_landmark(self.active_target_name)
                            radius = float(result.get("radius", 16.0))
                            alt = float(result.get("altitude", 10.0))
                            raw_msg = str(result.get("raw_message", "")).lower()
                            spd = float(result["speed"]) if (result.get("speed") and any(w in raw_msg for w in ["speed", "m/s", "mps"])) else self.flight.cruise_speed
                            facing = result.get("facing", "INWARD")
                            revs = result.get("revolutions")
                            direction = result.get("direction", "CCW")

                            # Ensure perception vocabulary recognizes target deterministically
                            vocab = list(dict.fromkeys(
                                [base_target.lower()] +
                                [self.dialogue.extract_base_landmark(s).lower() for s in self.subgoals] +
                                self.config.perception.default_classes
                            ))
                            self.perception.update_target_vocabulary(vocab)

                            pos = self.resolve_landmark_position(base_target)
                            if pos is None:
                                pos = self.resolve_landmark_position(self.active_target_name)
                            if pos is not None:
                                self.flight.set_orbit_target(
                                    center_xyz=pos,
                                    radius=radius,
                                    altitude=alt,
                                    speed=spd,
                                    facing=facing,
                                    revolutions=revs,
                                    direction=direction,
                                    target_name=self.active_target_name
                                )
                                self.flight_mode = "ORBIT"
                                self.narrator.emit_event(
                                    "ORBIT_STARTED",
                                    f"Operator commanded orbit: circling {self.active_target_name} at {radius:.1f}m radius, {alt:.1f}m altitude ({facing} facing).",
                                    result
                                )
                            else:
                                self.flight.set_target(None)
                                self.flight_mode = "NAVIGATING"
                                self.narrator.emit_event(
                                    "TARGET_SEARCH",
                                    f"Searching for '{base_target}' to initiate circular orbit..."
                                )

                        elif intent == "WAYPOINT_PATH":
                            wps = result.get("waypoints") or []
                            raw_msg = str(result.get("raw_message", "")).lower()
                            spd = float(result["speed"]) if (result.get("speed") and any(w in raw_msg for w in ["speed", "m/s", "mps"])) else self.flight.cruise_speed
                            facing = result.get("facing", "TANGENT")
                            if wps:
                                self.flight.set_trajectory_path(
                                    waypoints=wps,
                                    speed=spd,
                                    facing=facing,
                                    target_name=result.get("target", "waypoint_curve")
                                )
                                self.flight_mode = "CURVE"
                                self.active_target_name = result.get("target", "curve")
                                self.narrator.emit_event(
                                    "TRAJECTORY_FOLLOWING",
                                    f"Executing continuous waypoint curve with {len(wps)} 3D points at {spd:.1f} m/s.",
                                    {"waypoints_count": len(wps)}
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

            # 2. Periodic spatial memory maintenance & deduplication (every 3 seconds)
            now = time.time()
            if now - last_maintenance_time >= 3.0:
                last_maintenance_time = now
                self.memory.clean_registry(dedup=True, max_stale_sec=self.config.memory.max_retention_sec)
                self.memory.prune_unconfirmed(min_count=2, max_age_sec=15.0)

            # 3. Cadence commentary (every 4 seconds)
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

    def clean_spatial_memory(self) -> Dict[str, int]:
        """Perform on-demand deduplication and maintenance of spatial memory."""
        return self.memory.clean_registry(
            dedup=True,
            max_stale_sec=self.config.memory.max_retention_sec,
            prune_unconfirmed_older_than=30.0,
            min_confirmed_count=2
        )

    def _acquire_frames(self) -> Tuple[np.ndarray, np.ndarray]:
        """Fetch RGB and metric depth images from AirSim or generate synthetic frames."""
        w, h = self.config.airsim.image_width, self.config.airsim.image_height

        cam_client = self._camera_client or (self.flight.client if not self.flight.mock else None)
        if cam_client is not None and _AIRSIM_AVAILABLE and not self.config.mock_airsim:
            try:
                cam_name = self.config.airsim.camera_name
                requests = [
                    airsim.ImageRequest(cam_name, airsim.ImageType.Scene, False, False),
                    airsim.ImageRequest(cam_name, airsim.ImageType.DepthPlanar, True, False)
                ]
                if getattr(self.perception, "engine", "") == "segmentation":
                    requests.append(airsim.ImageRequest(cam_name, airsim.ImageType.Segmentation, False, False))

                responses = cam_client.simGetImages(requests, vehicle_name=self.flight.vehicle_name)
                if (not responses or len(responses) < 2 or len(responses[0].image_data_uint8) == 0) and cam_name != "0":
                    # Fallback to default camera index "0"
                    fallback_requests = [
                        airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
                        airsim.ImageRequest("0", airsim.ImageType.DepthPlanar, True, False)
                    ]
                    if getattr(self.perception, "engine", "") == "segmentation":
                        fallback_requests.append(airsim.ImageRequest("0", airsim.ImageType.Segmentation, False, False))
                    responses = cam_client.simGetImages(fallback_requests, vehicle_name=self.flight.vehicle_name)
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

                    if len(responses) >= 3 and len(responses[2].image_data_uint8) > 0:
                        seg1d = np.frombuffer(responses[2].image_data_uint8, dtype=np.uint8)
                        if seg1d.size == resp_h * resp_w * 3:
                            seg = seg1d.reshape(resp_h, resp_w, 3)
                        elif seg1d.size == resp_h * resp_w * 4:
                            seg = seg1d.reshape(resp_h, resp_w, 4)[:, :, :3]
                        else:
                            seg = None
                        if seg is not None and (resp_w != w or resp_h != h):
                            seg = cv2.resize(seg, (w, h), interpolation=cv2.INTER_NEAREST)
                        self._last_airsim_seg = seg

                    if rgb is not None and depth_arr is not None:
                        # Resize to standard config resolution if AirSim returned different aspect/size
                        if resp_w != w or resp_h != h:
                            rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
                            depth_arr = cv2.resize(depth_arr, (w, h), interpolation=cv2.INTER_NEAREST)
                        self._last_airsim_rgb = rgb
                        self._last_airsim_depth = depth_arr
                        return rgb, depth_arr
            except Exception as e:
                # If network frame dropped momentarily, reuse last frame instead of falling back to synthetic
                if self._last_airsim_rgb is not None:
                    return self._last_airsim_rgb, self._last_airsim_depth

        if self._last_airsim_rgb is not None and not self.config.mock_airsim:
            return self._last_airsim_rgb, self._last_airsim_depth

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

        # Define 3D mock world landmarks (dynamically selected per environment)
        env_name = getattr(self.config.env, "name", "city").lower() if hasattr(self.config, "env") else "city"
        if env_name in ("blocks", "block"):
            mock_landmarks = [
                {"name": "orange block", "pos": np.array([25.0, 5.0, 0.0]), "color": (30, 130, 245), "w": 6.0, "h": 6.0},
                {"name": "blue cylinder", "pos": np.array([45.0, -10.0, 0.0]), "color": (230, 100, 30), "w": 6.0, "h": 10.0},
                {"name": "red pillar", "pos": np.array([60.0, 15.0, 0.0]), "color": (50, 50, 220), "w": 4.0, "h": 14.0},
                {"name": "green cube", "pos": np.array([35.0, -20.0, 0.0]), "color": (50, 200, 50), "w": 8.0, "h": 8.0},
                {"name": "yellow pyramid", "pos": np.array([50.0, 0.0, 0.0]), "color": (30, 210, 230), "w": 7.0, "h": 9.0},
            ]
        elif env_name in ("neighborhood", "nh", "suburb"):
            mock_landmarks = [
                {"name": "white car", "pos": np.array([25.0, 8.0, 0.0]), "color": (230, 230, 230), "w": 4.5, "h": 2.0},
                {"name": "stop sign", "pos": np.array([40.0, 0.0, 0.0]), "color": (50, 50, 220), "w": 1.2, "h": 3.0},
                {"name": "red car", "pos": np.array([55.0, -12.0, 0.0]), "color": (30, 30, 210), "w": 4.5, "h": 2.0},
                {"name": "house", "pos": np.array([50.0, 15.0, 0.0]), "color": (60, 130, 210), "w": 14.0, "h": 8.0},
                {"name": "tree", "pos": np.array([30.0, -20.0, 0.0]), "color": (40, 160, 40), "w": 5.0, "h": 7.0},
                {"name": "bench", "pos": np.array([20.0, -5.0, 0.0]), "color": (40, 90, 140), "w": 2.5, "h": 1.2},
            ]
        else:
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

        # If orbiting, override distance with radial clearance to target circle
        if self.flight.is_orbiting and self.flight.orbit_center is not None:
            dx = curr_x - self.flight.orbit_center[0]
            dy = curr_y - self.flight.orbit_center[1]
            dist = abs(math.hypot(dx, dy) - self.flight.orbit_radius)
            mode = "ORBIT"
        elif self.flight.is_following_path:
            mode = "CURVE"

        orbit_telem = self.flight.get_orbit_telemetry()

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
            "projection_latency_ms": round(self.perception.last_projection_latency_ms, 2),
            "is_orbiting": orbit_telem["is_orbiting"],
            "orbit_center": orbit_telem["orbit_center"],
            "orbit_radius": orbit_telem["orbit_radius"],
            "orbit_target": orbit_telem["orbit_target"],
            "orbit_facing": orbit_telem["orbit_facing"],
            "revolution_count": orbit_telem["revolution_count"],
            "is_following_path": orbit_telem["is_following_path"],
            "environment": getattr(self.config.env, "name", "city") if hasattr(self.config, "env") else "city",
            "perception_engine": getattr(self.perception, "engine", "auto"),
        }

    def start(self, start_dashboard: Optional[bool] = None, start_voice_mic: bool = False) -> None:
        """Start all orchestrator threads, dashboard server, and optional voice mic listener."""
        self.running = True

        if start_dashboard is None:
            start_dashboard = self.config.dashboard.enabled

        # 1. Start Dashboard Web Server if enabled
        if start_dashboard:
            self._dashboard_server = DashboardServer(
                self,
                host=self.config.dashboard.host,
                port=self.config.dashboard.port
            )
            self._dashboard_server.start()

        # 2. Launch optional background local microphone listener
        if start_voice_mic:
            self._voice_listener = VoiceListenerThread(
                command_callback=self.queue_command,
                transcriber=self.voice_transcriber,
                sample_rate=self.config.voice.mic_sample_rate,
                energy_threshold=self.config.voice.vad_energy_threshold,
                device_index=self.config.voice.mic_device_index
            )
            self._voice_listener.start()

        # 3. Launch high-frequency perception worker
        self._perception_thread = threading.Thread(
            target=self._perception_loop, daemon=True, name="PerceptionWorkerThread"
        )
        self._perception_thread.start()

        # 4. Launch high-frequency flight control loop (20-30 Hz)
        self._flight_thread = threading.Thread(
            target=self._flight_loop, daemon=True, name="FlightControlLoopThread"
        )
        self._flight_thread.start()

        # 5. Launch asynchronous dialogue & cognitive loop
        self._cognitive_thread = threading.Thread(
            target=self._cognitive_loop, daemon=True, name="CognitiveWorkerThread"
        )
        self._cognitive_thread.start()

        if start_dashboard:
            print("\n========================================================")
            print(" CityNav Real-Time Autonomous Flight System Active!")
            print(f" Dashboard URL: http://localhost:{self.config.dashboard.port}")
            if start_voice_mic:
                print(" Local Voice Microphone Listener: ACTIVE")
            print(" Press Ctrl+C in terminal to stop.")
            print("========================================================\n")

    def stop(self) -> None:
        """Signal all threads to shut down and land drone."""
        print("\n[Orchestrator] Stopping system...")
        self.running = False
        if self._voice_listener is not None:
            try:
                self._voice_listener.stop()
            except Exception:
                pass
            self._voice_listener = None
        try:
            self.flight.land()
        except Exception:
            pass
        print("[Orchestrator] System stopped cleanly.")


def main():
    parser = argparse.ArgumentParser(description="Real-Time Autonomous CityNav Drone System")
    parser.add_argument("--airsim-ip", "--ip", dest="airsim_ip", type=str, default="", help="AirSim host IP address")
    parser.add_argument("--airsim-port", dest="airsim_port", type=int, default=0, help="AirSim RPC port")
    parser.add_argument("--env", "--environment", dest="environment", type=str, default="city", choices=["city", "neighborhood", "nh", "blocks", "geometric"], help="Simulation environment preset (city, neighborhood/nh, blocks)")
    parser.add_argument("--engine", "--perception", dest="perception_engine", type=str, default="auto", choices=["auto", "coco", "color", "segmentation", "yolo_world"], help="Perception engine backend")
    parser.add_argument("--vehicle-name", dest="vehicle_name", type=str, default="", help="AirSim vehicle name")
    parser.add_argument("--mock", action="store_true", help="Force kinematic simulation mock mode")
    parser.add_argument("--speed", type=float, default=4.0, help="Cruise speed in m/s")
    parser.add_argument("--port", type=int, default=5000, help="Dashboard web port")
    parser.add_argument("--instruction", type=str, default="", help="Initial navigation instruction")
    parser.add_argument("--takeoff", "--auto-takeoff", dest="auto_takeoff", action="store_true", help="Automatically arm and take off on startup")
    parser.add_argument("--api-key", "--gemini-key", type=str, default="", help="Gemini 2.0 Flash API Key")
    parser.add_argument("--voice-mic", action="store_true", help="Enable background local microphone listener for hands-free voice control")
    args = parser.parse_args()

    cfg = SystemConfig()
    cfg = apply_environment_preset(cfg, args.environment, args.perception_engine)
    if args.airsim_ip:
        cfg.airsim.ip = args.airsim_ip
    if args.airsim_port:
        cfg.airsim.port = args.airsim_port
    if args.vehicle_name:
        cfg.airsim.vehicle_name = args.vehicle_name
    cfg.flight.cruise_speed = args.speed
    cfg.dashboard.port = args.port
    if args.api_key:
        cfg.cognitive.api_key = args.api_key
    if args.mock:
        cfg.mock_airsim = True
        cfg.mock_perception = True

    orchestrator = CityNavOrchestrator(cfg)

    def signal_handler(sig, frame):
        orchestrator.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    orchestrator.start(start_voice_mic=args.voice_mic)

    if args.instruction:
        time.sleep(1.0)
        orchestrator.start_mission(args.instruction)
    elif args.auto_takeoff:
        time.sleep(1.0)
        print("[Orchestrator] Auto-takeoff requested on startup...")
        alt = cfg.flight.takeoff_altitude
        orchestrator.flight.takeoff(altitude=alt)
        with orchestrator._state_lock:
            orchestrator.flight_mode = "HOVERING"
        orchestrator.narrator.emit_event(
            "TAKEOFF_COMPLETED",
            f"Autonomous takeoff completed. Hovering at {alt:.1f}m.",
            {"altitude": alt}
        )

    # Keep main thread alive
    try:
        while orchestrator.running:
            time.sleep(0.5)
    except KeyboardInterrupt:
        orchestrator.stop()


if __name__ == "__main__":
    main()
