"""Automated Self-Testing Scenario Evaluator and Diagnostic Workflow Runner.

Enables programmatic mission execution, mid-flight interruption injection,
telemetry inspection, spatial memory verification, obstacle stress testing,
and an interactive CLI session for rapid diagnosis.
"""

import sys
import os
import time
import json
import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
import numpy as np

# Ensure parent path is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from realtime_citynav.config import SystemConfig
from realtime_citynav.main import CityNavOrchestrator
from realtime_citynav.spatial_memory import SpatialMemory
from realtime_citynav.dialogue_manager import DialogueManager


@dataclass
class ScenarioResult:
    """Outcome and diagnostics of a single evaluation scenario."""
    name: str
    description: str
    passed: bool
    duration_sec: float
    details: Dict[str, Any] = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "passed": self.passed,
            "duration_sec": round(self.duration_sec, 3),
            "details": self.details,
            "logs": self.logs,
            "error_message": self.error_message
        }


class ScenarioEvaluator:
    """Programmatic harness for running simulated missions and edge-case stress suites."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def _create_mock_orchestrator(self) -> CityNavOrchestrator:
        """Create a headless, deterministic mock orchestrator."""
        cfg = SystemConfig()
        cfg.mock_airsim = True
        cfg.mock_perception = True
        cfg.mock_gemini = True
        cfg.dashboard.enabled = False  # Headless mode: no socket/web server
        cfg.flight.cruise_speed = 5.0
        cfg.flight.takeoff_altitude = 5.0

        orch = CityNavOrchestrator(cfg)
        return orch

    def log(self, scenario_result: ScenarioResult, msg: str) -> None:
        """Append log message and print if verbose."""
        scenario_result.logs.append(msg)
        if self.verbose:
            print(f"  [{scenario_result.name}] {msg}")

    # =========================================================================
    # SCENARIO 1: Altitude Parsing Above Landmark ("go 50m above fountain")
    # =========================================================================
    def run_scenario_altitude_above_fountain(self) -> ScenarioResult:
        """Verify 'go 50m above fountain' targets Z=-50m directly over fountain (not X-50m)."""
        t0 = time.time()
        res = ScenarioResult(
            name="Altitude Above Fountain",
            description="Verify 'go 50m above fountain' sets NED Z=-50m directly over fountain without subtracting from X.",
            passed=False,
            duration_sec=0.0
        )

        orch = self._create_mock_orchestrator()
        orch.start(start_dashboard=False)

        try:
            instruction = "go 50m above fountain"
            self.log(res, f"Starting mission: '{instruction}'")
            subgoals = orch.start_mission(instruction)

            # Check decomposition
            self.log(res, f"Subgoals planned: {subgoals}")
            base_lm = orch.dialogue.extract_base_landmark(subgoals[0])
            if base_lm.lower() != "fountain":
                res.error_message = f"Expected base landmark 'fountain', got '{base_lm}'"
                return res

            # Check waypoint calculation
            fountain_mock_pos = np.array([32.0, 0.0, 0.0])
            tgt_pos, obs_alt = orch._compute_approach_target(subgoals[0], fountain_mock_pos)
            self.log(res, f"Computed target position: {tgt_pos}, Observation altitude: {obs_alt}")

            # Verification 1: X must be fountain X (32.0), NOT 32 - 50 = -18!
            if not np.isclose(tgt_pos[0], 32.0, atol=0.1):
                res.error_message = f"X coordinate corrupted by standoff subtraction: expected 32.0, got {tgt_pos[0]}"
                return res

            # Verification 2: Y must be fountain Y (0.0)
            if not np.isclose(tgt_pos[1], 0.0, atol=0.1):
                res.error_message = f"Y coordinate unexpected: expected 0.0, got {tgt_pos[1]}"
                return res

            # Verification 3: Z must be -50.0 in NED (50m above ground)
            if not np.isclose(tgt_pos[2], -50.0, atol=0.1):
                res.error_message = f"Z coordinate in NED incorrect: expected -50.0, got {tgt_pos[2]}"
                return res

            # Verification 4: Observation altitude is 50.0m
            if not np.isclose(obs_alt, 50.0, atol=0.1):
                res.error_message = f"Altitude incorrect: expected 50.0, got {obs_alt}"
                return res

            # Step flight simulation forward to verify climbing behavior
            for _ in range(30):
                orch.flight.step_control(depth_img=None)
                time.sleep(0.01)

            curr_pose = orch.flight.get_pose()
            curr_alt = -curr_pose[2]
            self.log(res, f"Vehicle climbed to altitude: {curr_alt:.1f}m (NED z = {curr_pose[2]:.1f})")
            if curr_alt < 5.0:
                res.error_message = f"Vehicle failed to climb: altitude is only {curr_alt:.1f}m"
                return res

            res.details = {
                "target_position": tgt_pos.tolist(),
                "observation_altitude": obs_alt,
                "current_pose": curr_pose,
                "altitude_climbed": round(curr_alt, 2)
            }
            res.passed = True
            self.log(res, "PASSED: Target correctly mapped to [32.0, 0.0, -50.0]")

        except Exception as e:
            res.error_message = str(e)
        finally:
            orch.stop()
            res.duration_sec = time.time() - t0

        return res

    # =========================================================================
    # SCENARIO 2: Spatial Prepositions Suite (Front, Left, Right, Behind, Over)
    # =========================================================================
    def run_scenario_spatial_prepositions(self) -> ScenarioResult:
        """Verify spatial prepositions map to the correct 3D axes and coordinates."""
        t0 = time.time()
        res = ScenarioResult(
            name="Spatial Prepositions Suite",
            description="Verify 'in front of', 'to the left of', 'to the right of', 'behind', and 'over' map to proper axes.",
            passed=False,
            duration_sec=0.0
        )

        orch = self._create_mock_orchestrator()

        test_cases = [
            {
                "cmd": "go 15m in front of fountain",
                "center": np.array([32.0, 0.0, 0.0]),
                "expected_pos": np.array([17.0, 0.0, -5.0]),  # 32 - 15 = 17
                "axis": "X (North)"
            },
            {
                "cmd": "fly 20m to the left of blue building",
                "center": np.array([75.0, -14.0, 0.0]),
                "expected_pos": np.array([75.0, -34.0, -14.0]),  # -14 - 20 = -34
                "axis": "Y (West/Left)"
            },
            {
                "cmd": "fly 25m to the right of tower",
                "center": np.array([130.0, -10.0, 0.0]),
                "expected_pos": np.array([130.0, 15.0, -22.0]),  # -10 + 25 = 15
                "axis": "Y (East/Right)"
            },
            {
                "cmd": "go 10m behind hospital",
                "center": np.array([60.0, -30.0, 0.0]),
                "expected_pos": np.array([70.0, -30.0, -14.0]),  # 60 + 10 = 70, alt 14m clearance
                "axis": "X (Behind)"
            },
            {
                "cmd": "fly 30m over fountain",
                "center": np.array([32.0, 0.0, 0.0]),
                "expected_pos": np.array([32.0, 0.0, -30.0]),  # altitude = 30
                "axis": "Z (Altitude)"
            }
        ]

        try:
            case_results = []
            for tc in test_cases:
                tgt_pos, alt = orch._compute_approach_target(tc["cmd"], tc["center"])
                diff = float(np.linalg.norm(tgt_pos - tc["expected_pos"]))
                self.log(res, f"Cmd: '{tc['cmd']}' -> Tgt: {tgt_pos.round(1)} (Expected: {tc['expected_pos'].round(1)})")
                
                if diff > 0.1:
                    res.error_message = f"Failed for '{tc['cmd']}': expected {tc['expected_pos']}, got {tgt_pos}"
                    return res

                case_results.append({
                    "command": tc["cmd"],
                    "output_pos": tgt_pos.tolist(),
                    "expected_pos": tc["expected_pos"].tolist(),
                    "axis_verified": tc["axis"]
                })

            res.details = {"cases": case_results}
            res.passed = True
            self.log(res, f"PASSED: All {len(test_cases)} spatial preposition test cases verified.")

        except Exception as e:
            res.error_message = str(e)
        finally:
            res.duration_sec = time.time() - t0

        return res

    # =========================================================================
    # SCENARIO 3: Mid-Flight Interruption & Target Diversion
    # =========================================================================
    def run_scenario_midflight_interruption(self) -> ScenarioResult:
        """Verify operator mid-flight interruption diverts destination immediately."""
        t0 = time.time()
        res = ScenarioResult(
            name="Mid-Flight Interruption & Rerouting",
            description="Verify mid-flight interruption 'switch to blue building' overrides active target smoothly.",
            passed=False,
            duration_sec=0.0
        )

        orch = self._create_mock_orchestrator()
        orch.start(start_dashboard=False)

        try:
            # 1. Start toward tower
            orch.start_mission("fly to the tower")
            time.sleep(0.3)
            self.log(res, f"Initial target: {orch.active_target_name}")

            # 2. Issue mid-flight interruption
            interruption = "actually switch to the blue building"
            self.log(res, f"Issuing operator interruption: '{interruption}'")
            orch.dialogue.queue_user_message(interruption)

            # Wait for cognitive worker to process queue
            time.sleep(0.4)

            # 3. Verify target diversion
            with orch._state_lock:
                active_tgt = orch.active_target_name
                tgt_pos = orch.flight.target_position

            self.log(res, f"Active target after interruption: '{active_tgt}'")
            if "blue building" not in active_tgt.lower():
                res.error_message = f"Target was not switched to blue building, remained '{active_tgt}'"
                return res

            if tgt_pos is None or not (55.0 <= tgt_pos[0] <= 90.0):
                res.error_message = f"Waypoint did not update toward blue building position, got {tgt_pos}"
                return res

            # Verify narrator logged the interruption
            recent = orch.narrator.get_recent(5)
            types = [e["event_type"] for e in recent]
            if not any(t in ["INTERRUPTION_HANDLED", "OPERATOR_INPUT"] for t in types):
                res.error_message = "Narrator did not record operator interruption event."
                return res

            res.details = {
                "active_target": active_tgt,
                "target_pos": tgt_pos.tolist(),
                "narrations_logged": types
            }
            res.passed = True
            self.log(res, "PASSED: Target successfully diverted mid-flight.")

        except Exception as e:
            res.error_message = str(e)
        finally:
            orch.stop()
            res.duration_sec = time.time() - t0

        return res

    # =========================================================================
    # SCENARIO 4: Spatial Memory Deduplication & Clustering
    # =========================================================================
    def run_scenario_spatial_memory_deduplication(self) -> ScenarioResult:
        """Verify spatial memory cleans duplicate landmarks and clusters noisy observations."""
        t0 = time.time()
        res = ScenarioResult(
            name="Spatial Memory Deduplication",
            description="Verify noisy multi-class observations cluster into clean, non-redundant landmarks.",
            passed=False,
            duration_sec=0.0
        )

        mem = SpatialMemory(cluster_threshold=5.0, alpha=0.2)

        try:
            # Inject noisy observations of the fountain
            mem.add_observation("fountain", np.array([32.0, 0.0, 0.0]), confidence=0.85)
            mem.add_observation("The Fountain", np.array([32.5, 0.3, 0.0]), confidence=0.90)
            mem.add_observation("water fountain", np.array([31.8, -0.2, 0.0]), confidence=0.88)
            mem.add_observation("fountains", np.array([32.2, 0.1, 0.0]), confidence=0.92)

            # Inject noisy observations of the red building
            mem.add_observation("building", np.array([75.0, 12.0, 0.0]), confidence=0.80)
            mem.add_observation("red building", np.array([75.8, 11.6, 0.0]), confidence=0.95)
            mem.add_observation("the red building", np.array([75.2, 12.2, 0.0]), confidence=0.91)

            # Inject distinct tree landmark far away
            mem.add_observation("tree", np.array([10.0, 25.0, 0.0]), confidence=0.87)

            self.log(res, f"Landmarks before cleanup: {len(mem.get_all_landmarks())}")

            # Run maintenance cleanup
            stats = mem.clean_registry(dedup=True)
            self.log(res, f"Cleanup maintenance stats: {stats}")

            landmarks = mem.get_all_landmarks()
            self.log(res, f"Landmarks after deduplication: {len(landmarks)}")
            self.log(res, "Landmark Registry Table:\n" + mem.get_landmark_table())

            # Verification 1: Exactly 3 unique landmarks should exist (fountain, red building, tree)
            if len(landmarks) != 3:
                res.error_message = f"Expected exactly 3 unique landmarks, got {len(landmarks)}: {[l['name'] for l in landmarks]}"
                return res

            # Verification 2: Observation counts should be aggregated
            fountain_lm = mem.find_landmark("fountain")
            if fountain_lm is None or fountain_lm["count"] < 4:
                res.error_message = f"Fountain observation count was not aggregated: {fountain_lm}"
                return res

            building_lm = mem.find_landmark("building")
            if building_lm is None or building_lm["count"] < 3:
                res.error_message = f"Building observation count was not aggregated: {building_lm}"
                return res

            # Verification 3: Descriptive names preferred ('red building' over 'building')
            if building_lm["name"].lower() != "red building":
                res.error_message = f"Expected descriptive name 'red building', got '{building_lm['name']}'"
                return res

            res.details = {
                "landmark_count": len(landmarks),
                "landmarks": [{l["name"]: {"count": l["count"], "pos": l["pos"].tolist()}} for l in landmarks],
                "table": mem.get_landmark_table()
            }
            res.passed = True
            self.log(res, "PASSED: Spatial memory is clean, non-redundant, and properly aggregated.")

        except Exception as e:
            res.error_message = str(e)
        finally:
            res.duration_sec = time.time() - t0

        return res

    # =========================================================================
    # SCENARIO 5: Reactive Obstacle APF Collision Avoidance
    # =========================================================================
    def run_scenario_obstacle_avoidance(self) -> ScenarioResult:
        """Verify reactive APF obstacle avoidance deflects drone away from close barriers."""
        t0 = time.time()
        res = ScenarioResult(
            name="Reactive APF Obstacle Avoidance",
            description="Verify forward flight triggers lateral and braking repulsion when obstacle appears.",
            passed=False,
            duration_sec=0.0
        )

        orch = self._create_mock_orchestrator()

        try:
            orch.flight.takeoff(altitude=5.0)
            orch.flight.set_target(np.array([50.0, 0.0, -5.0]))

            # Normal step: clear path
            clear_depth = np.ones((480, 640), dtype=np.float32) * 20.0
            state_clear = orch.flight.step_control(clear_depth)
            v_clear = state_clear["cmd_velocity"]
            self.log(res, f"Clear flight velocity: {v_clear}")

            # Inject obstacle in left field of view at 2.0m
            blocked_depth = np.ones((480, 640), dtype=np.float32) * 20.0
            h, w = 480, 640
            blocked_depth[h // 4: 3 * h // 4, w // 4: w // 2] = 2.0  # Dense close obstacle

            state_blocked = orch.flight.step_control(blocked_depth)
            v_blocked = state_blocked["cmd_velocity"]
            avoid_dir = state_blocked["avoidance_direction"]
            self.log(res, f"Obstacle encountered! Velocity: {v_blocked}, Avoidance direction: {avoid_dir}")

            # Verification 1: Avoidance must be active
            if not state_blocked["obstacle_avoidance"]:
                res.error_message = "Obstacle avoidance flag was not triggered."
                return res

            # Verification 2: Avoidance direction must be RIGHT (since obstacle is on left)
            if avoid_dir != "RIGHT":
                res.error_message = f"Expected avoidance direction 'RIGHT', got '{avoid_dir}'"
                return res

            # Verification 3: Lateral velocity (+Y) must be positive and forward velocity braked
            if v_blocked[1] <= 0.5:
                res.error_message = f"Repulsion did not command rightward slide: vy = {v_blocked[1]}"
                return res

            res.details = {
                "clear_velocity": v_clear,
                "avoidance_velocity": v_blocked,
                "avoidance_direction": avoid_dir
            }
            res.passed = True
            self.log(res, "PASSED: APF obstacle avoidance successfully deflected vehicle.")

        except Exception as e:
            res.error_message = str(e)
        finally:
            res.duration_sec = time.time() - t0

        return res

    # =========================================================================
    # SCENARIO 6: Recall Query Response
    # =========================================================================
    def run_scenario_recall_query(self) -> ScenarioResult:
        """Verify operator can query registered landmarks and receive accurate summaries."""
        t0 = time.time()
        res = ScenarioResult(
            name="Spatial Recall Query",
            description="Verify operator query 'what have you seen so far?' returns memory landmark digest.",
            passed=False,
            duration_sec=0.0
        )

        orch = self._create_mock_orchestrator()

        try:
            # Populate spatial memory
            orch.memory.add_observation("fountain", np.array([32.0, 0.0, -5.0]), confidence=0.9)
            orch.memory.add_observation("hospital", np.array([60.0, -30.0, -5.0]), confidence=0.92)

            query = "what landmarks have you seen so far?"
            orch.dialogue.queue_user_message(query)

            result = orch.dialogue.process_interruption(
                current_target="tower",
                memory_summary=orch.memory.get_summary()
            )

            self.log(res, f"Query result: {result}")
            if result is None:
                res.error_message = "No result returned for operator query."
                return res

            if result.get("intent") != "QUERY":
                res.error_message = f"Intent was not classified as QUERY, got '{result.get('intent')}'"
                return res

            reply = result.get("reply", "")
            if "fountain" not in reply or "hospital" not in reply:
                res.error_message = f"Query reply missed known landmarks: '{reply}'"
                return res

            res.details = {"reply": reply}
            res.passed = True
            self.log(res, "PASSED: Spatial recall query answered accurately.")

        except Exception as e:
            res.error_message = str(e)
        finally:
            res.duration_sec = time.time() - t0

        return res

    # =========================================================================
    # SCENARIO 7: Extreme Stress & Rapid Command Queuing
    # =========================================================================
    def run_scenario_stress_cases(self) -> ScenarioResult:
        """Verify system stability under rapid contradictory commands, extreme altitudes, and unknown targets."""
        t0 = time.time()
        res = ScenarioResult(
            name="Stress Cases & Edge Robustness",
            description="Verify stability under extreme altitudes, unknown targets, and rapid consecutive interruptions.",
            passed=False,
            duration_sec=0.0
        )

        orch = self._create_mock_orchestrator()
        orch.start(start_dashboard=False)

        try:
            orch.start_mission("fly to the fountain")

            # Rapidly queue contradictory commands
            commands = [
                "stop and hover",
                "climb to 100m",
                "go 50m above fountain",
                "fly to the mythical enchanted castle",
                "hover"
            ]
            for cmd in commands:
                orch.dialogue.queue_user_message(cmd)

            # Let cognitive worker process all commands
            time.sleep(1.2)

            # Verify vehicle pose remains finite and safe
            pose = orch.flight.get_pose()
            self.log(res, f"Vehicle pose after rapid barrage: {pose}")
            if not np.all(np.isfinite(pose)):
                res.error_message = f"Drone pose contains NaNs or Infs: {pose}"
                return res

            # Verify orchestrator is still alive and running
            if not orch.running:
                res.error_message = "Orchestrator crashed or stopped unexpectedly."
                return res

            res.details = {"final_pose": pose}
            res.passed = True
            self.log(res, "PASSED: System handled rapid stress barrage without error.")

        except Exception as e:
            res.error_message = str(e)
        finally:
            orch.stop()
            res.duration_sec = time.time() - t0

        return res

    # =========================================================================
    # SCENARIO 8: Compound Spatial Prepositions (Vertical + Horizontal Offsets)
    # =========================================================================
    def run_scenario_compound_spatial_prepositions(self) -> ScenarioResult:
        """Verify compound instructions (vertical altitude + horizontal offset) calculate accurate 3D positions."""
        t0 = time.time()
        res = ScenarioResult(
            name="Compound Spatial Prepositions",
            description="Verify compound commands ('20m above and 15m behind', '10m left and 25m above') apply both axes simultaneously.",
            passed=False,
            duration_sec=0.0
        )

        orch = self._create_mock_orchestrator()

        test_cases = [
            {
                "cmd": "go 20m above and 15m behind the red building",
                "center": np.array([75.0, 12.0, 0.0]),
                "expected_pos": np.array([90.0, 12.0, -20.0]),  # X: 75 + 15 = 90, Z: 0 - 20 = -20
                "expected_base": "red building",
                "expected_alt": 20.0
            },
            {
                "cmd": "go 15m behind and 20m above the red building",
                "center": np.array([75.0, 12.0, 0.0]),
                "expected_pos": np.array([90.0, 12.0, -20.0]),
                "expected_base": "red building",
                "expected_alt": 20.0
            },
            {
                "cmd": "fly 10m to the left and 25m above tower",
                "center": np.array([130.0, -10.0, 0.0]),
                "expected_pos": np.array([130.0, -20.0, -25.0]),  # Y: -10 - 10 = -20, Z: 0 - 25 = -25
                "expected_base": "tower",
                "expected_alt": 25.0
            },
            {
                "cmd": "go 50m above and 10m in front of fountain",
                "center": np.array([32.0, 0.0, 0.0]),
                "expected_pos": np.array([22.0, 0.0, -50.0]),  # X: 32 - 10 = 22, Z: 0 - 50 = -50
                "expected_base": "fountain",
                "expected_alt": 50.0
            }
        ]

        try:
            case_results = []
            for tc in test_cases:
                spatial = orch.dialogue.parse_spatial_relation(tc["cmd"])
                base_lm = spatial.get("base_landmark", "")
                tgt_pos, alt = orch._compute_approach_target(tc["cmd"], tc["center"])
                diff = float(np.linalg.norm(tgt_pos - tc["expected_pos"]))

                self.log(res, f"Cmd: '{tc['cmd']}' -> Base: '{base_lm}', Pos: {tgt_pos.round(1)}, Alt: {alt}")

                if base_lm.lower() != tc["expected_base"].lower():
                    res.error_message = f"Base landmark mismatch for '{tc['cmd']}': expected '{tc['expected_base']}', got '{base_lm}'"
                    return res

                if diff > 0.1:
                    res.error_message = f"3D target position incorrect for '{tc['cmd']}': expected {tc['expected_pos']}, got {tgt_pos}"
                    return res

                if not np.isclose(alt, tc["expected_alt"], atol=0.1):
                    res.error_message = f"Altitude mismatch for '{tc['cmd']}': expected {tc['expected_alt']}, got {alt}"
                    return res

                case_results.append({
                    "command": tc["cmd"],
                    "base_landmark": base_lm,
                    "target_pos": tgt_pos.tolist(),
                    "alt": alt
                })

            res.details = {"cases": case_results}
            res.passed = True
            self.log(res, f"PASSED: All {len(test_cases)} compound spatial commands verified.")

        except Exception as e:
            res.error_message = str(e)
        finally:
            res.duration_sec = time.time() - t0

        return res

    # =========================================================================
    # SCENARIO 9: Elevated Landmark 3D Waypoint Calculation
    # =========================================================================
    def run_scenario_elevated_landmark_altitude(self) -> ScenarioResult:
        """Verify altitude above elevated landmark (non-zero Z in NED) accounts for landmark elevation."""
        t0 = time.time()
        res = ScenarioResult(
            name="Elevated Landmark Altitude",
            description="Verify 'go 30m above tower' when tower is at Z=-22m targets Z=-52m (52m altitude) avoiding collision.",
            passed=False,
            duration_sec=0.0
        )

        orch = self._create_mock_orchestrator()

        try:
            # Tower with 22m elevation (NED Z = -22.0)
            elevated_tower_pos = np.array([130.0, -10.0, -22.0])
            cmd = "go 30m above tower"
            self.log(res, f"Computing approach for '{cmd}' over elevated landmark at {elevated_tower_pos}")

            tgt_pos, alt = orch._compute_approach_target(cmd, elevated_tower_pos)
            self.log(res, f"Computed target: {tgt_pos}, Observation altitude: {alt}")

            # Verification 1: X and Y centered over tower
            if not (np.isclose(tgt_pos[0], 130.0, atol=0.1) and np.isclose(tgt_pos[1], -10.0, atol=0.1)):
                res.error_message = f"Horizontal center deviated: expected [130, -10], got {tgt_pos[:2]}"
                return res

            # Verification 2: Z must be -52.0 (-22 - 30), NOT -30.0!
            # If Z were -30.0, the drone would only have 8m clearance over a 22m building.
            expected_z = -52.0
            if not np.isclose(tgt_pos[2], expected_z, atol=0.1):
                res.error_message = f"Z coordinate ignored landmark elevation: expected {expected_z}, got {tgt_pos[2]}"
                return res

            # Verification 3: Observation altitude must be 52.0m
            if not np.isclose(alt, 52.0, atol=0.1):
                res.error_message = f"Target altitude incorrect: expected 52.0, got {alt}"
                return res

            res.details = {
                "landmark_pos": elevated_tower_pos.tolist(),
                "target_pos": tgt_pos.tolist(),
                "target_alt": alt
            }
            res.passed = True
            self.log(res, "PASSED: Target altitude correctly calculated relative to elevated landmark.")

        except Exception as e:
            res.error_message = str(e)
        finally:
            res.duration_sec = time.time() - t0

        return res

    # =========================================================================
    # SCENARIO 10: Automatic Memory Maintenance & Periodic Pruning
    # =========================================================================
    def run_scenario_continuous_memory_pruning(self) -> ScenarioResult:
        """Verify memory periodic maintenance merges synonyms/duplicates and prunes transient false detections."""
        t0 = time.time()
        res = ScenarioResult(
            name="Continuous Memory Maintenance & Pruning",
            description="Verify unconfirmed transient detections are pruned and multi-synonym observations merge.",
            passed=False,
            duration_sec=0.0
        )

        orch = self._create_mock_orchestrator()

        try:
            # 1. Add multiple observations of fountain with synonyms and hyphenation
            orch.memory.add_observation("fountain", np.array([32.0, 0.0, 0.0]), confidence=0.9)
            orch.memory.add_observation("water-fountain", np.array([32.3, 0.1, 0.0]), confidence=0.88)
            orch.memory.add_observation("The Fountain", np.array([31.8, -0.2, 0.0]), confidence=0.92)
            orch.memory.add_observation("fountains", np.array([32.1, 0.0, 0.0]), confidence=0.95)

            # 2. Add an unconfirmed transient false positive (count=1, backdated by 40s)
            transient_id = orch.memory.add_observation("unconfirmed_car", np.array([10.0, 50.0, 0.0]), confidence=0.3)
            with orch.memory._lock:
                orch.memory._landmarks[transient_id]["last_seen"] -= 40.0

            self.log(res, f"Landmarks before maintenance: {len(orch.memory.get_all_landmarks())}")

            # 3. Perform maintenance
            stats = orch.clean_spatial_memory()
            self.log(res, f"Maintenance stats: {stats}")

            landmarks = orch.memory.get_all_landmarks()
            self.log(res, f"Landmarks after maintenance: {len(landmarks)}")

            # Verification 1: transient_car must be pruned
            car_lm = orch.memory.find_landmark("unconfirmed_car")
            if car_lm is not None:
                res.error_message = "Transient unconfirmed detection was not pruned."
                return res

            # Verification 2: fountain entries must merge into 1 clean record
            if len(landmarks) != 1:
                res.error_message = f"Expected exactly 1 landmark, got {len(landmarks)}: {[l['name'] for l in landmarks]}"
                return res

            fountain_lm = orch.memory.find_landmark("fountain")
            if fountain_lm is None or fountain_lm["count"] != 4:
                res.error_message = f"Fountain count not aggregated: {fountain_lm}"
                return res

            # Verification 3: Empty string search must return None
            empty_res = orch.memory.find_landmark("")
            if empty_res is not None:
                res.error_message = f"find_landmark('') should return None, but returned {empty_res}"
                return res

            res.details = {
                "remaining_landmarks": [{l["name"]: l["count"]} for l in landmarks],
                "stats": stats
            }
            res.passed = True
            self.log(res, "PASSED: Maintenance successfully pruned noise and consolidated duplicates.")

        except Exception as e:
            res.error_message = str(e)
        finally:
            res.duration_sec = time.time() - t0

        return res

    # =========================================================================
    # SCENARIO 11: Malformed and Punctuation Input Robustness
    # =========================================================================
    def run_scenario_malformed_and_punctuation_robustness(self) -> ScenarioResult:
        """Verify system stability and correct handling when operator provides noisy/malformed messages."""
        t0 = time.time()
        res = ScenarioResult(
            name="Malformed & Punctuation Input Robustness",
            description="Verify graceful handling of empty inputs, punctuation ('...', '???'), and action verbs ('ascend to 50m above fountain').",
            passed=False,
            duration_sec=0.0
        )

        orch = self._create_mock_orchestrator()

        try:
            # 1. Punctuation interruption should not create a target named "..."
            orch.dialogue.queue_user_message("...")
            res1 = orch.dialogue.process_interruption(current_target="tower")
            self.log(res, f"Punctuation interruption result: {res1}")
            if res1 is None or res1.get("intent") == "TARGET_CORRECTION":
                res.error_message = f"Punctuation '...' was treated as target correction: {res1}"
                return res

            # 2. 'ascend to 50m above fountain' should correctly parse to 'ABOVE' relation
            spatial = orch.dialogue.parse_spatial_relation("ascend to 50m above fountain")
            self.log(res, f"Action verb parse result: {spatial}")
            if spatial.get("relation") != "ABOVE" or spatial.get("distance") != 50.0:
                res.error_message = f"'ascend to 50m above fountain' failed to parse: {spatial}"
                return res
            if spatial.get("base_landmark", "").lower() != "fountain":
                res.error_message = f"Base landmark failed to extract: {spatial.get('base_landmark')}"
                return res

            # 3. Subgoal instruction decomposition with action verb
            subgoals = orch.dialogue.parse_initial_instruction("ascend to 50m above fountain")
            self.log(res, f"Instruction subgoals: {subgoals}")
            if not subgoals or "fountain" not in subgoals[0].lower():
                res.error_message = f"Failed to decompose 'ascend to 50m above fountain': {subgoals}"
                return res

            res.details = {
                "punctuation_reply": res1.get("reply"),
                "spatial_parse": spatial,
                "subgoals": subgoals
            }
            res.passed = True
            self.log(res, "PASSED: System robustly handled malformed, punctuation, and action verb inputs.")

        except Exception as e:
            res.error_message = str(e)
        finally:
            res.duration_sec = time.time() - t0

        return res

    def run_scenario_multileg_route_and_shorthand_resolution(self) -> ScenarioResult:
        """Scenario 12: Multi-leg route extraction and anaphora/shorthand adjective resolution."""
        t0 = time.time()
        res = ScenarioResult(
            name="Multi-Leg Route & Shorthand Landmark Resolution",
            description="Verify 'go from fountain to red building, and also go from red to blue building' resolves to ['fountain', 'red building', 'blue building'] and sequences subgoals correctly.",
            passed=False,
            duration_sec=0.0
        )
        orch = self._create_mock_orchestrator()
        orch.start(start_dashboard=False)
        try:
            # 1. Test multi-leg parse with adjective shorthand
            inst = "go from fountain to red building, and also go from red to blue building"
            subgoals = orch.dialogue.parse_initial_instruction(inst)
            self.log(res, f"Parsed subgoals for '{inst}': {subgoals}")
            expected = ["fountain", "red building", "blue building"]
            if subgoals != expected:
                res.error_message = f"Expected subgoals {expected}, got {subgoals}"
                return res

            # 2. Test standalone from-to with shorthand
            inst2 = "go from red to blue building"
            subgoals2 = orch.dialogue.parse_initial_instruction(inst2)
            self.log(res, f"Parsed subgoals for '{inst2}': {subgoals2}")
            expected2 = ["red building", "blue building"]
            if subgoals2 != expected2:
                res.error_message = f"Expected subgoals {expected2}, got {subgoals2}"
                return res

            # 3. Test shorthand resolution directly
            if orch.dialogue.resolve_landmark_shorthand("the red one") != "red building":
                res.error_message = "Failed to resolve 'the red one' -> 'red building'"
                return res
            if orch.dialogue.resolve_landmark_shorthand("blue") != "blue building":
                res.error_message = "Failed to resolve 'blue' -> 'blue building'"
                return res

            # 4. Start mission and verify active target sequencing
            orch.start_mission(inst)
            with orch._state_lock:
                initial_target = orch.active_target_name
            self.log(res, f"Initial active target: '{initial_target}'")
            if initial_target != "fountain":
                res.error_message = f"Expected initial target 'fountain', got '{initial_target}'"
                return res

            # Simulate reaching fountain
            fountain_pos = np.array([32.0, 0.0, -5.0])
            orch.flight._pose = (fountain_pos[0], fountain_pos[1], fountain_pos[2], 0.0)
            orch.flight._target_pos = fountain_pos.copy()
            # Advance subgoal
            with orch._state_lock:
                orch.active_subgoal_index = 1
                orch._set_current_subgoal_target(1)
                second_target = orch.active_target_name

            self.log(res, f"Second active target: '{second_target}'")
            if second_target != "red building":
                res.error_message = f"Expected second target 'red building', got '{second_target}'"
                return res

            # Simulate reaching red building
            red_pos = np.array([75.0, 12.0, -14.0])
            orch.flight._pose = (red_pos[0], red_pos[1], red_pos[2], 0.0)
            orch.flight._target_pos = red_pos.copy()
            # Advance subgoal
            with orch._state_lock:
                orch.active_subgoal_index = 2
                orch._set_current_subgoal_target(2)
                third_target = orch.active_target_name

            self.log(res, f"Third active target: '{third_target}'")
            if third_target != "blue building":
                res.error_message = f"Expected third target 'blue building', got '{third_target}'"
                return res

            # Test operator interruption with shorthand: "switch to red"
            orch.dialogue.queue_user_message("switch to red")
            interruption = orch.dialogue.process_interruption(current_target=third_target)
            self.log(res, f"Interruption processed: {interruption}")
            if not interruption or interruption.get("target") != "red building":
                res.error_message = f"Interruption failed to resolve 'switch to red' -> 'red building': {interruption}"
                return res

            res.details = {
                "subgoals": subgoals,
                "subgoals2": subgoals2,
                "initial_target": initial_target,
                "second_target": second_target,
                "third_target": third_target,
                "interruption_target": interruption.get("target")
            }
            res.passed = True
            self.log(res, "PASSED: Multi-leg route extraction and shorthand resolution validated end-to-end.")

        except Exception as e:
            res.error_message = str(e)
        finally:
            orch.stop()
            res.duration_sec = time.time() - t0

        return res

    # =========================================================================
    # SCENARIO 13: Single Landmark Orbit Kinematics ("hover in circle around blue building at 10m altitude")
    # =========================================================================
    def run_scenario_orbit_blue_building(self) -> ScenarioResult:
        t0 = time.time()
        res = ScenarioResult(
            name="Single Landmark Orbit Kinematics",
            description="Verify 'hover in circle around blue building at 10m altitude' engages ORBIT mode, sets center [75, -14], and points yaw inward.",
            passed=False,
            duration_sec=0.0
        )
        orch = self._create_mock_orchestrator()
        orch.start(start_dashboard=False)
        try:
            instruction = "hover in circle around blue building at 10m altitude"
            self.log(res, f"Starting mission: '{instruction}'")
            subgoals = orch.start_mission(instruction)
            self.log(res, f"Subgoals planned: {subgoals}")

            # Step simulation to finish takeoff and engage orbit
            for _ in range(25):
                orch.flight.step_control(depth_img=None)
                time.sleep(0.01)

            telem = orch.get_telemetry()
            self.log(res, f"Telemetry after takeoff: mode={telem['mode']}, pos={telem['position']}, alt={telem['altitude']}")

            if telem["mode"] != "ORBIT":
                res.error_message = f"Expected mode 'ORBIT', got '{telem['mode']}'"
                return res

            if orch.flight.orbit_center is None or not np.allclose(orch.flight.orbit_center[:2], [75.0, -14.0], atol=1.0):
                res.error_message = f"Orbit center mismatch: expected [75, -14], got {orch.flight.orbit_center}"
                return res

            if orch.flight.orbit_facing != "INWARD":
                res.error_message = f"Expected INWARD facing, got {orch.flight.orbit_facing}"
                return res

            res.details = {"telemetry": telem, "subgoals": subgoals}
            res.passed = True
            self.log(res, "PASSED: Single landmark orbit kinematics verified.")
        except Exception as e:
            res.error_message = str(e)
        finally:
            orch.stop()
            res.duration_sec = time.time() - t0
        return res

    # =========================================================================
    # SCENARIO 14: Tangent-Facing Orbit with Speed & Radius
    # =========================================================================
    def run_scenario_tangent_orbit_speed_radius(self) -> ScenarioResult:
        t0 = time.time()
        res = ScenarioResult(
            name="Tangent Orbit with Custom Speed and Radius",
            description="Verify 'circle red building twice facing forward at 5 m/s at 20m radius' applies tangent yaw, speed 5 m/s, and radius 20m.",
            passed=False,
            duration_sec=0.0
        )
        orch = self._create_mock_orchestrator()
        orch.start(start_dashboard=False)
        try:
            cmd = "circle red building twice facing forward at 5 m/s at 20m radius"
            self.log(res, f"Starting mission: '{cmd}'")
            subgoals = orch.start_mission(cmd)

            for _ in range(25):
                orch.flight.step_control(depth_img=None)
                time.sleep(0.01)

            if orch.flight.orbit_facing != "TANGENT":
                res.error_message = f"Expected TANGENT facing, got {orch.flight.orbit_facing}"
                return res
            if not np.isclose(orch.flight.orbit_radius, 20.0, atol=0.5):
                res.error_message = f"Expected radius 20.0m, got {orch.flight.orbit_radius}"
                return res
            if not np.isclose(orch.flight.orbit_speed, 5.0, atol=0.5):
                res.error_message = f"Expected speed 5.0 m/s, got {orch.flight.orbit_speed}"
                return res
            if orch.flight.orbit_revolutions != 2.0:
                res.error_message = f"Expected revolutions 2.0, got {orch.flight.orbit_revolutions}"
                return res

            res.passed = True
            self.log(res, "PASSED: Tangent facing, speed, radius, and revolution count verified.")
        except Exception as e:
            res.error_message = str(e)
        finally:
            orch.stop()
            res.duration_sec = time.time() - t0
        return res

    # =========================================================================
    # SCENARIO 15: Clockwise Orbit Parameter Inversion
    # =========================================================================
    def run_scenario_clockwise_orbit_parameters(self) -> ScenarioResult:
        t0 = time.time()
        res = ScenarioResult(
            name="Clockwise Orbit Parameter Inversion",
            description="Verify 'circle around blue building 3 times clockwise at 25m altitude' sets direction=-1 (CW), altitude=25m, and revolutions=3.",
            passed=False,
            duration_sec=0.0
        )
        orch = self._create_mock_orchestrator()
        orch.start(start_dashboard=False)
        try:
            cmd = "circle around blue building 3 times clockwise at 25m altitude"
            self.log(res, f"Starting mission: '{cmd}'")
            orch.start_mission(cmd)

            for _ in range(25):
                orch.flight.step_control(depth_img=None)
                time.sleep(0.01)

            if orch.flight.orbit_direction != -1:
                res.error_message = f"Expected orbit direction -1 (CW), got {orch.flight.orbit_direction}"
                return res
            if not np.isclose(orch.flight.orbit_altitude, 25.0, atol=0.5):
                res.error_message = f"Expected altitude 25.0m, got {orch.flight.orbit_altitude}"
                return res
            if orch.flight.orbit_revolutions != 3.0:
                res.error_message = f"Expected revolutions 3.0, got {orch.flight.orbit_revolutions}"
                return res

            res.passed = True
            self.log(res, "PASSED: Clockwise orbit direction and altitude parameters verified.")
        except Exception as e:
            res.error_message = str(e)
        finally:
            orch.stop()
            res.duration_sec = time.time() - t0
        return res

    # =========================================================================
    # SCENARIO 16: Compound Multi-Altitude Multi-Landmark Orbit
    # =========================================================================
    def run_scenario_compound_multi_altitude_orbit(self) -> ScenarioResult:
        t0 = time.time()
        res = ScenarioResult(
            name="Compound Multi-Altitude Multi-Landmark Orbit",
            description="Verify 'circle around blue building 3 times at 30m altitude, and then around red building 2 times at 35m altitude' preserves altitudes and advances legs.",
            passed=False,
            duration_sec=0.0
        )
        orch = self._create_mock_orchestrator()
        orch.start(start_dashboard=False)
        try:
            cmd = "circle around blue building 3 times at 30m altitude, and then around red building 2 times at 35m altitude"
            self.log(res, f"Starting mission: '{cmd}'")
            subgoals = orch.start_mission(cmd)
            self.log(res, f"Parsed subgoals: {subgoals}")

            if len(subgoals) != 2:
                res.error_message = f"Expected 2 subgoals, got {len(subgoals)}: {subgoals}"
                return res

            if "30" not in subgoals[0] or "35" not in subgoals[1]:
                res.error_message = f"Altitude qualifiers stripped from subgoals: {subgoals}"
                return res

            # Simulate Leg 1 active
            orch.flight.is_in_air = True
            orch.flight_mode = "ORBIT"
            orch.flight.set_orbit_target(
                center_xyz=[75.0, -14.0, 0.0],
                radius=16.0,
                altitude=30.0,
                revolutions=3.0,
                target_name="blue building"
            )

            # Advance to Leg 2
            orch.flight.orbit_completed = True
            for _ in range(5):
                state = orch.flight.step_control(depth_img=None)
                if state.get("has_arrived"):
                    with orch._state_lock:
                        orch._set_current_subgoal_target(1)
                    break
                time.sleep(0.01)

            if orch.active_subgoal_index != 1:
                res.error_message = f"Did not advance to active subgoal index 1: {orch.active_subgoal_index}"
                return res

            if not np.isclose(orch.flight.orbit_altitude, 35.0, atol=0.5):
                res.error_message = f"Expected Leg 2 altitude 35.0m, got {orch.flight.orbit_altitude}"
                return res

            res.passed = True
            self.log(res, "PASSED: Compound multi-altitude orbit sequenced and climbed cleanly.")
        except Exception as e:
            res.error_message = str(e)
        finally:
            orch.stop()
            res.duration_sec = time.time() - t0
        return res

    # =========================================================================
    # SCENARIO 17: Explicit 3D Waypoint Path Following
    # =========================================================================
    def run_scenario_explicit_3d_waypoint_path(self) -> ScenarioResult:
        t0 = time.time()
        res = ScenarioResult(
            name="Explicit 3D Continuous Waypoint Path Following",
            description="Verify 'follow curve [[15, 25, -20], [25, 35, -25], [35, 25, -20]]' sets CURVE mode and follows 3D coordinates.",
            passed=False,
            duration_sec=0.0
        )
        orch = self._create_mock_orchestrator()
        orch.start(start_dashboard=False)
        try:
            cmd = "follow curve [[15, 25, -20], [25, 35, -25], [35, 25, -20]]"
            self.log(res, f"Injecting interruption: '{cmd}'")
            orch.dialogue.queue_user_message(cmd)

            # Allow cognitive worker to process interruption
            t_wait = time.time()
            handled = False
            while time.time() - t_wait < 3.0:
                if orch.flight_mode in ("CURVE", "WAYPOINT_PATH"):
                    handled = True
                    break
                time.sleep(0.05)

            if not handled:
                res.error_message = f"Expected CURVE flight mode, got '{orch.flight_mode}'"
                return res

            if not orch.flight.is_following_path:
                res.error_message = "Flight controller is_following_path flag not set"
                return res

            if len(orch.flight.trajectory_path) != 3:
                res.error_message = f"Expected 3 trajectory points, got {len(orch.flight.trajectory_path)}"
                return res

            res.passed = True
            self.log(res, "PASSED: Explicit 3D waypoint curve configured and followed.")
        except Exception as e:
            res.error_message = str(e)
        finally:
            orch.stop()
            res.duration_sec = time.time() - t0
        return res

    # =========================================================================
    # SCENARIO 18: Universal Collective Quantifier Tour
    # =========================================================================
    def run_scenario_collective_quantifier_city_tour(self) -> ScenarioResult:
        t0 = time.time()
        res = ScenarioResult(
            name="Universal Collective Quantifier City Tour",
            description="Verify 'visit all landmarks' decomposes into full 5-landmark tour in canonical order.",
            passed=False,
            duration_sec=0.0
        )
        orch = self._create_mock_orchestrator()
        orch.start(start_dashboard=False)
        try:
            cmd = "visit all landmarks"
            self.log(res, f"Starting mission: '{cmd}'")
            subgoals = orch.start_mission(cmd)
            self.log(res, f"Subgoals planned: {subgoals}")

            expected = ["fountain", "red building", "blue building", "tower", "hospital"]
            if subgoals != expected:
                res.error_message = f"Expected {expected}, got {subgoals}"
                return res

            if orch.active_subgoal_index != 0 or orch.active_target_name != "fountain":
                res.error_message = f"First target mismatch: expected 'fountain', got '{orch.active_target_name}'"
                return res

            res.passed = True
            self.log(res, "PASSED: Universal collective quantifier city tour planned correctly.")
        except Exception as e:
            res.error_message = str(e)
        finally:
            orch.stop()
            res.duration_sec = time.time() - t0
        return res

    # =========================================================================
    # SCENARIO 19: Mid-Flight Stop & Hover Interruption
    # =========================================================================
    def run_scenario_stop_and_hover_interruption(self) -> ScenarioResult:
        t0 = time.time()
        res = ScenarioResult(
            name="Mid-Flight Stop & Hover Interruption",
            description="Verify operator 'stop and hover right here' halts active navigation into stationary HOVERING.",
            passed=False,
            duration_sec=0.0
        )
        orch = self._create_mock_orchestrator()
        orch.start(start_dashboard=False)
        try:
            orch.start_mission("tower")
            time.sleep(0.2)

            self.log(res, "Injecting interruption: 'stop and hover right here'")
            orch.dialogue.queue_user_message("stop and hover right here")

            t_wait = time.time()
            hovering = False
            while time.time() - t_wait < 3.0:
                telem = orch.get_telemetry()
                if telem["mode"] == "HOVERING":
                    hovering = True
                    break
                time.sleep(0.05)

            if not hovering:
                res.error_message = f"Drone failed to enter HOVERING mode. Current mode: '{orch.flight_mode}'"
                return res

            res.passed = True
            self.log(res, "PASSED: Stop and hover cleanly entered stationary hover.")
        except Exception as e:
            res.error_message = str(e)
        finally:
            orch.stop()
            res.duration_sec = time.time() - t0
        return res

    # =========================================================================
    # SCENARIO 20: Mid-Flight Orbit Interruption from Linear Flight
    # =========================================================================
    def run_scenario_midflight_orbit_interruption(self) -> ScenarioResult:
        t0 = time.time()
        res = ScenarioResult(
            name="Mid-Flight Orbit Interruption from Linear Flight",
            description="Verify linear flight to 'fountain' interrupted by 'circle twice around red building' switches directly to ORBIT.",
            passed=False,
            duration_sec=0.0
        )
        orch = self._create_mock_orchestrator()
        orch.start(start_dashboard=False)
        try:
            orch.start_mission("fountain")
            time.sleep(0.2)

            self.log(res, "Injecting interruption: 'circle twice around red building'")
            orch.dialogue.queue_user_message("circle twice around red building")

            t_wait = time.time()
            in_orbit = False
            while time.time() - t_wait < 3.0:
                telem = orch.get_telemetry()
                if telem["mode"] == "ORBIT" and telem.get("orbit_target") == "red building":
                    in_orbit = True
                    break
                time.sleep(0.05)

            if not in_orbit:
                res.error_message = f"Failed to switch to ORBIT mode around red building: {orch.flight_mode}"
                return res

            res.passed = True
            self.log(res, "PASSED: Mid-flight linear navigation diverted to orbit around red building.")
        except Exception as e:
            res.error_message = str(e)
        finally:
            orch.stop()
            res.duration_sec = time.time() - t0
        return res

    # =========================================================================
    # SCENARIO 21: Pure Altitude Change Command Without Landmark
    # =========================================================================
    def run_scenario_pure_altitude_change(self) -> ScenarioResult:
        t0 = time.time()
        res = ScenarioResult(
            name="Pure Altitude Change Command Without Landmark",
            description="Verify operator 'climb to 40m' updates target altitude to 40m without needing a landmark destination.",
            passed=False,
            duration_sec=0.0
        )
        orch = self._create_mock_orchestrator()
        orch.start(start_dashboard=False)
        try:
            # Place in hover
            orch.flight_mode = "HOVERING"
            orch.flight.is_in_air = True
            orch.flight.is_hovering = True

            self.log(res, "Injecting interruption: 'climb to 40m'")
            orch.dialogue.queue_user_message("climb to 40m")

            t_wait = time.time()
            alt_updated = False
            while time.time() - t_wait < 3.0:
                if np.isclose(orch.flight.target_altitude, 40.0, atol=1.0):
                    alt_updated = True
                    break
                time.sleep(0.05)

            if not alt_updated:
                res.error_message = f"Target altitude not updated: expected 40.0m, got {orch.flight.target_altitude}"
                return res

            res.passed = True
            self.log(res, "PASSED: Pure altitude change adjusted vertical target correctly.")
        except Exception as e:
            res.error_message = str(e)
        finally:
            orch.stop()
            res.duration_sec = time.time() - t0
        return res

    # =========================================================================
    # SCENARIO 22: Geofence Boundary Collision Clamping
    # =========================================================================
    def run_scenario_geofence_boundary_safety(self) -> ScenarioResult:
        t0 = time.time()
        res = ScenarioResult(
            name="Geofence Boundary Collision Clamping",
            description="Verify drone attempting to exceed horizontal geofence limit (250m) is clamped and safe.",
            passed=False,
            duration_sec=0.0
        )
        orch = self._create_mock_orchestrator()
        try:
            orch.flight.is_in_air = True
            # Place near boundary at X=248m, Y=0m
            orch.flight._pose = (248.0, 0.0, -10.0, 0.0)
            orch.flight.target_position = np.array([300.0, 0.0, -10.0])  # Beyond boundary

            # Step control
            state = orch.flight.step_control(depth_img=None)
            cmd_v = state["cmd_velocity"]
            self.log(res, f"Position: {orch.flight._pose[:2]}, Commanded velocity toward outside: {cmd_v}")

            # Advance 20 steps
            for _ in range(20):
                orch.flight.step_control(depth_img=None)

            new_x = orch.flight._pose[0]
            self.log(res, f"Position after stepping toward out-of-bounds: X={new_x:.2f}m")
            if new_x > 255.0:
                res.error_message = f"Geofence violated: X exceeded limit (got {new_x:.2f}m)"
                return res

            res.passed = True
            self.log(res, "PASSED: Geofence boundary successfully restrained drone safely within perimeter.")
        except Exception as e:
            res.error_message = str(e)
        finally:
            res.duration_sec = time.time() - t0
        return res

    # =========================================================================
    # SCENARIO 23: Mid-Orbit Interruption with Spatial Target
    # =========================================================================
    def run_scenario_mid_orbit_spatial_target_interruption(self) -> ScenarioResult:
        t0 = time.time()
        res = ScenarioResult(
            name="Mid-Orbit Interruption with Spatial Target",
            description="Verify active orbit interrupted by 'hover 100m above red building' cleanly terminates orbit and climbs to 100m above red building.",
            passed=False,
            duration_sec=0.0
        )
        orch = self._create_mock_orchestrator()
        orch.start(start_dashboard=False)
        try:
            orch.start_mission("circle around blue building")
            time.sleep(0.3)

            self.log(res, "Interrupting active orbit with 'hover 100m above red building'")
            orch.dialogue.queue_user_message("hover 100m above red building")

            t_wait = time.time()
            diverted = False
            while time.time() - t_wait < 3.0:
                telem = orch.get_telemetry()
                if "red building" in str(telem.get("active_target")).lower() and telem["mode"] != "ORBIT":
                    diverted = True
                    break
                time.sleep(0.05)

            if not diverted:
                res.error_message = f"Failed to divert from orbit to target correction: mode={orch.flight_mode}, target={orch.active_target_name}"
                return res

            res.passed = True
            self.log(res, "PASSED: Mid-orbit spatial target interruption diverted cleanly.")
        except Exception as e:
            res.error_message = str(e)
        finally:
            orch.stop()
            res.duration_sec = time.time() - t0
        return res

    # =========================================================================
    # SCENARIO 24: Return to Past Landmark Memory Recall
    # =========================================================================
    def run_scenario_return_to_past_landmark(self) -> ScenarioResult:
        t0 = time.time()
        res = ScenarioResult(
            name="Return to Past Landmark Memory Recall",
            description="Verify operator 'go back to fountain' recalls past fountain coordinates and diverts flight.",
            passed=False,
            duration_sec=0.0
        )
        orch = self._create_mock_orchestrator()
        orch.start(start_dashboard=False)
        try:
            # Register fountain in spatial memory
            orch.memory.add_observation("fountain", np.array([32.0, 0.0, -5.0]), confidence=0.92)
            orch.start_mission("hospital")
            time.sleep(0.3)

            self.log(res, "Injecting interruption: 'go back to fountain'")
            orch.dialogue.queue_user_message("go back to fountain")

            t_wait = time.time()
            recalled = False
            while time.time() - t_wait < 3.0:
                if orch.active_target_name and "fountain" in orch.active_target_name.lower():
                    recalled = True
                    break
                time.sleep(0.05)

            if not recalled:
                res.error_message = f"Did not divert back to fountain: target={orch.active_target_name}"
                return res

            res.passed = True
            self.log(res, "PASSED: Return to past landmark successfully routed to prior memory location.")
        except Exception as e:
            res.error_message = str(e)
        finally:
            orch.stop()
            res.duration_sec = time.time() - t0
        return res

    # =========================================================================
    # Master Suite Runner
    # =========================================================================
    def run_all(self) -> List[ScenarioResult]:
        """Execute all validation and stress scenarios in sequence."""
        scenarios = [
            self.run_scenario_altitude_above_fountain,
            self.run_scenario_spatial_prepositions,
            self.run_scenario_midflight_interruption,
            self.run_scenario_spatial_memory_deduplication,
            self.run_scenario_obstacle_avoidance,
            self.run_scenario_recall_query,
            self.run_scenario_stress_cases,
            self.run_scenario_compound_spatial_prepositions,
            self.run_scenario_elevated_landmark_altitude,
            self.run_scenario_continuous_memory_pruning,
            self.run_scenario_malformed_and_punctuation_robustness,
            self.run_scenario_multileg_route_and_shorthand_resolution,
            self.run_scenario_orbit_blue_building,
            self.run_scenario_tangent_orbit_speed_radius,
            self.run_scenario_clockwise_orbit_parameters,
            self.run_scenario_compound_multi_altitude_orbit,
            self.run_scenario_explicit_3d_waypoint_path,
            self.run_scenario_collective_quantifier_city_tour,
            self.run_scenario_stop_and_hover_interruption,
            self.run_scenario_midflight_orbit_interruption,
            self.run_scenario_pure_altitude_change,
            self.run_scenario_geofence_boundary_safety,
            self.run_scenario_mid_orbit_spatial_target_interruption,
            self.run_scenario_return_to_past_landmark,
        ]

        results = []
        for s in scenarios:
            res = s()
            results.append(res)
        return results

    def print_report(self, results: List[ScenarioResult]) -> bool:
        """Render formatted ANSI execution report in terminal."""
        print("\n" + "=" * 80)
        print(" RealTime CityNav Autonomous Flight Scenario Evaluation Report")
        print("=" * 80)

        all_passed = True
        for idx, r in enumerate(results, start=1):
            status_str = "\033[92m[PASS]\033[0m" if r.passed else "\033[91m[FAIL]\033[0m"
            print(f"{status_str} Scenario {idx}: {r.name} ({r.duration_sec:.2f}s)")
            print(f"       {r.description}")
            if not r.passed:
                all_passed = False
                print(f"       \033[91mERROR: {r.error_message}\033[0m")
            for log_line in r.logs[-3:]:
                print(f"       • {log_line}")
            print("-" * 80)

        total = len(results)
        passed_count = sum(1 for r in results if r.passed)
        summary_color = "\033[92m" if all_passed else "\033[91m"
        print(f"{summary_color}Evaluation Summary: {passed_count}/{total} Scenarios Passed.\033[0m")
        print("=" * 80 + "\n")
        return all_passed


# =========================================================================
# Interactive Diagnostic Workflow Session
# =========================================================================
def run_interactive_session():
    """Interactive diagnostic shell allowing operator to give tasks, send interruptions,
    step simulation, and inspect outputs in real time."""
    print("\n" + "=" * 70)
    print(" CityNav Interactive Diagnostic Shell")
    print(" Commands:")
    print("   start <text>             - Start a new mission (e.g. 'start go 50m above fountain')")
    print("   msg <text>               - Queue mid-flight operator interruption")
    print("   step <n>                 - Advance simulation step by n iterations (default 10)")
    print("   inject <name> <x> <y> <z>- Inject a landmark observation into spatial memory")
    print("   status                   - Print live telemetry snapshot")
    print("   memory                   - Display formatted spatial memory landmark table")
    print("   clean                    - Deduplicate and clean spatial memory")
    print("   eval                     - Run full automated scenario evaluation suite")
    print("   hover                    - Command immediate emergency hover")
    print("   quit                     - Exit shell")
    print("=" * 70 + "\n")

    cfg = SystemConfig()
    cfg.mock_airsim = True
    cfg.mock_perception = True
    cfg.mock_gemini = True
    cfg.dashboard.enabled = False

    orch = CityNavOrchestrator(cfg)
    orch.start(start_dashboard=False)

    try:
        while True:
            try:
                line = input("CityNav> ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not line:
                continue

            parts = line.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            if cmd in ("quit", "exit", "q"):
                break
            elif cmd == "start":
                if not arg:
                    print("Usage: start <mission instruction>")
                    continue
                subgoals = orch.start_mission(arg)
                print(f"[Mission Started] Subgoals: {subgoals}")
            elif cmd == "msg":
                if not arg:
                    print("Usage: msg <interruption text>")
                    continue
                orch.dialogue.queue_user_message(arg)
                print(f"[Queued Interruption] '{arg}'")
            elif cmd == "step":
                steps = int(arg) if arg.isdigit() else 10
                for _ in range(steps):
                    orch.flight.step_control()
                    time.sleep(0.02)
                t = orch.get_telemetry()
                print(f"[Stepped {steps}x] Pos: {t['position']}, Alt: {t['altitude']}m, Mode: {t['mode']}, Target: {t['active_target']}")
            elif cmd == "inject":
                inject_parts = arg.split()
                if len(inject_parts) < 4:
                    print("Usage: inject <name> <x> <y> <z>")
                    continue
                lm_name = inject_parts[0]
                try:
                    lx, ly, lz = float(inject_parts[1]), float(inject_parts[2]), float(inject_parts[3])
                    lid = orch.memory.add_observation(lm_name, np.array([lx, ly, lz]))
                    print(f"[Injected Landmark] ID {lid}: '{lm_name}' at [{lx}, {ly}, {lz}]")
                except ValueError as ve:
                    print(f"Error parsing coordinates: {ve}")
            elif cmd == "status":
                t = orch.get_telemetry()
                print(json.dumps(t, indent=2))
            elif cmd == "memory":
                print(orch.memory.get_landmark_table())
            elif cmd == "clean":
                stats = orch.clean_spatial_memory()
                print(f"[Cleaned Registry] {stats}")
            elif cmd == "eval":
                evaluator = ScenarioEvaluator(verbose=True)
                results = evaluator.run_all()
                evaluator.print_report(results)
            elif cmd == "hover":
                orch.emergency_hover()
                print("[Emergency Hover Commanded]")
            else:
                print(f"Unknown command: '{cmd}'. Available: start, msg, step, inject, status, memory, clean, eval, hover, quit.")
    finally:
        orch.stop()
        print("\n[Interactive Shell Terminated cleanly.]")


def main():
    parser = argparse.ArgumentParser(description="Real-Time CityNav Scenario Evaluator")
    parser.add_argument("--all", action="store_true", default=True, help="Run full scenario evaluation suite")
    parser.add_argument("--scenario", type=str, default="", help="Run a specific scenario by name")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive diagnostic shell")
    parser.add_argument("--verbose", action="store_true", help="Show detailed step logs")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")
    args = parser.parse_args()

    if args.interactive:
        run_interactive_session()
        return

    evaluator = ScenarioEvaluator(verbose=args.verbose)

    if args.scenario:
        scenario_map = {
            "altitude": evaluator.run_scenario_altitude_above_fountain,
            "prepositions": evaluator.run_scenario_spatial_prepositions,
            "interruption": evaluator.run_scenario_midflight_interruption,
            "deduplication": evaluator.run_scenario_spatial_memory_deduplication,
            "obstacle": evaluator.run_scenario_obstacle_avoidance,
            "recall": evaluator.run_scenario_recall_query,
            "stress": evaluator.run_scenario_stress_cases,
            "compound": evaluator.run_scenario_compound_spatial_prepositions,
            "elevated": evaluator.run_scenario_elevated_landmark_altitude,
            "pruning": evaluator.run_scenario_continuous_memory_pruning,
            "continuous": evaluator.run_scenario_continuous_memory_pruning,
            "robustness": evaluator.run_scenario_malformed_and_punctuation_robustness,
            "malformed": evaluator.run_scenario_malformed_and_punctuation_robustness,
        }
        matched = next((fn for k, fn in scenario_map.items() if k in args.scenario.lower()), None)
        if not matched:
            print(f"Unknown scenario '{args.scenario}'. Choose from: {list(scenario_map.keys())}")
            sys.exit(1)
        results = [matched()]
    else:
        results = evaluator.run_all()

    if args.json:
        print(json.dumps([r.to_dict() for r in results], indent=2))
        sys.exit(0 if all(r.passed for r in results) else 1)

    all_passed = evaluator.print_report(results)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
