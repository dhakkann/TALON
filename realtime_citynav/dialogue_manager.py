"""Asynchronous Dialogue Manager and Gemini 2.0 Flash Operator Interruption Engine."""

import os
import re
import json
import math
import queue
from typing import Dict, List, Optional, Any
import numpy as np

try:
    from google import genai
    from google.genai import types
    _GENAI_AVAILABLE = True
except ImportError:
    _GENAI_AVAILABLE = False


class DialogueManager:
    """Non-blocking operator dialogue and command interruption manager.
    
    Uses Gemini 2.0 Flash for semantic subgoal decomposition and mid-flight
    command classification with a resilient heuristic fallback engine.
    """

    KNOWN_DEFAULT_LANDMARKS = [
        "fountain",
        "red building",
        "blue building",
        "tower",
        "hospital",
    ]

    KNOWN_LANDMARK_CENTERS = {
        "fountain": np.array([32.0, 0.0, 0.0]),
        "red building": np.array([75.0, 12.0, 0.0]),
        "blue building": np.array([75.0, -14.0, 0.0]),
        "tower": np.array([130.0, -10.0, 0.0]),
        "hospital": np.array([60.0, -30.0, 0.0]),
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-3.5-flash-lite",
        mock: bool = False
    ):
        self.mock = mock
        if mock or api_key is None or api_key == "":
            self.api_key = ""
        else:
            self.api_key = api_key

        self.model_name = model_name
        self.command_queue: queue.Queue[str] = queue.Queue()
        self.client = None

        if not self.mock and _GENAI_AVAILABLE and self.api_key:
            try:
                self.client = genai.Client(api_key=self.api_key)
            except Exception as e:
                print(f"[DialogueManager] Could not initialize Gemini client: {e}. Falling back to heuristic mode.")
                self.client = None

    def _generate_with_gemini(self, prompt: str, response_mime_type: str = "application/json") -> str:
        """Call Gemini model with automatic resilient fallback across supported flash models."""
        candidates = [self.model_name, "gemini-3.5-flash-lite", "gemini-3.5-flash", "gemini-2.5-flash-lite", "gemini-2.5-flash"]
        seen = set()
        models_to_try = [m for m in candidates if not (m in seen or seen.add(m))]

        last_err = None
        for m in models_to_try:
            try:
                cfg = types.GenerateContentConfig(response_mime_type=response_mime_type) if _GENAI_AVAILABLE else None
                response = self.client.models.generate_content(
                    model=m,
                    contents=prompt,
                    config=cfg
                )
                self.model_name = m  # Lock in working model
                raw_text = response.text.strip()
                if raw_text.startswith("```"):
                    raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
                    raw_text = re.sub(r"\s*```$", "", raw_text)
                return raw_text.strip()
            except Exception as e:
                last_err = e
                err_str = str(e).lower()
                if "not_found" in err_str or "404" in err_str or "no longer available" in err_str or "not supported" in err_str:
                    continue
                raise e
        if last_err:
            raise last_err
        return ""

    def queue_user_message(self, text: str) -> None:
        """Enqueue an operator message without blocking active flight control."""
        if text and text.strip():
            self.command_queue.put(text.strip())

    def has_pending_commands(self) -> bool:
        """Check if any operator commands are waiting in queue."""
        return not self.command_queue.empty()

    def parse_initial_instruction(self, instruction: str) -> List[str]:
        """Decompose natural language route instruction into ordered landmark subgoals.
        
        Args:
            instruction: Full instruction string (e.g., 'Fly to the fountain, then the red building').
            
        Returns:
            Ordered list of landmark subgoals (e.g., ['fountain', 'red building']).
        """
        if not instruction or not instruction.strip():
            return []

        # Check if instruction is a multi-step sequence with transition keywords or multiple landmarks
        has_transition = bool(re.search(r'\b(?:then|next|after that|afterwards|finally|followed by|and then|, and)\b', instruction, re.IGNORECASE))
        lower_inst = instruction.lower()
        lm_matches = sum(1 for lm in self.KNOWN_DEFAULT_LANDMARKS if lm in lower_inst or any(w in lower_inst.split() for w in lm.split()))

        # Fast single orbit command check (only if not a multi-step sequence)
        if not has_transition and lm_matches <= 1 and self.is_orbit_command(instruction):
            orbit_info = self.parse_orbit_command(instruction)
            if orbit_info and orbit_info.get("target"):
                rev_part = f" {int(orbit_info['revolutions'])} times" if orbit_info.get("revolutions") else ""
                facing_part = " facing forward" if orbit_info.get("facing") == "TANGENT" else ""
                return [f"circle around {orbit_info['target']}{rev_part}{facing_part}".strip()]

        # Attempt Gemini Flash decomposition if client available
        if self.client is not None:
            try:
                known_str = ", ".join(self.KNOWN_DEFAULT_LANDMARKS)
                prompt = (
                    "You are the navigation planner for an autonomous urban drone. "
                    "Extract the ordered sequence of physical landmark subgoals from this instruction. "
                    "Resolve shorthand adjectives (e.g. 'red' -> 'red building') to full landmark names. "
                    f"Known environment landmarks: {known_str}. "
                    "If the instruction requests visiting all or every landmark (e.g. 'go to every landmark', 'visit all landmarks', 'tour the city', 'explore each landmark', 'visit all buildings'), expand this into the full ordered list of all known environment landmarks: [\"fountain\", \"red building\", \"blue building\", \"tower\", \"hospital\"]. "
                    "If an instruction specifies a spatial relation or altitude offset (e.g. 'hover 100m above red building', 'go 50m above fountain', '20m in front of tower'), PRESERVE the spatial relation and numeric distance in the subgoal name (e.g. '100m above red building', '50m above fountain', '20m in front of tower'). Do NOT strip altitude or prepositions. "
                    "If the instruction specifies an orbit or circular inspection around a landmark (e.g. 'hover in circle around blue building', 'circle twice around red building', 'orbit tower'), PRESERVE the orbit instruction in the subgoal name (e.g. 'circle around blue building'). "
                    "Return ONLY a JSON object formatted as: {\"subgoals\": [\"<landmark1>\", \"<landmark2>\"]}\n\n"
                    f"Instruction: \"{instruction}\""
                )
                raw_text = self._generate_with_gemini(prompt, response_mime_type="application/json")
                data = json.loads(raw_text)
                subgoals = data.get("subgoals", [])
                if isinstance(subgoals, list) and subgoals:
                    resolved_subgoals = []
                    for s in subgoals:
                        res_s = self.resolve_landmark_shorthand(str(s).strip(), resolved_subgoals)
                        if res_s and (not resolved_subgoals or resolved_subgoals[-1] != res_s):
                            resolved_subgoals.append(res_s)

                    # Deterministic safeguard: if raw instruction specifies an orbit command or spatial relations that were stripped by LLM, restore/enrich them
                    if self.is_orbit_command(instruction):
                        heuristic_subgoals = self._heuristic_parse_instruction(instruction)
                        if any(self.is_orbit_command(hs) for hs in heuristic_subgoals):
                            resolved_subgoals = heuristic_subgoals
                    elif len(resolved_subgoals) == 1:
                        spatial = self.parse_spatial_relation(instruction)
                        if spatial.get("relation") != "DEFAULT" and spatial.get("base_landmark"):
                            if spatial["base_landmark"].lower() == self.extract_base_landmark(resolved_subgoals[0]).lower():
                                clean_inst = re.sub(r'^(?:hover|fly|go|climb|ascend|descend)\s+(?:at\s+|to\s+)?', '', instruction, flags=re.IGNORECASE).strip()
                                resolved_subgoals = [self.resolve_landmark_shorthand(clean_inst)]

                    if resolved_subgoals:
                        return resolved_subgoals
            except Exception as e:
                print(f"[DialogueManager] Gemini instruction parsing failed: {e}. Using heuristic fallback.")

        # Heuristic fallback parser
        return self._heuristic_parse_instruction(instruction)

    def process_interruption(
        self,
        current_target: Optional[str] = None,
        memory_summary: str = ""
    ) -> Optional[Dict[str, Any]]:
        """Process the next operator interruption from queue (non-blocking).
        
        Classifies intent into:
        - TARGET_CORRECTION: Operator changes current target
        - RETURN_TO_PAST: Operator requests navigating back to a prior landmark
        - QUERY: Operator queries drone memory or status
        - HOVER_STOP: Operator commands vehicle to halt or hover
        
        Returns:
            Dict containing 'intent', 'target', and 'reply', or None if queue is empty.
        """
        if self.command_queue.empty():
            return None

        msg = self.command_queue.get_nowait()
        cur_target_str = current_target or "None"

        # Guard against empty messages or messages containing only punctuation (e.g. "...", "???")
        alphanumeric = re.sub(r'[^\w\s]', '', msg).strip()
        if not alphanumeric:
            return {
                "intent": "QUERY",
                "target": None,
                "reply": "No clear command received. Please specify a destination, query, or hover.",
                "raw_message": msg
            }

        # Check for HOVER_STOP first (e.g. "stop and hover right here", "halt")
        if self.is_hover_stop_command(msg):
            return {
                "intent": "HOVER_STOP",
                "target": None,
                "reply": "Holding position and hovering.",
                "raw_message": msg
            }

        # Check for explicit 3D waypoint coordinate path [[x, y, z], ...]
        if self.is_explicit_waypoint_path(msg):
            wp_match = re.findall(r'\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]', msg)
            wps = [[float(x), float(y), float(z)] for x, y, z in wp_match]
            facing = "INWARD" if "inward" in msg.lower() else "TANGENT"
            return {
                "intent": "WAYPOINT_PATH",
                "waypoints": wps,
                "speed": 3.0,
                "facing": facing,
                "target": "waypoint_curve",
                "reply": f"Executing continuous 3D waypoint curve ({len(wps)} points).",
                "raw_message": msg
            }

        # Check if operator message is a compound multi-subgoal route/mission
        subgoals = self.parse_initial_instruction(msg)
        if len(subgoals) > 1:
            return {
                "intent": "MULTI_SUBGOAL_MISSION",
                "subgoals": subgoals,
                "instruction": msg,
                "reply": f"Understood. Starting multi-step mission ({len(subgoals)} subgoals): {' -> '.join(subgoals)}.",
                "raw_message": msg
            }

        # Attempt Gemini Flash classification if client available
        if self.client is not None:
            try:
                prompt = (
                    "An operator interrupted an autonomous drone mid-flight.\n"
                    f"Active Target: \"{cur_target_str}\"\n"
                    f"Known Landmarks in Spatial Memory:\n{memory_summary}\n\n"
                    f"Operator message: \"{msg}\"\n\n"
                    "Classify this operator command into exactly one intent:\n"
                    "1. \"ORBIT_CIRCLE\": Operator commands circling, orbiting, or revolving around a landmark (e.g. 'hover in circle around blue building', 'circle twice around red building', 'orbit tower facing forward').\n"
                    "2. \"WAYPOINT_PATH\": Operator provides explicit 3D waypoint coordinate sequences or asks for a discrete waypoint curve path (e.g. 'follow path [[10, 20, -5], ...]').\n"
                    "3. \"TARGET_CORRECTION\": Divert flight to a newly specified target or spatial waypoint (e.g. '100m above red building', 'fountain', 'in front of tower').\n"
                    "4. \"RETURN_TO_PAST\": Return to a previously registered landmark in memory.\n"
                    "5. \"QUERY\": Inquire about landmarks, surroundings, or status.\n"
                    "6. \"HOVER_STOP\": Pause, hover in place, or stop navigation (ONLY when no new destination, spatial waypoint, orbit, or landmark is specified).\n\n"
                    "If the operator specifies a spatial offset, altitude, or preposition (e.g. 'hover 100m above red building', 'go 20m in front of tower'), PRESERVE the spatial relation and numeric distance in 'target' (e.g. '100m above red building') and classify as TARGET_CORRECTION.\n\n"
                    "Respond with JSON:\n"
                    "{\n"
                    "  \"intent\": \"ORBIT_CIRCLE\" | \"WAYPOINT_PATH\" | \"TARGET_CORRECTION\" | \"RETURN_TO_PAST\" | \"QUERY\" | \"HOVER_STOP\",\n"
                    "  \"target\": \"<target landmark name or spatial waypoint or null>\",\n"
                    "  \"radius\": <float or null>,\n"
                    "  \"altitude\": <float or null>,\n"
                    "  \"speed\": <float or null>,\n"
                    "  \"facing\": \"INWARD\" | \"TANGENT\",\n"
                    "  \"revolutions\": <float or null>,\n"
                    "  \"direction\": \"CCW\" | \"CW\",\n"
                    "  \"waypoints\": <list of [x, y, z] or null>,\n"
                    "  \"reply\": \"<concise confirmation to operator>\"\n"
                    "}"
                )
                raw_text = self._generate_with_gemini(prompt, response_mime_type="application/json")
                data = json.loads(raw_text)
                data["raw_message"] = msg

                if data.get("intent") == "ORBIT_CIRCLE" or self.is_orbit_command(msg):
                    orbit_info = self.parse_orbit_command(msg)
                    if orbit_info and orbit_info.get("target"):
                        data["intent"] = "ORBIT_CIRCLE"
                        data["target"] = orbit_info["target"]
                        data["base_landmark"] = orbit_info["base_landmark"]
                        data["radius"] = data.get("radius") or orbit_info["radius"]
                        data["altitude"] = data.get("altitude") or orbit_info["altitude"]
                        data["speed"] = data.get("speed") or orbit_info["speed"]
                        data["facing"] = data.get("facing") or orbit_info["facing"]
                        data["revolutions"] = data.get("revolutions") if "revolutions" in data else orbit_info["revolutions"]
                        data["direction"] = data.get("direction") or orbit_info["direction"]
                        data["waypoints"] = orbit_info.get("waypoints")
                        if not data.get("reply"):
                            rev_str = f" for {orbit_info['revolutions']:.0f} revolutions" if orbit_info['revolutions'] else " continuously"
                            data["reply"] = f"Orbiting {orbit_info['target']}{rev_str} at {data['radius']:.1f}m radius, {data['altitude']:.1f}m altitude."
                        return data

                raw_spatial = self.parse_spatial_relation(msg)
                tgt = data.get("target") or msg
                resolved_tgt = self.resolve_landmark_shorthand(tgt)

                # Deterministic safeguard: if raw message had a spatial constraint that was stripped, restore it
                if raw_spatial.get("relation") != "DEFAULT" and raw_spatial.get("base_landmark"):
                    if raw_spatial["base_landmark"].lower() == self.extract_base_landmark(resolved_tgt).lower():
                        data["intent"] = "TARGET_CORRECTION"
                        clean_msg = re.sub(r'^(?:hover|fly|go|climb|ascend|descend)\s+(?:at\s+|to\s+)?', '', msg, flags=re.IGNORECASE).strip()
                        resolved_tgt = self.resolve_landmark_shorthand(clean_msg)

                data["target"] = resolved_tgt
                data["spatial"] = self.parse_spatial_relation(resolved_tgt)
                if data["spatial"].get("base_landmark"):
                    data["base_landmark"] = data["spatial"]["base_landmark"]
                return data
            except Exception as e:
                print(f"[DialogueManager] Gemini interruption handling failed: {e}. Using heuristic fallback.")

        # Heuristic fallback classifier
        return self._heuristic_classify_interruption(msg, cur_target_str, memory_summary)

    @classmethod
    def is_orbit_command(cls, text: str) -> bool:
        """Detect if natural language instruction specifies a circular orbit or curve maneuver."""
        if not text or not text.strip():
            return False
        lower = text.lower().strip()
        patterns = [
            r'\b(?:circle|orbit|circling|orbiting|revolve|revolving)\b',
            r'\b(?:in\s+a?\s*circle|in\s+circles)\b',
            r'\b(?:circular\s+(?:path|trajectory|curve|pattern|orbit|route))\b',
            r'\b(?:continuous\s+curve)\b',
            r'\b(?:laps?|turns?)\s+around\b',
            r'\baround\s+[\w\s]+\s+\d+\s*(?:times|revolutions?|laps?|turns?)\b',
            r'\baround\s+(?:the\s+|a\s+|an\s+)?(?:fountain|building|tower|hospital|red|blue)\b',
        ]
        return any(re.search(pat, lower) for pat in patterns)

    @classmethod
    def is_hover_stop_command(cls, text: str) -> bool:
        """Detect if natural language instruction specifies a halt/hover stop command without a landmark."""
        if not text or not text.strip():
            return False
        lower = text.lower().strip()
        if not any(w in lower for w in ["stop", "hover", "halt", "pause", "wait"]):
            return False
        if cls.is_orbit_command(text) or any(w in lower for w in ["circle", "orbit", "revolve", "curve", "waypoints"]):
            return False
        if re.search(r'\[\s*-?\d+', text):
            return False
        spatial = cls.parse_spatial_relation(text)
        if spatial.get("relation") != "DEFAULT" and spatial.get("base_landmark"):
            return False
        if any(lm in lower for lm in cls.KNOWN_DEFAULT_LANDMARKS):
            return False
        if any(w in lower for w in ["fountain", "building", "tower", "hospital", "tree", "car", "statue", "bench", "bridge", "lamp"]):
            return False
        return True

    @classmethod
    def is_explicit_waypoint_path(cls, text: str) -> bool:
        """Detect if instruction contains explicit 3D waypoint coordinate array [[x, y, z], ...]."""
        if not text:
            return False
        matches = re.findall(r'\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]', text)
        return len(matches) >= 2

    @classmethod
    def generate_orbit_waypoints(
        cls,
        center: np.ndarray,
        radius: float = 16.0,
        altitude: float = 10.0,
        num_points: int = 16,
        direction: str = "CCW"
    ) -> List[List[float]]:
        """Generate 3D waypoint coordinate sequence along circular curve.
        
        NED convention: Z is -altitude.
        """
        xc, yc = float(center[0]), float(center[1])
        target_z = -float(altitude)
        points = []
        dir_mult = 1.0 if direction.upper() in ("CCW", "COUNTER_CLOCKWISE") else -1.0
        for i in range(num_points):
            theta = dir_mult * (2.0 * math.pi * i / num_points)
            x = round(xc + radius * math.cos(theta), 2)
            y = round(yc + radius * math.sin(theta), 2)
            points.append([x, y, round(target_z, 2)])
        return points

    @classmethod
    def parse_orbit_command(cls, text: str) -> Optional[Dict[str, Any]]:
        """Parse structured parameters from natural language orbit/circling command.
        
        Supports:
        - Landmark: 'blue building', 'red building', shorthand 'blue', etc.
        - Facing: 'INWARD' (default for landmark inspection) or 'TANGENT' ('facing forward', 'looking ahead')
        - Radius: specified in meters or default 16.0m
        - Altitude: specified in meters or default 10.0m
        - Speed: specified in m/s or default 3.0 m/s
        - Revolutions: specified (e.g. 'twice' -> 2, '3 times' -> 3) or None (continuous indefinite loop)
        - Direction: 'CCW' (default) or 'CW'
        - Waypoints: optional discretized 3D coordinate sequence
        """
        if not text or not cls.is_orbit_command(text):
            return None

        lower = text.lower().strip()

        # 1. Facing: 'TANGENT' vs 'INWARD'
        if re.search(r'\b(?:facing\s+forward|looking\s+ahead|look\s+ahead|tangent\s+facing|facing\s+tangent|tangent)\b', lower):
            facing = "TANGENT"
        else:
            facing = "INWARD"

        # 2. Revolutions count
        revolutions: Optional[float] = None
        if re.search(r'\b(?:once|1\s*time|1\s*lap|1\s*revolution|single\s*lap)\b', lower):
            revolutions = 1.0
        elif re.search(r'\b(?:twice|2\s*times|2\s*laps|2\s*revolutions|two\s*laps|two\s*times)\b', lower):
            revolutions = 2.0
        elif re.search(r'\b(?:thrice|3\s*times|3\s*laps|3\s*revolutions|three\s*laps|three\s*times)\b', lower):
            revolutions = 3.0
        else:
            m_rev = re.search(r'(\d+(?:\.\d+)?)\s*(?:times|revolutions?|laps?|turns?)', lower)
            if m_rev:
                revolutions = float(m_rev.group(1))

        # 3. Radius
        radius = 16.0
        m_rad = re.search(r'\b(?:radius|r)\s*(?:of|=|:)?\s*(\d+(?:\.\d+)?)\s*(?:m|meters?)?\b', lower)
        if not m_rad:
            m_rad = re.search(r'(\d+(?:\.\d+)?)\s*(?:m|meters?)?\s*radius\b', lower)
        if m_rad:
            radius = float(m_rad.group(1))

        # 4. Altitude
        altitude = 10.0
        m_alt = re.search(r'\b(?:altitude|alt|height)\s*(?:of|=|:)?\s*(\d+(?:\.\d+)?)\s*(?:m|meters?)?\b', lower)
        if not m_alt:
            m_alt = re.search(r'(\d+(?:\.\d+)?)\s*(?:m|meters?)?\s*(?:altitude|height)\b', lower)
        if not m_alt:
            m_alt = re.search(r'\bat\s+(\d+(?:\.\d+)?)\s*(?:m|meters?)\s+(?:altitude|height)\b', lower)
        if not m_alt:
            m_at = re.search(r'\bat\s+(\d+(?:\.\d+)?)\s*(?:m|meters?)\b(?!\s*radius)(?!\s*/\s*s)(?!\s*ps)\b', lower)
            if m_at and (not m_rad or float(m_at.group(1)) != radius):
                altitude = float(m_at.group(1))
        else:
            altitude = float(m_alt.group(1))

        # 5. Speed
        speed = 3.0
        m_spd = re.search(r'\b(?:speed|velocity)\s*(?:of|=|:)?\s*(\d+(?:\.\d+)?)\s*(?:m/s|mps)?\b', lower)
        if not m_spd:
            m_spd = re.search(r'(\d+(?:\.\d+)?)\s*(?:m/s|mps)\b', lower)
        if m_spd:
            speed = float(m_spd.group(1))

        # 6. Direction: CW vs CCW
        if re.search(r'\b(?:clockwise|cw)\b', lower) and not re.search(r'\b(?:counter-clockwise|counterclockwise|ccw|anticlockwise)\b', lower):
            direction = "CW"
        else:
            direction = "CCW"

        # 7. Landmark Extraction
        matched_target = None
        cands = []
        for k in cls.KNOWN_DEFAULT_LANDMARKS:
            idx = lower.find(k)
            if idx != -1:
                cands.append((idx, k))
        if cands:
            cands.sort(key=lambda t: t[0])
            matched_target = cands[0][1]

        if not matched_target:
            short_cands = []
            for k in ["blue", "red", "fountain", "tower", "hospital"]:
                m = re.search(r'\b' + k + r'\b', lower)
                if m:
                    short_cands.append((m.start(), cls.resolve_landmark_shorthand(k)))
            if short_cands:
                short_cands.sort(key=lambda t: t[0])
                matched_target = short_cands[0][1]

        if not matched_target:
            m_target = re.search(r'\b(?:around|round|orbiting|circling|circle|orbit)\s+(?:the\s+|a\s+|an\s+)?([a-zA-Z0-9_\s-]+?)(?=\s+(?:at|with|facing|looking|twice|once|thrice|\d+\s*(?:times|revolutions?|laps?|m|meters?|radius|altitude|speed)|clockwise|counter-clockwise|ccw|cw)|$)', lower)
            if m_target:
                cand = cls.clean_landmark_name(m_target.group(1))
                matched_target = cls.resolve_landmark_shorthand(cand)
            else:
                matched_target = "blue building"

        base_landmark = cls.extract_base_landmark(matched_target)

        # 8. Coordinate waypoints generation
        waypoints = None
        center_coords = None
        for k, coords in cls.KNOWN_LANDMARK_CENTERS.items():
            if k in base_landmark.lower() or base_landmark.lower() in k:
                center_coords = coords
                break
        if center_coords is not None:
            waypoints = cls.generate_orbit_waypoints(
                center=center_coords,
                radius=radius,
                altitude=altitude,
                num_points=16,
                direction=direction
            )

        requested_coordinates = any(w in lower for w in ["coordinate", "coords", "path", "curve", "waypoints"])

        return {
            "target": matched_target,
            "base_landmark": base_landmark,
            "radius": radius,
            "altitude": altitude,
            "speed": speed,
            "facing": facing,
            "revolutions": revolutions,
            "direction": direction,
            "waypoints": waypoints,
            "requested_coordinates": requested_coordinates,
            "raw": text
        }

    @classmethod
    def clean_landmark_name(cls, s: str) -> str:
        """Strip command verbs, prepositions, articles, and filler words from landmark phrases."""
        s = s.strip()
        s = re.sub(
            r'^(?:fly\s+to|navigate\s+to|head\s+towards?|go\s+to|proceed\s+to|ascend\s+to|climb\s+to|hover\s+at|hover\s+over|hover\s+around|hover\s+in\s+(?:a\s+)?circle\s+around|circle\s+around|circle|orbit\s+around|orbit|hover|stop\s+at|stop\s+near|pass|reach|find|locate|search\s+for)\s+',
            '', s, flags=re.IGNORECASE
        ).strip()
        s = re.sub(r'^(?:also\s+)?(?:go|fly|head|proceed|hover|stop|circle|orbit)\s+(?:from|to|at|near|over|around)?\s*', '', s, flags=re.IGNORECASE).strip()
        s = re.sub(r'^(?:from|to|around|near)\s+', '', s, flags=re.IGNORECASE).strip()
        s = re.sub(r'^(?:the|a|an)\s+', '', s, flags=re.IGNORECASE).strip()
        s = re.sub(r'\s+ones?$', '', s, flags=re.IGNORECASE).strip()
        s = re.sub(r'[.,;:]+$', '', s).strip()
        return s

    @classmethod
    def resolve_landmark_shorthand(
        cls,
        name: str,
        context_landmarks: Optional[List[str]] = None
    ) -> str:
        """Resolve shorthand adjectives or elliptical references to full landmark names.
        
        Examples:
            'red' -> 'red building'
            'the red one' -> 'red building'
            'blue' -> 'blue building'
            '50m above red' -> '50m above red building'
        """
        if not name or not name.strip():
            return name or ""

        cleaned = cls.clean_landmark_name(name)
        if not cleaned:
            return name.strip()

        # Check for spatial relation in cleaned phrase (e.g. '50m above red', '15m behind blue')
        m_spatial = re.match(
            r'^(?:(?:at\s+)?(?:an\s+altitude\s+of\s+)?\d+(?:\.\d+)?\s*(?:m|meter|meters)?\s*(?:altitude\s+)?(?:above|over)\s+(?:and\s+|,\s*|\s*with\s+)?(?:(?:\d+(?:\.\d+)?\s*(?:m|meter|meters)?\s+)?(?:in\s+front(?:\s+of)?|ahead(?:\s+of)?|behind|in\s+back(?:\s+of)?|to\s+the\s+left(?:\s+of)?|left(?:\s+of)?|to\s+the\s+right(?:\s+of)?|right(?:\s+of)?)\s+)?(?:the\s+|a\s+|an\s+)?)(.+)$',
            cleaned, re.IGNORECASE
        )
        if not m_spatial:
            m_spatial = re.match(
                r'^(?:(?:\d+(?:\.\d+)?\s*(?:m|meter|meters)?\s+)?(?:above|over|in\s+front(?:\s+of)?|ahead(?:\s+of)?|behind|in\s+back(?:\s+of)?|to\s+the\s+left(?:\s+of)?|left(?:\s+of)?|to\s+the\s+right(?:\s+of)?|right(?:\s+of)?)\s+(?:the\s+|a\s+|an\s+)?)(.+)$',
                cleaned, re.IGNORECASE
            )
        if m_spatial:
            base = m_spatial.group(1).strip()
            resolved_base = cls.resolve_landmark_shorthand(base, context_landmarks)
            prefix_str = cleaned[:len(cleaned) - len(base)].strip()
            return f"{prefix_str} {resolved_base}".strip()

        low = cleaned.lower()
        all_known = list(dict.fromkeys((context_landmarks or []) + cls.KNOWN_DEFAULT_LANDMARKS))

        for k in all_known:
            if low == k.lower():
                return k

        matches = []
        for k in all_known:
            words = k.lower().split()
            if low in words or low in k.lower():
                matches.append(k)

        if len(matches) == 1:
            return matches[0]
        elif len(matches) > 1:
            first_word_matches = [k for k in matches if k.lower().startswith(low)]
            if len(first_word_matches) == 1:
                return first_word_matches[0]
            return matches[0]

        return cleaned

    @classmethod
    def parse_spatial_relation(cls, text: str) -> Dict[str, Any]:
        """Parse spatial relation, distance offset, and base landmark from natural language.
        
        Supports:
        - Compound prepositions (vertical + horizontal):
            e.g. 'go 20m above and 15m behind the red building', '10m to the left and 25m above tower'
        - Vertical only:
            'above' / 'over' / 'at X altitude': e.g., 'go 50m above fountain', 'fly 20m over red building'
        - Horizontal only:
            - 'in front of' / 'ahead of': e.g., 'go 15m in front of fountain'
            - 'behind' / 'in back of': e.g., 'go 10m behind hospital'
            - 'to the left of' / 'left of': e.g., 'go 20m to the left of blue building'
            - 'to the right of' / 'right of': e.g., 'go 25m to the right of tower'
        - Default: 'fountain', 'blue building'
        
        Returns:
            Dict containing:
                'relation': 'ABOVE' | 'IN_FRONT' | 'BEHIND' | 'LEFT' | 'RIGHT' | 'COMPOUND' | 'DEFAULT'
                'distance': float or None (in meters)
                'altitude_offset': float or None (in meters)
                'horizontal_relation': 'IN_FRONT' | 'BEHIND' | 'LEFT' | 'RIGHT' | None
                'horizontal_distance': float or None (in meters)
                'base_landmark': cleaned landmark name (e.g. 'fountain', 'red building')
                'raw': original text
        """
        if not text or not text.strip():
            return {
                "relation": "DEFAULT",
                "distance": None,
                "altitude_offset": None,
                "horizontal_relation": None,
                "horizontal_distance": None,
                "base_landmark": "",
                "raw": text or ""
            }

        raw = text.strip()
        cleaned = raw

        # Strip leading action verbs and conversational prefixes
        cleaned = re.sub(
            r'^(?:please\s+)?(?:fly|navigate|head|go|proceed|move|climb|ascend|descend|hover)\s+(?:to\s+|towards\s+)?',
            '', cleaned, flags=re.IGNORECASE
        ).strip()
        cleaned = re.sub(
            r'^(?:no,?|actually,?|instead,?|switch\s+to|change\s+to)\s+',
            '', cleaned, flags=re.IGNORECASE
        ).strip()

        # Helper to classify horizontal direction keyword
        def _classify_h_dir(h_str: str) -> str:
            h_low = h_str.lower()
            if "front" in h_low or "ahead" in h_low:
                return "IN_FRONT"
            elif "behind" in h_low or "back" in h_low:
                return "BEHIND"
            elif "left" in h_low:
                return "LEFT"
            else:
                return "RIGHT"

        def _resolve_lm(raw_lm: str) -> str:
            cleaned_raw = re.sub(r'[.,;:]+$', '', raw_lm).strip()
            return cls.resolve_landmark_shorthand(cleaned_raw)

        # --- COMPOUND PATTERNS ---
        # Pattern 1: <vert> (and|,|with) <horiz> <landmark>
        # e.g., '20m above and 15m behind the red building', 'above and 10m to the left of tower'
        m_vert_horiz = re.match(
            r'^(?:at\s+)?(?:an\s+altitude\s+of\s+)?(\d+(?:\.\d+)?)\s*(?:m|meter|meters)?\s*(?:altitude\s+)?(?:above|over)\s+(?:and\s+|,\s*|\s*with\s+)?'
            r'(?:(\d+(?:\.\d+)?)\s*(?:m|meter|meters)?\s+)?(in\s+front(?:\s+of)?|ahead(?:\s+of)?|behind|in\s+back(?:\s+of)?|to\s+the\s+left(?:\s+of)?|left(?:\s+of)?|to\s+the\s+right(?:\s+of)?|right(?:\s+of)?)\s+(?:the\s+|a\s+|an\s+)?(.+)$',
            cleaned, re.IGNORECASE
        )
        if m_vert_horiz:
            alt = float(m_vert_horiz.group(1))
            h_dist = float(m_vert_horiz.group(2)) if m_vert_horiz.group(2) else None
            h_rel = _classify_h_dir(m_vert_horiz.group(3))
            lm = _resolve_lm(m_vert_horiz.group(4))
            return {
                "relation": "COMPOUND",
                "distance": alt,
                "altitude_offset": alt,
                "horizontal_relation": h_rel,
                "horizontal_distance": h_dist,
                "base_landmark": lm,
                "raw": raw
            }

        # Pattern 2: <horiz> (and|,|with) <vert> <landmark>
        # e.g., '15m behind and 20m above the red building', '10m to the left and 25m above tower'
        m_horiz_vert = re.match(
            r'^(?:(\d+(?:\.\d+)?)\s*(?:m|meter|meters)?\s+)?(in\s+front(?:\s+of)?|ahead(?:\s+of)?|behind|in\s+back(?:\s+of)?|to\s+the\s+left(?:\s+of)?|left(?:\s+of)?|to\s+the\s+right(?:\s+of)?|right(?:\s+of)?)\s+(?:and\s+|,\s*|\s*with\s+)?'
            r'(?:at\s+)?(?:an\s+altitude\s+of\s+)?(\d+(?:\.\d+)?)\s*(?:m|meter|meters)?\s*(?:altitude\s+)?(?:above|over)\s+(?:the\s+|a\s+|an\s+)?(.+)$',
            cleaned, re.IGNORECASE
        )
        if m_horiz_vert:
            h_dist = float(m_horiz_vert.group(1)) if m_horiz_vert.group(1) else None
            h_rel = _classify_h_dir(m_horiz_vert.group(2))
            alt = float(m_horiz_vert.group(3))
            lm = _resolve_lm(m_horiz_vert.group(4))
            return {
                "relation": "COMPOUND",
                "distance": alt,
                "altitude_offset": alt,
                "horizontal_relation": h_rel,
                "horizontal_distance": h_dist,
                "base_landmark": lm,
                "raw": raw
            }

        # Pattern 3: <horiz> <landmark> at Xm altitude
        # e.g., '15m in front of fountain at 50m altitude'
        m_horiz_at_alt = re.match(
            r'^(?:(\d+(?:\.\d+)?)\s*(?:m|meter|meters)?\s+)?(in\s+front(?:\s+of)?|ahead(?:\s+of)?|behind|in\s+back(?:\s+of)?|to\s+the\s+left(?:\s+of)?|left(?:\s+of)?|to\s+the\s+right(?:\s+of)?|right(?:\s+of)?)\s+(?:the\s+|a\s+|an\s+)?(.+?)\s+at\s+(\d+(?:\.\d+)?)\s*(?:m|meter|meters)?(?:\s+(?:altitude|height|elevation))?$',
            cleaned, re.IGNORECASE
        )
        if m_horiz_at_alt:
            h_dist = float(m_horiz_at_alt.group(1)) if m_horiz_at_alt.group(1) else None
            h_rel = _classify_h_dir(m_horiz_at_alt.group(2))
            lm = _resolve_lm(m_horiz_at_alt.group(3))
            alt = float(m_horiz_at_alt.group(4))
            return {
                "relation": "COMPOUND",
                "distance": alt,
                "altitude_offset": alt,
                "horizontal_relation": h_rel,
                "horizontal_distance": h_dist,
                "base_landmark": lm,
                "raw": raw
            }

        # --- SINGLE PREPOSITION PATTERNS ---
        # 1. Check "at X altitude / height above/over <landmark>" or "Xm above/over <landmark>"
        m = re.match(
            r'^(?:at\s+)?(?:an\s+altitude\s+of\s+)?(\d+(?:\.\d+)?)\s*(?:m|meter|meters)?\s*(?:altitude\s+)?(?:above|over)\s+(?:the\s+|a\s+|an\s+)?(.+)$',
            cleaned, re.IGNORECASE
        )
        if m:
            dist = float(m.group(1))
            lm = _resolve_lm(m.group(2))
            return {
                "relation": "ABOVE",
                "distance": dist,
                "altitude_offset": dist,
                "horizontal_relation": None,
                "horizontal_distance": None,
                "base_landmark": lm,
                "raw": raw
            }

        # 2. Check "above/over <landmark> at/by Xm [altitude]"
        m = re.match(
            r'^(?:above|over)\s+(?:the\s+|a\s+|an\s+)?(.+?)\s+(?:at|by)\s+(\d+(?:\.\d+)?)\s*(?:m|meter|meters)?(?:\s+(?:altitude|height|elevation))?$',
            cleaned, re.IGNORECASE
        )
        if m:
            lm = _resolve_lm(m.group(1))
            dist = float(m.group(2))
            return {
                "relation": "ABOVE",
                "distance": dist,
                "altitude_offset": dist,
                "horizontal_relation": None,
                "horizontal_distance": None,
                "base_landmark": lm,
                "raw": raw
            }

        # 3. Check "<landmark> at Xm altitude/height" or "to <landmark> at Xm altitude"
        m = re.match(
            r'^(?:to\s+)?(?:the\s+|a\s+|an\s+)?(.+?)\s+at\s+(\d+(?:\.\d+)?)\s*(?:m|meter|meters)?(?:\s+(?:altitude|height|elevation))?$',
            cleaned, re.IGNORECASE
        )
        if m:
            lm = _resolve_lm(m.group(1))
            dist = float(m.group(2))
            return {
                "relation": "ABOVE",
                "distance": dist,
                "altitude_offset": dist,
                "horizontal_relation": None,
                "horizontal_distance": None,
                "base_landmark": lm,
                "raw": raw
            }

        # 4. Check "above/over <landmark>" without explicit numeric distance
        m = re.match(
            r'^(?:above|over)\s+(?:the\s+|a\s+|an\s+)?(.+)$',
            cleaned, re.IGNORECASE
        )
        if m:
            lm = _resolve_lm(m.group(1))
            return {
                "relation": "ABOVE",
                "distance": None,
                "altitude_offset": None,
                "horizontal_relation": None,
                "horizontal_distance": None,
                "base_landmark": lm,
                "raw": raw
            }

        # 5. Check "Xm in front of <landmark>" or "in front of / ahead of <landmark>"
        m = re.match(
            r'^(?:(\d+(?:\.\d+)?)\s*(?:m|meter|meters)?\s+)?(?:in\s+front(?:\s+of)?|ahead(?:\s+of)?)\s+(?:the\s+|a\s+|an\s+)?(.+)$',
            cleaned, re.IGNORECASE
        )
        if m:
            dist = float(m.group(1)) if m.group(1) else None
            lm = _resolve_lm(m.group(2))
            return {
                "relation": "IN_FRONT",
                "distance": dist,
                "altitude_offset": None,
                "horizontal_relation": "IN_FRONT",
                "horizontal_distance": dist,
                "base_landmark": lm,
                "raw": raw
            }

        # 6. Check "Xm behind <landmark>" or "behind / in back of <landmark>"
        m = re.match(
            r'^(?:(\d+(?:\.\d+)?)\s*(?:m|meter|meters)?\s+)?(?:behind|in\s+back(?:\s+of)?)\s+(?:the\s+|a\s+|an\s+)?(.+)$',
            cleaned, re.IGNORECASE
        )
        if m:
            dist = float(m.group(1)) if m.group(1) else None
            lm = _resolve_lm(m.group(2))
            return {
                "relation": "BEHIND",
                "distance": dist,
                "altitude_offset": None,
                "horizontal_relation": "BEHIND",
                "horizontal_distance": dist,
                "base_landmark": lm,
                "raw": raw
            }

        # 7. Check "Xm to the left of <landmark>" or "left of <landmark>"
        m = re.match(
            r'^(?:(\d+(?:\.\d+)?)\s*(?:m|meter|meters)?\s+)?(?:to\s+the\s+left(?:\s+of)?|left(?:\s+of)?)\s+(?:the\s+|a\s+|an\s+)?(.+)$',
            cleaned, re.IGNORECASE
        )
        if m:
            dist = float(m.group(1)) if m.group(1) else None
            lm = _resolve_lm(m.group(2))
            return {
                "relation": "LEFT",
                "distance": dist,
                "altitude_offset": None,
                "horizontal_relation": "LEFT",
                "horizontal_distance": dist,
                "base_landmark": lm,
                "raw": raw
            }

        # 8. Check "Xm to the right of <landmark>" or "right of <landmark>"
        m = re.match(
            r'^(?:(\d+(?:\.\d+)?)\s*(?:m|meter|meters)?\s+)?(?:to\s+the\s+right(?:\s+of)?|right(?:\s+of)?)\s+(?:the\s+|a\s+|an\s+)?(.+)$',
            cleaned, re.IGNORECASE
        )
        if m:
            dist = float(m.group(1)) if m.group(1) else None
            lm = _resolve_lm(m.group(2))
            return {
                "relation": "RIGHT",
                "distance": dist,
                "altitude_offset": None,
                "horizontal_relation": "RIGHT",
                "horizontal_distance": dist,
                "base_landmark": lm,
                "raw": raw
            }

        # Default fallback: clean articles and punctuation
        fallback_lm = re.sub(r'^(?:to\s+)?(?:the\s+|a\s+|an\s+)', '', cleaned, flags=re.IGNORECASE).strip()
        fallback_lm = _resolve_lm(fallback_lm)

        return {
            "relation": "DEFAULT",
            "distance": None,
            "altitude_offset": None,
            "horizontal_relation": None,
            "horizontal_distance": None,
            "base_landmark": fallback_lm or raw,
            "raw": raw
        }

    @classmethod
    def extract_base_landmark(cls, text: str) -> str:
        """Extract clean base landmark noun phrase without prepositions or distance qualifiers."""
        spatial = cls.parse_spatial_relation(text)
        return spatial.get("base_landmark", text.strip())

    def _parse_chunks(self, text: str, context: Optional[List[str]] = None) -> List[str]:
        """Split text on transition keywords and commas into ordered subgoals."""
        chunks = re.split(r'\b(?:then|next|after that|afterwards|finally|followed by|and then)\b', text, flags=re.IGNORECASE)
        res = list(context or [])

        pattern = re.compile(
            r'^(?:fly\s+to|navigate\s+to|head\s+towards?|go\s+to|proceed\s+to|ascend\s+to|climb\s+to|pass|reach|find|locate|search\s+for|the|a|an)\s+',
            re.IGNORECASE
        )
        verb_prefix = re.compile(
            r'^(?:fly|navigate|head|go|proceed|climb|ascend|descend|hover)\s+',
            re.IGNORECASE
        )

        orbit_context_active = self.is_orbit_command(text)

        for chunk in chunks:
            chunk = chunk.strip().strip(",.;:")
            if not chunk:
                continue

            # Orbit command detection for chunk (including elliptical coordination e.g. "and then around red building 2 times")
            is_orb = self.is_orbit_command(chunk) or (orbit_context_active and bool(re.search(r'\baround\b', chunk, re.IGNORECASE)))
            if is_orb:
                parse_chunk = chunk if self.is_orbit_command(chunk) else f"circle {chunk}"
                orbit_info = self.parse_orbit_command(parse_chunk)
                if orbit_info and orbit_info.get("target"):
                    rev_part = f" {int(orbit_info['revolutions'])} times" if orbit_info.get("revolutions") else ""
                    facing_part = " facing forward" if orbit_info.get("facing") == "TANGENT" else ""
                    speed_part = f" at {orbit_info['speed']} m/s" if (orbit_info.get("speed") and orbit_info["speed"] != 3.0) else ""
                    radius_part = f" at {orbit_info['radius']}m radius" if (orbit_info.get("radius") and orbit_info["radius"] != 16.0) else ""
                    altitude_part = f" at {orbit_info['altitude']}m altitude" if (orbit_info.get("altitude") and orbit_info["altitude"] != 10.0) else ""
                    direction_part = " clockwise" if orbit_info.get("direction") == "CW" else ""
                    orbit_subgoal = f"circle around {orbit_info['target']}{rev_part}{facing_part}{speed_part}{radius_part}{altitude_part}{direction_part}".strip()
                    if not res or res[-1] != orbit_subgoal:
                        res.append(orbit_subgoal)
                    orbit_context_active = True
                    continue

            split_pattern = r'[,;]|\b(?<!above\s)(?<!over\s)(?<!front\s)(?<!ahead\s)(?<!behind\s)(?<!left\s)(?<!right\s)and\b'
            parts = re.split(split_pattern, chunk, flags=re.IGNORECASE)
            stop_words = {"stop", "hover", "halt", "pause", "wait", "here", "right here", "there", "now"}
            for part in parts:
                p = part.strip()
                if p.lower() in stop_words:
                    continue
                while True:
                    cleaned = pattern.sub('', p).strip()
                    if cleaned == p:
                        break
                    p = cleaned
                if p.lower() in stop_words:
                    continue

                if p and any(k in p.lower() for k in ["above", "over", "in front", "left of", "right of", "behind", "altitude", "elevation"]):
                    p = verb_prefix.sub('', p).strip()
                    p = re.sub(r'^(?:the|a|an)\s+', '', p, flags=re.IGNORECASE).strip()

                resolved = self.resolve_landmark_shorthand(p, res)
                if resolved.lower() in stop_words:
                    continue
                if resolved and len(resolved) > 2 and resolved not in res:
                    res.append(resolved)

        return res

    def _heuristic_parse_instruction(self, text: str) -> List[str]:
        """Rule-based extraction of subgoals from flight text.
        
        Supports:
        - Collective / universal quantifiers: e.g. 'go to every landmark', 'visit all landmarks', 'tour the city'
        - Multi-leg routes: e.g. 'go from fountain to red building, and also go from red to blue building'
        - Ellipsis and shorthand resolution: e.g. 'from red to blue building' -> 'red building' to 'blue building'
        - Transition sequences: e.g. 'fly to fountain, then 20m above red building, then blue building'
        """
        lower = text.lower().strip()

        # Hover / stop commands and explicit waypoint curves should not be split into multiple subgoals
        if self.is_hover_stop_command(text):
            return [text.strip()]
        if self.is_explicit_waypoint_path(text):
            return [text.strip()]

        # Collective quantifier check: "go to every landmark", "visit all landmarks", "tour the city", etc.
        collective_patterns = [
            r'\b(?:every|all|each)\s+(?:the\s+)?(?:landmarks?|buildings?)\b',
            r'\b(?:visit|tour|explore|check|fly\s+to|go\s+to|navigate\s+to)\s+(?:all|every|each)(?:\s+the)?(?:\s+landmarks?|\s+buildings?)?\b',
            r'\b(?:tour|explore)\s+(?:the\s+)?(?:city|town|world|area)\b',
            r'\ball\s+of\s+the\s+(?:landmarks?|buildings?)\b',
        ]
        if any(re.search(pat, lower) for pat in collective_patterns):
            return list(self.KNOWN_DEFAULT_LANDMARKS)

        # 1. Match 'from <origin> to <destination>' clauses
        from_to_pat = r'\bfrom\s+(.+?)\s+to\s+(.+?)(?=(?:\s*,\s*|\s+and\s+(?:also\s+)?|\s+then\s+|\s+after\s+that\s+|\s+next\s+|\s+to\s+)|$)'
        matches = list(re.finditer(from_to_pat, text, re.IGNORECASE))

        if matches:
            subgoals: List[str] = []
            for m in matches:
                orig_raw = m.group(1).strip()
                dest_raw = m.group(2).strip()
                orig_res = self.resolve_landmark_shorthand(orig_raw, subgoals)
                dest_res = self.resolve_landmark_shorthand(dest_raw, subgoals)

                # Chain waypoints without duplicating consecutive intermediate landmarks
                if not subgoals:
                    subgoals.append(orig_res)
                elif subgoals[-1].lower() != orig_res.lower() and self.extract_base_landmark(subgoals[-1]).lower() != self.extract_base_landmark(orig_res).lower():
                    subgoals.append(orig_res)

                if subgoals[-1].lower() != dest_res.lower() and self.extract_base_landmark(subgoals[-1]).lower() != self.extract_base_landmark(dest_res).lower():
                    subgoals.append(dest_res)

            # Check for trailing waypoints after the last from-to match (e.g. 'and then to blue building')
            last_end = matches[-1].end()
            trailing = text[last_end:].strip()
            trailing = re.sub(r'^(?:,\s*|\s+and\s+(?:also\s+)?|\s+then\s+|\s+to\s+)+', '', trailing, flags=re.IGNORECASE).strip()
            if trailing:
                subgoals = self._parse_chunks(trailing, context=subgoals)

            return subgoals

        # 2. Standard transition chunk parsing
        subgoals = self._parse_chunks(text)
        if not subgoals:
            subgoals = [self.resolve_landmark_shorthand(text.strip())]
        return subgoals

    def _heuristic_classify_interruption(
        self,
        msg: str,
        current_target: str,
        memory_summary: str
    ) -> Dict[str, Any]:
        """Heuristic rule engine for classifying operator interruptions."""
        lower = msg.lower().strip()

        # Guard against empty messages or messages containing only punctuation (e.g. "...", "???")
        alphanumeric = re.sub(r'[^\w\s]', '', msg).strip()
        if not alphanumeric:
            return {
                "intent": "QUERY",
                "target": None,
                "reply": "No clear command received. Please specify a destination, query, or hover.",
                "raw_message": msg
            }

        # Check for HOVER_STOP first
        if self.is_hover_stop_command(msg):
            return {
                "intent": "HOVER_STOP",
                "target": None,
                "reply": "Holding position and hovering.",
                "raw_message": msg
            }

        # Explicit 3D waypoint coordinate path [[x, y, z], ...]
        if self.is_explicit_waypoint_path(msg):
            wp_match = re.findall(r'\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]', msg)
            wps = [[float(x), float(y), float(z)] for x, y, z in wp_match]
            facing = "INWARD" if "inward" in lower else "TANGENT"
            return {
                "intent": "WAYPOINT_PATH",
                "waypoints": wps,
                "speed": 3.0,
                "facing": facing,
                "target": "waypoint_curve",
                "reply": f"Executing continuous 3D waypoint curve ({len(wps)} points).",
                "raw_message": msg
            }

        # Check if operator message specifies a multi-subgoal route/mission
        subgoals = self.parse_initial_instruction(msg)
        if len(subgoals) > 1:
            return {
                "intent": "MULTI_SUBGOAL_MISSION",
                "subgoals": subgoals,
                "instruction": msg,
                "reply": f"Understood. Starting multi-step mission ({len(subgoals)} subgoals): {' -> '.join(subgoals)}.",
                "raw_message": msg
            }

        # Check for Orbit / Circling command
        if self.is_orbit_command(msg):
            orbit_info = self.parse_orbit_command(msg)
            if orbit_info and orbit_info.get("target"):
                rev_str = f" for {orbit_info['revolutions']:.0f} revolutions" if orbit_info['revolutions'] else " continuously"
                facing_str = f" ({orbit_info['facing'].lower()} facing)"
                intent_type = "WAYPOINT_PATH" if orbit_info.get("requested_coordinates") and orbit_info.get("waypoints") else "ORBIT_CIRCLE"
                return {
                    "intent": intent_type,
                    "target": orbit_info["target"],
                    "base_landmark": orbit_info["base_landmark"],
                    "radius": orbit_info["radius"],
                    "altitude": orbit_info["altitude"],
                    "speed": orbit_info["speed"],
                    "facing": orbit_info["facing"],
                    "revolutions": orbit_info["revolutions"],
                    "direction": orbit_info["direction"],
                    "waypoints": orbit_info.get("waypoints"),
                    "reply": f"Orbiting {orbit_info['target']}{rev_str} at {orbit_info['radius']:.1f}m radius, {orbit_info['altitude']:.1f}m altitude{facing_str}.",
                    "raw_message": msg
                }

        # Explicit 3D waypoint coordinate path [[x, y, z], ...]
        wp_match = re.findall(r'\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]', msg)
        if wp_match and len(wp_match) >= 2:
            wps = [[float(x), float(y), float(z)] for x, y, z in wp_match]
            facing = "INWARD" if "inward" in lower else "TANGENT"
            return {
                "intent": "WAYPOINT_PATH",
                "waypoints": wps,
                "speed": 3.0,
                "facing": facing,
                "target": "waypoint_curve",
                "reply": f"Executing continuous 3D waypoint curve ({len(wps)} points).",
                "raw_message": msg
            }

        # Check if message contains a spatial relation relative to a landmark or destination
        spatial_candidate = self.parse_spatial_relation(msg)
        has_landmark = bool(
            spatial_candidate["relation"] != "DEFAULT"
            or any(w in lower for w in [
                "fountain", "building", "tower", "hospital", "tree", "car", "statue", "bench", "bridge", "lamp"
            ])
            or any(lm in lower for lm in self.KNOWN_DEFAULT_LANDMARKS)
        )

        # Stop / Hover command (ONLY when not commanding flight to a specific landmark or spatial waypoint)
        if not has_landmark and spatial_candidate["relation"] == "DEFAULT":
            if any(w in lower for w in ["stop", "hover", "halt", "pause", "wait"]):
                return {
                    "intent": "HOVER_STOP",
                    "target": None,
                    "reply": "Holding position and hovering.",
                    "raw_message": msg
                }

        # Return to past command
        if any(w in lower for w in ["go back", "return", "previous", "earlier", "past"]):
            # Extract landmark name after 'to' or 'back to'
            match = re.search(r'(?:go back to|return to|back to|to the|to)\s+([a-zA-Z0-9\s_-]+)', lower)
            raw_target = match.group(1).strip() if match else None
            resolved_target = None
            if raw_target:
                raw_target = self.clean_landmark_name(raw_target)
                resolved_target = self.resolve_landmark_shorthand(raw_target)
            return {
                "intent": "RETURN_TO_PAST",
                "target": resolved_target or raw_target,
                "reply": f"Understood. Returning to previous landmark '{resolved_target or raw_target or 'prior'}'.",
                "raw_message": msg
            }

        # Pure altitude change command (ONLY when no landmark is targeted)
        # e.g. "climb to 15m", "fly at 12m", "altitude 18m", "fly higher", "fly lower"
        if not has_landmark:
            alt_match = re.search(r'(?:climb\s+to|fly\s+at|altitude\s+to|altitude|height\s+to|height|ascend\s+to)\s*(\d+(?:\.\d+)?)\s*(?:m|meters)?', lower)
            if alt_match:
                new_alt = float(alt_match.group(1))
                return {
                    "intent": "ALTITUDE_CHANGE",
                    "altitude": new_alt,
                    "target": None,
                    "reply": f"Adjusting cruise altitude to {new_alt:.1f} meters.",
                    "raw_message": msg
                }
            if "fly higher" in lower or "climb" in lower or "go up" in lower or "ascend" in lower:
                return {
                    "intent": "ALTITUDE_CHANGE",
                    "altitude": 16.0,
                    "target": None,
                    "reply": "Climbing to elevated clearance altitude (16.0m).",
                    "raw_message": msg
                }
            if "fly lower" in lower or "descend" in lower or "go down" in lower:
                return {
                    "intent": "ALTITUDE_CHANGE",
                    "altitude": 5.0,
                    "target": None,
                    "reply": "Descending to 5.0m observation altitude.",
                    "raw_message": msg
                }

        # Query command
        if any(w in lower for w in ["what", "where", "list", "how many", "status", "tell me", "seen"]):
            return {
                "intent": "QUERY",
                "target": None,
                "reply": f"Known landmarks in memory:\n{memory_summary or 'No landmarks registered yet.'}",
                "raw_message": msg
            }

        # Target correction (default for statements specifying new destination or spatial waypoint)
        # e.g., "no, the blue building", "switch to fountain", "fly to tower", "go 50m above fountain", "hover 100m above red building"
        prefix = re.compile(
            r'^(?:no,?|actually,?|instead,?|switch\s+to|change\s+to|fly\s+to|go\s+to|head\s+to|ascend\s+to|climb\s+to|hover\s+at|hover\s+over|hover\s+around|hover|stop\s+at|stop\s+near|the|a|an)\s+',
            re.IGNORECASE
        )
        t = lower
        while True:
            c = prefix.sub('', t).strip()
            if c == t:
                break
            t = c
        clean_target = t.strip().strip(".")

        resolved_target = self.resolve_landmark_shorthand(clean_target or msg)
        spatial_info = self.parse_spatial_relation(resolved_target)
        base_landmark = spatial_info.get("base_landmark") or resolved_target

        return {
            "intent": "TARGET_CORRECTION",
            "target": resolved_target,
            "base_landmark": base_landmark,
            "spatial": spatial_info,
            "reply": f"Acknowledged. Diverting target to '{resolved_target}'.",
            "raw_message": msg
        }
