"""Asynchronous Dialogue Manager and Gemini 2.0 Flash Operator Interruption Engine."""

import os
import re
import json
import queue
from typing import Dict, List, Optional, Any

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

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.0-flash"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.model_name = model_name
        self.command_queue: queue.Queue[str] = queue.Queue()
        self.client = None

        if _GENAI_AVAILABLE and self.api_key:
            try:
                self.client = genai.Client(api_key=self.api_key)
            except Exception as e:
                print(f"[DialogueManager] Could not initialize Gemini client: {e}. Falling back to heuristic mode.")
                self.client = None

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

        # Attempt Gemini 2.0 Flash decomposition if client available
        if self.client is not None:
            try:
                prompt = (
                    "You are the navigation planner for an autonomous urban drone. "
                    "Extract the ordered sequence of physical landmark subgoals from this instruction. "
                    "Return ONLY a JSON object formatted as: {\"subgoals\": [\"<landmark1>\", \"<landmark2>\"]}\n\n"
                    f"Instruction: \"{instruction}\""
                )
                cfg = types.GenerateContentConfig(response_mime_type="application/json") if _GENAI_AVAILABLE else None
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=cfg
                )
                raw_text = response.text.strip()
                if raw_text.startswith("```"):
                    raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
                    raw_text = re.sub(r"\s*```$", "", raw_text)
                data = json.loads(raw_text.strip())
                subgoals = data.get("subgoals", [])
                if isinstance(subgoals, list) and subgoals:
                    return [str(s).strip() for s in subgoals if str(s).strip()]
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

        # Attempt Gemini 2.0 Flash classification if client available
        if self.client is not None:
            try:
                prompt = (
                    "An operator interrupted an autonomous drone mid-flight.\n"
                    f"Active Target: \"{cur_target_str}\"\n"
                    f"Known Landmarks in Spatial Memory:\n{memory_summary}\n\n"
                    f"Operator message: \"{msg}\"\n\n"
                    "Classify this operator command into exactly one intent:\n"
                    "1. \"TARGET_CORRECTION\": Divert flight to a newly specified target.\n"
                    "2. \"RETURN_TO_PAST\": Return to a previously registered landmark in memory.\n"
                    "3. \"QUERY\": Inquire about landmarks, surroundings, or status.\n"
                    "4. \"HOVER_STOP\": Pause, hover, or stop navigation.\n\n"
                    "Respond with JSON:\n"
                    "{\n"
                    "  \"intent\": \"TARGET_CORRECTION\" | \"RETURN_TO_PAST\" | \"QUERY\" | \"HOVER_STOP\",\n"
                    "  \"target\": \"<target landmark name or null>\",\n"
                    "  \"reply\": \"<concise confirmation to operator>\"\n"
                    "}"
                )
                cfg = types.GenerateContentConfig(response_mime_type="application/json") if _GENAI_AVAILABLE else None
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=cfg
                )
                raw_text = response.text.strip()
                if raw_text.startswith("```"):
                    raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
                    raw_text = re.sub(r"\s*```$", "", raw_text)
                data = json.loads(raw_text.strip())
                data["raw_message"] = msg
                return data
            except Exception as e:
                print(f"[DialogueManager] Gemini interruption handling failed: {e}. Using heuristic fallback.")

        # Heuristic fallback classifier
        return self._heuristic_classify_interruption(msg, cur_target_str, memory_summary)

    def _heuristic_parse_instruction(self, text: str) -> List[str]:
        """Rule-based extraction of subgoals from flight text."""
        # Split on transition keywords (then, next, after, and finally, followed by)
        chunks = re.split(r'\b(?:then|next|after that|afterwards|finally|followed by|and then)\b', text, flags=re.IGNORECASE)
        subgoals = []

        # Common stop words/verbs to strip iteratively
        pattern = re.compile(
            r'^(?:fly\s+to|navigate\s+to|head\s+towards?|go\s+to|proceed\s+to|pass|reach|find|locate|search\s+for|the|a|an)\s+',
            re.IGNORECASE
        )

        for chunk in chunks:
            chunk = chunk.strip().strip(",.;:")
            if not chunk:
                continue
            # Split within chunk on commas or 'and'
            parts = re.split(r'[,;]|\band\b', chunk, flags=re.IGNORECASE)
            for part in parts:
                p = part.strip()
                # Iteratively strip leading command verbs and articles
                while True:
                    cleaned = pattern.sub('', p).strip()
                    if cleaned == p:
                        break
                    p = cleaned
                if p and len(p) > 2 and p not in subgoals:
                    subgoals.append(p)

        if not subgoals:
            # Fallback: extract nouns/phrases
            subgoals = [text.strip()]
        return subgoals

    def _heuristic_classify_interruption(
        self,
        msg: str,
        current_target: str,
        memory_summary: str
    ) -> Dict[str, Any]:
        """Heuristic rule engine for classifying operator interruptions."""
        lower = msg.lower().strip()

        # Stop / Hover command
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
            if raw_target:
                # Strip leading articles
                raw_target = re.sub(r'^(?:the|a|an)\s+', '', raw_target).strip()
            return {
                "intent": "RETURN_TO_PAST",
                "target": raw_target,
                "reply": f"Understood. Returning to previous landmark '{raw_target or 'prior'}'.",
                "raw_message": msg
            }

        # Altitude change command (e.g. "climb to 15m", "fly at 12m", "altitude 18m", "fly higher", "fly lower")
        alt_match = re.search(r'(?:climb\s+to|fly\s+at|altitude\s+to|altitude|height\s+to|height)\s*(\d+(?:\.\d+)?)\s*(?:m|meters)?', lower)
        if alt_match:
            new_alt = float(alt_match.group(1))
            return {
                "intent": "ALTITUDE_CHANGE",
                "altitude": new_alt,
                "target": None,
                "reply": f"Adjusting cruise altitude to {new_alt:.1f} meters.",
                "raw_message": msg
            }
        if "fly higher" in lower or "climb" in lower or "go up" in lower:
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

        # Target correction (default for statements specifying new destination)
        # e.g., "no, the blue building", "switch to fountain", "fly to tower"
        prefix = re.compile(
            r'^(?:no,?|actually,?|instead,?|switch\s+to|change\s+to|fly\s+to|go\s+to|head\s+to|the|a|an)\s+',
            re.IGNORECASE
        )
        t = lower
        while True:
            c = prefix.sub('', t).strip()
            if c == t:
                break
            t = c
        clean_target = t.strip().strip(".")

        return {
            "intent": "TARGET_CORRECTION",
            "target": clean_target or msg,
            "reply": f"Acknowledged. Diverting target to '{clean_target or msg}'.",
            "raw_message": msg
        }
