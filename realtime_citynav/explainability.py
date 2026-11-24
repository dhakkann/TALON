"""Event-Driven and Cadence-Based Live Rationale Narration Generator."""

import time
import threading
from typing import Callable, Dict, List, Optional, Any
from dataclasses import dataclass, field


@dataclass
class NarrationEvent:
    """A timestamped explanation or rationale statement."""
    timestamp: float
    event_type: str
    message: str
    telemetry: Dict[str, Any] = field(default_factory=dict)
    spoken_text: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "message": self.message,
            "telemetry": self.telemetry,
            "spoken_text": self.spoken_text
        }


class LiveRationaleNarrator:
    """Generates real-time explainability stream for the operator.
    
    Supports both low-latency deterministic rationale synthesis (<0.5 ms)
    and optional LLM-augmented situational commentary.
    """

    def __init__(self, max_history: int = 200):
        self.max_history = max_history
        self._history: List[NarrationEvent] = []
        self._listeners: List[Callable[[NarrationEvent], None]] = []
        self._lock = threading.Lock()
        self._last_cadence_time: float = 0.0

    def add_listener(self, callback: Callable[[NarrationEvent], None]) -> None:
        """Register a callback for real-time streaming (e.g. WebSocket / SSE)."""
        with self._lock:
            self._listeners.append(callback)

    def emit_event(
        self,
        event_type: str,
        message: str,
        telemetry: Optional[Dict[str, Any]] = None,
        spoken_text: Optional[str] = None
    ) -> NarrationEvent:
        """Emit an event-driven narration and broadcast to listeners."""
        # Synthesize concise spoken feedback for critical events if not explicitly passed
        if spoken_text is None:
            if event_type == "EMERGENCY_HOVER":
                spoken_text = "Emergency hover engaged. Holding position."
            elif event_type == "TARGET_ACQUIRED":
                tgt = (telemetry or {}).get("target", "landmark")
                spoken_text = f"Target acquired: {tgt}."
            elif event_type == "ORBIT_STARTED":
                tgt = (telemetry or {}).get("target", "landmark")
                spoken_text = f"Orbiting {tgt}."
            elif event_type == "OBSTACLE_AVOIDANCE":
                spoken_text = "Obstacle detected. Avoidance maneuver active."
            elif event_type == "VOICE_COMMAND":
                cmd = (telemetry or {}).get("transcript", "")
                spoken_text = f"Acknowledged: {cmd}" if cmd else "Voice command received."
            elif event_type == "QUERY_REPLY":
                spoken_text = message[:120]

        event = NarrationEvent(
            timestamp=time.time(),
            event_type=event_type,
            message=message,
            telemetry=telemetry or {},
            spoken_text=spoken_text
        )
        
        with self._lock:
            self._history.append(event)
            if len(self._history) > self.max_history:
                self._history.pop(0)
            listeners_copy = list(self._listeners)

        # Notify listeners outside the lock
        for listener in listeners_copy:
            try:
                listener(event)
            except Exception:
                pass  # Ignore failing listeners

        return event

    def notify_target_acquired(
        self,
        target_name: str,
        distance: float,
        world_pos: Any
    ) -> NarrationEvent:
        """Narrate acquisition of a landmark target."""
        pos_str = f"[{world_pos[0]:.1f}, {world_pos[1]:.1f}, {world_pos[2]:.1f}]" if world_pos is not None else "unknown"
        msg = f"Locked target: '{target_name}' at distance {distance:.1f}m {pos_str}. Aligning velocity vector."
        return self.emit_event(
            "TARGET_ACQUIRED",
            msg,
            {"target": target_name, "distance": distance}
        )

    def notify_waypoint_reached(self, target_name: str) -> NarrationEvent:
        """Narrate waypoint arrival."""
        msg = f"Arrived at subgoal '{target_name}'. Transitioning to next objective."
        return self.emit_event(
            "WAYPOINT_REACHED",
            msg,
            {"target": target_name}
        )

    def notify_obstacle_avoidance(
        self,
        direction: str,
        obstacle_distance: float
    ) -> NarrationEvent:
        """Narrate reactive APF safety intervention."""
        msg = f"Proximity warning! Obstacle detected at {obstacle_distance:.1f}m. APF vector deflected {direction}."
        return self.emit_event(
            "OBSTACLE_AVOIDANCE",
            msg,
            {"direction": direction, "distance": obstacle_distance}
        )

    def notify_interruption(self, intent: str, details: str) -> NarrationEvent:
        """Narrate mid-flight operator intervention."""
        msg = f"Operator Interruption [{intent}]: {details}"
        return self.emit_event(
            "INTERRUPTION_HANDLED",
            msg,
            {"intent": intent, "details": details}
        )

    def notify_cadence(
        self,
        current_target: Optional[str],
        current_speed: float,
        altitude: float,
        dist_to_target: Optional[float]
    ) -> Optional[NarrationEvent]:
        """Periodic heartbeat status update."""
        dist_str = f"dist: {dist_to_target:.1f}m" if dist_to_target is not None else "searching"
        tgt_str = current_target if current_target else "search pattern"
        msg = f"Cruising at {current_speed:.1f} m/s | Alt: {altitude:.1f}m | Target: '{tgt_str}' ({dist_str})."
        return self.emit_event(
            "CADENCE_UPDATE",
            msg,
            {
                "speed": current_speed,
                "altitude": altitude,
                "target": current_target,
                "distance": dist_to_target
            }
        )

    def get_recent(self, count: int = 20) -> List[Dict[str, Any]]:
        """Retrieve recent narration history."""
        with self._lock:
            return [e.to_dict() for e in self._history[-count:]]
