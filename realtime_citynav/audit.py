"""Audit and Verification Logger for AR511 Mission Commands and Gemini LLM Pipeline."""

import os
import json
import time
import threading
from datetime import datetime

_LOCK = threading.Lock()
LOG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "mission_audit.log"))


def log_event(category: str, details: dict):
    """Thread-safe append of mission events to mission_audit.log."""
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
        "category": category,
        **details
    }
    line = json.dumps(entry, ensure_ascii=False)
    with _LOCK:
        try:
            with open(LOG_PATH, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception as e:
            print(f"[AuditLogger] Failed to write log: {e}")


def get_latest_events(n: int = 20):
    """Read the latest n events from the audit log."""
    if not os.path.exists(LOG_PATH):
        return []
    with _LOCK:
        try:
            with open(LOG_PATH, "r", encoding="utf-8") as f:
                lines = [l.strip() for l in f if l.strip()]
            return [json.loads(l) for l in lines[-n:]]
        except Exception:
            return []
