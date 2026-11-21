"""Persistent 3D Spatial Landmark Registry with Euclidean Clustering."""

import time
import threading
from typing import Dict, List, Optional, Any
import numpy as np


class SpatialMemory:
    """Thread-safe persistent 3D spatial landmark registry.
    
    Clusters landmark observations within Euclidean distance threshold,
    maintains running-average 3D positions, observation counts, and timestamps.
    """

    def __init__(
        self,
        cluster_threshold: float = 5.0,
        alpha: float = 0.2,
        max_retention_sec: float = 3600.0
    ):
        """Initialize spatial memory registry.
        
        Args:
            cluster_threshold: Max Euclidean distance in meters to cluster into same landmark.
            alpha: Exponential moving average update weight for new position observations.
            max_retention_sec: Maximum retention period in seconds before stale landmark pruning.
        """
        self.cluster_threshold = cluster_threshold
        self.alpha = alpha
        self.max_retention_sec = max_retention_sec
        self._landmarks: Dict[int, Dict[str, Any]] = {}
        self._next_id: int = 1
        self._lock = threading.Lock()

    def add_observation(
        self,
        name: str,
        world_pos: np.ndarray,
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """Add or merge a landmark observation into persistent memory.
        
        Args:
            name: Landmark class label or name (e.g. 'fountain', 'building').
            world_pos: 3D world coordinate [X, Y, Z] in meters.
            confidence: Perception detection confidence score (0 to 1).
            metadata: Optional additional metadata dict.
            
        Returns:
            The landmark ID (either matched existing or newly created).
        """
        pos = np.asarray(world_pos, dtype=float).flatten()
        if len(pos) != 3:
            raise ValueError(f"world_pos must be a 3D coordinate, got shape {pos.shape}")
        if not np.all(np.isfinite(pos)):
            raise ValueError(f"world_pos coordinates must be finite, got {pos}")

        clean_name = name.strip().lower()
        now = time.time()

        with self._lock:
            best_id: Optional[int] = None
            min_dist = float("inf")

            # Look for existing landmark of same class within cluster distance
            for lid, info in self._landmarks.items():
                if info["name"].lower() == clean_name:
                    dist = float(np.linalg.norm(info["pos"] - pos))
                    if dist <= self.cluster_threshold and dist < min_dist:
                        min_dist = dist
                        best_id = lid

            if best_id is not None:
                # Update existing landmark with running-average position
                entry = self._landmarks[best_id]
                entry["pos"] = (1.0 - self.alpha) * entry["pos"] + self.alpha * pos
                entry["last_seen"] = now
                entry["count"] += 1
                entry["confidence"] = 0.7 * entry["confidence"] + 0.3 * confidence
                if metadata:
                    entry.setdefault("metadata", {}).update(metadata)
                return best_id
            else:
                # Register new landmark
                lid = self._next_id
                self._next_id += 1
                self._landmarks[lid] = {
                    "id": lid,
                    "name": name.strip(),
                    "pos": pos.copy(),
                    "first_seen": now,
                    "last_seen": now,
                    "count": 1,
                    "confidence": float(confidence),
                    "metadata": metadata or {}
                }
                return lid

    def add_observations(self, detections: List[Dict[str, Any]]) -> List[int]:
        """Batch process detections from perception layer.
        
        Args:
            detections: List of dicts with 'name', 'world_pos', and optional 'conf'.
            
        Returns:
            List of landmark IDs updated or created.
        """
        lids = []
        for det in detections:
            name = det.get("name")
            pos = det.get("world_pos")
            conf = det.get("conf", 1.0)
            if name is not None and pos is not None:
                lids.append(self.add_observation(name, pos, confidence=conf))
        return lids

    def find_landmark(self, query_name: str) -> Optional[Dict[str, Any]]:
        """Find the most prominent landmark matching a query string.
        
        Matches if query is a substring of the landmark name or vice versa.
        Returns the landmark with highest observation count among matches.
        """
        q = query_name.strip().lower()
        with self._lock:
            matches = [
                info for info in self._landmarks.values()
                if q in info["name"].lower() or info["name"].lower() in q
            ]
            if not matches:
                return None
            best = max(matches, key=lambda x: x["count"])
            return self._copy_entry(best)

    def get_closest_landmark(
        self,
        world_pos: np.ndarray,
        query_name: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Find closest landmark to a given 3D position, optionally filtered by name."""
        pos = np.asarray(world_pos, dtype=float).flatten()
        q = query_name.strip().lower() if query_name else None

        with self._lock:
            candidates = self._landmarks.values()
            if q:
                candidates = [
                    c for c in candidates
                    if q in c["name"].lower() or c["name"].lower() in q
                ]
            if not candidates:
                return None

            best = min(candidates, key=lambda c: float(np.linalg.norm(c["pos"] - pos)))
            return self._copy_entry(best)

    def get_all_landmarks(self) -> List[Dict[str, Any]]:
        """Return list of copies of all registered landmarks."""
        with self._lock:
            return [self._copy_entry(info) for info in self._landmarks.values()]

    def get_summary(self) -> str:
        """Generate human and LLM readable summary of known landmarks."""
        with self._lock:
            if not self._landmarks:
                return "No landmarks recorded yet."
            lines = []
            for l in sorted(self._landmarks.values(), key=lambda x: x["count"], reverse=True):
                p = l["pos"]
                lines.append(
                    f"- {l['name']} (ID {l['id']}) at [{p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f}], "
                    f"observed {l['count']}x, last seen {time.time() - l['last_seen']:.1f}s ago"
                )
            return "\n".join(lines)

    def prune_stale(self, max_age_sec: Optional[float] = None) -> int:
        """Remove landmarks that haven't been observed within max_age_sec.
        
        Returns:
            Number of pruned landmarks.
        """
        cutoff = time.time() - (max_age_sec if max_age_sec is not None else self.max_retention_sec)
        with self._lock:
            stale_ids = [lid for lid, info in self._landmarks.items() if info["last_seen"] < cutoff]
            for lid in stale_ids:
                del self._landmarks[lid]
            return len(stale_ids)

    def clear(self) -> None:
        """Clear all stored landmarks."""
        with self._lock:
            self._landmarks.clear()
            self._next_id = 1

    @staticmethod
    def _copy_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
        """Create a deep-enough copy of a landmark entry to avoid mutation outside lock."""
        c = dict(entry)
        c["pos"] = entry["pos"].copy()
        c["metadata"] = dict(entry.get("metadata", {}))
        return c
