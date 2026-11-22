"""Persistent 3D Spatial Landmark Registry with Euclidean Clustering."""

import re
import time
import threading
from typing import Dict, List, Optional, Any, Set
import numpy as np


class SpatialMemory:
    """Thread-safe persistent 3D spatial landmark registry.
    
    Clusters landmark observations within Euclidean distance threshold,
    maintains running-average 3D positions, observation counts, and timestamps.
    Includes cross-class fuzzy deduplication, name normalization, and observation pruning.
    """

    SYNONYM_GROUPS: List[Set[str]] = [
        {"fountain", "water fountain", "water feature", "fountains", "pool"},
        {"building", "skyscraper", "structure", "edifice", "buildings", "complex"},
        {"car", "vehicle", "automobile", "sedan", "cars", "vehicles", "truck", "suv", "van"},
        {"tree", "plant", "bush", "trees", "foliage", "hedge"},
        {"traffic light", "stoplight", "signal light", "signal", "traffic lights"},
        {"street lamp", "street light", "lamp post", "lantern", "street lamps", "lamppost"},
        {"bus stop", "bus station", "transit shelter", "bus stops"},
        {"statue", "monument", "sculpture", "statues", "monuments"},
        {"bench", "park bench", "seating", "benches"},
    ]

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

    @staticmethod
    def normalize_name(name: str) -> str:
        """Normalize landmark name by stripping articles, punctuation, hyphens, and plural endings."""
        if not name:
            return ""
        n = name.strip().lower()
        # Replace hyphens and underscores with space
        n = re.sub(r'[-_]+', ' ', n)
        # Strip leading articles
        n = re.sub(r'^(?:the|a|an)\s+', '', n)
        # Strip punctuation
        n = re.sub(r'[^\w\s]', '', n).strip()
        # Collapse multiple spaces
        n = re.sub(r'\s+', ' ', n).strip()
        # Normalize simple plurals
        if n.endswith("ies") and len(n) > 4:
            n = n[:-3] + "y"
        elif n.endswith("ees"):
            n = n[:-1]
        elif n.endswith("sses"):
            n = n[:-2]
        elif n.endswith("es") and len(n) > 4 and not n.endswith(("bus", "glass", "grass", "cross")):
            n = n[:-2]
        elif n.endswith("s") and len(n) > 3 and not n.endswith(("bus", "status", "gas", "lens", "grass", "glass")):
            n = n[:-1]
        return n

    @classmethod
    def names_match(cls, name1: str, name2: str) -> bool:
        """Determine whether two landmark labels refer to the same object category."""
        n1 = cls.normalize_name(name1)
        n2 = cls.normalize_name(name2)
        if not n1 or not n2:
            return False
        if n1 == n2:
            return True

        # Check semantic synonym groups
        for group in cls.SYNONYM_GROUPS:
            norm_group = {cls.normalize_name(g) for g in group}
            if n1 in norm_group and n2 in norm_group:
                return True

        # Word-level containment (e.g. "red building" contains "building", "water fountain" contains "fountain")
        words1 = set(w for w in n1.split() if len(w) > 1)
        words2 = set(w for w in n2.split() if len(w) > 1)
        if words1 and words2:
            if words1.issubset(words2) or words2.issubset(words1):
                return True

        return False

    def is_spatial_match(self, pos1: np.ndarray, pos2: np.ndarray, name: str = "", name2: str = "") -> bool:
        """Check if two positions fall within the clustering distance threshold."""
        dist_3d = float(np.linalg.norm(pos1 - pos2))
        dist_xy = float(np.linalg.norm(pos1[:2] - pos2[:2]))
        dz = float(abs(pos1[2] - pos2[2]))

        # Adaptive radius: allow larger cluster threshold for expansive landmarks like buildings
        clean = (name + " " + name2).lower()
        is_large_structure = any(s in clean for s in ["building", "tower", "hospital", "skyscraper", "structure", "complex"])
        thresh = self.cluster_threshold * 2.5 if is_large_structure else self.cluster_threshold

        if dist_3d <= thresh:
            return True
        if dist_xy <= thresh * 1.2 and dz <= 15.0:
            return True
        return False

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

        clean_name = re.sub(r'^(?:the|a|an)\s+', '', name.strip(), flags=re.IGNORECASE).strip()
        now = time.time()

        with self._lock:
            best_id: Optional[int] = None
            min_dist = float("inf")

            # Look for existing landmark of matching class within cluster distance
            for lid, info in self._landmarks.items():
                if self.names_match(info["name"], clean_name) and self.is_spatial_match(info["pos"], pos, info["name"], clean_name):
                    dist = float(np.linalg.norm(info["pos"] - pos))
                    if dist < min_dist:
                        min_dist = dist
                        best_id = lid

            if best_id is not None:
                # Update existing landmark with running-average position
                entry = self._landmarks[best_id]
                entry["pos"] = (1.0 - self.alpha) * entry["pos"] + self.alpha * pos
                entry["last_seen"] = now
                entry["count"] += 1
                entry["confidence"] = 0.7 * entry["confidence"] + 0.3 * confidence
                # Keep more specific/descriptive name (e.g. 'red building' over 'building')
                # Count descriptive words (ignoring leading articles)
                norm_new = self.normalize_name(clean_name)
                norm_cur = self.normalize_name(entry["name"])
                if len(norm_new.split()) > len(norm_cur.split()) or (len(norm_new) > len(norm_cur) and norm_cur in norm_new):
                    entry["name"] = clean_name
                if metadata:
                    entry.setdefault("metadata", {}).update(metadata)
                return best_id
            else:
                # Register new landmark
                lid = self._next_id
                self._next_id += 1
                self._landmarks[lid] = {
                    "id": lid,
                    "name": clean_name,
                    "pos": pos.copy(),
                    "first_seen": now,
                    "last_seen": now,
                    "count": 1,
                    "confidence": float(confidence),
                    "metadata": metadata or {}
                }
                return lid

    def deduplicate(self) -> int:
        """Scan all stored landmarks and merge redundant / duplicate entries.
        
        Returns:
            Number of redundant landmark entries pruned/merged.
        """
        with self._lock:
            merged_count = 0
            to_delete = set()
            lids = list(self._landmarks.keys())

            for i in range(len(lids)):
                lid1 = lids[i]
                if lid1 in to_delete:
                    continue
                lm1 = self._landmarks[lid1]

                for j in range(i + 1, len(lids)):
                    lid2 = lids[j]
                    if lid2 in to_delete:
                        continue
                    lm2 = self._landmarks[lid2]

                    if self.names_match(lm1["name"], lm2["name"]) and self.is_spatial_match(lm1["pos"], lm2["pos"], lm1["name"], lm2["name"]):
                        # Merge lid2 into lid1
                        c1 = lm1["count"]
                        c2 = lm2["count"]
                        total_c = max(1, c1 + c2)
                        # Weighted position average based on observation counts
                        lm1["pos"] = (lm1["pos"] * c1 + lm2["pos"] * c2) / total_c
                        lm1["count"] = total_c
                        lm1["confidence"] = max(lm1["confidence"], lm2["confidence"])
                        lm1["last_seen"] = max(lm1["last_seen"], lm2["last_seen"])
                        lm1["first_seen"] = min(lm1["first_seen"], lm2["first_seen"])
                        # Prefer longer / more descriptive name without leading articles
                        norm1 = self.normalize_name(lm1["name"])
                        norm2 = self.normalize_name(lm2["name"])
                        clean2 = re.sub(r'^(?:the|a|an)\s+', '', lm2["name"].strip(), flags=re.IGNORECASE).strip()
                        if len(norm2.split()) > len(norm1.split()) or (len(norm2) > len(norm1) and norm1 in norm2):
                            lm1["name"] = clean2
                        else:
                            lm1["name"] = re.sub(r'^(?:the|a|an)\s+', '', lm1["name"].strip(), flags=re.IGNORECASE).strip()
                        if lm2.get("metadata"):
                            lm1.setdefault("metadata", {}).update(lm2["metadata"])

                        to_delete.add(lid2)
                        merged_count += 1

            for lid in to_delete:
                del self._landmarks[lid]

            return merged_count

    def prune_unconfirmed(self, min_count: int = 2, max_age_sec: Optional[float] = 30.0) -> int:
        """Prune unconfirmed single-observation landmarks that are older than max_age_sec."""
        cutoff = time.time() - (max_age_sec if max_age_sec is not None else 30.0)
        with self._lock:
            pruned_ids = [
                lid for lid, info in self._landmarks.items()
                if info["count"] < min_count and info["last_seen"] < cutoff
            ]
            for lid in pruned_ids:
                del self._landmarks[lid]
            return len(pruned_ids)

    def clean_registry(
        self,
        dedup: bool = True,
        max_stale_sec: Optional[float] = None,
        prune_unconfirmed_older_than: Optional[float] = 30.0,
        min_confirmed_count: int = 2
    ) -> Dict[str, int]:
        """Perform comprehensive maintenance: prune stale landmarks, prune unconfirmed noise, and deduplicate."""
        stale = self.prune_stale(max_stale_sec)
        unconfirmed = self.prune_unconfirmed(
            min_count=min_confirmed_count,
            max_age_sec=prune_unconfirmed_older_than
        ) if prune_unconfirmed_older_than is not None else 0
        dedup_count = self.deduplicate() if dedup else 0
        with self._lock:
            remaining = len(self._landmarks)
        return {
            "stale_pruned": stale,
            "unconfirmed_pruned": unconfirmed,
            "deduplicated": dedup_count,
            "total_landmarks": remaining
        }

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
        
        Matches if query is a substring of the landmark name or vice versa,
        or via semantic class matching.
        Returns the landmark with highest observation count among matches.
        """
        if not query_name or not query_name.strip():
            return None
        q = self.normalize_name(query_name)
        if not q:
            return None
        with self._lock:
            matches = [
                info for info in self._landmarks.values()
                if self.names_match(info["name"], query_name)
                or q in self.normalize_name(info["name"])
                or self.normalize_name(info["name"]) in q
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
        q = self.normalize_name(query_name) if query_name else None

        with self._lock:
            candidates = list(self._landmarks.values())
            if q:
                candidates = [
                    c for c in candidates
                    if self.names_match(c["name"], query_name)
                    or q in self.normalize_name(c["name"])
                    or self.normalize_name(c["name"]) in q
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

    def get_landmark_table(self) -> str:
        """Format registered landmarks into a clean Markdown table."""
        with self._lock:
            if not self._landmarks:
                return "No landmarks recorded."
            lines = [
                "| ID | Landmark Name | X (m) | Y (m) | Z (m) | Obs Count | Conf | Last Seen |",
                "|---|---|---|---|---|---|---|---|"
            ]
            now = time.time()
            for l in sorted(self._landmarks.values(), key=lambda x: x["count"], reverse=True):
                p = l["pos"]
                age = now - l["last_seen"]
                lines.append(
                    f"| {l['id']} | {l['name']} | {p[0]:.1f} | {p[1]:.1f} | {p[2]:.1f} | "
                    f"{l['count']} | {l['confidence']:.2f} | {age:.1f}s ago |"
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
