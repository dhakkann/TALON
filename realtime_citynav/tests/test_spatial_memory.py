"""Unit tests for Persistent 3D Spatial Landmark Registry."""

import unittest
import threading
import numpy as np
from realtime_citynav.spatial_memory import SpatialMemory


class TestSpatialMemory(unittest.TestCase):
    def setUp(self):
        self.memory = SpatialMemory(cluster_threshold=5.0, alpha=0.2)

    def test_single_observation_registration(self):
        """Registering a landmark observation creates an entry with ID 1."""
        pos = np.array([10.0, 20.0, -5.0])
        lid = self.memory.add_observation("fountain", pos, confidence=0.9)
        self.assertEqual(lid, 1)

        entry = self.memory.find_landmark("fountain")
        self.assertIsNotNone(entry)
        self.assertEqual(entry["name"], "fountain")
        self.assertEqual(entry["count"], 1)
        np.testing.assert_allclose(entry["pos"], pos)

    def test_euclidean_clustering_merge(self):
        """Observations within cluster threshold (5m) merge and update running average."""
        pos1 = np.array([10.0, 20.0, -5.0])
        lid1 = self.memory.add_observation("fountain", pos1)
        self.assertEqual(lid1, 1)

        # Second observation 2 meters away (well within 5m threshold)
        pos2 = np.array([12.0, 20.0, -5.0])
        lid2 = self.memory.add_observation("fountain", pos2)
        self.assertEqual(lid2, 1, "Should merge into same landmark ID")

        entry = self.memory.find_landmark("fountain")
        self.assertEqual(entry["count"], 2)
        # Expected position: (1 - 0.2) * [10, 20, -5] + 0.2 * [12, 20, -5] = [10.4, 20.0, -5.0]
        expected_pos = np.array([10.4, 20.0, -5.0])
        np.testing.assert_allclose(entry["pos"], expected_pos, atol=1e-3)

    def test_distinct_landmarks_separated_by_distance(self):
        """Same class beyond cluster threshold (e.g. 20m apart) creates distinct landmarks."""
        pos1 = np.array([0.0, 0.0, -5.0])
        pos2 = np.array([50.0, 50.0, -5.0])

        lid1 = self.memory.add_observation("building", pos1)
        lid2 = self.memory.add_observation("building", pos2)

        self.assertNotEqual(lid1, lid2)
        all_lms = self.memory.get_all_landmarks()
        self.assertEqual(len(all_lms), 2)

    def test_different_classes_do_not_merge(self):
        """Different classes at the same coordinates remain distinct."""
        pos = np.array([10.0, 10.0, -5.0])
        lid1 = self.memory.add_observation("car", pos)
        lid2 = self.memory.add_observation("tree", pos)

        self.assertNotEqual(lid1, lid2)
        self.assertEqual(len(self.memory.get_all_landmarks()), 2)

    def test_closest_landmark_query(self):
        """Querying closest landmark returns nearest geographic neighbor."""
        self.memory.add_observation("tower", np.array([100.0, 0.0, -5.0]))
        self.memory.add_observation("statue", np.array([10.0, 0.0, -5.0]))

        query_pos = np.array([12.0, 1.0, -5.0])
        closest = self.memory.get_closest_landmark(query_pos)
        self.assertIsNotNone(closest)
        self.assertEqual(closest["name"], "statue")

    def test_summary_formatting(self):
        """Summary contains all landmark records."""
        self.memory.add_observation("fountain", np.array([5.0, 10.0, -5.0]))
        summary = self.memory.get_summary()
        self.assertIn("fountain", summary)
        self.assertIn("observed 1x", summary)

    def test_concurrent_multithreaded_writes(self):
        """Thread safety under concurrent writes from multiple threads."""
        def worker(thread_id):
            for i in range(50):
                self.memory.add_observation(
                    f"landmark_{thread_id}",
                    np.array([float(thread_id * 10), float(i), -5.0])
                )

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        lms = self.memory.get_all_landmarks()
        self.assertGreater(len(lms), 0)

    def test_nan_coordinates_rejected(self):
        """Registering landmarks with NaN or Inf coordinates raises ValueError."""
        with self.assertRaises(ValueError):
            self.memory.add_observation("bad_landmark", np.array([np.nan, 0.0, -5.0]))
        with self.assertRaises(ValueError):
            self.memory.add_observation("bad_landmark", np.array([0.0, np.inf, -5.0]))
        self.assertEqual(len(self.memory.get_all_landmarks()), 0)

    def test_prune_stale_landmarks(self):
        """Stale landmarks older than retention period are successfully pruned."""
        self.memory.add_observation("old_fountain", np.array([10.0, 10.0, -5.0]))
        # Manually backdate timestamp
        self.memory._landmarks[1]["last_seen"] -= 4000.0
        self.memory.add_observation("fresh_tower", np.array([50.0, 50.0, -5.0]))

        pruned = self.memory.prune_stale(max_age_sec=3600.0)
        self.assertEqual(pruned, 1)
        remaining = self.memory.get_all_landmarks()
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0]["name"], "fresh_tower")

    def test_empty_query_returns_none(self):
        """Empty or whitespace queries return None instead of matching all landmarks."""
        self.memory.add_observation("fountain", np.array([10.0, 10.0, -5.0]))
        self.assertIsNone(self.memory.find_landmark(""))
        self.assertIsNone(self.memory.find_landmark("   "))

    def test_hyphenated_and_synonym_normalization(self):
        """Hyphenated words and synonyms match and deduplicate seamlessly."""
        self.assertTrue(self.memory.names_match("water-fountain", "fountain"))
        self.assertTrue(self.memory.names_match("The Red-Building", "red building"))
        self.assertTrue(self.memory.names_match("buildings", "building"))
        self.assertTrue(self.memory.names_match("cars", "car"))

        self.memory.add_observation("water-fountain", np.array([32.0, 0.0, 0.0]))
        self.memory.add_observation("The Fountain", np.array([32.2, 0.1, 0.0]))
        all_lms = self.memory.get_all_landmarks()
        self.assertEqual(len(all_lms), 1)
        self.assertEqual(all_lms[0]["count"], 2)

    def test_prune_unconfirmed_noise(self):
        """Unconfirmed single-observation landmarks older than threshold are pruned."""
        lid1 = self.memory.add_observation("spurious_car", np.array([10.0, 10.0, -5.0]))
        self.memory._landmarks[lid1]["last_seen"] -= 60.0

        lid2 = self.memory.add_observation("confirmed_fountain", np.array([20.0, 20.0, -5.0]))
        self.memory.add_observation("confirmed_fountain", np.array([20.1, 20.0, -5.0]))

        pruned = self.memory.prune_unconfirmed(min_count=2, max_age_sec=30.0)
        self.assertEqual(pruned, 1)
        remaining = self.memory.get_all_landmarks()
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0]["name"], "confirmed_fountain")


if __name__ == "__main__":
    unittest.main()
