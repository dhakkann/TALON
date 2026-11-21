"""Unit tests for Dialogue Manager and Interruption Classification."""

import unittest
from realtime_citynav.dialogue_manager import DialogueManager


class TestDialogueManager(unittest.TestCase):
    def setUp(self):
        # Initialized without API key to test resilient heuristic engine
        self.dm = DialogueManager(api_key=None)

    def test_instruction_decomposition(self):
        """Natural language navigation instructions decompose into ordered subgoals."""
        instruction = "Fly to the fountain, then pass the red building, and navigate to the tower"
        subgoals = self.dm.parse_initial_instruction(instruction)
        
        self.assertIsInstance(subgoals, list)
        self.assertEqual(len(subgoals), 3)
        self.assertEqual(subgoals, ["fountain", "red building", "tower"])

    def test_target_correction_interruption(self):
        """Interruption with target correction switches destination and cleans prefixes."""
        self.dm.queue_user_message("no, the blue building")
        self.assertTrue(self.dm.has_pending_commands())

        result = self.dm.process_interruption(current_target="red building")
        self.assertIsNotNone(result)
        self.assertEqual(result["intent"], "TARGET_CORRECTION")
        self.assertEqual(result["target"], "blue building")

    def test_return_to_past_interruption(self):
        """Interruption commanding return to past landmark routes back and strips articles."""
        self.dm.queue_user_message("go back to the fountain")
        result = self.dm.process_interruption(
            current_target="tower",
            memory_summary="- fountain at [10, 20, -5]"
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["intent"], "RETURN_TO_PAST")
        self.assertEqual(result["target"], "fountain")

    def test_memory_query_interruption(self):
        """Operator query returns spatial memory summary."""
        self.dm.queue_user_message("what have you seen so far?")
        result = self.dm.process_interruption(
            current_target="tower",
            memory_summary="- fountain (ID 1)\n- red building (ID 2)"
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["intent"], "QUERY")
        self.assertIn("fountain", result["reply"])

    def test_hover_stop_interruption(self):
        """Operator stop command triggers hover."""
        self.dm.queue_user_message("stop and hover right here")
        result = self.dm.process_interruption(current_target="tower")
        self.assertIsNotNone(result)
        self.assertEqual(result["intent"], "HOVER_STOP")

    def test_empty_queue_non_blocking(self):
        """Processing with an empty queue returns None immediately."""
        self.assertFalse(self.dm.has_pending_commands())
        result = self.dm.process_interruption()
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
