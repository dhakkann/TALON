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

    def test_compound_spatial_prepositions_parsing(self):
        """Compound prepositions (vertical + horizontal) extract offsets and base landmark cleanly."""
        res1 = self.dm.parse_spatial_relation("go 20m above and 15m behind the red building")
        self.assertEqual(res1["relation"], "COMPOUND")
        self.assertEqual(res1["altitude_offset"], 20.0)
        self.assertEqual(res1["horizontal_relation"], "BEHIND")
        self.assertEqual(res1["horizontal_distance"], 15.0)
        self.assertEqual(res1["base_landmark"], "red building")

        res2 = self.dm.parse_spatial_relation("fly 10m to the left and 25m above tower")
        self.assertEqual(res2["relation"], "COMPOUND")
        self.assertEqual(res2["altitude_offset"], 25.0)
        self.assertEqual(res2["horizontal_relation"], "LEFT")
        self.assertEqual(res2["horizontal_distance"], 10.0)
        self.assertEqual(res2["base_landmark"], "tower")

    def test_action_verbs_parsing(self):
        """Action verbs like 'ascend to' parse spatial altitude relations correctly."""
        res = self.dm.parse_spatial_relation("ascend to 50m above fountain")
        self.assertEqual(res["relation"], "ABOVE")
        self.assertEqual(res["distance"], 50.0)
        self.assertEqual(res["base_landmark"], "fountain")

        subgoals = self.dm.parse_initial_instruction("ascend to 50m above fountain")
        self.assertEqual(subgoals, ["50m above fountain"])

    def test_punctuation_only_message_does_not_divert(self):
        """Punctuation-only messages ('...', '???') are handled gracefully as queries, not target diversions."""
        self.dm.queue_user_message("...")
        res = self.dm.process_interruption(current_target="tower")
        self.assertIsNotNone(res)
        self.assertNotEqual(res["intent"], "TARGET_CORRECTION")
        self.assertIsNone(res["target"])

    def test_multi_leg_route_from_to_parsing(self):
        """Multi-leg routes with 'from X to Y' and shorthand resolve into ordered subgoals."""
        # Multi-leg route where second leg refers to origin by shorthand adjective 'red'
        subgoals1 = self.dm.parse_initial_instruction(
            "go from fountain to red building, and also go from red to blue building"
        )
        self.assertEqual(subgoals1, ["fountain", "red building", "blue building"])

        # Standalone from-to route with adjective ellipsis
        subgoals2 = self.dm.parse_initial_instruction("go from red to blue building")
        self.assertEqual(subgoals2, ["red building", "blue building"])

        # Chained route with 'and then to'
        subgoals3 = self.dm.parse_initial_instruction(
            "fly from the fountain to the red building and then to the blue building"
        )
        self.assertEqual(subgoals3, ["fountain", "red building", "blue building"])

        # Route with trailing comma-separated waypoints
        subgoals4 = self.dm.parse_initial_instruction(
            "from fountain to red building, then blue building, then hospital"
        )
        self.assertEqual(subgoals4, ["fountain", "red building", "blue building", "hospital"])

    def test_shorthand_landmark_resolution(self):
        """Adjective and pronoun shorthands ('red', 'the blue one') resolve to full landmark names."""
        self.assertEqual(self.dm.resolve_landmark_shorthand("the red one"), "red building")
        self.assertEqual(self.dm.resolve_landmark_shorthand("red"), "red building")
        self.assertEqual(self.dm.resolve_landmark_shorthand("the blue one"), "blue building")
        self.assertEqual(self.dm.resolve_landmark_shorthand("blue"), "blue building")
        self.assertEqual(self.dm.resolve_landmark_shorthand("50m above red"), "50m above red building")

    def test_interruption_with_shorthand(self):
        """Operator interruptions using shorthand ('switch to red', 'return to blue') resolve accurately."""
        self.dm.queue_user_message("switch to red")
        res1 = self.dm.process_interruption(current_target="fountain")
        self.assertIsNotNone(res1)
        self.assertEqual(res1["intent"], "TARGET_CORRECTION")
        self.assertEqual(res1["target"], "red building")
        self.assertEqual(res1["base_landmark"], "red building")

        self.dm.queue_user_message("go back to blue")
        res2 = self.dm.process_interruption(current_target="red building")
        self.assertIsNotNone(res2)
        self.assertEqual(res2["intent"], "RETURN_TO_PAST")
        self.assertEqual(res2["target"], "blue building")

    def test_collective_quantifier_expansion(self):
        """Collective phrases like 'go to every landmark' expand to full ordered landmark list."""
        all_landmarks = ["fountain", "red building", "blue building", "tower", "hospital"]
        
        for phrase in [
            "go to every landmark",
            "visit all landmarks",
            "tour the city",
            "explore each landmark",
            "fly to every building"
        ]:
            subgoals = self.dm.parse_initial_instruction(phrase)
            self.assertEqual(subgoals, all_landmarks, f"Failed for phrase: '{phrase}'")

    def test_hover_with_spatial_relation(self):
        """Commands like 'hover 100m above red building' route as TARGET_CORRECTION, not HOVER_STOP."""
        # As an interruption
        self.dm.queue_user_message("hover 100m above red building")
        res = self.dm.process_interruption(current_target="fountain")
        self.assertIsNotNone(res)
        self.assertEqual(res["intent"], "TARGET_CORRECTION")
        self.assertEqual(res["target"], "100m above red building")
        self.assertEqual(res["base_landmark"], "red building")
        self.assertEqual(res["spatial"]["relation"], "ABOVE")
        self.assertEqual(res["spatial"]["distance"], 100.0)

        # As an initial instruction
        subgoals = self.dm.parse_initial_instruction("hover 100m above red building")
        self.assertEqual(subgoals, ["100m above red building"])


if __name__ == "__main__":
    unittest.main()
