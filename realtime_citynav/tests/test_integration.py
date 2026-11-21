"""Integration tests for the Real-Time CityNav Orchestrator and Web Endpoints."""

import unittest
import time
import numpy as np
from realtime_citynav.config import SystemConfig
from realtime_citynav.main import CityNavOrchestrator
from realtime_citynav.dashboard.app import create_app


class TestEndToEndIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Configure for fast headless integration testing
        cls.config = SystemConfig()
        cls.config.mock_airsim = True
        cls.config.mock_perception = True
        cls.config.mock_gemini = True
        cls.config.dashboard.port = 5055  # Use non-default port for tests

        cls.orchestrator = CityNavOrchestrator(cls.config)
        cls.orchestrator.start()
        time.sleep(0.3)
        # Start initial mission
        cls.orchestrator.start_mission("Fly to the fountain, then pass the red building, and navigate to the tower")
        time.sleep(0.3)  # Allow background threads to spin up

    @classmethod
    def tearDownClass(cls):
        cls.orchestrator.stop()

    def test_mission_start_and_subgoal_planning(self):
        """Starting a mission decomposes instruction and commands takeoff."""
        instruction = "Fly to the fountain, then pass the red building, and navigate to the tower"
        subgoals = self.orchestrator.start_mission(instruction)

        self.assertGreaterEqual(len(subgoals), 2)
        self.assertTrue(self.orchestrator.flight.is_in_air)
        self.assertEqual(self.orchestrator.flight_mode, "NAVIGATING")
        self.assertIsNotNone(self.orchestrator.active_target_name)

    def test_live_telemetry_generation(self):
        """Telemetry snapshot produces valid flight and perception metrics."""
        telemetry = self.orchestrator.get_telemetry()
        self.assertIn("position", telemetry)
        self.assertIn("altitude", telemetry)
        self.assertIn("speed", telemetry)
        self.assertIn("active_target", telemetry)
        self.assertTrue(telemetry["is_mock"])
        self.assertGreaterEqual(telemetry["altitude"], 4.0)

    def test_mid_flight_interruption_correction(self):
        """Operator mid-flight target correction redirects active objective."""
        # Operator interrupts: "no, the blue building"
        self.orchestrator.dialogue.queue_user_message("no, the blue building")

        # Allow cognitive loop to process queue
        time.sleep(0.4)

        with self.orchestrator._state_lock:
            current_target = self.orchestrator.active_target_name

        self.assertIsNotNone(current_target)
        self.assertIn("blue building", current_target.lower())

    def test_spatial_memory_population_and_query(self):
        """Registered landmarks populate in memory and answer queries."""
        # Add a landmark
        pos = np.array([15.0, -10.0, -5.0])
        self.orchestrator.memory.add_observation("park bench", pos)

        # Operator queries: "what have you seen?"
        self.orchestrator.dialogue.queue_user_message("what have you seen?")
        time.sleep(0.4)

        recent_narrations = self.orchestrator.narrator.get_recent(5)
        # Check that one of the recent narrations acknowledges operator query
        types = [n["event_type"] for n in recent_narrations]
        self.assertTrue(any(t in ["INTERRUPTION_HANDLED", "OPERATOR_INPUT"] for t in types))

    def test_dashboard_api_endpoints(self):
        """Flask dashboard endpoints return valid HTTP responses."""
        client = self.orchestrator._dashboard_server.app.test_client()

        # Test index page
        res = client.get("/")
        self.assertEqual(res.status_code, 200)

        # Test telemetry endpoint
        res = client.get("/api/telemetry")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("altitude", data)

        # Test landmarks endpoint
        res = client.get("/api/landmarks")
        self.assertEqual(res.status_code, 200)
        self.assertIsInstance(res.get_json(), list)

        # Test chat API
        res = client.post("/api/chat", json={"message": "hover"})
        self.assertEqual(res.status_code, 200)

    def test_video_frame_rendering(self):
        """Latest annotated frame is generated without error."""
        # Wait for perception loop to render at least one frame
        time.sleep(0.2)
        frame = self.orchestrator.get_latest_annotated_frame()
        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape, (480, 640, 3))

    def test_emergency_hover(self):
        """Emergency hover immediately brakes vehicle and switches flight mode to HOVERING."""
        self.orchestrator.emergency_hover()
        self.assertEqual(self.orchestrator.flight_mode, "HOVERING")
        self.assertTrue(self.orchestrator.flight.is_hovering)
        self.assertIsNone(self.orchestrator.flight.target_position)
        state = self.orchestrator.flight.step_control()
        self.assertAlmostEqual(state["cmd_velocity"][0], 0.0, places=2)
        self.assertAlmostEqual(state["cmd_velocity"][1], 0.0, places=2)


if __name__ == "__main__":
    unittest.main()
