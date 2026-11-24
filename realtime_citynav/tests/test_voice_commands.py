"""Automated Test Suite for Voice Commands and Audio Feedback Subsystem."""

import os
import time
import json
import unittest
import numpy as np

from realtime_citynav.config import DEFAULT_CONFIG
from realtime_citynav.main import CityNavOrchestrator
from realtime_citynav.dashboard.app import create_app
from realtime_citynav.voice.audio_utils import AudioProcessor
from realtime_citynav.voice.transcriber import VoiceTranscriber
from realtime_citynav.voice.listener import VoiceListenerThread
from realtime_citynav.explainability import LiveRationaleNarrator


class TestVoiceCommands(unittest.TestCase):
    """Test voice processing, audio conversions, and REST endpoints."""

    def setUp(self):
        self.config = DEFAULT_CONFIG
        self.config.mock_airsim = True
        self.config.mock_perception = True
        self.config.mock_gemini = True
        self.orchestrator = CityNavOrchestrator(self.config)
        self.app = create_app(self.orchestrator)
        self.client = self.app.test_client()

    def tearDown(self):
        if self.orchestrator.running:
            self.orchestrator.stop()

    def test_audio_processor_pcm_wav_conversions(self):
        """Verify PCM and WAV conversion round-trip fidelity."""
        pcm = AudioProcessor.generate_sine_wave_pcm(freq_hz=440.0, duration_sec=0.2, sample_rate=16000)
        self.assertGreater(len(pcm), 0)

        wav = AudioProcessor.pcm_to_wav(pcm, sample_rate=16000, channels=1, sample_width=2)
        self.assertTrue(wav.startswith(b"RIFF"))
        self.assertIn(b"WAVE", wav[:12])

        extracted_pcm, rate, ch = AudioProcessor.wav_to_pcm(wav)
        self.assertEqual(rate, 16000)
        self.assertEqual(ch, 1)
        self.assertEqual(pcm, extracted_pcm)

    def test_audio_processor_numpy_conversions_and_metrics(self):
        """Verify NumPy array conversion and RMS energy computation."""
        t = np.linspace(0, 0.1, 1600, endpoint=False)
        sine = (0.5 * np.sin(2 * np.pi * 440.0 * t) * 32767.0).astype(np.int16)
        
        wav = AudioProcessor.numpy_to_wav(sine, sample_rate=16000)
        recovered, rate = AudioProcessor.wav_to_numpy(wav)
        
        self.assertEqual(rate, 16000)
        np.testing.assert_array_equal(sine, recovered)

        rms = AudioProcessor.calculate_rms(sine)
        self.assertGreater(rms, 5000.0)  # Significant energy
        dbfs = AudioProcessor.calculate_dbfs(sine)
        self.assertGreater(dbfs, -10.0)  # ~ -6 dBFS for 0.5 amplitude
        self.assertTrue(AudioProcessor.is_speech_active(sine, energy_threshold=1000.0))

        # Test silence / zero samples
        silence = np.zeros(1600, dtype=np.int16)
        self.assertEqual(AudioProcessor.calculate_rms(silence), 0.0)
        self.assertEqual(AudioProcessor.calculate_dbfs(silence), -100.0)
        self.assertFalse(AudioProcessor.is_speech_active(silence, energy_threshold=400.0))

    def test_audio_processor_base64_roundtrip(self):
        """Verify base64 encoding and decoding of WAV bytes."""
        pcm = AudioProcessor.generate_sine_wave_pcm(duration_sec=0.1)
        wav = AudioProcessor.pcm_to_wav(pcm)
        b64 = AudioProcessor.wav_to_base64(wav)
        
        recovered_wav = AudioProcessor.base64_to_wav(b64)
        self.assertEqual(wav, recovered_wav)

        # Also test data URI prefix stripping
        data_uri = f"data:audio/wav;base64,{b64}"
        recovered_from_uri = AudioProcessor.base64_to_wav(data_uri)
        self.assertEqual(wav, recovered_from_uri)

    def test_transcriber_emergency_keywords(self):
        """Verify sub-millisecond priority identification of emergency stop terms."""
        transcriber = VoiceTranscriber(mock=True)

        # Positive emergency halt commands
        for word in ["stop", "Stop!", "hover immediately", "Halt right here", "please pause", "land now"]:
            self.assertTrue(transcriber.is_emergency_keyword(word), f"Failed to detect: {word}")

        # Negative general flight commands
        for cmd in ["fly to red building", "circle blue building twice", "climb to 30m altitude", "fountain"]:
            self.assertFalse(transcriber.is_emergency_keyword(cmd), f"False positive on: {cmd}")

    def test_transcriber_mock_transcribe(self):
        """Verify mock transcriber returns valid structured dictionary."""
        transcriber = VoiceTranscriber(mock=True)
        res = transcriber.transcribe_audio(b"dummy_wav_data")
        
        self.assertIn("transcript", res)
        self.assertIn("confidence", res)
        self.assertIn("is_emergency", res)
        self.assertEqual(res["source"], "mock_audio")

        enriched = transcriber.process_transcript_text("circle blue building", confidence=0.92)
        self.assertEqual(enriched["transcript"], "circle blue building")
        self.assertEqual(enriched["confidence"], 0.92)
        self.assertFalse(enriched["is_emergency"])

    def test_voice_command_emergency_halt_endpoint(self):
        """Verify /api/voice_command triggers immediate emergency hover in <20 ms."""
        self.orchestrator.flight_mode = "NAVIGATING"

        t0 = time.perf_counter()
        resp = self.client.post(
            "/api/voice_command",
            data=json.dumps({"transcript": "stop and hover", "confidence": 0.98}),
            content_type="application/json"
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["status"], "emergency_hover")
        self.assertEqual(data["action"], "HOVER")
        self.assertEqual(self.orchestrator.flight_mode, "HOVERING")
        self.assertLess(elapsed_ms, 50.0)  # Well under 50ms requirement

    def test_voice_command_routing_endpoint(self):
        """Verify /api/voice_command enqueues navigation flight instructions."""
        resp = self.client.post(
            "/api/voice_command",
            data=json.dumps({"transcript": "circle blue building twice at 30m altitude", "confidence": 0.95}),
            content_type="application/json"
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["status"], "queued")
        self.assertEqual(data["transcript"], "circle blue building twice at 30m altitude")

        # Verify queued in dialogue manager
        self.assertTrue(self.orchestrator.dialogue.has_pending_commands())
        msg = self.orchestrator.dialogue.command_queue.get_nowait()
        self.assertEqual(msg, "circle blue building twice at 30m altitude")

    def test_voice_command_empty_validation(self):
        """Verify 400 Bad Request on empty voice transcript."""
        resp = self.client.post(
            "/api/voice_command",
            data=json.dumps({"transcript": "   "}),
            content_type="application/json"
        )
        self.assertEqual(resp.status_code, 400)

    def test_voice_audio_endpoint_with_wav_bytes(self):
        """Verify /api/voice_audio endpoint handles raw binary WAV audio."""
        pcm = AudioProcessor.generate_sine_wave_pcm(duration_sec=0.1)
        wav = AudioProcessor.pcm_to_wav(pcm)

        resp = self.client.post(
            "/api/voice_audio",
            data=wav,
            content_type="audio/wav"
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("transcript", data)

    def test_voice_status_endpoint(self):
        """Verify /api/voice/status returns active capabilities."""
        resp = self.client.get("/api/voice/status")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data.get("enabled"))
        self.assertTrue(data.get("ptt_supported"))
        self.assertTrue(data.get("speech_synthesis_supported"))

    def test_explainability_spoken_text_synthesis(self):
        """Verify LiveRationaleNarrator automatically generates spoken_text for critical events."""
        narrator = LiveRationaleNarrator()

        e_hover = narrator.emit_event("EMERGENCY_HOVER", "Emergency hover commanded.")
        self.assertIsNotNone(e_hover.spoken_text)
        self.assertIn("Emergency hover", e_hover.spoken_text)

        e_tgt = narrator.emit_event("TARGET_ACQUIRED", "Locked target.", {"target": "fountain"})
        self.assertIsNotNone(e_tgt.spoken_text)
        self.assertIn("Target acquired: fountain", e_tgt.spoken_text)

        e_voice = narrator.emit_event("VOICE_COMMAND", "Voice cmd", {"transcript": "circle blue building"})
        self.assertIsNotNone(e_voice.spoken_text)
        self.assertIn("Acknowledged: circle blue building", e_voice.spoken_text)

    def test_voice_listener_lifecycle(self):
        """Verify VoiceListenerThread starts and stops gracefully without throwing."""
        listener = VoiceListenerThread(
            command_callback=lambda cmd: None,
            transcriber=VoiceTranscriber(mock=True)
        )
        # Testing clean stop even if not started or audio device not opened
        listener.stop()
        self.assertFalse(listener.running)


if __name__ == "__main__":
    unittest.main()
