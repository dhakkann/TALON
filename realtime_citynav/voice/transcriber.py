"""Voice Transcriber integrating Gemini Multimodal Audio and heuristic fallbacks.

Transcribes audio streams into structured text flight instructions,
extracting emergency halt commands with sub-millisecond priority checks.
"""

import os
import re
import json
import time
from typing import Optional, Dict, Any

try:
    from google import genai
    from google.genai import types
    _GENAI_AVAILABLE = True
except ImportError:
    _GENAI_AVAILABLE = False


class VoiceTranscriber:
    """Multi-tier Speech-to-Text transcriber for flight commands."""

    EMERGENCY_WORDS = {"stop", "hover", "halt", "pause", "wait", "land"}

    def __init__(
        self,
        api_key: str = "",
        model_name: str = "gemini-3.5-flash",
        mock: bool = False
    ):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self.model_name = model_name
        self.mock = mock or (not self.api_key and not os.environ.get("GEMINI_API_KEY"))
        self.client = None

        if not self.mock and _GENAI_AVAILABLE and self.api_key:
            try:
                self.client = genai.Client(api_key=self.api_key)
                print(f"[VoiceTranscriber] Initialized Gemini client with model '{self.model_name}'.")
            except Exception as e:
                print(f"[VoiceTranscriber] Gemini client init failed ({e}), falling back to mock mode.")
                self.mock = True
        else:
            self.mock = True

    def is_emergency_keyword(self, text: str) -> bool:
        """Sub-millisecond priority check for immediate emergency flight halt."""
        if not text:
            return False
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        return any(w in self.EMERGENCY_WORDS for w in words)

    def transcribe_audio(
        self,
        audio_bytes: bytes,
        mime_type: str = "audio/wav"
    ) -> Dict[str, Any]:
        """Transcribe raw audio bytes into text and structured flight intent.
        
        Args:
            audio_bytes: Binary audio data (typically 16kHz mono WAV).
            mime_type: MIME type of the audio stream.
            
        Returns:
            Dict containing:
                'transcript': Transcribed natural language text.
                'confidence': Confidence score (0.0 - 1.0).
                'is_emergency': True if halt/stop was commanded.
                'source': 'gemini_audio' or 'mock_audio'.
        """
        t0 = time.perf_counter()

        if self.mock or self.client is None:
            # Deterministic offline mock path
            return {
                "transcript": "stop and hover",
                "confidence": 1.0,
                "is_emergency": True,
                "source": "mock_audio",
                "latency_ms": (time.perf_counter() - t0) * 1000.0
            }

        prompt = (
            "Listen carefully to this spoken drone operator flight instruction. "
            "Return a JSON object with: "
            "'transcript' (the exact spoken English words), "
            "'confidence' (float 0.0 to 1.0), "
            "'is_emergency' (boolean, true if operator commanded stop, hover, halt, or land)."
        )

        try:
            part = types.Part.from_bytes(data=audio_bytes, mime_type=mime_type)
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[part, prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.0
                )
            )
            raw = response.text.strip()
            # Clean markdown JSON wrapping if present
            if raw.startswith("```json"):
                raw = raw[7:]
            if raw.endswith("```"):
                raw = raw[:-3]
            parsed = json.loads(raw.strip())
            transcript = str(parsed.get("transcript", "")).strip()
            confidence = float(parsed.get("confidence", 0.95))
            is_emergency = bool(parsed.get("is_emergency", False)) or self.is_emergency_keyword(transcript)

            return {
                "transcript": transcript,
                "confidence": confidence,
                "is_emergency": is_emergency,
                "source": "gemini_audio",
                "latency_ms": (time.perf_counter() - t0) * 1000.0
            }
        except Exception as e:
            print(f"[VoiceTranscriber] Gemini audio transcription error: {e}")
            return {
                "transcript": "",
                "confidence": 0.0,
                "is_emergency": False,
                "error": str(e),
                "source": "error_fallback",
                "latency_ms": (time.perf_counter() - t0) * 1000.0
            }

    def process_transcript_text(
        self,
        transcript: str,
        confidence: float = 1.0
    ) -> Dict[str, Any]:
        """Validate and enrich client-side Web Speech transcript."""
        clean = transcript.strip()
        is_emergency = self.is_emergency_keyword(clean)
        return {
            "transcript": clean,
            "confidence": max(0.0, min(1.0, float(confidence))),
            "is_emergency": is_emergency,
            "source": "web_speech"
        }
