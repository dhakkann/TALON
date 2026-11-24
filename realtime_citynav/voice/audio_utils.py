"""Audio processing utilities for CityNav voice command system.

Provides PCM/WAV conversions, base64 encoding/decoding, energy-based
Voice Activity Detection (VAD), and audio buffer management.
"""

import io
import wave
import base64
import math
from typing import Optional, Tuple, List
import numpy as np


class AudioProcessor:
    """Utilities for audio decoding, normalization, and energy-based VAD."""

    DEFAULT_SAMPLE_RATE = 16000
    DEFAULT_CHANNELS = 1
    DEFAULT_SAMPLE_WIDTH = 2  # 16-bit PCM

    @classmethod
    def pcm_to_wav(
        cls,
        pcm_bytes: bytes,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        channels: int = DEFAULT_CHANNELS,
        sample_width: int = DEFAULT_SAMPLE_WIDTH
    ) -> bytes:
        """Wrap raw PCM bytes with a standard WAV RIFF header."""
        out_buf = io.BytesIO()
        with wave.open(out_buf, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_bytes)
        return out_buf.getvalue()

    @classmethod
    def wav_to_pcm(cls, wav_bytes: bytes) -> Tuple[bytes, int, int]:
        """Extract raw PCM bytes, sample rate, and channels from WAV bytes."""
        in_buf = io.BytesIO(wav_bytes)
        with wave.open(in_buf, "rb") as wf:
            channels = wf.getnchannels()
            sample_rate = wf.getframerate()
            pcm_frames = wf.readframes(wf.getnframes())
        return pcm_frames, sample_rate, channels

    @classmethod
    def numpy_to_wav(
        cls,
        samples: np.ndarray,
        sample_rate: int = DEFAULT_SAMPLE_RATE
    ) -> bytes:
        """Convert a 1D numpy array of audio samples into standard WAV bytes."""
        # Ensure int16
        if samples.dtype != np.int16:
            # Normalize float to int16 if needed
            if np.issubdtype(samples.dtype, np.floating):
                samples = np.clip(samples, -1.0, 1.0)
                samples = (samples * 32767.0).astype(np.int16)
            else:
                samples = samples.astype(np.int16)
        return cls.pcm_to_wav(samples.tobytes(), sample_rate=sample_rate)

    @classmethod
    def wav_to_numpy(cls, wav_bytes: bytes) -> Tuple[np.ndarray, int]:
        """Convert WAV bytes into a 1D float32 or int16 numpy array."""
        pcm_bytes, sample_rate, channels = cls.wav_to_pcm(wav_bytes)
        samples = np.frombuffer(pcm_bytes, dtype=np.int16)
        if channels > 1:
            # Downmix to mono by averaging channels
            samples = samples.reshape(-1, channels).mean(axis=1).astype(np.int16)
        return samples, sample_rate

    @classmethod
    def base64_to_wav(cls, b64_str: str) -> bytes:
        """Decode base64 string to WAV bytes, stripping data URI prefix if present."""
        if "," in b64_str:
            b64_str = b64_str.split(",", 1)[1]
        return base64.b64decode(b64_str.strip())

    @classmethod
    def wav_to_base64(cls, wav_bytes: bytes) -> str:
        """Encode WAV bytes to a base64 string."""
        return base64.b64encode(wav_bytes).decode("ascii")

    @classmethod
    def calculate_rms(cls, samples: np.ndarray) -> float:
        """Calculate Root-Mean-Square (RMS) amplitude of audio samples."""
        if len(samples) == 0:
            return 0.0
        float_samples = samples.astype(np.float64)
        mean_square = np.mean(float_samples ** 2)
        return float(np.sqrt(mean_square))

    @classmethod
    def calculate_dbfs(cls, samples: np.ndarray) -> float:
        """Calculate Decibels relative to Full Scale (dBFS). Max is 0.0 dBFS."""
        rms = cls.calculate_rms(samples)
        if rms <= 1e-6:
            return -100.0
        # For 16-bit audio, full scale square wave max is 32767
        db = 20.0 * math.log10(rms / 32767.0)
        return max(-100.0, float(db))

    @classmethod
    def is_speech_active(
        cls,
        samples: np.ndarray,
        energy_threshold: float = 400.0
    ) -> bool:
        """Determine if audio chunk contains speech based on energy threshold."""
        return cls.calculate_rms(samples) >= energy_threshold

    @classmethod
    def generate_sine_wave_pcm(
        cls,
        freq_hz: float = 440.0,
        duration_sec: float = 1.0,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        amplitude: float = 0.5
    ) -> bytes:
        """Generate synthetic sine wave PCM audio bytes for test harnesses."""
        t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
        wave_data = amplitude * np.sin(2 * np.pi * freq_hz * t)
        samples = (wave_data * 32767.0).astype(np.int16)
        return samples.tobytes()
