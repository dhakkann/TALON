"""Background Microphone Listener Thread using sounddevice.

Enables hands-free voice control directly from the host laptop microphone
even when the browser Cockpit window is minimized or unfocused.
"""

import time
import threading
from typing import Optional, Callable
import numpy as np

from .audio_utils import AudioProcessor
from .transcriber import VoiceTranscriber

try:
    import sounddevice as sd
    _SOUNDDEVICE_AVAILABLE = True
except (ImportError, OSError):
    _SOUNDDEVICE_AVAILABLE = False


class VoiceListenerThread:
    """Captures laptop microphone input and dispatches voice commands to the orchestrator."""

    def __init__(
        self,
        command_callback: Optional[Callable[[str], None]] = None,
        transcriber: Optional[VoiceTranscriber] = None,
        sample_rate: int = 16000,
        energy_threshold: float = 500.0,
        device_index: Optional[int] = None
    ):
        self.command_callback = command_callback
        self.transcriber = transcriber or VoiceTranscriber(mock=True)
        self.sample_rate = sample_rate
        self.energy_threshold = energy_threshold
        self.device_index = device_index

        self.running = False
        self.thread: Optional[threading.Thread] = None
        self._stream = None

    def start(self) -> bool:
        """Start background microphone listening thread."""
        if not _SOUNDDEVICE_AVAILABLE:
            print("[VoiceListener] sounddevice not available or no audio device found. Listener disabled.")
            return False

        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True, name="VoiceListenerThread")
        self.thread.start()
        print(f"[VoiceListener] Local microphone listening active on device {self.device_index or 'default'}.")
        return True

    def stop(self) -> None:
        """Stop listening and release audio device."""
        self.running = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        print("[VoiceListener] Microphone listener stopped.")

    def _run_loop(self) -> None:
        """Continuous audio capture and energy VAD loop."""
        chunk_duration = 0.1  # 100ms per block
        block_size = int(self.sample_rate * chunk_duration)

        buffer = []
        speech_started = False
        silence_start_time = None

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="int16",
                blocksize=block_size,
                device=self.device_index
            ) as stream:
                self._stream = stream
                while self.running:
                    data, overflowed = stream.read(block_size)
                    if overflowed:
                        pass
                    
                    samples = data.flatten()
                    energy = AudioProcessor.calculate_rms(samples)

                    if energy >= self.energy_threshold:
                        if not speech_started:
                            speech_started = True
                            buffer = []
                        buffer.append(samples)
                        silence_start_time = None
                    elif speech_started:
                        buffer.append(samples)
                        if silence_start_time is None:
                            silence_start_time = time.time()
                        elif time.time() - silence_start_time > 0.6:
                            # 600ms silence detected after speech -> process utterance
                            speech_started = False
                            silence_start_time = None
                            if buffer:
                                full_audio = np.concatenate(buffer)
                                buffer = []
                                # Only process utterances longer than 0.5 sec
                                if len(full_audio) >= int(self.sample_rate * 0.5):
                                    wav_bytes = AudioProcessor.numpy_to_wav(full_audio, self.sample_rate)
                                    result = self.transcriber.transcribe_audio(wav_bytes)
                                    text = result.get("transcript", "").strip()
                                    if text and self.command_callback:
                                        print(f"[VoiceListener] Recognized speech: \"{text}\"")
                                        self.command_callback(text)
        except Exception as e:
            print(f"[VoiceListener] Audio input stream error: {e}")
            self.running = False
