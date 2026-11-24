"""Flask and SSE Web Dashboard Server for Real-Time CityNav Drone Operations."""

import time
import json
import logging
import threading
from typing import Optional, Dict, Any
from flask import Flask, render_template, Response, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np

from realtime_citynav.voice import AudioProcessor, VoiceTranscriber

# Suppress verbose Flask request logging to keep console clean
logging.getLogger("werkzeug").setLevel(logging.ERROR)


def create_app(system_orchestrator=None):
    """Factory creating Flask dashboard application."""
    app = Flask(__name__, template_folder="templates")
    CORS(app)

    # Reference to orchestrator holding flight controller, perception, memory, dialogue
    app.orchestrator = system_orchestrator

    # Initialize voice transcriber
    transcriber = None
    if app.orchestrator is not None:
        transcriber = getattr(app.orchestrator, "voice_transcriber", None)
    if transcriber is None:
        mock_mode = app.orchestrator.config.mock_gemini if app.orchestrator else True
        api_key = app.orchestrator.config.cognitive.api_key if app.orchestrator else ""
        transcriber = VoiceTranscriber(api_key=api_key, mock=mock_mode)
    app.voice_transcriber = transcriber

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/video_feed")
    def video_feed():
        """MJPEG live camera streaming endpoint."""
        def generate_frames():
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(
                blank, "Awaiting Camera Feed...", (160, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2
            )
            _, blank_encoded = cv2.imencode(".jpg", blank)
            blank_bytes = blank_encoded.tobytes()

            while True:
                frame_bytes = None
                if app.orchestrator is not None:
                    annotated = app.orchestrator.get_latest_annotated_frame()
                    if annotated is not None:
                        success, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
                        if success:
                            frame_bytes = buf.tobytes()

                if frame_bytes is None:
                    frame_bytes = blank_bytes

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                )
                time.sleep(0.04)  # ~25 FPS stream cap

        return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/events")
    def sse_events():
        """Server-Sent Events streaming telemetry and live narrations."""
        def event_stream():
            last_event_count = 0
            while True:
                if app.orchestrator is not None:
                    telemetry = app.orchestrator.get_telemetry()
                    narrations = app.orchestrator.narrator.get_recent(15)
                    landmarks = app.orchestrator.memory.get_all_landmarks()

                    # Convert numpy arrays to lists
                    for lm in landmarks:
                        if isinstance(lm.get("pos"), np.ndarray):
                            lm["pos"] = lm["pos"].tolist()

                    payload = {
                        "telemetry": telemetry,
                        "narrations": narrations,
                        "landmarks": landmarks,
                        "timestamp": time.time()
                    }
                    data_str = json.dumps(payload)
                    yield f"data: {data_str}\n\n"
                else:
                    yield f"data: {json.dumps({'status': 'idle'})}\n\n"
                time.sleep(0.1)  # 10 Hz telemetry push

        return Response(event_stream(), mimetype="text/event-stream")

    @app.route("/api/chat", methods=["POST"])
    def chat_command():
        """Operator chat input endpoint."""
        body = request.get_json(force=True, silent=True) or {}
        message = body.get("message", "").strip()
        if not message:
            return jsonify({"status": "error", "message": "Empty message"}), 400

        if app.orchestrator is not None:
            app.orchestrator.dialogue.queue_user_message(message)
            app.orchestrator.narrator.emit_event(
                "OPERATOR_INPUT",
                f"Operator: \"{message}\"",
                {"raw_message": message}
            )
            return jsonify({"status": "queued", "message": message})
        return jsonify({"status": "ok", "message": message})

    @app.route("/api/voice_command", methods=["POST"])
    def voice_command():
        """Process transcribed speech input from Web Speech API or client STT."""
        body = request.get_json(force=True, silent=True) or {}
        transcript = body.get("transcript", "").strip()
        confidence = float(body.get("confidence", 1.0))

        if not transcript:
            return jsonify({"status": "error", "message": "Empty transcript"}), 400

        if app.orchestrator is not None:
            # 1. Instant emergency halt fast-path (<1ms)
            if app.orchestrator.dialogue.is_hover_stop_command(transcript):
                app.orchestrator.emergency_hover()
                app.orchestrator.narrator.emit_event(
                    "EMERGENCY_HOVER",
                    f"Voice Emergency Halt: \"{transcript}\"",
                    {"transcript": transcript, "confidence": confidence},
                    spoken_text="Emergency hover engaged. Holding position."
                )
                return jsonify({
                    "status": "emergency_hover",
                    "action": "HOVER",
                    "transcript": transcript,
                    "confidence": confidence
                })

            # 2. General flight command routing
            app.orchestrator.queue_command(transcript)
            app.orchestrator.narrator.emit_event(
                "VOICE_COMMAND",
                f"Voice Operator ({confidence * 100:.0f}%): \"{transcript}\"",
                {"transcript": transcript, "confidence": confidence},
                spoken_text=f"Acknowledged: {transcript}"
            )
            return jsonify({
                "status": "queued",
                "transcript": transcript,
                "confidence": confidence
            })

        return jsonify({"status": "ok", "transcript": transcript})

    @app.route("/api/voice_audio", methods=["POST"])
    def voice_audio():
        """Process raw audio payload (WAV/WebM) via server-side VoiceTranscriber."""
        audio_bytes = b""
        mime_type = request.content_type or "audio/wav"

        # Check if multipart form
        if request.files and "audio" in request.files:
            audio_bytes = request.files["audio"].read()
        elif request.data:
            # Raw binary body
            audio_bytes = request.data
        else:
            body = request.get_json(force=True, silent=True) or {}
            b64_str = body.get("audio_base64", "")
            if b64_str:
                audio_bytes = AudioProcessor.base64_to_wav(b64_str)

        if not audio_bytes:
            return jsonify({"status": "error", "message": "No audio payload provided"}), 400

        transcriber = getattr(app, "voice_transcriber", None)
        if transcriber is None:
            transcriber = VoiceTranscriber(mock=True)

        res = transcriber.transcribe_audio(audio_bytes, mime_type=mime_type)
        transcript = res.get("transcript", "").strip()
        confidence = res.get("confidence", 0.0)

        if transcript and app.orchestrator is not None:
            if app.orchestrator.dialogue.is_hover_stop_command(transcript):
                app.orchestrator.emergency_hover()
                app.orchestrator.narrator.emit_event(
                    "EMERGENCY_HOVER",
                    f"Audio Emergency Halt: \"{transcript}\"",
                    {"transcript": transcript, "confidence": confidence},
                    spoken_text="Emergency hover engaged. Holding position."
                )
                return jsonify({
                    "status": "emergency_hover",
                    "action": "HOVER",
                    "transcript": transcript,
                    "confidence": confidence,
                    "latency_ms": res.get("latency_ms", 0.0)
                })

            app.orchestrator.queue_command(transcript)
            app.orchestrator.narrator.emit_event(
                "VOICE_COMMAND",
                f"Audio Operator ({confidence * 100:.0f}%): \"{transcript}\"",
                {"transcript": transcript, "confidence": confidence},
                spoken_text=f"Acknowledged: {transcript}"
            )

        return jsonify({
            "status": "processed",
            "transcript": transcript,
            "confidence": confidence,
            "latency_ms": res.get("latency_ms", 0.0)
        })

    @app.route("/api/voice/status")
    def voice_status():
        """Retrieve voice subsystem capabilities and status."""
        transcriber = getattr(app, "voice_transcriber", None)
        is_mock = transcriber.mock if transcriber else True
        return jsonify({
            "enabled": True,
            "engine": "mock" if is_mock else "gemini_multimodal",
            "ptt_supported": True,
            "speech_synthesis_supported": True
        })

    @app.route("/api/start", methods=["POST"])
    def start_mission():
        """Start a new navigation mission with route instruction."""
        body = request.get_json(force=True, silent=True) or {}
        instruction = body.get("instruction", "").strip()
        if not instruction:
            env_name = getattr(app.orchestrator.config.env, "name", "city") if app.orchestrator else "city"
            if env_name == "blocks":
                instruction = "Fly to the orange block, then circle the blue cylinder, and inspect the red pillar"
            elif env_name in ("neighborhood", "nh"):
                instruction = "Fly to the white car, then pass the stop sign, and navigate to the house"
            else:
                instruction = "Fly to the fountain, then pass the red building, and navigate to the tower"

        if app.orchestrator is not None:
            subgoals = app.orchestrator.start_mission(instruction)
            return jsonify({"status": "started", "instruction": instruction, "subgoals": subgoals})
        return jsonify({"status": "started", "subgoals": ["fountain", "building"]})

    @app.route("/api/stop", methods=["POST"])
    def stop_mission():
        """Trigger emergency hover."""
        if app.orchestrator is not None:
            app.orchestrator.emergency_hover()
        return jsonify({"status": "hovering"})

    @app.route("/api/takeoff", methods=["POST"])
    def takeoff_command():
        """Trigger takeoff to cruise altitude."""
        if app.orchestrator is not None:
            alt = app.orchestrator.config.flight.takeoff_altitude
            success = app.orchestrator.flight.takeoff(altitude=alt)
            with app.orchestrator._state_lock:
                app.orchestrator.flight_mode = "HOVERING"
            app.orchestrator.narrator.emit_event(
                "TAKEOFF_COMPLETED",
                f"Manual takeoff initiated. Climbing to {alt:.1f}m.",
                {"altitude": alt}
            )
            return jsonify({"status": "takeoff", "altitude": alt, "success": success})
        return jsonify({"status": "error", "message": "No orchestrator"}), 500

    @app.route("/api/land", methods=["POST"])
    def land_command():
        """Trigger landing sequence."""
        if app.orchestrator is not None:
            success = app.orchestrator.flight.land()
            with app.orchestrator._state_lock:
                app.orchestrator.flight_mode = "LANDING"
            app.orchestrator.narrator.emit_event(
                "LANDING_INITIATED",
                "Manual landing sequence initiated."
            )
            return jsonify({"status": "landing", "success": success})
        return jsonify({"status": "error", "message": "No orchestrator"}), 500

    @app.route("/api/telemetry")
    def get_telemetry():
        if app.orchestrator is not None:
            return jsonify(app.orchestrator.get_telemetry())
        return jsonify({"mode": "mock", "speed": 0.0})

    @app.route("/api/landmarks")
    def get_landmarks():
        if app.orchestrator is not None:
            lms = app.orchestrator.memory.get_all_landmarks()
            for l in lms:
                if isinstance(l.get("pos"), np.ndarray):
                    l["pos"] = l["pos"].tolist()
            return jsonify(lms)
        return jsonify([])

    return app


class DashboardServer:
    """Thread-safe runner for dashboard server."""

    def __init__(self, orchestrator, host: str = "0.0.0.0", port: int = 5000):
        self.orchestrator = orchestrator
        self.host = host
        self.port = port
        self.app = create_app(orchestrator)
        self.thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start Flask dashboard in background daemon thread."""
        def run():
            # Run without reloader in background thread, load_dotenv=False avoids Windows console handle collision
            self.app.run(host=self.host, port=self.port, debug=False, use_reloader=False, threaded=True, load_dotenv=False)

        self.thread = threading.Thread(target=run, daemon=True, name="DashboardServerThread")
        self.thread.start()
        print(f"[DashboardServer] Live web interface active at http://localhost:{self.port}")

    def stop(self) -> None:
        """Mark server as stopped."""
        pass
