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

# Suppress verbose Flask request logging to keep console clean
logging.getLogger("werkzeug").setLevel(logging.ERROR)


def create_app(system_orchestrator=None):
    """Factory creating Flask dashboard application."""
    app = Flask(__name__, template_folder="templates")
    CORS(app)

    # Reference to orchestrator holding flight controller, perception, memory, dialogue
    app.orchestrator = system_orchestrator

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

    @app.route("/api/start", methods=["POST"])
    def start_mission():
        """Start a new navigation mission with route instruction."""
        body = request.get_json(force=True, silent=True) or {}
        instruction = body.get("instruction", "").strip()
        if not instruction:
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
            # Run without reloader in background thread
            self.app.run(host=self.host, port=self.port, debug=False, use_reloader=False, threaded=True)

        self.thread = threading.Thread(target=run, daemon=True, name="DashboardServerThread")
        self.thread.start()
        print(f"[DashboardServer] Live web interface active at http://localhost:{self.port}")

    def stop(self) -> None:
        """Mark server as stopped."""
        pass
