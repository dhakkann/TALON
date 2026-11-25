import argparse
import queue
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template_string, request

try:
    from airsim_wrapper import AirSimWrapper, colorize_depth
    from gemini_airsim import (
        DEFAULT_MODEL,
        DroneChatAgent,
        execute_drone_code,
        write_chat_history,
    )
except ImportError:
    from src.airsim_wrapper import AirSimWrapper, colorize_depth
    from src.gemini_airsim import (
        DEFAULT_MODEL,
        DroneChatAgent,
        execute_drone_code,
        write_chat_history,
    )


class AirSimWorker:
    """Single-thread owner for AirSimWrapper RPC calls (msgpackrpc thread-safe)."""

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self._cmd_queue: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self.connect_error: str | None = None
        self.aw: AirSimWrapper | None = None
        self.flight_busy: bool = False
        self.last_code: str = ""
        self.last_exec_ok: bool = True
        self.last_exec_output: str = ""
        self._dry_rgb_jpg: bytes | None = None
        self._dry_depth_jpg: bytes | None = None
        self._dry_telemetry = {
            "position": [0.0, 0.0, 0.0],
            "yaw": 0.0,
            "front_distance": 42.5,
            "status": "DRY RUN",
            "avoiding": False,
            "timestamp": time.time(),
        }
        if dry_run:
            self._init_dry_frames()
            self._ready_event.set()
        else:
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()
            self._ready_event.wait(timeout=15.0)

    def _init_dry_frames(self):
        rgb = np.zeros((240, 320, 3), dtype=np.uint8)
        rgb[:, :] = (32, 40, 52)
        cv2.putText(rgb, "FPV DRY-RUN", (70, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 235, 250), 2)
        _, b1 = cv2.imencode(".jpg", rgb)
        self._dry_rgb_jpg = b1.tobytes()

        depth_arr = np.full((240, 320), 25.0, dtype=np.float32)
        depth_arr[80:160, 110:210] = 8.5
        depth_bgr = colorize_depth(depth_arr)
        _, b2 = cv2.imencode(".jpg", depth_bgr)
        self._dry_depth_jpg = b2.tobytes()

    def _connect_sim(self) -> bool:
        try:
            self.aw = AirSimWrapper(connect=True)
            self.aw.client.startRecording()
            self.connect_error = None
            return True
        except Exception as exc:
            self.aw = None
            self.connect_error = f"{type(exc).__name__}: {exc}"
            return False

    def _run_loop(self):
        self._connect_sim()
        self._ready_event.set()
        last_poll = 0.0
        last_retry = time.time()
        while not self._stop_event.is_set():
            if self.aw is None:
                if time.time() - last_retry >= 2.0:
                    self._connect_sim()
                    last_retry = time.time()
                time.sleep(0.2)
                continue
            try:
                task = self._cmd_queue.get(timeout=0.1)
            except queue.Empty:
                now = time.time()
                if now - last_poll >= 0.35:
                    self.aw.refresh_snapshot()
                    last_poll = now
                continue

            kind, payload = task
            if kind == "exec":
                self.flight_busy = True
                self.last_code = payload
                ok, out = execute_drone_code(self.aw, payload)
                self.last_exec_ok = ok
                self.last_exec_output = out
                self.aw.refresh_snapshot()
                self.flight_busy = False
            elif kind == "land":
                self.flight_busy = True
                self.last_code = "aw.land()"
                try:
                    self.aw.clear_abort()
                    self.aw.land()
                    self.last_exec_ok = True
                    self.last_exec_output = "Emergency land completed."
                except Exception as exc:
                    self.last_exec_ok = False
                    self.last_exec_output = f"{type(exc).__name__}: {exc}"
                self.flight_busy = False

    def submit_code(self, code: str):
        if self.dry_run:
            self.last_code = code
            self.last_exec_ok = True
            self.last_exec_output = "[Dry-Run] Code parsed cleanly (simulator not attached)."
            self._dry_telemetry["timestamp"] = time.time()
            return
        if self.aw is not None:
            self.aw.clear_abort()
        self._cmd_queue.put(("exec", code))

    def emergency_land(self):
        if self.dry_run:
            self.last_code = "aw.land()"
            self.last_exec_ok = True
            self.last_exec_output = "[Dry-Run] Landed."
            self._dry_telemetry["status"] = "LANDED"
            return
        if self.aw is not None:
            self.aw.request_abort()
        self._cmd_queue.put(("land", None))

    def get_snapshot(self) -> tuple[bytes | None, bytes | None, dict]:
        if self.dry_run:
            t = dict(self._dry_telemetry)
            t.update(
                {
                    "connected": False,
                    "dry_run": True,
                    "flight_busy": False,
                    "last_code": self.last_code,
                    "last_exec_ok": self.last_exec_ok,
                    "last_exec_output": self.last_exec_output,
                }
            )
            return self._dry_rgb_jpg, self._dry_depth_jpg, t

        if self.aw is None:
            return (
                None,
                None,
                {
                    "position": [0.0, 0.0, 0.0],
                    "yaw": 0.0,
                    "front_distance": 0.0,
                    "status": "DISCONNECTED",
                    "avoiding": False,
                    "connected": False,
                    "connect_error": self.connect_error,
                    "flight_busy": False,
                    "last_code": self.last_code,
                    "last_exec_ok": self.last_exec_ok,
                    "last_exec_output": self.last_exec_output,
                },
            )

        rgb, depth, tel = self.aw.get_cached_frames()
        tel.update(
            {
                "connected": True,
                "dry_run": False,
                "flight_busy": self.flight_busy or not self._cmd_queue.empty(),
                "queue_depth": self._cmd_queue.qsize(),
                "last_code": self.last_code,
                "last_exec_ok": self.last_exec_ok,
                "last_exec_output": self.last_exec_output,
            }
        )
        return rgb, depth, tel

    def stop(self):
        self._stop_event.set()
        if self.aw is not None and self.aw.client is not None:
            try:
                self.aw.client.stopRecording()
            except Exception:
                pass


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>AR511 AirSim LLM Drone Cockpit</title>
<style>
  :root {
    --bg: #0b111a; --panel: #131c29; --panel-2: #192434; --border: #26354a;
    --text: #e6eef8; --muted: #8fa3bd; --accent: #38bdf8; --green: #22c55e;
    --amber: #f59e0b; --red: #ef4444; --code-bg: #070b11;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
    background: var(--bg); color: var(--text); height: 100vh; display: flex; flex-direction: column;
  }
  header {
    display: flex; align-items: center; justify-content: space-between; gap: 8px; flex-wrap: wrap;
    padding: 10px 14px; background: var(--panel); border-bottom: 1px solid var(--border);
  }
  .title { font-weight: 700; font-size: 15px; letter-spacing: 0.3px; }
  .badges { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
  .badge {
    font-size: 11px; font-weight: 600; padding: 4px 8px; border-radius: 999px;
    border: 1px solid var(--border); background: var(--panel-2);
  }
  .badge.ok { color: var(--green); border-color: rgba(34,197,94,0.4); }
  .badge.warn { color: var(--amber); border-color: rgba(245,158,11,0.4); }
  .badge.err { color: var(--red); border-color: rgba(239,68,68,0.4); }
  button {
    cursor: pointer; border: 1px solid var(--border); background: var(--panel-2); color: var(--text);
    border-radius: 6px; padding: 6px 10px; font-size: 12px; font-weight: 600;
  }
  button:hover { border-color: var(--accent); }
  button.primary { background: #0284c7; border-color: #38bdf8; }
  button.danger { background: #991b1b; border-color: #ef4444; }
  main {
    flex: 1; min-height: 0; display: grid; grid-template-columns: 1fr 1fr; gap: 10px; padding: 10px;
  }
  @media (max-width: 900px) { main { grid-template-columns: 1fr; overflow-y: auto; } }
  .col { display: flex; flex-direction: column; gap: 10px; min-height: 0; }
  .card {
    background: var(--panel); border: 1px solid var(--border); border-radius: 8px;
    padding: 10px; display: flex; flex-direction: column; gap: 8px; min-height: 0;
  }
  .card h3 { margin: 0; font-size: 12px; text-transform: uppercase; letter-spacing: 0.6px; color: var(--muted); }
  .presets { display: flex; flex-wrap: wrap; gap: 6px; }
  .presets button { font-size: 11px; padding: 5px 8px; }
  #chat-log {
    flex: 1; min-height: 180px; overflow-y: auto; display: flex; flex-direction: column; gap: 8px;
    padding-right: 4px;
  }
  .msg { padding: 8px 10px; border-radius: 8px; font-size: 13px; line-height: 1.4; white-space: pre-wrap; }
  .msg.user { background: #1e3a5f; align-self: flex-end; max-width: 88%; }
  .msg.bot { background: var(--panel-2); align-self: flex-start; max-width: 94%; border: 1px solid var(--border); }
  .msg.question { border-color: var(--amber); background: rgba(245, 158, 11, 0.12); }
  .msg .meta { font-size: 10px; color: var(--muted); margin-bottom: 3px; }
  .quick-replies { display: flex; gap: 6px; flex-wrap: wrap; }
  .quick-replies button { background: rgba(245,158,11,0.18); border-color: var(--amber); color: #fde68a; }
  .input-row { display: flex; gap: 8px; }
  .input-row input {
    flex: 1; padding: 8px 10px; border-radius: 6px; border: 1px solid var(--border);
    background: var(--code-bg); color: var(--text); font-size: 13px;
  }
  .telemetry-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 6px; }
  .stat { background: var(--code-bg); border: 1px solid var(--border); border-radius: 6px; padding: 6px 8px; }
  .stat .label { font-size: 10px; color: var(--muted); }
  .stat .val { font-size: 14px; font-weight: 700; margin-top: 2px; font-variant-numeric: tabular-nums; }
  .views { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
  .view-box { background: var(--code-bg); border: 1px solid var(--border); border-radius: 6px; overflow: hidden; }
  .view-box .cap { font-size: 11px; color: var(--muted); padding: 4px 8px; border-bottom: 1px solid var(--border); }
  .view-box img { width: 100%; height: 190px; object-fit: cover; display: block; background: #05080c; }
  pre {
    margin: 0; background: var(--code-bg); border: 1px solid var(--border); border-radius: 6px;
    padding: 8px; font-size: 12px; color: #93c5fd; overflow: auto; max-height: 150px;
  }
  .exec-out { font-size: 11px; color: var(--muted); min-height: 16px; }
</style>
</head>
<body>
<header>
  <div class="title">AR511 Language-Guided Drone Cockpit</div>
  <div class="badges">
    <span id="sim-badge" class="badge">CONNECTING...</span>
    <span id="mode-badge" class="badge">MODE</span>
    <button id="toggle-mode-btn" onclick="toggleMode()">Toggle Offline Mode</button>
    <button onclick="saveHistory()">Save Chat</button>
    <button class="danger" onclick="emergencyLand()">LAND NOW</button>
  </div>
</header>
<main>
  <div class="col">
    <div class="card">
      <h3>One-Click Demo Prompts</h3>
      <div class="presets">
        <button onclick="sendText('Take off and hover at 8 meters')">Take Off</button>
        <button onclick="sendText('Fly safely to solarpanels and hover 12m above')">Solar Panels</button>
        <button onclick="sendText('Fly to the turbine')">Turbine (Ambiguity)</button>
        <button onclick="sendText('Fly safely to turbine1')">Turbine 1</button>
        <button onclick="sendText('Fly safely to the car and hover 8m above')">Inspect Car</button>
        <button onclick="sendText('Fly a 12m radius circle orbit')">Orbit</button>
        <button onclick="sendText('Check front obstacle distance')">Check Distance</button>
        <button onclick="sendText('Land the drone safely')">Land</button>
      </div>
    </div>
    <div class="card" style="flex:1;">
      <h3>Interactive Command &amp; Clarification Chat</h3>
      <div id="chat-log"></div>
      <div id="quick-replies" class="quick-replies"></div>
      <form class="input-row" onsubmit="event.preventDefault(); sendInput();">
        <input id="cmd-input" type="text" placeholder="Type a flight command (e.g., Fly to the turbine / Inspect car)..." autocomplete="off" />
        <button class="primary" type="submit">Send</button>
      </form>
    </div>
  </div>
  <div class="col">
    <div class="card">
      <h3>Live Telemetry &amp; Obstacle Clearance</h3>
      <div class="telemetry-grid">
        <div class="stat"><div class="label">POSITION (X, Y, Z)</div><div class="val" id="t-pos">0, 0, 0</div></div>
        <div class="stat"><div class="label">YAW (DEG)</div><div class="val" id="t-yaw">0.0°</div></div>
        <div class="stat"><div class="label">FRONT CLEARANCE</div><div class="val" id="t-dist">-- m</div></div>
        <div class="stat"><div class="label">STATUS</div><div class="val" id="t-status">IDLE</div></div>
      </div>
    </div>
    <div class="card">
      <h3>Live FPV RGB + DepthPlanar Sensor</h3>
      <div class="views">
        <div class="view-box"><div class="cap">FPV RGB Camera</div><img id="img-rgb" alt="FPV RGB" /></div>
        <div class="view-box"><div class="cap">DepthPlanar Heatmap</div><img id="img-depth" alt="DepthPlanar" /></div>
      </div>
    </div>
    <div class="card" style="flex:1;">
      <h3>Generated Python Flight Code</h3>
      <pre id="code-box"># Generated aw.* commands will appear here...</pre>
      <div class="exec-out" id="exec-out">Ready.</div>
    </div>
  </div>
</main>
<script>
let offlineMode = {{ 'true' if offline else 'false' }};

function appendMsg(role, text, meta, isQuestion) {
  const log = document.getElementById('chat-log');
  const div = document.createElement('div');
  div.className = 'msg ' + (role === 'user' ? 'user' : ('bot' + (isQuestion ? ' question' : '')));
  if (meta) {
    const m = document.createElement('div');
    m.className = 'meta';
    m.textContent = meta;
    div.appendChild(m);
  }
  const body = document.createElement('div');
  body.textContent = text;
  div.appendChild(body);
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
}

function renderQuickReplies(text, isQuestion) {
  const qr = document.getElementById('quick-replies');
  qr.innerHTML = '';
  if (!isQuestion) return;
  const low = text.toLowerCase();
  const options = low.includes('turbine')
    ? ['turbine1', 'turbine2']
    : (low.includes('tower') ? ['tower1', 'tower2', 'tower3'] : []);
  options.forEach(opt => {
    const b = document.createElement('button');
    b.type = 'button';
    b.textContent = opt;
    b.onclick = () => sendText(opt);
    qr.appendChild(b);
  });
}

async function sendInput() {
  const inp = document.getElementById('cmd-input');
  const txt = inp.value.trim();
  if (!txt) return;
  inp.value = '';
  await sendText(txt);
}

async function sendText(txt) {
  appendMsg('user', txt, 'YOU');
  document.getElementById('quick-replies').innerHTML = '';
  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({message: txt})
    });
    const data = await res.json();
    const label = data.source === 'gemini' ? 'GEMINI' : (data.source === 'offline_fallback' ? 'OFFLINE FALLBACK' : 'OFFLINE PRESET');
    appendMsg('bot', data.text, label, data.is_question);
    renderQuickReplies(data.text, data.is_question);
    if (data.code) {
      document.getElementById('code-box').textContent = data.code;
      document.getElementById('exec-out').textContent = 'Executing in AirSim...';
    }
    updateModeBadge();
  } catch (e) {
    appendMsg('bot', 'Request failed: ' + e, 'ERROR', false);
  }
}

async function toggleMode() {
  const res = await fetch('/api/mode', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({offline: !offlineMode})
  });
  const data = await res.json();
  offlineMode = data.offline;
  updateModeBadge();
}

function updateModeBadge() {
  const b = document.getElementById('mode-badge');
  b.textContent = offlineMode ? 'OFFLINE PRESET' : 'GEMINI ONLINE';
  b.className = 'badge ' + (offlineMode ? 'warn' : 'ok');
}

async function emergencyLand() {
  await fetch('/api/land', {method: 'POST'});
  appendMsg('bot', 'Emergency land command sent to AirSim.', 'SAFETY');
}

async function saveHistory() {
  const res = await fetch('/api/save', {method: 'POST'});
  const data = await res.json();
  appendMsg('bot', 'Saved chat history to: ' + data.path, 'SYSTEM');
}

async function pollTelemetry() {
  try {
    const res = await fetch('/api/telemetry');
    const t = await res.json();
    const simB = document.getElementById('sim-badge');
    if (t.connected) {
      simB.textContent = 'AIRSIM CONNECTED';
      simB.className = 'badge ok';
    } else if (t.dry_run) {
      simB.textContent = 'DRY-RUN MODE';
      simB.className = 'badge warn';
    } else {
      simB.textContent = 'AIRSIM DISCONNECTED';
      simB.className = 'badge err';
    }
    if (typeof t.offline === 'boolean') {
      offlineMode = t.offline;
      updateModeBadge();
    }
    const p = t.position || [0,0,0];
    document.getElementById('t-pos').textContent = `${p[0].toFixed(1)}, ${p[1].toFixed(1)}, ${p[2].toFixed(1)}`;
    document.getElementById('t-yaw').textContent = `${(t.yaw || 0).toFixed(1)}°`;
    const dEl = document.getElementById('t-dist');
    const d = Number(t.front_distance ?? 100);
    dEl.textContent = `${d.toFixed(1)} m`;
    dEl.style.color = d < 4.5 ? '#ef4444' : (d < 7.0 ? '#f59e0b' : '#22c55e');
    const stEl = document.getElementById('t-status');
    stEl.textContent = t.avoiding ? 'AVOIDING' : (t.flight_busy ? 'EXECUTING' : (t.status || 'IDLE'));
    stEl.style.color = t.avoiding ? '#ef4444' : '#e6eef8';
    if (t.last_code) {
      document.getElementById('code-box').textContent = t.last_code;
      document.getElementById('exec-out').textContent = t.flight_busy
        ? 'Executing in AirSim...'
        : (t.last_exec_ok ? (t.last_exec_output || 'Execution completed.') : ('Error: ' + t.last_exec_output));
    }
    const ts = Date.now();
    document.getElementById('img-rgb').src = '/api/frame/rgb.jpg?t=' + ts;
    document.getElementById('img-depth').src = '/api/frame/depth.jpg?t=' + ts;
  } catch (e) {}
}

updateModeBadge();
pollTelemetry();
setInterval(pollTelemetry, 450);
</script>
</body>
</html>
"""


def create_app(
    model: str = DEFAULT_MODEL,
    offline: bool = False,
    dry_run: bool = False,
    testname: str = "web_cockpit",
    agent: DroneChatAgent | None = None,
    worker: AirSimWorker | None = None,
) -> Flask:
    app = Flask(__name__)
    chat_agent = agent or DroneChatAgent(model=model, offline=offline)
    sim_worker = worker or AirSimWorker(dry_run=dry_run)
    chat_lock = threading.Lock()

    app.config["AGENT"] = chat_agent
    app.config["WORKER"] = sim_worker

    @app.get("/")
    def index():
        return render_template_string(HTML_PAGE, offline=chat_agent.offline)

    @app.post("/api/chat")
    def api_chat():
        payload = request.get_json(silent=True) or {}
        message = str(payload.get("message", "")).strip()
        if not message:
            return jsonify({"error": "Empty message"}), 400
        with chat_lock:
            result = chat_agent.ask(message)
        if result["code"]:
            sim_worker.submit_code(result["code"])
        result["executing"] = bool(result["code"])
        result["offline"] = chat_agent.offline
        return jsonify(result)

    @app.post("/api/mode")
    def api_mode():
        payload = request.get_json(silent=True) or {}
        want_offline = bool(payload.get("offline", True))
        with chat_lock:
            chat_agent.set_offline(want_offline)
        return jsonify({"offline": chat_agent.offline, "error": chat_agent.last_error})

    @app.post("/api/land")
    def api_land():
        sim_worker.emergency_land()
        return jsonify({"ok": True})

    @app.post("/api/save")
    def api_save():
        with chat_lock:
            saved = write_chat_history(chat_agent.model, testname, chat_agent.history)
        return jsonify({"ok": True, "path": str(saved)})

    @app.get("/api/telemetry")
    def api_telemetry():
        _, _, tel = sim_worker.get_snapshot()
        tel["offline"] = chat_agent.offline
        return jsonify(tel)

    @app.get("/api/frame/rgb.jpg")
    def api_rgb():
        rgb, _, _ = sim_worker.get_snapshot()
        if rgb is None:
            return Response(status=204)
        return Response(rgb, mimetype="image/jpeg")

    @app.get("/api/frame/depth.jpg")
    def api_depth():
        _, depth, _ = sim_worker.get_snapshot()
        if depth is None:
            return Response(status=204)
        return Response(depth, mimetype="image/jpeg")

    return app


def main():
    parser = argparse.ArgumentParser(description="AR511 AirSim LLM Web Chat Cockpit")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5050)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--testname", type=str, default="web_cockpit")
    args = parser.parse_args()

    app = create_app(
        model=args.model,
        offline=args.offline,
        dry_run=args.dry_run,
        testname=args.testname,
    )
    print(f"Starting Web Chat Cockpit at http://{args.host}:{args.port}")
    try:
        app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)
    finally:
        app.config["WORKER"].stop()


if __name__ == "__main__":
    main()
