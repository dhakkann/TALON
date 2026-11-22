# Real-Time Autonomous City Navigation (`realtime_citynav`)

**AR511 Autonomous Mobile Robotics Course Project**  
*Developers*: Dharmesh Sahu (B24033, SBB) & Mihir Yadav (B24341, SCEE)  

A continuous-flight, real-time autonomous drone navigation system in AirSim 3D city environments featuring:
- **Continuous 20–30 Hz Flight Controller**: Native `moveByVelocityAsync` smooth velocity vectors with proportional waypoint guidance and reactive depth-based Artificial Potential Field (APF) obstacle avoidance.
- **Sub-Millisecond 3D Vision**: Ultralytics YOLO-World v2 open-vocabulary landmark detection with vectorized NumPy pinhole camera depth back-projection (<0.06 ms latency).
- **Persistent 3D Spatial Memory**: Euclidean clustering (5m threshold) registry tracking observed landmarks, running-average coordinates, and timestamps.
- **Asynchronous Cognitive Guidance**: Gemini 2.0 Flash client with robust semantic heuristic fallback for route decomposition and non-blocking mid-flight interruption classification (`TARGET_CORRECTION`, `RETURN_TO_PAST`, `QUERY`, `HOVER_STOP`).
- **Live Explainability & Dashboard**: Single-page modern dark-mode UI with live MJPEG camera feed, Server-Sent Events (SSE) telemetry & rationale stream, 3D landmark memory table, and mid-flight operator chat.

---

## Architecture

```
realtime_citynav/
├── __init__.py
├── config.py                 # Centralized configuration (AirSim, camera, APF, Gemini, dashboard)
├── perception.py             # YOLO-World v2 open-vocab detector & vectorized 3D projection
├── flight_controller.py      # 20-30 Hz continuous velocity flight loop & APF safety field
├── spatial_memory.py         # Persistent 3D spatial landmark registry & Euclidean clustering
├── dialogue_manager.py       # Asynchronous Gemini 2.0 Flash parser & interruption queue
├── explainability.py         # Event-driven and cadence-based live rationale narration
├── dashboard/
│   ├── __init__.py
│   ├── app.py                # Flask MJPEG & SSE telemetry web server
│   └── templates/
│       └── index.html        # Futuristic dark-mode operator mission control UI
├── main.py                   # High-level multithreaded orchestrator
└── tests/                    # Automated test suite (32 unit & integration tests)
    ├── __init__.py
    ├── test_perception.py
    ├── test_spatial_memory.py
    ├── test_dialogue_manager.py
    ├── test_flight_controller.py
    └── test_integration.py
```

---

## Quickstart

### 1. Run Automated Test Suite
To execute all 61 unit, integration, and scenario tests:
```bash
pytest realtime_citynav/tests
```

### 2. Run Automated Scenario Evaluator Suite
To programmatically run the 11 validation and stress scenarios (testing 3D spatial prepositions, compound directions, elevated landmark altitudes, continuous memory maintenance, deduplication, mid-flight interruptions, APF obstacle avoidance, and extreme inputs):
```bash
python -m realtime_citynav.scenario_evaluator
```
Or run in interactive diagnostic shell mode to give tasks, inject landmarks, send mid-flight interruptions, step flight, and inspect live memory tables:
```bash
python -m realtime_citynav.scenario_evaluator --interactive
```

### 3. Launch Standalone Flight Operations (with Web Dashboard)
To run the full system in simulation mode (works with or without AirSim):
```bash
python -m realtime_citynav.main --speed 4.0 --port 5000 --instruction "Fly to the fountain, then pass the red building, and navigate to the tower"
```
Then open your browser to **http://localhost:5000** to view the live dashboard.

### 3. Connect to Native AirSim Environment
When AirSim is running:
```bash
python -m realtime_citynav.main --speed 4.0 --port 5000
```
The orchestrator automatically handshakes with AirSim RPC on `127.0.0.1:41451` and uses live camera and physics.
