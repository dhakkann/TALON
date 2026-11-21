# TALON: Tactical Autonomous Language-guided Operations and Navigation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Platform: AirSim / Windows / Linux](https://img.shields.io/badge/Platform-AirSim%20%7C%20Windows%20%7C%20Linux-orange.svg)]()

> **AR511 Autonomous Mobile Robotics Course Project**  
> **Developers**: Dharmesh Sahu & Mihir Yadav  

**TALON** is a continuous-flight, real-time autonomous drone navigation system operating in 3D urban environments (AirSim / Unreal Engine). It transforms language-guided aerial navigation from static, turn-based discrete stepping into a **real-time 20–30 Hz continuous flight control loop** paired with **30+ FPS open-vocabulary visual perception** and **asynchronous LLM cognitive guidance**.

---

## Key Highlights

- **Continuous 20–30 Hz Flight Controller**: Native `moveByVelocityAsync` velocity vector commands with active yaw heading alignment and reactive 3D Artificial Potential Field (APF) obstacle avoidance (vertical climb + lateral deflection + forward braking).
- **Sub-Millisecond 3D Pinhole Projection**: Vectorized NumPy projection (<0.08 ms) mapping 2D bounding box centroids and metric depth into 3D World NED coordinates.
- **Open-Vocabulary Perception**: Ultralytics YOLO-World v2 (`yolov8s-worldv2`) running zero-shot landmark recognition at 30+ FPS on GPU.
- **Persistent 3D Spatial Memory**: Euclidean clustering (5m radius) tracking observed landmarks, running-average coordinates, timestamps, and observation persistence counts.
- **Asynchronous Cognitive Guidance**: Non-blocking Gemini 2.0 Flash worker for natural-language instruction decomposition and live mid-flight interruption classification (`TARGET_CORRECTION`, `ALTITUDE_CHANGE`, `RETURN_TO_PAST`, `QUERY`, `HOVER_STOP`).
- **Live Cockpit UI**: Single-page dark-mode web dashboard (`http://localhost:5000`) with live MJPEG camera feed, Server-Sent Events (SSE) telemetry stream, 3D landmark memory table, and mid-flight operator chat.

---

## System Architecture

```
                                  +------------------------------------+
                                  |     OPERATOR / WEB DASHBOARD       |
                                  |   (Live Feed, Telemetry, Chat)     |
                                  +-----------------+------------------+
                                                    |
                                       Natural Text / Corrections
                                                    v
+----------------------------------------------------------------------------------------------------+
| TIER 3: COGNITIVE & INTERACTION LAYER (Asynchronous Worker / Event-Driven)                         |
|  - Dialogue Manager: Non-blocking command queue (Gemini 2.0 Flash + rule fallback)                 |
|  - Semantic Planner: Decomposes natural language instructions into landmark subgoals               |
|  - Explainability Layer: Real-time rationale narration streamed to dashboard                       |
+---------------------------------------------------+------------------------------------------------+
                                                    |
                                          Active 3D Waypoint / Velocity Bias
                                                    v
+----------------------------------------------------------------------------------------------------+
| TIER 1: HIGH-FREQUENCY FLIGHT CONTROL LOOP (20–30 Hz Dedicated Thread)                             |
|  - Continuous non-blocking flight via moveByVelocityAsync(vx, vy, vz, duration=0.1)                |
|  - Proportional guidance tracking with safe facade standoff distances                              |
|  - 3D APF Reactive Obstacle Avoidance: Vertical climb & lateral deflection                         |
|  - Impenetrable physical boundary barriers in kinematic simulation mode                            |
+---------------------------------------------------+------------------------------------------------+
                                                    ^
                                                    | Live 3D Detections & Positions
                                                    |
+---------------------------------------------------+------------------------------------------------+
| TIER 2: REAL-TIME PERCEPTION PIPELINE (30+ FPS Dedicated Thread)                                   |
|  - Frame acquisition via simGetImages (RGB + DepthPlanar)                                          |
|  - Open-Vocabulary Detector: YOLO-World v2 (yolov8s-worldv2)                                       |
|  - Vectorized Depth Back-Projection (<0.08 ms per call)                                            |
|  - Dynamic 3D perspective terrain grid and landmark rendering in simulation mode                   |
+---------------------------------------------------+------------------------------------------------+
                                                    |
                                                    v
+----------------------------------------------------------------------------------------------------+
| TIER 4: QUERYABLE PERSISTENT 3D SPATIAL MEMORY (Thread-Safe Shared State)                          |
|  - Dynamic spatial registry: {id, class, pos[x,y,z], first_seen, last_seen, count}                 |
|  - Euclidean spatial clustering with exponential running average coordinates                       |
|  - Instant lookup and backtracking trajectory planner for return-to-past commands                  |
+----------------------------------------------------------------------------------------------------+
```

---

## Repository Structure

```
TALON/
├── realtime_citynav/
│   ├── config.py                 # Centralized configuration (AirSim, speeds, APF, Gemini)
│   ├── flight_controller.py      # 20 Hz continuous velocity flight loop & APF safety field
│   ├── perception.py             # 30+ FPS YOLO-World v2 detector & vectorized 3D projection
│   ├── spatial_memory.py         # Persistent 3D spatial landmark registry & Euclidean clustering
│   ├── dialogue_manager.py       # Asynchronous Gemini 2.0 Flash parser & interruption queue
│   ├── explainability.py         # Event-driven and cadence live rationale narration
│   ├── main.py                   # Main multithreaded system orchestrator
│   ├── dashboard/
│   │   ├── app.py                # Flask MJPEG & SSE telemetry web server
│   │   └── templates/
│   │       └── index.html        # Single-page operator mission control UI
│   └── tests/                    # Automated test suite (43 unit & integration tests)
├── implementation_plan.md        # Technical architecture design document
├── project_presentation.pptx     # AR511 Project Presentation
├── Project_Report.pdf            # AR511 Project Technical Report
├── README.md                     # Project documentation & guide
└── .gitignore                    # Git exclusions (weights, caches, environments)
```

---

## Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/dhakkann/TALON.git
cd TALON
```

### 2. Set Up Virtual Environment
```bash
# Recommended: Python 3.10 or 3.11
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux:
source .venv/bin/activate

pip install -r realtime_citynav/requirements.txt  # or: pip install airsim ultralytics opencv-python flask flask-cors numpy google-genai
```

### 3. Run Automated Tests
Verify system integrity across all 43 tests:
```bash
python -m unittest discover -s realtime_citynav/tests -p "test_*.py"
```

---

## Running TALON

### Simulation / Mock Mode (Standalone, No GPU/AirSim Required)
Run the full system immediately in mock simulation mode:
```bash
python -m realtime_citynav.main --mock --speed 4.0 --port 5000
```
Open **[http://localhost:5000](http://localhost:5000)** in your browser to access the cockpit interface.

### Running with Unreal Engine / AirSim
1. Launch your AirSim Unreal environment (e.g. `CityEnviron.exe` or `AirSimNH.exe`).
2. (Optional) Set your Google Gemini API key:
   ```bash
   # Windows PowerShell
   $env:GEMINI_API_KEY="your-api-key"
   # Linux / macOS
   export GEMINI_API_KEY="your-api-key"
   ```
3. Launch the TALON orchestrator:
   ```bash
   python -m realtime_citynav.main --speed 4.0 --port 5000
   ```

---

## Mid-Flight Operator Commands

While the drone is airborne, interact via the dashboard chat box in real time:

- **Target Correction**: `"no, the blue building"`, `"switch to tower"`, `"fly to hospital"`
- **Altitude Adjustment**: `"climb to 15m"`, `"fly higher"`, `"fly at 20m altitude"`, `"descend to 5m"`
- **Memory Query**: `"what have you seen?"`, `"list landmarks"`
- **Return to Past**: `"go back to the fountain"`, `"return to red building"`
- **Emergency Hover**: `"hover"`, `"stop"`, `"halt"`

---

## Authors & Acknowledgments

- **Dharmesh Sahu** (B24033) - Department of Electrical Engineering & Computer Science
- **Mihir Yadav** (B24341) - Department of Electrical Engineering & Computer Science
- Developed for **AR511 Autonomous Mobile Robot**, under faculty guidance.
