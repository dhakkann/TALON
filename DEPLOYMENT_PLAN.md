# Autonomous Drone Navigation in AirSim: Flight Planning & Deployment Guide

**System Name**: TALON (Tactical Autonomous Language-guided Operations and Navigation)  
**Project**: AR511 Autonomous Mobile Robot Course Project  
**Authors**: Mihir Yadav (B24341) & Dharmesh Sahu (B24033)  
**Target Platform**: Microsoft AirSim on Unreal Engine 4 / Native Python 3.10–3.12  

---

## 1. System Architecture & 4-Tier Pipeline Overview

TALON replaces legacy discrete-step benchmarks (e.g., AirVLN, CityNavAgent) with a real-time, event-driven, multithreaded autonomy stack operating across four tightly integrated tiers:

```
===================================================================================
                           TIER 4: OPERATOR INTERACTION
  * Web Cockpit Dashboard (Flask + MJPEG Stream + SSE Telemetry @ localhost:5000)
  * Mid-Flight Operator Chat & Live Rationale Narration (LiveRationaleNarrator)
===================================================================================
                                          │ Operator Commands / Telemetry
                                          ▼
===================================================================================
                       TIER 3: COGNITIVE GUIDANCE & PLANNING
  * Dialogue Manager: Gemini 2.0/3.5 Flash + Deterministic Heuristic Regex Engine
  * Multi-Subgoal Mission Decomposition & Spatial Preposition Parser
  * Mid-Flight Interruption Queue (Target Correction, Orbit, Waypoint Path, Hover)
===================================================================================
        │ Active Subgoal & Orbit Config         ▲ Query Past Landmarks
        ▼                                       │
===================================================================================
                    TIER 2: PERCEPTION & 3D SPATIAL MEMORY
  * Open-Vocabulary Detector: YOLO-World v2 (yolov8s-worldv2.pt) @ 30+ FPS
  * Vectorized Pinhole Depth Back-Projection (<0.08 ms, body & world NED frames)
  * Persistent Spatial Memory: 3D Registry, Euclidean Clustering (5m), Pruning
===================================================================================
        │ Detected 3D Landmark Positions
        ▼
===================================================================================
                      TIER 1: HIGH-FREQUENCY FLIGHT CONTROL
  * 20–30 Hz Continuous Velocity Control Loop (moveByVelocityAsync)
  * Proportional Waypoint Guidance & Standoff Approach Coordinates
  * Mathematical Orbit & 3D Spline Waypoint Follower
  * Reactive 3D APF Obstacle Avoidance (Vertical climb + Lateral deflection + Stop)
  * Dual Mode: AirSim Multirotor RPC (:41451) <-> Kinematic Simulator Fallback
===================================================================================
                                          │ RPC Commands (:41451)
                                          ▼
===================================================================================
                    PHYSICAL / SIMULATION ENVIRONMENT (AirSim)
  * Unreal Engine Multirotor Environment (CityEnviron / AirSimNH / Blocks)
  * Standardized settings.json (SimMode: Multirotor, SimpleFlight, front_center)
===================================================================================
```

### 1.1 Tier 1: Continuous Flight Controller (`flight_controller.py`)
- **Control Rate**: 20–30 Hz continuous closed loop.
- **Actuation Primitive**: Native AirSim RPC `moveByVelocityAsync(vx, vy, vz, duration=0.1, drivetrain=MaxDegreeOfFreedom, yaw_mode=YawMode(is_rate=False, yaw_or_rate=target_yaw))`.
- **Guidance Modes**:
  1. *Proportional Waypoint Navigation*: Computes direct Euclidean velocity vectors to 3D standoff targets with speed saturation ($v_{\max} = 6.0\text{ m/s}$).
  2. *Continuous Mathematical Orbit*: Closed-form circle tracking using tangential velocity $\mathbf{v}_{\text{tan}}$ and radial correction $v_{\text{rad}} = k_p (R - r)\hat{\mathbf{u}}_r$. Supports dynamic yaw: `INWARD` (camera locked onto target center) and `TANGENT` (camera aligned with forward curve tangent).
  3. *Continuous 3D Spline / Waypoint Curve Follower*: Follows arbitrary discrete 3D waypoints ($N \ge 2$) with arrival radii and heading alignment.
  4. *Reactive Artificial Potential Field (APF)*: 3D obstacle avoidance that slides left/right ($2.0\text{ m/s}$), brakes forward velocity ($v_x = -1.5\text{ m/s}$), and climbs vertically ($v_z = -2.5\text{ m/s}$) when obstacles enter the central 50% ROI at $<4.0\text{m}$.
- **Kinematic Fallback**: If AirSim is unreachable within 3.0s, seamlessly transitions to an internal 20 Hz kinematic simulation with collision boundaries, allowing 100% offline testing.

### 1.2 Tier 2: Perception & 3D Spatial Memory (`perception.py`, `spatial_memory.py`)
- **Detector**: Open-vocabulary YOLO-World v2 (`yolov8s-worldv2.pt`) running at 30+ FPS (GPU) or synthetic 3D perspective projection (CPU mock). Dynamically updates text vocabulary when novel landmarks are referenced in commands.
- **Vectorized Pinhole Back-Projection**: Converts pixel centroid $(u, v)$ and median depth $z_{\text{metric}}$ to 3D camera coordinates in $<0.08\text{ ms}$, then applies camera-to-body NED transformation:
  $$\begin{bmatrix} x_{\text{body}} \\ y_{\text{body}} \\ z_{\text{body}} \end{bmatrix} = \begin{bmatrix} z_{\text{cam}} \\ x_{\text{cam}} \\ y_{\text{cam}} \end{bmatrix}$$
  and rotates into World NED via drone yaw $\psi$.
- **3D Spatial Memory**: Maintains persistent landmark registry with Euclidean spatial clustering (5.0m merge radius), exponential moving average smoothing ($\alpha = 0.2$), and time-decay retention (up to 3600s).

### 1.3 Tier 3: Cognitive Guidance & Planning (`dialogue_manager.py`)
- **Natural Language Planner**: Leverages Gemini Flash (`gemini-3.5-flash-lite` / `gemini-3.5-flash`) with full heuristic fallback.
- **Multi-Subgoal Decomposition**: Automatically sequences complex multi-step instructions (e.g. *"fly to fountain, then circle red building 2 times at 30m altitude, then hover"*).
- **Spatial Preposition Parser**: Accurately maps prepositions (*"20m above"*, *"15m in front of"*, *"behind"*, *"left of"*, *"compass north-east"*) to 3D NED coordinates with building height allowances.
- **Mid-Flight Interruption Queue**: Non-blocking thread-safe queue supporting mid-mission flight corrections, emergency stops, orbit parameter shifts, and past landmark recall.

### 1.4 Tier 4: Operator Cockpit Dashboard (`realtime_citynav/dashboard/`)
- **Web UI**: Modern single-page cockpit dashboard hosted via Flask at `http://localhost:5000`.
- **Live Video Feed**: High-cadence MJPEG stream (`/video_feed`) showing 30+ FPS camera imagery with 2D bounding boxes, landmark labels, and distance overlays.
- **Real-Time Telemetry**: Server-Sent Events (`/events`) streaming altitude, speed, coordinates $(X, Y, Z)$, flight mode, active subgoal, and registered memory landmarks.
- **Interactive Controls**: Natural language chat input, Emergency Hover button, Mission Start/Stop buttons, and live reasoning narration log.

---

## 2. AirSim Simulation Requirements

### 2.1 Unreal Engine Environment
- **Compatible Environments**:
  - `CityEnviron` (recommended for full city navigation and urban buildings)
  - `AirSimNH` (Neighborhood environment with houses, trees, streets)
  - `Blocks` (Lightweight testing environment)
  - Custom Unreal Engine 4.27 projects with AirSim plugin enabled.
- **Execution Flags**:
  ```powershell
  # Launch windowed at 1280x720 for optimal performance alongside dashboard
  .\CityEnviron.exe -windowed -ResX=1280 -ResY=720
  ```

### 2.2 Network Ports & Protocols
| Port | Protocol | Purpose | Direction |
|------|----------|---------|-----------|
| **41451** | TCP (RPC) | AirSim RPC API control and sensor data acquisition | Python Client -> AirSim Host |
| **41452+** | TCP (RPC) | Additional vehicles (if multi-vehicle mode configured) | Python Client -> AirSim Host |
| **5000** | HTTP / SSE | TALON Operator Cockpit Web Dashboard | Web Browser -> Python Host |

### 2.3 AirSim `settings.json` Placement & Configuration
AirSim reads its configuration on startup from the user's `Documents` directory:
- **Windows Path**: `C:\Users\<Username>\Documents\AirSim\settings.json`
- **Copy Command**:
  ```powershell
  Copy-Item -Path ".\airsim_settings\settings.json" -Destination "$HOME\Documents\AirSim\settings.json" -Force
  ```

#### Exact Verified Configuration (`airsim_settings/settings.json`):
```json
{
  "SeeDocsAt": "https://microsoft.github.io/AirSim/settings/",
  "SettingsVersion": 1.2,
  "SimMode": "Multirotor",
  "ClockSpeed": 1.0,
  "ApiServerPort": 41451,
  "Vehicles": {
    "SimpleFlight": {
      "VehicleType": "SimpleFlight",
      "AutoCreate": true,
      "Cameras": {
        "front_center": {
          "CaptureSettings": [
            {
              "ImageType": 0,
              "Width": 640,
              "Height": 480,
              "FOV_Degrees": 90,
              "TargetGamma": 1.0
            },
            {
              "ImageType": 2,
              "Width": 640,
              "Height": 480,
              "FOV_Degrees": 90
            }
          ],
          "X": 0.35,
          "Y": 0.0,
          "Z": -0.1,
          "Pitch": 0.0,
          "Roll": 0.0,
          "Yaw": 0.0
        }
      }
    }
  },
  "CameraDefaults": {
    "CaptureSettings": [
      {
        "ImageType": 0,
        "Width": 640,
        "Height": 480,
        "FOV_Degrees": 90,
        "TargetGamma": 1.0
      },
      {
        "ImageType": 2,
        "Width": 640,
        "Height": 480,
        "FOV_Degrees": 90
      }
    ]
  }
}
```

---

## 3. Operational Procedure for Mihir's Laptop

This section describes step-by-step how Mihir can run live tests directly on his laptop (either standalone on a single machine, or with AirSim running on a secondary host across LAN).

### 3.1 Pre-Flight Setup & Environment Verification
1. Open PowerShell terminal in the repository root:
   ```powershell
   cd "c:\Users\holme\Desktop\Robotics\AR511 Autonomous Mobile Robot"
   ```
2. Verify Python environment (Python 3.10 to 3.12 recommended):
   ```powershell
   python --version
   ```
3. Ensure required packages are installed:
   ```powershell
   pip install -r requirements.txt
   ```
4. Copy `settings.json` to AirSim folder:
   ```powershell
   if (!(Test-Path "$HOME\Documents\AirSim")) { New-Item -ItemType Directory -Path "$HOME\Documents\AirSim" }
   Copy-Item -Path ".\airsim_settings\settings.json" -Destination "$HOME\Documents\AirSim\settings.json" -Force
   ```

### 3.2 Launching AirSim
1. Start the Unreal Engine simulation binary:
   ```powershell
   # Example: launching CityEnviron binary
   & "C:\Path\To\CityEnviron\CityEnviron.exe" -windowed -ResX=1280 -ResY=720
   ```
2. When prompted by AirSim *"Would you like to use default settings? (Select No if you configured settings.json)"*, select **No** to use our configured `settings.json`.
3. Press **F1** or **Backspace** to verify the vehicle is spawned in `SimpleFlight` multirotor mode.

### 3.3 Launching the TALON Navigation Stack

#### Option A: Localhost Live Simulation (AirSim on same laptop)
```powershell
python -m realtime_citynav.main --speed 4.0 --port 5000
```

#### Option B: Remote LAN Simulation (AirSim on external desktop/laptop)
If Unreal Engine runs on another machine on the LAN (e.g., IP `192.168.1.50`):
```powershell
python -m realtime_citynav.main --airsim-ip 192.168.1.50 --speed 4.0 --port 5000
```
*Note: You can also set environment variables:*
```powershell
$env:AIRSIM_IP = "192.168.1.50"
$env:AIRSIM_PORT = "41451"
python -m realtime_citynav.main --speed 4.0
```

#### Option C: Standalone Offline Mock Mode (No AirSim needed)
If running without Unreal Engine (pure kinematic simulation & synthetic cityscape rendering):
```powershell
python -m realtime_citynav.main --mock --speed 4.0 --port 5000
```

### 3.4 Operator Cockpit Dashboard Controls
1. Open Google Chrome, Edge, or Firefox and navigate to:
   ```
   http://localhost:5000
   ```
2. **Dashboard Features**:
   - **Video Stream**: Central panel displaying real-time annotated camera feed with detection boxes, median depths, and active target crosshairs.
   - **Telemetry Panel**: Live displays for Altitude, Horizontal Speed, NED Coordinates $(X, Y, Z)$, Heading (Yaw degrees), Active Subgoal, and Flight Mode.
   - **Spatial Memory Registry**: Real-time list of all 3D landmarks discovered, their estimated coordinates, observation count, and last-seen timestamp.
   - **Live Commentary Box**: Streaming explanations from `LiveRationaleNarrator` describing current actions, APF deflections, and arrival events.
   - **Operator Chat Box**: Command input for sending mid-flight commands to the drone.

### 3.5 Example In-Flight Natural Language Commands

| Command String | Expected Behavior |
|----------------|-------------------|
| `"take off and fly to fountain"` | Drone takes off to 5.0m, accelerates to cruise speed (4.0 m/s), approaches fountain at 8m standoff. |
| `"circle 2 times around red building at 30m altitude"` | Climbs to 30.0m altitude, enters 16m radius orbit around red building facing inward, hovers after 2 laps. |
| `"circle around blue building facing forward at 25m altitude"` | Enters orbit around blue building at 25.0m with tangential forward-looking camera heading. |
| `"fly 20m above and 15m in front of tower"` | Positions at compound spatial offset: 15m in front of tower entrance at 20m observation altitude. |
| `"visit every landmark in the city"` | Automatically decomposes into a multi-subgoal route touring fountain, red building, blue building, and tower. |
| `"return to fountain"` | Queries spatial memory for fountain coordinates and routes back directly. |
| `"hold position and hover"` | Immediately drops velocity to 0, holding current 3D position and altitude. |
| `"climb to 25m"` | Adjusts cruise altitude to 25.0m while continuing active mission path. |

### 3.6 Emergency Procedures
- **Immediate Software Hover**: Click the red **"EMERGENCY HOVER"** button on the web dashboard, or type `"stop"` / `"hover"` into the chat box.
- **Graceful Stack Shutdown**: Press `Ctrl+C` in the PowerShell terminal running `main.py`. The orchestrator will command immediate soft landing (`landAsync()`), disarm the multirotor, and close background threads.
- **AirSim Hardware Reset**: Press **Backspace** in the AirSim window to instantly reset the vehicle to the origin spawn point.

---

## 4. Offline-to-Online Transition Protocol

To transition seamlessly between offline verification and live AirSim flights:

```
[Offline Development]                        [Live Flight Deployment]
---------------------                        ------------------------
1. Run pytest test suite                     1. Check Windows Firewall (Port 41451)
2. Run scenario_evaluator                    2. Start CityEnviron.exe
3. Run massive scenario matrix               3. Verify settings.json loaded
4. Test kinematic fallback                   4. Launch main.py with --airsim-ip
                                             5. Open http://localhost:5000
```

### Step-by-Step Transition Checklist:
1. **Offline Validation Gate**:
   - Run `python -m pytest` -> Must report 116 passed.
   - Run `python -m realtime_citynav.scenario_evaluator` -> Must report 24/24 passed.
   - Run `python scratch/run_massive_scenario_matrix.py` -> Must report 30/30 passed.
   - Run `python scratch/test_battery_independent_commands.py` -> Must report clean pass.
2. **Pre-Connection Socket Check**:
   - Verify that TCP port 41451 is open and listening before launching `main.py`:
     ```powershell
     Test-NetConnection -ComputerName 127.0.0.1 -Port 41451
     ```
3. **Connection Handshake**:
   - Launch `python -m realtime_citynav.main`.
   - Watch console logs for:
     ```
     [FlightController] Connected to AirSim successfully!
     [FlightController] API Control Enabled & Multirotor Armed.
     ```
   - If this log appears, TALON is operating in live AirSim mode. If AirSim is offline, it cleanly logs:
     ```
     [FlightController] AirSim unreachable (...). Switching to mock kinematic simulator.
     ```
     and continues running without crashing.

---

## 5. Comprehensive Troubleshooting Matrix

| Symptom / Error | Probable Root Cause | Resolution Step |
|-----------------|---------------------|-----------------|
| `AirSim unreachable (Connection refused)` | AirSim binary is not running or listening on a different port/IP. | 1. Ensure `CityEnviron.exe` is running.<br>2. Check `settings.json` has `"ApiServerPort": 41451`.<br>3. Verify IP via `--airsim-ip 127.0.0.1`. |
| `Cannot connect across LAN` | Windows Firewall blocking inbound TCP port 41451 on AirSim host. | On the host running Unreal, run in admin PowerShell:<br>`New-NetFirewallRule -DisplayName "AirSim RPC" -Direction Inbound -LocalPort 41451 -Protocol TCP -Action Allow` |
| `Drone does not take off in AirSim` | AirSim spawned in ComputerVision mode or disarmed. | Verify `settings.json` has `"SimMode": "Multirotor"` and `"VehicleType": "SimpleFlight"`. Ensure `enableApiControl(True)` and `armDisarm(True)` were called. |
| `Camera image black or empty in dashboard` | Camera name mismatch in `settings.json`. | Ensure `settings.json` defines camera with name `"front_center"`. In `config.py`, verify `camera_name: str = "front_center"`. |
| `YOLO-World falls back to CPU/mock` | CUDA not available or PyTorch CPU-only build. | TALON automatically falls back to CPU/mock without errors. For GPU acceleration, install CUDA-enabled PyTorch: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`. |
| `Gemini API key warning` | `GEMINI_API_KEY` environment variable not set. | TALON runs deterministically via heuristic regex fallback with 100% functionality. To enable live LLM, run: `$env:GEMINI_API_KEY="your-key-here"` or pass `--api-key your-key-here`. |
| `Port 5000 already in use` | Another Flask or web service running on port 5000. | Launch TALON on an alternate port: `python -m realtime_citynav.main --port 5050` and access `http://localhost:5050`. |
| `Vehicle crashes into buildings` | APF obstacle repulsion threshold or altitude clearance too low. | In `config.py`, increase `obstacle_distance_threshold = 5.0` or increase takeoff altitude: `--speed 3.0`. |
