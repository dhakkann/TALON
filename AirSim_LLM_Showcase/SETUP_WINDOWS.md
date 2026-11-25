# Windows Setup & Demo Guide (AR511 Showcase)

## 1. Install Python Dependencies (Python 3.12+)
```powershell
pip install -r requirements.txt
```

## 2. Download & Extract the Precompiled Simulator (Outside Git)
Download `AirSimInspection.zip` (~1.04 GB) from the upstream release and extract it to `%USERPROFILE%\Downloads\AirSimInspection\` so `AirSimInspection.exe` sits at:
```text
C:\Users\<username>\Downloads\AirSimInspection\AirSimInspection.exe
```
Download URL:
`https://github.com/BashMocha/Prompting-LLMs-for-Aerial-Navigation/releases/download/simulation/AirSimInspection.zip`

## 3. Configure Gemini API Key (No Secrets in Repo)
Set your `GEMINI_API_KEY` as a Windows user environment variable (or in the current PowerShell session):
```powershell
setx GEMINI_API_KEY "your-gemini-api-key"
```
> Keep `src/config.json` with `"API-KEY"` placeholders when committing or sharing the repository.
> If Wi-Fi or Gemini API access is unavailable during the live demo, pass `--offline` or click **Toggle Offline Mode** in the Web Cockpit to use the built-in offline preset & clarification fallback.

## 4. One-Click Launch (Recommended for Showcase)
Double-click `launch_showcase.bat` (or run `.\launch_showcase.bat` in PowerShell). It automatically:
1. Syncs `settings.json` (Multirotor `SimpleFlight`, RGB `ImageType: 0`, `DepthPlanar` `ImageType: 1`) to `%USERPROFILE%\Documents\AirSim\settings.json`.
2. Launches `AirSimInspection.exe` in windowed mode with `-unattended` and lightweight Intel Iris Xe scalability flags (`sg.GlobalIlluminationQuality 0`, `sg.ReflectionQuality 0`, `sg.ResolutionQuality 75`) so integrated GPUs run smoothly. Pass `.\launch_showcase.bat --hq` on a dedicated GPU (e.g., RTX 4060) for full UE5 quality.
3. Starts the Web Chat Cockpit at `http://127.0.0.1:5050` for side-by-side control, live FPV + DepthPlanar telemetry (`320x240` RGB + `80x60` DepthPlanar for low-latency RPC polling), and generated Python code inspection.

For interactive terminal CLI mode instead:
```powershell
.\launch_cli.bat
```
(Supports `--offline`, `--model gemini-3.6-flash`, `!status`, `!offline`, `!online`, `!quit`.)

To record a smooth 24 FPS showcase video (uses stepped simulation `simContinueForTime` so integrated GPUs produce zero dropped frames):
```powershell
python src\record_showcase_video.py
```

## 5. Scene Targets & Obstacle Avoidance
- **Named scene actors:** `turbine1`, `turbine2`, `solarpanels`, `car`, `crowd`, `tower1`, `tower2`, `tower3`.
- **Ambiguity clarification:** Asking `"Fly to the turbine"` or `"Fly to the tower"` triggers a `"Question - ..."` clarification prompt instead of guessing.
- **Obstacle avoidance (`aw.fly_safe` / `aw.get_distance`):** Uses the forward `DepthPlanar` camera central corridor (10th percentile clearance in meters). With default `min_clearance=7.0 m` (triggering lookahead avoidance below `10.0 m` when an obstacle blocks the path), the drone sidesteps right (6 m) and climbs (3 m) before resuming toward its waypoint.
