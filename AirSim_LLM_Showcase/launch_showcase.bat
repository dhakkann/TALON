@echo off
setlocal
cd /d "%~dp0"

echo ==================================================
echo   AR511 Language-Guided Drone Showcase Launcher
echo ==================================================

if not exist "%USERPROFILE%\Documents\AirSim" mkdir "%USERPROFILE%\Documents\AirSim"
copy /Y "%~dp0settings.json" "%USERPROFILE%\Documents\AirSim\settings.json" >nul
echo [1/3] Synced AirSim settings.json to %USERPROFILE%\Documents\AirSim\settings.json

set "SIM_EXE=%USERPROFILE%\Downloads\AirSimInspection\AirSimInspection.exe"
if not exist "%SIM_EXE%" (
    if exist "%~dp0..\AirSimInspection\AirSimInspection.exe" set "SIM_EXE=%~dp0..\AirSimInspection\AirSimInspection.exe"
)
for %%I in ("%SIM_EXE%") do set "SIM_DIR=%%~dpI"

set "EXEC_CMDS=sg.ResolutionQuality 75,sg.ViewDistanceQuality 1,sg.AntiAliasingQuality 1,sg.ShadowQuality 1,sg.GlobalIlluminationQuality 0,sg.ReflectionQuality 0,sg.PostProcessQuality 1,sg.EffectsQuality 0,sg.FoliageQuality 0,sg.ShadingQuality 1,r.Lumen.DiffuseIndirect.Allow 0,r.Lumen.Reflections.Allow 0,DisableAllScreenMessages"
if /I "%~1"=="--hq" (
    set "EXEC_CMDS=DisableAllScreenMessages"
    shift
)

powershell -NoProfile -Command "try { $c = New-Object System.Net.Sockets.TcpClient('127.0.0.1', 41451); $c.Close(); exit 0 } catch { exit 1 }"
if %errorlevel% neq 0 (
    if exist "%SIM_EXE%" (
        echo [2/3] Launching AirSimInspection.exe in windowed unattended mode...
        start "" /D "%SIM_DIR%" "%SIM_EXE%" -windowed -ResX=800 -ResY=600 -unattended -ExecCmds="%EXEC_CMDS%"
        echo Waiting for AirSim RPC port 41451...
        powershell -NoProfile -Command "for($i=0;$i -lt 45;$i++){ try{$c=New-Object System.Net.Sockets.TcpClient('127.0.0.1',41451);$c.Close();exit 0}catch{Start-Sleep -Seconds 1} }; exit 1"
    ) else (
        echo [WARNING] AirSimInspection.exe not found at %SIM_EXE%.
        echo           Start AirSimInspection manually, then refresh the cockpit.
    )
) else (
    echo [2/3] AirSim RPC port 41451 is already listening.
)

echo [3/3] Starting Web Chat Cockpit on http://127.0.0.1:5050 ...
start "" /B powershell -NoProfile -Command "for($i=0;$i -lt 30;$i++){ try{$c=New-Object System.Net.Sockets.TcpClient('127.0.0.1',5050);$c.Close();Start-Process 'http://127.0.0.1:5050';exit 0}catch{Start-Sleep -Milliseconds 300} }"
python "%~dp0src\web_cockpit.py" %*
endlocal
