@echo off
setlocal
cd /d "%~dp0"

if not exist "%USERPROFILE%\Documents\AirSim" mkdir "%USERPROFILE%\Documents\AirSim"
copy /Y "%~dp0settings.json" "%USERPROFILE%\Documents\AirSim\settings.json" >nul

set "SIM_EXE=%USERPROFILE%\Downloads\AirSimInspection\AirSimInspection.exe"
if not exist "%SIM_EXE%" (
    if exist "%~dp0..\AirSimInspection\AirSimInspection.exe" set "SIM_EXE=%~dp0..\AirSimInspection\AirSimInspection.exe"
)
for %%I in ("%SIM_EXE%") do set "SIM_DIR=%%~dpI"

powershell -NoProfile -Command "try { $c = New-Object System.Net.Sockets.TcpClient('127.0.0.1', 41451); $c.Close(); exit 0 } catch { exit 1 }"
if %errorlevel% neq 0 (
    if exist "%SIM_EXE%" (
        echo Launching AirSimInspection.exe...
        start "" /D "%SIM_DIR%" "%SIM_EXE%" -windowed -ResX=800 -ResY=600 -unattended -ExecCmds="DisableAllScreenMessages"
        powershell -NoProfile -Command "for($i=0;$i -lt 45;$i++){ try{$c=New-Object System.Net.Sockets.TcpClient('127.0.0.1',41451);$c.Close();exit 0}catch{Start-Sleep -Seconds 1} }; exit 1"
    )
)

python "%~dp0src\gemini_airsim.py" %*
endlocal
