@echo off
REM Windows script to restart the agent server

echo.
echo ============================================
echo   Restarting MCP Agent Server
echo ============================================
echo.

echo 1. Stopping existing server...
taskkill /F /IM python.exe /FI "WINDOWTITLE eq *agent.py*" >nul 2>&1
if errorlevel 1 (
    echo    No existing process found
) else (
    echo    Server stopped
)

timeout /t 2 /nobreak >nul

echo.
echo 2. Starting new server...
echo    Server will run on http://localhost:8000
echo    Press Ctrl+C to stop
echo.

py agent.py
