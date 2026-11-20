@echo off
cd /d "%~dp0"
start "" http://127.0.0.1:8760
".venv\Scripts\python.exe" -m uvicorn server.app:app --host 127.0.0.1 --port 8760
pause
