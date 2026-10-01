@echo off
setlocal
start "Sustainable Beam API" cmd /k "cd /d %~dp0backend && python -m uvicorn app.main:app --reload --port 8000"
start "Sustainable Beam UI" cmd /k "cd /d %~dp0frontend && npm run dev"
endlocal
