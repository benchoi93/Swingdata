@echo off
REM Quick manual trigger — run the daily agent right now
REM Usage: double-click or run from command prompt

echo Starting Claude Code agent session...
echo.
powershell -ExecutionPolicy Bypass -File "%~dp0run_daily.ps1"
echo.
echo Session complete. Check reports\ for today's report.
pause
