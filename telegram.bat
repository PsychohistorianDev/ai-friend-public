@echo off
cd /d "%~dp0"
:again
py engine\telegram.py
if %errorlevel%==75 (
    echo.
    echo (restarting the bridge on the current engine code...)
    goto again
)
pause
