@echo off
cd /d "%~dp0"
py engine\web.py %*
pause
