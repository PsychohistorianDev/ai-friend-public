@echo off
cd /d "%~dp0"
py engine\condense.py %*
pause
