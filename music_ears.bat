@echo off
cd /d "%~dp0"
rem uses the engine's Python; if the ear lives in another one, run e.g.:  py -3.12 engine\music_ears.py
py engine\music_ears.py %*
pause
