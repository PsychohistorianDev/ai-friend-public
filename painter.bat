@echo off
cd /d "%~dp0"
rem uses the engine's Python; if the painter lives in another one, run e.g.:  py -3.12 engine\painter.py
rem   painter.bat --test "a violet bloom"   paints once and exits
py engine\painter.py %*
pause
