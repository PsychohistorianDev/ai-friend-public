@echo off
cd /d "%~dp0"
py engine\blog.py --deploy
pause
