@echo off
cd /d "%~dp0"
py engine\backfill_creations.py %*
pause
