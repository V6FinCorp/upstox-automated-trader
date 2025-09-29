@echo off
setlocal
set UAT_CONFIG=%~dp0..\src\config\config.json
python %~dp0..\src\main.py
endlocal
