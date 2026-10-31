@echo off
rem Double-click to start SpinStudio on quasi08 (pass quasi09 as argument for the other node).
set NODE=%1
if "%NODE%"=="" set NODE=quasi08
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0SpinStudio.ps1" -Node %NODE%
pause
