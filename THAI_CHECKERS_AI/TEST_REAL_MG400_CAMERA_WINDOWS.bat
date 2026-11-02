@echo off
chcp 65001 >nul
cd /d "%~dp0"
py -3 TEST_REAL_MG400_READONLY.py --camera
if errorlevel 1 echo Camera/network preflight INCOMPLETE - inspect the report before using MG400.
pause
