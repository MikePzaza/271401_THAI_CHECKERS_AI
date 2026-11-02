@echo off
chcp 65001 >nul
cd /d "%~dp0"
py -3 TEST_REAL_MG400_READONLY.py
if errorlevel 1 echo Hardware preflight INCOMPLETE - inspect the report before using MG400.
pause
