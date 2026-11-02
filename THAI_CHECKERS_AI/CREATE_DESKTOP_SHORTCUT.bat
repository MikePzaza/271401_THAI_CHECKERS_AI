@echo off
setlocal
cd /d "%~dp0"
echo Creating a desktop shortcut for AI ADOM VS YOU...
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0Create_Desktop_Shortcut.ps1"
if errorlevel 1 (
  echo.
  echo Shortcut creation failed. Check the error above and verify that the Desktop folder exists.
) else (
  echo.
  echo Done. Look for AI ADOM VS YOU on your desktop.
)
echo.
pause
