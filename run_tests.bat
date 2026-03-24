@echo off
setlocal
cd /d "%~dp0"

python -c "import pytest" 2>nul
if errorlevel 1 (
    echo Installing pytest...
    python -m pip install pytest
    if errorlevel 1 exit /b 1
)

python -m pytest test\ -q %*
set EXITCODE=%ERRORLEVEL%
exit /b %EXITCODE%
