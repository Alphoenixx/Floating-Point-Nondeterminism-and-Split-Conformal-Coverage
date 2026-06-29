@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

set OMP_NUM_THREADS=1
set OPENBLAS_NUM_THREADS=1
set MKL_NUM_THREADS=1
set NUMEXPR_NUM_THREADS=1
set VECLIB_MAXIMUM_THREADS=1
set PYTHONHASHSEED=0
set PYTHONUNBUFFERED=1

echo ============================================================
echo  Floating-Point Nondeterminism and Split-Conformal Coverage
echo  One-click reproduction (Windows)
echo ============================================================
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python was not found on PATH.
    echo         Install Python 3 from https://www.python.org and re-run this file.
    echo.
    pause
    exit /b 1
)

echo [Stage 1 of 2] Installing dependencies ...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo [ERROR] Dependency installation failed.
    echo         If you see SSL or ConnectionAborted 10053, a firewall, antivirus,
    echo         or VPN is blocking pip. Pause it once and re-run this file.
    echo.
    pause
    exit /b 1
)
echo Dependencies are ready.
echo.

echo [Stage 2 of 2] Running all experiments (progress shown per task) ...
python -u run_all.py
if errorlevel 1 (
    echo.
    echo [ERROR] A required experiment failed. See the messages above.
    echo.
    pause
    exit /b 1
)

echo.
echo Done. All JSON results and figures are in the results folder.
echo The recorded environment and the PASS/FAIL line are printed above.
echo These digits are specific to this machine and environment, which is expected.
echo.
pause
