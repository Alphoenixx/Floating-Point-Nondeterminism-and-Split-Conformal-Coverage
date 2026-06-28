@echo off
setlocal
cd /d "%~dp0"
set OMP_NUM_THREADS=1
set OPENBLAS_NUM_THREADS=1
set MKL_NUM_THREADS=1
set NUMEXPR_NUM_THREADS=1
set VECLIB_MAXIMUM_THREADS=1
set PYTHONHASHSEED=0

where python >nul 2>nul
if errorlevel 1 (
    echo Python was not found on PATH. Install Python 3 from python.org and re-run.
    pause
    exit /b 1
)

python -c "import numpy" >nul 2>nul
if not errorlevel 1 goto checkplot
echo numpy could not be imported. Attempting a one-time install...
python -m pip install numpy==2.4.6 matplotlib==3.11.0
python -c "import numpy" >nul 2>nul
if not errorlevel 1 goto checkplot
echo.
echo numpy is present on disk but Python cannot import it. The exact error follows:
echo ----------------------------------------------------------------------
python -c "import numpy"
echo ----------------------------------------------------------------------
echo If this mentions a blocked download, SSL, or ConnectionAborted 10053, your
echo antivirus/firewall/VPN is blocking pip; pause it once and run:
echo     python -m pip install numpy==2.4.6 matplotlib==3.11.0
pause
exit /b 1

:checkplot
python -c "import matplotlib" >nul 2>nul
if not errorlevel 1 goto run
echo.
echo NOTE: matplotlib could not be imported, so figure PDFs will be skipped.
echo All numeric results (results.json, results_real.json) are still produced.
echo The figures are already included in the LaTeX source, so this is harmless
echo for reproducing the paper numbers. To enable figures, repair matplotlib with:
echo     python -m pip install --force-reinstall --no-cache-dir matplotlib==3.11.0
echo.

:run
if exist results rmdir /s /q results
python synthetic_experiment.py
if errorlevel 1 (
    echo synthetic_experiment.py failed.
    pause
    exit /b 1
)
python real_experiment.py
if errorlevel 1 (
    echo real_experiment.py failed.
    pause
    exit /b 1
)
echo.
echo Finished. JSON and figures are in the results folder.
python verify.py
python -c "import json;r=json.load(open('results/results.json'));e=r['environment'];print('ran on        :',e['platform'],'|',e['machine'],'| python',e['python_version'],'| numpy',e['numpy_version'])"
python -c "import json;r=json.load(open('results/results.json'));print('synthetic fp32 eps =',r['eps_fp32_max'])"
python -c "import json;r=json.load(open('results/results_real.json'));print('real eps_max       =',r['eps_max'])"
echo These numbers are specific to this machine and environment, which is expected for this study.
pause
