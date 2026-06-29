#!/bin/sh
set -e
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTHONHASHSEED=0
export PYTHONUNBUFFERED=1
PY="${PYTHON:-python3}"

echo "[Stage 1 of 2] Installing dependencies ..."
"$PY" -m pip install --upgrade pip
"$PY" -m pip install -r requirements.txt
echo "Dependencies are ready."
echo

echo "[Stage 2 of 2] Running all experiments (progress shown per task) ..."
"$PY" -u run_all.py
