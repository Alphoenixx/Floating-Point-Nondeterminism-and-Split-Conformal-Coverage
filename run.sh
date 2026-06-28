#!/bin/sh
set -e
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTHONHASHSEED=0
PY="${PYTHON:-python3}"
"$PY" -m pip install -r requirements.txt
rm -rf results
"$PY" synthetic_experiment.py
"$PY" real_experiment.py
"$PY" verify.py
