# Floating-Point Nondeterminism and Split-Conformal Coverage

Reproducibility code for the paper *When the Threshold Moves but the Model Does
Not: Floating-Point Nondeterminism Can Silently Erode Split-Conformal Coverage Guarantees*
(Ankan Sadhu).

The code reproduces every numerical result and every figure in the paper using
only NumPy and Matplotlib. No GPU and no external data download are required:
the real dataset is bundled in `data/gapminder.csv`.

## Contents

| Path | Description |
| --- | --- |
| `run.bat` | One-command entry point on Windows. Runs both experiments, then `verify.py`. |
| `run.sh` | One-command entry point on Linux/macOS. Installs deps, runs both experiments, then `verify.py`. |
| `Dockerfile` | Pinned container (Python 3.13, NumPy 2.4.6, Matplotlib 3.11.0) reproducing Environment A; regenerates all results and figures from a single `docker run`. |
| `synthetic_experiment.py` | Synthetic 10-class study. Produces Tables 1-2 and 5, Figures 1-3 and 5 (`fig5_above_floor.pdf`), the large-n stress regime, and `results/results.json`. |
| `real_experiment.py` | Gapminder 4-continent study. Produces Table 3, Figure 4, and `results/results_real.json`. |
| `verify.py` | Reads both results files and prints a single pass/fail line confirming every certificate check (the digit values are environment-specific by design). |
| `environment.py` | Collects the execution environment that is embedded in each results file. |
| `data/gapminder.csv` | The exact real dataset used in the paper. |
| `results/` | Output directory (created on first run): figures and JSON. |
| `requirements.txt` | Python dependencies (NumPy 2.4.6, Matplotlib 3.11.0). |

## How to run (Windows)

Double-click `run.bat`, or run `run.bat` from a terminal. It sets single-thread
math environment variables, makes sure NumPy is available (installing it once if
missing; Matplotlib is optional and only needed for the figure PDFs), deletes any
stale `results/`, runs both experiments, runs `verify.py`, and prints the
environment and the pass/fail line so you can confirm a correct run.

The install step only runs the first time. After NumPy and Matplotlib are
installed, `run.bat` works fully offline. If you see repeated
`ConnectionAborted 10053` errors, an antivirus, firewall, or VPN is blocking
pip; pause that protection once, run
`python -m pip install numpy==2.4.6 matplotlib==3.11.0`, then run `run.bat`
again.

## How to run (Linux/macOS or Docker)

Linux/macOS, one command:

```
sh run.sh
```

Or build the pinned container (reproduces Environment A) and run it:

```
docker build -t fpnd-conformal .
docker run --rm fpnd-conformal
```

## How to run (any platform, manual)

```
python -m pip install -r requirements.txt
python synthetic_experiment.py
python real_experiment.py
python verify.py
```

`verify.py` prints a single line such as `ALL CERTIFICATE CHECKS PASSED (3/3
alpha synthetic + 3/3 alpha real, stress ok) | top-1 flips = 0 | eps ~ 1e-03
(fp16)`. Because the low-order digits are environment-specific by design, this
pass/fail line is what confirms a correct reproduction, not the exact values.

All outputs are written to `results/`:

- `fig1_jitter_vs_width.pdf`, `fig2_certificate.pdf`, `fig3_score_band.pdf`
- `fig4_realdata_certificate.pdf`, `fig5_above_floor.pdf`
- `results.json`, `results_real.json`

## Recorded execution environment

Every run records the full execution environment into the results JSON under the
`environment` key and prints a short summary at the start of each experiment:
operating system, machine architecture, CPU, Python version and build, NumPy and
Matplotlib versions, the BLAS backend, the single-thread settings, and the exact
version of every installed package. Because the effect studied here is itself
environment-dependent, this metadata makes each reported result self-describing.
The exact least-significant digits of the results depend on the operating
system, CPU, and math library, which is the phenomenon this paper studies; the
qualitative findings and all certificate checks hold on every platform.

## Determinism

Every random draw is seeded (`numpy.random.default_rng`), and BLAS threading is
pinned to a single thread at import time, so runs are bit-for-bit identical
across repeated executions on the same machine. Reduction products inside the
emulated kernels are accumulated in `float64` and then rounded to the target
precision, so the only surviving variability within one machine is the
deliberately injected schedule reordering rather than the host BLAS library. The
floating-point nondeterminism studied in the paper is introduced deliberately
and reproducibly, by evaluating the same trained weights under two fixed
reduction schedules (natural order versus a seeded chunk permutation) with an
`fp16` accumulator. This isolates non-associativity of floating-point summation,
the documented source of run-to-run and cross-kernel nondeterminism, without
relying on hardware variability.

## Method summary

A low-capacity random-feature softmax classifier is trained in `fp32`. The same
weights and inputs are then evaluated under two execution schedules that are
equal in exact arithmetic but differ in summation order and accumulator
precision. Split-conformal calibration fixes the threshold on schedule A; the
resulting prediction sets are compared across schedules A and B to measure
coverage drift and decision churn, which are then checked against the
certificate derived in the paper.

## Data

`data/gapminder.csv` is the public Gapminder dataset (per-country socioeconomic
indicators by year). See https://www.gapminder.org/data/ for the upstream
source and licensing.
