# Floating-Point Nondeterminism and Split-Conformal Coverage

Reproducibility code for the paper *When the Threshold Moves but the Model Does
Not: Floating-Point Nondeterminism Breaks Split-Conformal Coverage Guarantees*
(Ankan Sadhu).

The code reproduces every numerical result and every figure in the paper using
only NumPy and Matplotlib. No GPU, no network access, and no external data
download are required: the real dataset is bundled in `data/gapminder.csv`.

## Contents

| Path | Description |
| --- | --- |
| `synthetic_experiment.py` | Synthetic 10-class study. Produces Tables 1-2, Figures 1-3, and `results/results.json`. |
| `real_experiment.py` | Gapminder 4-continent study. Produces Table 3, Figure 4, and `results/results_real.json`. |
| `run_all.py` | Runs both experiments in sequence. |
| `data/gapminder.csv` | The exact real dataset used in the paper. |
| `results/` | Output directory (created on first run): figures and JSON. |
| `requirements.txt` | Python dependencies. |

## Setup

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.13, NumPy 2.4, and Matplotlib 3.11.

## Reproduce everything

```
python3 run_all.py
```

or run the two experiments individually:

```
python3 synthetic_experiment.py
python3 real_experiment.py
```

All outputs are written to `results/`:

- `fig1_jitter_vs_width.pdf`, `fig2_certificate.pdf`, `fig3_score_band.pdf`
- `fig4_realdata_certificate.pdf`
- `results.json`, `results_real.json`

## Determinism

Every random draw is seeded (`numpy.random.default_rng`), so repeated runs on
the same NumPy build are bit-for-bit identical. The floating-point
nondeterminism studied in the paper is introduced deliberately and
reproducibly, by evaluating the same trained weights under two fixed reduction
schedules (natural order versus a seeded chunk permutation) with an `fp16`
accumulator. This isolates non-associativity of floating-point summation, the
documented source of run-to-run and cross-kernel nondeterminism, without
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
