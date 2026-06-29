# Floating-Point Nondeterminism and Split-Conformal Coverage

Reproducibility code for the paper *When the Threshold Moves but the Model Does
Not: Floating-Point Nondeterminism Can Silently Erode Split-Conformal Coverage
Guarantees*.

This copy is anonymized for double-blind peer review. Author, affiliation, and the
public repository and Zenodo archive links will be restored upon acceptance.

The code reproduces every numerical result and every figure in the paper. The
main synthetic and real-data experiments use only NumPy and Matplotlib; one
optional supporting experiment (`extras/vit_envelope_experiment.py`) additionally
uses PyTorch. No external data download is required: the real dataset is bundled
in `data/gapminder.csv`.

## One command

- **Windows:** double-click `run.bat` (or run it from a terminal).
- **Linux/macOS:** `sh run.sh`.
- **Docker (pinned Linux environment):** `docker build -t fpnd . && docker run --rm fpnd`.

Each entry point installs dependencies, then runs `run_all.py`, which executes
every experiment in order with a per-task progress bar and fully unbuffered
output, regenerates all figures, records the full execution environment into every
results file, and finally runs `verify.py` to print one pass/fail line.

## Project layout

```
run.bat / run.sh        one-click entry points (install + run, with progress bar)
run_all.py              orchestrator: runs every step, stamps provenance, verifies
requirements.txt        loose dependency bounds for the one-command install
Dockerfile              pinned, citable Linux environment
src/                    all experiment and library code
  environment.py        execution-environment capture
  synthetic_experiment.py
  real_experiment.py
  multiseed_experiment.py
  multiseed_real.py
  realkernel_experiment.py
  int8_ptq_experiment.py
  make_figures.py       optional Figures 6 and 7
  verify.py             one-line pass/fail certificate check
extras/                 optional PyTorch-only experiment
  vit_envelope_experiment.py
data/gapminder.csv      bundled public real dataset
results/                generated JSON results and figures (overwritten each run)
```

## Table and figure map (matches the paper)

The paper's Table 1 is the notation glossary; the data tables are Tables 2-7.

| Script | Produces |
| --- | --- |
| `src/synthetic_experiment.py` | Tables 2, 3, 6, 7; Figures 1-3 and 5; `results/results.json` (synthetic envelope, coverage/churn, large-n stress regime and its remedy). |
| `src/real_experiment.py` | Table 4; Figure 4; `results/results_real.json` (Gapminder real-data validation). |
| `src/multiseed_experiment.py` | Table 6 means +/- standard errors and 95% CIs over 50 resamples; `results/results_multiseed.json`. |
| `src/multiseed_real.py` | Real-data means +/- standard errors over 50 splits (quoted in Section 6.1); `results/results_multiseed_real.json`. |
| `src/realkernel_experiment.py` | Table 5 (stock float32/float16 library matmul envelope); `results/results_realkernel.json`. |
| `src/int8_ptq_experiment.py` | Table 5 (standard int8 PTQ, round-to-nearest vs truncating; the benign-vs-biased dichotomy); `results/results_int8_ptq.json`. |
| `extras/vit_envelope_experiment.py` | Production-kernel section (pretrained ViT-B/16 envelope and its growth with reduction length); `results/results_vit.json`. Requires `pip install torch torchvision`. |
| `src/make_figures.py` | Optional Figures 6 (kernel dichotomy) and 7 (ViT sqrt-D growth) from the JSON above; `results/fig6_dichotomy.pdf`, `results/fig7_vit_sqrtD.pdf`. |

**Table 5 (the numerical-envelope table)** reports the envelope of the stock
`float32`, `float16`, and `int8` library kernels, all measured within a single
execution environment. It is produced by `src/realkernel_experiment.py` and
`src/int8_ptq_experiment.py`; a single run regenerates it along with every other
table and all certificate checks.

## Dependencies

`requirements.txt` uses loose lower bounds (`numpy>=2.0,<3`, `matplotlib>=3.7`)
so the one-command install never fails on a supported Python. The exact versions
actually used are recorded into every results file (and printed at the end of the
run), which is what the paper reports. The `Dockerfile` pins the exact
combination used for the reported numbers (`numpy==2.4.1`, `matplotlib==3.10.8`)
to provide a fixed, citable Linux environment.

## Recorded execution environment

Every results file carries an `environment` block: operating system, machine
architecture, CPU, Python version and build, NumPy and Matplotlib versions, the
BLAS backend, the single-thread settings, and the version of every installed
package. Because the effect studied here is itself environment-dependent, this
metadata makes each reported number self-describing. The *magnitude* of the
envelope depends on the deployed kernel and numerical precision --- that is the
phenomenon this paper studies --- while the certificate outcomes (all checks
satisfied, zero top-1 flips) hold on every platform.

## Determinism

Every random draw is seeded (`numpy.random.default_rng`) and BLAS threading is
pinned to a single thread, so runs are bit-for-bit identical across repeated
executions on the same machine. The injected nondeterminism is deliberate and
reproducible: the same trained weights are evaluated under two fixed reduction
schedules (natural order versus a seeded chunk permutation) with the target
accumulator precision, isolating non-associativity of floating-point summation.

## Verification

`src/verify.py` reads the results files and prints one line, e.g. `ALL
CERTIFICATE CHECKS PASSED ... | top-1 flips = 0 | eps ~ 1e-03 (fp16)`. Because
the low-order digits are environment-specific by design, this pass/fail line is
what confirms a correct reproduction, not the exact values.

## Data

`data/gapminder.csv` is the public Gapminder dataset (per-country socioeconomic
indicators by year). See https://www.gapminder.org/data/ for the upstream source
and licensing.
