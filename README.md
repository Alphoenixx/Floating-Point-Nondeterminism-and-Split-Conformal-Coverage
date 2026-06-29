# Floating-Point Nondeterminism and Split-Conformal Coverage

Reproducibility code for the paper *When the Threshold Moves but the Model Does
Not: Floating-Point Nondeterminism Can Silently Erode Split-Conformal Coverage
Guarantees*.

**Author:** Ankan Sadhu, Government College of Engineering and Ceramic Technology (GCECT), Kolkata, India.

**Repository:** https://github.com/Alphoenixx/Floating-Point-Nondeterminism-and-Split-Conformal-Coverage  
**Archived snapshot (Zenodo):** https://doi.org/10.5281/zenodo.20979968

Released under the Apache License 2.0 (see `LICENSE`).

The code reproduces every numerical result and every figure in the paper. The
from-scratch synthetic and real-data experiments (Tables 2-5, 8, 9 and Figures
1-7) use only NumPy and Matplotlib. The two device-specific measured tables ---
Table 6 (the ViT `float32`->`float16` coverage) and Table 7 (the cross-device
drift) --- are recorded measurements on named hardware; `run_all.py` reproduces
their published numbers offline from shipped, SHA-256-frozen score artifacts, with
no GPU and no PyTorch required. Regenerating those raw artifacts on real hardware
(`extras/vit_envelope_experiment.py` and the `extras/crossdevice` /
`extras/vit_crossdevice` device scripts) additionally uses PyTorch. No external
data download is required: the real dataset is bundled in `data/gapminder.csv`.

## One command

- **Windows:** double-click `run.bat` (or run it from a terminal).
- **Linux/macOS:** `sh run.sh`.
- **Docker (pinned Linux environment):** `docker build -t fpnd . && docker run --rm fpnd`.

Each entry point installs dependencies, then runs `run_all.py`, which executes
every experiment in order with a per-task progress bar and fully unbuffered
output, regenerates all figures, reproduces the cross-device (Table 7) and ViT
(Table 6) certificates offline from the shipped frozen score artifacts, records
the full execution environment into every results file, and finally runs
`verify.py` to print one pass/fail line.

## Project layout

```
run.bat / run.sh        one-click entry points (install + run, with progress bar)
run_all.py              orchestrator: runs every step, stamps provenance, verifies
requirements.txt        loose dependency bounds for the one-command install
Dockerfile              pinned, citable Linux environment
src/                    all from-scratch experiment and library code
  environment.py            execution-environment capture
  synthetic_experiment.py   Tables 2, 3, 8, 9; Figures 1-3, 5
  real_experiment.py        Table 4; Figure 4
  multiseed_experiment.py   50-seed means/SE behind Tables 3, 8, 9; feeds Figure 6
  multiseed_real.py         50-split real-data means/SE (Section 6.1)
  realkernel_experiment.py  Table 5 (float32/float16 rows)
  int8_ptq_experiment.py    Table 5 (int8 rows; bias-variance dichotomy)
  make_figures.py           Figures 6 and 7 from the JSON above
  verify.py                 one-line pass/fail certificate check
extras/                 optional on-hardware measurements (PyTorch)
  vit_envelope_experiment.py        ViT envelope + sqrt-D sweep -> results_vit.json (Figure 7)
  crossdevice/                      Table 7 cross-device drift (devices A/B/C)
    device_scores.py                per-device native scores (CPU NumPy or CUDA PyTorch)
    pair_offline.py                 CPU-only offline pairing -> reproduces Table 7
    frozen_artifact.npz             SHA-256-frozen weights+features shared across devices
    scores_*.npz, *_summary.json    recorded per-device score artifacts
  vit_crossdevice/                  Table 6 ViT float32->float16 coverage
    vit_device_scores.py            records GPU fp32/fp16 ViT scores
    vit_verify.py                   CPU-only certificate check -> reproduces Table 6
    scores_B_vit_gpu.npz, *.json    recorded GPU score artifacts
data/gapminder.csv      bundled public real dataset
results/                generated JSON results and figures (overwritten each run;
                        results_vit.json is preserved so Figure 7 rebuilds without a GPU)
```

## Table and figure map (matches the paper)

The paper's Table 1 is the notation glossary; the data tables are Tables 2
through 9. Tables 2-5, 8, and 9 are reproduced from scratch; Tables 6 and 7 are
device-specific measurements whose published numbers are reproduced offline from
shipped frozen score artifacts (no GPU required).

| Script | Produces |
| --- | --- |
| `src/synthetic_experiment.py` | Tables 2, 3, 8, 9; Figures 1-3 and 5; `results/results.json` (synthetic envelope, coverage/churn, large-n stress regime and its precision-robust remedy). |
| `src/real_experiment.py` | Table 4; Figure 4; `results/results_real.json` (Gapminder real-data validation). |
| `src/multiseed_experiment.py` | The 50-resample means +/- standard errors and 95% CIs reported for the synthetic and stress regimes (Tables 3, 8, 9); `results/results_multiseed.json`; also feeds Figure 6. |
| `src/multiseed_real.py` | Real-data means +/- standard errors over 50 splits (quoted in Section 6.1, behind Table 4); `results/results_multiseed_real.json`. |
| `src/realkernel_experiment.py` | Table 5, the stock `float32`/`float16` library-matmul rows; `results/results_realkernel.json`. |
| `src/int8_ptq_experiment.py` | Table 5, the standard `int8` PTQ rows (round-to-nearest vs truncating; the benign-vs-biased dichotomy); `results/results_int8_ptq.json`. |
| `src/make_figures.py` | Figures 6 (kernel dichotomy) and 7 (ViT sqrt-D growth) from the JSON above; `results/fig6_dichotomy.pdf`, `results/fig7_vit_sqrtD.pdf`. |
| `extras/vit_envelope_experiment.py` | `results/results_vit.json`: the ViT-B/16 logit envelope and the reduction-length sqrt-D sweep behind Figure 7 and the per-element growth numbers of Section 6.7. Needs `pip install torch torchvision`; a recorded run is shipped and reused (so Figure 7 still rebuilds) when PyTorch is absent. |
| `extras/crossdevice/` (`device_scores.py` + `pair_offline.py`) | Table 7, the measured cross-device drift on the SHA-256-frozen artifact (devices A/B/C). `run_all.py` runs `pair_offline.py` on the shipped per-device score files to reproduce the published coverages (`0.9433/0.8984/0.8004`) and per-pair envelopes on any CPU. |
| `extras/vit_crossdevice/` (`vit_device_scores.py` + `vit_verify.py`) | Table 6, the ViT-B/16 `float32`->`float16` coverage on a real cuBLAS kernel. `run_all.py` runs `vit_verify.py` on the shipped recorded GPU scores to reproduce the published coverages and certificate on any CPU. |

**Table 5 (the single-environment numerical-envelope table)** reports the
envelope of the stock `float32`, `float16`, and `int8` library kernels, all
measured within one execution environment. It is produced by
`src/realkernel_experiment.py` and `src/int8_ptq_experiment.py`.

**Tables 6 and 7 (the measured-hardware tables)** are recorded once on named
devices (an NVIDIA GTX 1660 SUPER for the ViT kernel switch; an AMD desktop CPU,
the same GPU, and an Intel laptop CPU for the cross-device study). Regenerating the
raw scores needs that hardware, but the conformal certificate is a property of the
recorded scores, so the shipped, SHA-256-frozen `.npz` score files let `run_all.py`
recompute every published coverage, envelope, and certificate check on any CPU ---
no GPU, no PyTorch. The same-host thread-count control in the last row of Table 7
(OpenBLAS one versus eight threads, envelope `3.8e-7`) is a calibration-score
ablation recorded in the device summary JSON files.

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
what confirms a correct reproduction, not the exact values. In the same run,
`extras/crossdevice/pair_offline.py` prints `RESULT cross-device: ALL PAIRS WITHIN
CERTIFICATE` and `extras/vit_crossdevice/vit_verify.py` prints `VIT REPRODUCTION
CHECK: PASS`, reproducing Tables 7 and 6 from the frozen score artifacts.

## Data

`data/gapminder.csv` is the public Gapminder dataset (per-country socioeconomic
indicators by year). See https://www.gapminder.org/data/ for the upstream source
and licensing.

## License

This code is released under the Apache License 2.0; see the `LICENSE` file. The bundled Gapminder dataset (`data/gapminder.csv`) is the public Gapminder data; see https://www.gapminder.org/data/ for its upstream source and terms.

## Citation

If you use this code or its results, please cite the paper and the archived snapshot:

> Ankan Sadhu. *When the Threshold Moves but the Model Does Not: Floating-Point Nondeterminism Can Silently Erode Split-Conformal Coverage Guarantees.* 2026. Zenodo. https://doi.org/10.5281/zenodo.20979968
