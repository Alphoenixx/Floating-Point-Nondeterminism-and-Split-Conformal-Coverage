# Cross-device coverage measurement (A/B/C)

Turns calibrate-here / deploy-there from emulated into measured on physically distinct hardware, reusing the frozen 512-feature synthetic model. The only thing that varies across devices is the deployment matmul's reduction order/precision; the model weights and inputs are byte-identical (SHA-256 verified).

## Files

- `device_scores.py` — loads the frozen artifact, computes native logits/scores on one device. `--device cpu` (default, NumPy/BLAS) or `--device cuda` (PyTorch, computes fp32 and fp16).
- `pair_offline.py` — CPU-only. For each ordered pair (source calibrate -> target deploy) reports eps_max, eps_99, eps_mean, coverage in source and target, churn, top-1 flips, and certificate checks.

## Procedure

### Step 1 - Device A (desktop CPU), builds the canonical artifact

```
python extras/crossdevice/device_scores.py --build --label A_desktop_cpu
```

Produces `extras/crossdevice/frozen_artifact.npz` (copy this file, unchanged, to B and C) and `scores_A_desktop_cpu.npz`.

### Step 2 - Device B (desktop GPU)

```
python extras/crossdevice/device_scores.py --label B_desktop_gpu --device cuda
```

Reuses the same `frozen_artifact.npz` already on the desktop. Produces `scores_B_desktop_gpu.npz` with fp32 and fp16 variants.

### Step 3 - Device C (laptop CPU)

Copy the whole repo (including `frozen_artifact.npz`) to the laptop, then:

```
python extras/crossdevice/device_scores.py --label C_laptop_cpu
```

Produces `scores_C_laptop_cpu.npz`.

### Step 4 - Offline pairing (any CPU)

```
python extras/crossdevice/pair_offline.py \
  --files scores_A_desktop_cpu.npz scores_B_desktop_gpu.npz scores_C_laptop_cpu.npz \
  --pairs "A_desktop_cpu:fp32->B_desktop_gpu:fp32" \
          "A_desktop_cpu:fp32->C_laptop_cpu:fp32" \
          "B_desktop_gpu:fp32->B_desktop_gpu:fp16"
```

Writes `crossdevice_results.json`. A GPU-less reviewer can rerun Step 1 and Step 4 on CPU and read the GPU/laptop rows from the recorded score files with zero GPU recompute.
