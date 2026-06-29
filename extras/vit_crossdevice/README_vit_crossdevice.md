# Optional ViT cross-device measurement (quarantined)

This path is **not** part of the core reproduction (`run.sh` / `run_all.py`) and is
**not** required to verify any result in the paper. The canonical artifact is CPU-only
and dependency-light; this directory exists solely to record a single, frozen real-model
measurement on a CUDA GPU.

## What it does

`vit_device_scores.py` loads a pinned torchvision ViT-B/16 (`ViT_B_16_Weights.IMAGENET1K_V1`),
pushes a fixed, seeded synthetic input set (no ImageNet download) through the network in
fp32 and fp16, and saves split-conformal nonconformity scores in the same NPZ schema as
the `extras/crossdevice` pipeline. Reference labels are the fp32 argmax (kernel-independent),
so the fp32->fp16 pair isolates the effect of the divergent cuBLAS half-precision kernel.

The heavy dependencies (torch, torchvision) are isolated in `requirements_vit.txt`.

## Run once (GPU host)

```sh
pip install -r requirements_vit.txt
python vit_device_scores.py --label B_vit_gpu --device cuda
python ../crossdevice/pair_offline.py \
    --files scores_B_vit_gpu.npz \
    --pairs "B_vit_gpu:fp32->B_vit_gpu:fp16" \
    --out vit_crossdevice_results.json
python vit_verify.py vit_crossdevice_results.json
```

## Reproducing the certificate without a GPU

A reviewer without a GPU does **not** recompute the network. The recorded
`scores_B_vit_gpu.npz` is shipped; `vit_verify.py` re-checks that the conformal
certificate held on those saved outputs.

## Notes

- GTX 16-series (Turing TU116) has **no tensor cores**; fp16 GEMM runs on standard CUDA
  cores. The divergence measured here is a genuine distinct kernel, but it is not
  tensor-core fused. Tensor-core hardware may show a different (often larger) envelope.
- Inputs are fixed seeded tensors. This isolates kernel divergence on a real pretrained
  network; it is not a statement about natural-image accuracy.
