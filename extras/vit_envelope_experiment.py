import os
import json
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)

try:
    import torch
    from torchvision.models import vit_b_16, ViT_B_16_Weights
except Exception as e:
    print("TORCHVISION_MISSING", repr(e))
    print("PASTE_BACK_BEGIN")
    print(json.dumps({"error": "torchvision_missing", "detail": repr(e)}))
    print("PASTE_BACK_END")
    raise SystemExit(0)

def env_stats(a, b):
    d = (a.astype(np.float64) - b.astype(np.float64))
    ad = np.abs(d)
    return {"eps_max": float(ad.max()), "eps_mean": float(ad.mean()),
            "eps_99": float(np.quantile(ad, 0.99)), "signed_mean": float(d.mean())}

def softmax_np(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)

torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
results = {"provenance": "Single-device logit envelope and reduction-length sqrt-D sweep behind Figure 7 and the Section 6.7 per-element growth numbers. The paper headline network-level worst-output envelope (7.6e-3 logit, 2.1e-4 score) and the Table 6 coverage are the discrete-GPU (GTX 1660 SUPER) cuBLAS cross-device measurement archived in extras/vit_crossdevice/, not this file; the device field below records where this file was generated.", "device": dev, "cuda_name": torch.cuda.get_device_name(0) if dev == "cuda" else None,
           "torch_version": torch.__version__}

try:
    weights = ViT_B_16_Weights.IMAGENET1K_V1
    model = vit_b_16(weights=weights)
except Exception as e:
    print("WEIGHTS_DOWNLOAD_FAILED", repr(e))
    print("PASTE_BACK_BEGIN")
    print(json.dumps({"error": "weights_download_failed", "detail": repr(e)}))
    print("PASTE_BACK_END")
    raise SystemExit(0)

model.eval()
N = 256
bs = 32
g = torch.Generator().manual_seed(123)
X = torch.randn(N, 3, 224, 224, generator=g)

def run_logits(m, x, device, half):
    m = m.to(device)
    if half:
        m = m.half()
    outs = []
    with torch.no_grad():
        for i in range(0, x.shape[0], bs):
            xb = x[i:i + bs].to(device)
            xb = xb.half() if half else xb.float()
            outs.append(m(xb).float().cpu().numpy())
    return np.concatenate(outs, 0)

L_cpu32 = run_logits(vit_b_16(weights=weights).eval(), X, "cpu", False)

if dev == "cuda":
    L_gpu32 = run_logits(vit_b_16(weights=weights).eval(), X, "cuda", False)
    L_gpu16 = run_logits(vit_b_16(weights=weights).eval(), X, "cuda", True)
    results["logit_envelope"] = {
        "gpu_fp16_vs_gpu_fp32": env_stats(L_gpu16, L_gpu32),
        "gpu_fp32_vs_cpu_fp32": env_stats(L_gpu32, L_cpu32),
    }
    results["score_envelope_gpu_fp16_vs_fp32"] = env_stats(
        softmax_np(L_gpu16), softmax_np(L_gpu32))
    ref_for_sweep = "cuda"
else:
    L_cpu16 = run_logits(vit_b_16(weights=weights).eval(), X, "cpu", True)
    results["logit_envelope"] = {"cpu_fp16_vs_cpu_fp32": env_stats(L_cpu16, L_cpu32)}
    results["score_envelope_cpu_fp16_vs_fp32"] = env_stats(
        softmax_np(L_cpu16), softmax_np(L_cpu32))
    ref_for_sweep = "cpu"

mlp_w = None
for name, p in model.named_parameters():
    if name.endswith("mlp.0.weight"):
        mlp_w = p.detach().float()
        break
if mlp_w is None:
    mlp_w = model.heads.head.weight.detach().float()

out_dim, in_dim = mlp_w.shape
Ms = 4096
ga = torch.Generator().manual_seed(7)
A_full = torch.randn(Ms, in_dim, generator=ga)
sweep = []
Ds = [d for d in [64, 128, 256, 384, 512, 768, in_dim] if d <= in_dim]
Ds = sorted(set(Ds))
for D in Ds:
    A = A_full[:, :D].contiguous()
    Wd = mlp_w[:, :D].contiguous()
    if ref_for_sweep == "cuda":
        Ag = A.cuda()
        Wg = Wd.cuda()
        ref = (Ag.double() @ Wg.double().t()).float().cpu().numpy()
        f16 = (Ag.half() @ Wg.half().t()).float().cpu().numpy()
    else:
        ref = (A.double() @ Wd.double().t()).numpy()
        f16 = (A.half() @ Wd.half().t()).float().numpy()
    st = env_stats(f16, ref)
    st["reduction_length"] = int(D)
    st["eps_max_over_sqrtD"] = st["eps_max"] / (D ** 0.5)
    sweep.append(st)

results["reduction_length_sweep"] = {"weight_used": [int(out_dim), int(in_dim)], "points": sweep}

json.dump(results, open(os.path.join(OUT, "results_vit.json"), "w"), indent=2)
print("PASTE_BACK_BEGIN")
print(json.dumps(results, indent=2))
print("PASTE_BACK_END")
le = results["logit_envelope"]
for k, v in le.items():
    print("DONE logit {}: eps_max={:.3e} eps_mean={:.3e} signed={:+.3e}".format(k, v["eps_max"], v["eps_mean"], v["signed_mean"]))
for p in sweep:
    print("SWEEP D={:5d} eps_max={:.3e} eps_max/sqrtD={:.3e}".format(p["reduction_length"], p["eps_max"], p["eps_max_over_sqrtD"]))
