import os
import sys
import json
import hashlib
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def score_vec(logits, labels):
    P = softmax(logits)
    return 1.0 - P[np.arange(len(labels)), labels]


def env_stats(a, b):
    d = a.astype(np.float64) - b.astype(np.float64)
    ad = np.abs(d)
    return {"eps_max": float(ad.max()), "eps_mean": float(ad.mean()),
            "eps_99": float(np.quantile(ad, 0.99)), "signed_mean": float(d.mean())}


def run_logits(model, n, base_seed, bs, device, half):
    import torch
    m = torch.tensor(MEAN).view(1, 3, 1, 1)
    s = torch.tensor(STD).view(1, 3, 1, 1)
    outs = []
    done = 0
    bi = 0
    while done < n:
        b = min(bs, n - done)
        g = torch.Generator().manual_seed(base_seed + bi)
        x = torch.randn(b, 3, 224, 224, generator=g)
        x = (x - m) / s
        x = x.to(device)
        x = x.half() if half else x.float()
        with torch.no_grad():
            lo = model(x)
        outs.append(lo.float().cpu().numpy())
        done += b
        bi += 1
    return np.concatenate(outs, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="B_vit_gpu")
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--n-cal", type=int, default=4000)
    ap.add_argument("--n-test", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=20240601)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    import torch
    from torchvision.models import vit_b_16, ViT_B_16_Weights

    weights = ViT_B_16_Weights.IMAGENET1K_V1
    wtag = "ViT_B_16_Weights.IMAGENET1K_V1"
    print("WEIGHTS tag = " + wtag, flush=True)

    dev = args.device
    if dev == "cuda" and not torch.cuda.is_available():
        print("ERROR cuda requested but not available", flush=True)
        sys.exit(3)
    gpu_name = torch.cuda.get_device_name(0) if dev == "cuda" else None
    print("DEVICE = {} name = {}".format(dev, gpu_name), flush=True)
    print("DEVICE torch = {} cuda = {}".format(torch.__version__, torch.version.cuda), flush=True)
    tensor_cores = False
    print("NOTE tensor_cores = {} (fp16 GEMM runs on standard CUDA cores on GTX 16-series)".format(tensor_cores), flush=True)

    torch.manual_seed(args.seed)
    model = vit_b_16(weights=weights)
    model.eval()
    device = torch.device(dev)
    model = model.to(device)

    cal_seed = args.seed + 1000
    test_seed = args.seed + 5000

    print("COMPUTE fp32 cal ...", flush=True)
    lc32 = run_logits(model, args.n_cal, cal_seed, args.batch, device, False)
    print("COMPUTE fp32 test ...", flush=True)
    lt32 = run_logits(model, args.n_test, test_seed, args.batch, device, False)

    model16 = model.half()
    print("COMPUTE fp16 cal ...", flush=True)
    lc16 = run_logits(model16, args.n_cal, cal_seed, args.batch, device, True)
    print("COMPUTE fp16 test ...", flush=True)
    lt16 = run_logits(model16, args.n_test, test_seed, args.batch, device, True)

    y_cal = lc32.argmax(1).astype(np.int64)
    y_te = lt32.argmax(1).astype(np.int64)

    variants = {"fp32": (lc32, lt32), "fp16": (lc16, lt16)}
    save = {}
    out = {"label": args.label, "device": dev, "weights_tag": wtag, "gpu_name": gpu_name,
           "tensor_cores": tensor_cores, "seed": args.seed, "n_cal": int(args.n_cal),
           "n_test": int(args.n_test), "batch": int(args.batch),
           "torch_version": torch.__version__, "torch_cuda": torch.version.cuda,
           "variants": {}}
    for name in ["fp32", "fp16"]:
        lc, lt = variants[name]
        sc = score_vec(lc, y_cal)
        st = score_vec(lt, y_te)
        am = lt.argmax(1).astype(np.int64)
        acc = float((lt.argmax(1) == y_te).mean())
        save["scores_cal__" + name] = sc.astype(np.float64)
        save["scores_te__" + name] = st.astype(np.float64)
        save["argmax_te__" + name] = am
        out["variants"][name] = {"scores_cal_mean": float(sc.mean()),
                                 "scores_te_mean": float(st.mean()),
                                 "test_self_agree": acc}
        print("VARIANT {} score_te_mean={:.8f} self_agree={:.4f}".format(name, float(st.mean()), acc), flush=True)

    env = {"logit_te_fp16_vs_fp32": env_stats(lt16, lt32),
           "score_te_fp16_vs_fp32": env_stats(score_vec(lt16, y_te), score_vec(lt32, y_te))}
    out["envelope"] = env
    print("ENVELOPE logit_te fp16_vs_fp32: eps_max={:.3e} eps_99={:.3e} eps_mean={:.3e}".format(
        env["logit_te_fp16_vs_fp32"]["eps_max"], env["logit_te_fp16_vs_fp32"]["eps_99"], env["logit_te_fp16_vs_fp32"]["eps_mean"]), flush=True)
    print("ENVELOPE score_te fp16_vs_fp32: eps_max={:.3e} eps_99={:.3e} eps_mean={:.3e}".format(
        env["score_te_fp16_vs_fp32"]["eps_max"], env["score_te_fp16_vs_fp32"]["eps_99"], env["score_te_fp16_vs_fp32"]["eps_mean"]), flush=True)

    fp = hashlib.sha256()
    fp.update(wtag.encode())
    fp.update(np.array([args.seed, args.n_cal, args.n_test, args.batch]).tobytes())
    fp.update(y_cal.tobytes())
    fp.update(y_te.tobytes())
    art_sha = fp.hexdigest()
    print("ARTIFACT fingerprint sha256 = " + art_sha, flush=True)

    save["__label"] = np.array([args.label])
    save["__device"] = np.array([dev])
    save["__artifact_sha256"] = np.array([art_sha])
    outpath = args.out or os.path.join(HERE, "scores_" + args.label + ".npz")
    np.savez(outpath, **save)
    out["artifact_sha256"] = art_sha
    sumpath = os.path.join(HERE, "scores_" + args.label + "_summary.json")
    json.dump(out, open(sumpath, "w"), indent=2)
    print("WROTE " + outpath, flush=True)
    print("WROTE " + sumpath, flush=True)
    print("PASTE_BACK_BEGIN", flush=True)
    print(json.dumps(out, indent=2), flush=True)
    print("PASTE_BACK_END", flush=True)
    print("DONE label=" + args.label, flush=True)


if __name__ == "__main__":
    main()
