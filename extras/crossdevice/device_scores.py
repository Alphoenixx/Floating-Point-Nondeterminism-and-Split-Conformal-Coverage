import os
import sys
import json
import hashlib
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def train_softmax(Phi, yy, K, iters=200, lr=0.4, l2=0.2):
    n, D = Phi.shape
    W = np.zeros((D, K))
    Y = np.eye(K)[yy]
    for _ in range(iters):
        P = softmax(Phi @ W)
        W -= lr * (Phi.T @ (P - Y) / n + l2 * W)
    return W


def make_features(X, D, d, seed=7):
    r = np.random.default_rng(seed)
    Rmat = r.normal(0, 1.0 / np.sqrt(d), size=(d, D))
    b = r.normal(0, 0.1, size=(D,))
    Phi = np.maximum(0.0, X @ Rmat + b)
    return (Phi / (Phi.std(0) + 1e-8)).astype(np.float64)


def conformal_qhat(cal, alpha):
    n = len(cal)
    k = min(int(np.ceil((n + 1) * (1 - alpha))), n)
    return np.sort(cal)[k - 1]


def score_vec(logits, labels):
    P = softmax(logits)
    return 1.0 - P[np.arange(len(labels)), labels]


def build_artifact(path):
    RNG = np.random.default_rng(1234)
    K, d = 10, 100
    N_train, N_cal, N_test = 4000, 8000, 10000
    N = N_train + N_cal + N_test
    CENTER_SCALE = 0.55
    centers = RNG.normal(0, 1.0, size=(K, d)) * CENTER_SCALE
    y = RNG.integers(0, K, size=N)
    X = centers[y] + RNG.normal(0, 1.0, size=(N, d))
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    D_main = 512
    Phi = make_features(X, D_main, d)
    Phi_tr = Phi[:N_train]
    Phi_cal = Phi[N_train:N_train + N_cal]
    Phi_te = Phi[N_train + N_cal:]
    y_tr = y[:N_train]
    y_cal = y[N_train:N_train + N_cal]
    y_te = y[N_train + N_cal:]
    W = train_softmax(Phi_tr, y_tr, K)
    np.savez(path,
             Phi_cal=Phi_cal.astype(np.float32),
             Phi_te=Phi_te.astype(np.float32),
             W=W.astype(np.float32),
             y_cal=y_cal.astype(np.int64),
             y_te=y_te.astype(np.int64))
    return path


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def matmul_cpu(Phi32, W32):
    return (Phi32 @ W32).astype(np.float64)


def matmul_cuda(Phi32, W32, half):
    import torch
    dev = torch.device("cuda")
    t = torch.from_numpy(Phi32).to(dev)
    w = torch.from_numpy(W32).to(dev)
    if half:
        out = (t.half() @ w.half()).float()
    else:
        out = t @ w
    torch.cuda.synchronize()
    return out.cpu().numpy().astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--artifact", default=os.path.join(HERE, "frozen_artifact.npz"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    if args.build:
        print("BUILD building frozen artifact ...", flush=True)
        build_artifact(args.artifact)
        print("BUILD wrote " + args.artifact, flush=True)

    if not os.path.exists(args.artifact):
        print("ERROR artifact not found: " + args.artifact, flush=True)
        sys.exit(2)

    art_sha = sha256_file(args.artifact)
    print("ARTIFACT path   = " + args.artifact, flush=True)
    print("ARTIFACT sha256 = " + art_sha, flush=True)

    z = np.load(args.artifact)
    Phi_cal = z["Phi_cal"]
    Phi_te = z["Phi_te"]
    W = z["W"]
    y_cal = z["y_cal"]
    y_te = z["y_te"]
    print("LOADED Phi_cal{} Phi_te{} W{}".format(Phi_cal.shape, Phi_te.shape, W.shape), flush=True)

    ENV = {}
    env_lines = []
    try:
        sys.path.insert(0, os.path.join(REPO, "src"))
        import environment
        ENV = environment.collect()
        env_lines = environment.summary_lines(ENV)
    except Exception as e:
        ENV = {"note": "environment.py not importable: " + repr(e)}
    for ln in env_lines:
        print("ENV " + ln, flush=True)

    variants = {}
    if args.device == "cuda":
        import torch
        print("DEVICE cuda_available = {}".format(torch.cuda.is_available()), flush=True)
        if not torch.cuda.is_available():
            print("ERROR cuda requested but not available", flush=True)
            sys.exit(3)
        print("DEVICE name = {}".format(torch.cuda.get_device_name(0)), flush=True)
        print("DEVICE torch = {} cuda = {}".format(torch.__version__, torch.version.cuda), flush=True)
        ENV["torch_version"] = torch.__version__
        ENV["torch_cuda"] = torch.version.cuda
        ENV["gpu_name"] = torch.cuda.get_device_name(0)
        print("COMPUTE cuda fp32 cal ...", flush=True)
        lc32 = matmul_cuda(Phi_cal, W, False)
        print("COMPUTE cuda fp32 test ...", flush=True)
        lt32 = matmul_cuda(Phi_te, W, False)
        print("COMPUTE cuda fp16 cal ...", flush=True)
        lc16 = matmul_cuda(Phi_cal, W, True)
        print("COMPUTE cuda fp16 test ...", flush=True)
        lt16 = matmul_cuda(Phi_te, W, True)
        variants["fp32"] = (lc32, lt32)
        variants["fp16"] = (lc16, lt16)
    else:
        print("COMPUTE cpu fp32 cal ...", flush=True)
        lc32 = matmul_cpu(Phi_cal, W)
        print("COMPUTE cpu fp32 test ...", flush=True)
        lt32 = matmul_cpu(Phi_te, W)
        variants["fp32"] = (lc32, lt32)

    out = {"label": args.label, "device": args.device, "artifact_sha256": art_sha,
           "n_cal": int(Phi_cal.shape[0]), "n_te": int(Phi_te.shape[0]),
           "environment": ENV, "variants": {}}
    save = {}
    for name in variants:
        lc, lt = variants[name]
        sc = score_vec(lc, y_cal)
        st = score_vec(lt, y_te)
        am = lt.argmax(1).astype(np.int64)
        acc = float((lt.argmax(1) == y_te).mean())
        out["variants"][name] = {"scores_cal_mean": float(sc.mean()),
                                 "scores_te_mean": float(st.mean()),
                                 "test_accuracy": acc}
        save["scores_cal__" + name] = sc.astype(np.float64)
        save["scores_te__" + name] = st.astype(np.float64)
        save["argmax_te__" + name] = am
        print("VARIANT {} acc={:.4f} score_te_mean={:.8f}".format(name, acc, float(st.mean())), flush=True)

    outpath = args.out or os.path.join(HERE, "scores_" + args.label + ".npz")
    save["__label"] = np.array([args.label])
    save["__device"] = np.array([args.device])
    save["__artifact_sha256"] = np.array([art_sha])
    np.savez(outpath, **save)
    sumpath = os.path.join(HERE, "scores_" + args.label + "_summary.json")
    json.dump(out, open(sumpath, "w"), indent=2)
    print("WROTE " + outpath, flush=True)
    print("WROTE " + sumpath, flush=True)
    print("DONE label=" + args.label + " device=" + args.device, flush=True)


if __name__ == "__main__":
    main()
