import os
os.environ["OMP_NUM_THREADS"] = "1"
import json
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)

N_SEEDS = 40
ALPHAS = [0.05, 0.10, 0.20]

def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)

def scores(L, y):
    P = softmax(L)
    return 1.0 - P[np.arange(len(y)), y]

def qhat(c, a):
    n = len(c)
    k = min(int(np.ceil((n + 1) * (1 - a))), n)
    return np.sort(c)[k - 1]

def train(Phi, yy, K, it=300, lr=0.5, l2=0.05):
    n, D = Phi.shape
    W = np.zeros((D, K))
    Y = np.eye(K)[yy]
    for _ in range(it):
        P = softmax(Phi @ W)
        W -= lr * (Phi.T @ (P - Y) / n + l2 * W)
    return W

def agg(vals):
    a = np.asarray(vals, dtype=np.float64)
    m = float(a.mean())
    sd = float(a.std(ddof=1)) if len(a) > 1 else 0.0
    se = sd / np.sqrt(len(a))
    return {"mean": m, "sd": sd, "se": se, "ci95_lo": m - 1.96 * se, "ci95_hi": m + 1.96 * se}

def int8_ptq(P, W, mode):
    sw = np.abs(W).max(0) / 127.0
    sw[sw == 0] = 1.0
    sx = np.abs(P).max(1) / 127.0
    sx[sx == 0] = 1.0
    if mode == "round":
        Wq = np.clip(np.round(W / sw), -127, 127)
        Xq = np.clip(np.round(P / sx[:, None]), -127, 127)
    else:
        Wq = np.clip(np.trunc(W / sw), -127, 127)
        Xq = np.clip(np.trunc(P / sx[:, None]), -127, 127)
    acc = Xq.astype(np.int64) @ Wq.astype(np.int64)
    return acc.astype(np.float64) * sx[:, None] * sw[None, :]

def build(cs):
    g = np.random.default_rng(20240517)
    K, d, D = 10, 100, 512
    Ntr, n_cal, n_test = 8000, 20000, 20000
    cen = g.normal(0, 1.0, size=(K, d)) * cs
    Ntot = Ntr + n_cal + n_test
    y = g.integers(0, K, size=Ntot)
    X = cen[y] + g.normal(0, 1.0, size=(Ntot, d))
    mu, sd = X[:Ntr].mean(0), X[:Ntr].std(0) + 1e-8
    X = (X - mu) / sd
    rr = np.random.default_rng(7)
    Rm = rr.normal(0, 1.0 / np.sqrt(d), size=(d, D))
    bb = rr.normal(0, 0.1, size=(D,))
    Ph = np.maximum(0.0, X @ Rm + bb)
    pm, ps = Ph[:Ntr].mean(0), Ph[:Ntr].std(0) + 1e-8
    Ph = (Ph - pm) / ps
    W = train(Ph[:Ntr], y[:Ntr], K)
    return Ph[Ntr:], y[Ntr:], W, n_cal, n_test

def envelope(sd, sref):
    d = sd - sref
    return {"eps_max": float(np.abs(d).max()), "eps_mean": float(np.abs(d).mean()),
            "signed_mean": float(d.mean()), "eps_99": float(np.quantile(np.abs(d), 0.99))}

def run_regime(cs):
    Pp, yp, W, n_cal, n_test = build(cs)
    Lref = Pp.astype(np.float64) @ W.astype(np.float64)
    mtp = float(softmax(Lref).max(1).mean())
    acc = float((Lref.argmax(1) == yp).mean())
    sref = scores(Lref, yp)
    pool = len(yp)
    kernels = {"int8_round": int8_ptq(Pp, W, "round"), "int8_trunc": int8_ptq(Pp, W, "trunc")}
    out = {"center_scale": cs, "accuracy": acc, "mean_top_softmax": mtp,
           "n_cal": n_cal, "n_test": n_test, "kernels": {}}
    for name, Lk in kernels.items():
        sk = scores(Lk, yp)
        env = envelope(sk, sref)
        quant_ok = bool(env["eps_max"] > 1e-6)
        flips = float((Lk.argmax(1) != Lref.argmax(1)).mean())
        per_alpha = {}
        for a in ALPHAS:
            loss, churn = [], []
            for s in range(N_SEEDS):
                rng = np.random.default_rng(60000 + s)
                idx = rng.permutation(pool)
                cal, te = idx[:n_cal], idx[n_cal:n_cal + n_test]
                qh = qhat(sref[cal], a)
                cr = np.mean(sref[te] <= qh)
                cd = np.mean(sk[te] <= qh)
                loss.append(cr - cd)
                churn.append(float(np.mean((sref[te] <= qh) != (sk[te] <= qh))))
            floor = float(np.sqrt(a * (1 - a) / n_test))
            per_alpha[str(a)] = {"sampling_floor": floor, "cov_loss": agg(loss),
                                 "loss_over_floor": agg(loss)["mean"] / floor,
                                 "churn": agg(churn), "churn_over_floor": agg(churn)["mean"] / floor}
        out["kernels"][name] = {"quant_ok": quant_ok, "envelope": env,
                                "top1_flips": flips, "per_alpha": per_alpha}
    return out

results = {"n_seeds": N_SEEDS, "alphas": ALPHAS, "regimes": {}}
for tag, cs in [("saturated", 1.0), ("moderate", 0.40)]:
    results["regimes"][tag] = run_regime(cs)
json.dump(results, open(os.path.join(OUT, "results_int8_ptq.json"), "w"), indent=2)
for tag in results["regimes"]:
    for name, k in results["regimes"][tag]["kernels"].items():
        print("QUANT_OK {} {}: {} eps_max={:.3e}".format(tag, name, k["quant_ok"], k["envelope"]["eps_max"]))
print("PASTE_BACK_BEGIN")
print(json.dumps(results, indent=2))
print("PASTE_BACK_END")
for tag in results["regimes"]:
    r = results["regimes"][tag]
    for name, k in r["kernels"].items():
        print("DONE {} {}: mtp={:.3f} eps_max={:.3e} signed={:+.3e} loss/floor(a=.1)={:+.2f} churn/floor(a=.1)={:.2f}".format(
            tag, name, r["mean_top_softmax"], k["envelope"]["eps_max"], k["envelope"]["signed_mean"],
            k["per_alpha"]["0.1"]["loss_over_floor"], k["per_alpha"]["0.1"]["churn_over_floor"]))
