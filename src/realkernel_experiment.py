import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
import json
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)

N_SEEDS = 30
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

def k_f64(P, W):
    return P.astype(np.float64) @ W.astype(np.float64)

def k_f32(P, W):
    return (P.astype(np.float32) @ W.astype(np.float32)).astype(np.float64)

def k_f16_native(P, W):
    return (P.astype(np.float16) @ W.astype(np.float16)).astype(np.float64)

def k_f16_blocked(P, W, nb=8):
    A = P.astype(np.float16)
    B = W.astype(np.float16)
    Dd = A.shape[1]
    step = Dd // nb
    bnds = [(i, min(i + step, Dd)) for i in range(0, Dd, step)]
    acc = np.zeros((A.shape[0], B.shape[1]), np.float16)
    for a, b in bnds:
        acc = (acc + (A[:, a:b] @ B[a:b, :]).astype(np.float16)).astype(np.float16)
    return acc.astype(np.float64)

def agg(vals):
    a = np.asarray(vals, dtype=np.float64)
    m = float(a.mean())
    sd = float(a.std(ddof=1)) if len(a) > 1 else 0.0
    se = sd / np.sqrt(len(a))
    return {"mean": m, "sd": sd, "se": se,
            "ci95_lo": m - 1.96 * se, "ci95_hi": m + 1.96 * se, "n": int(len(a))}

def build(cs):
    g = np.random.default_rng(20240517)
    K, d, D = 10, 100, 512
    Ntr, n_cal, n_test = 8000, 20000, 20000
    Npool = n_cal + n_test
    cen = g.normal(0, 1.0, size=(K, d)) * cs
    Ntot = Ntr + Npool
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
    Pp, yp = Ph[Ntr:], y[Ntr:]
    Lref = k_f64(Pp, W)
    mtp = float(softmax(Lref).max(1).mean())
    acc = float((Lref.argmax(1) == yp).mean())
    kernels = {"f32": k_f32(Pp, W), "f16_native": k_f16_native(Pp, W),
               "f16_blocked": k_f16_blocked(Pp, W)}
    sref = scores(Lref, yp)
    sker = {name: scores(L, yp) for name, L in kernels.items()}
    flips = {name: float((L.argmax(1) != Lref.argmax(1)).mean()) for name, L in kernels.items()}
    eps = {name: {"eps_max": float(np.abs(sker[name] - sref).max()),
                  "eps_mean": float(np.abs(sker[name] - sref).mean()),
                  "signed_mean": float(np.mean(sker[name] - sref))} for name in kernels}
    return {"cs": cs, "acc": acc, "mtp": mtp, "n_cal": n_cal, "n_test": n_test,
            "pool": len(yp), "sref": sref, "sker": sker, "flips": flips, "eps": eps}

def scenarios(block):
    sref = block["sref"]
    sN = block["sker"]["f16_native"]
    sB = block["sker"]["f16_blocked"]
    s32 = block["sker"]["f32"]
    n_cal, n_test, pool = block["n_cal"], block["n_test"], block["pool"]
    defs = [("crossbackend_native_vs_blocked", sN, sN, sB),
            ("precision_f32cal_f16deploy", s32, s32, sN),
            ("f64ref_f16deploy", sref, sref, sN)]
    out = {}
    for name, scal_src, sref_te, sdep_te in defs:
        acc = {a: {"gap": [], "churn": [], "loss": []} for a in ALPHAS}
        for s in range(N_SEEDS):
            rng = np.random.default_rng(40000 + s)
            idx = rng.permutation(pool)
            cal, te = idx[:n_cal], idx[n_cal:n_cal + n_test]
            for a in ALPHAS:
                qh = qhat(scal_src[cal], a)
                cr = float(np.mean(sref_te[te] <= qh))
                cd = float(np.mean(sdep_te[te] <= qh))
                acc[a]["gap"].append(abs(cr - cd))
                acc[a]["loss"].append(cr - cd)
                acc[a]["churn"].append(float(np.mean((sref_te[te] <= qh) != (sdep_te[te] <= qh))))
        out[name] = {}
        for a in ALPHAS:
            floor = float(np.sqrt(a * (1 - a) / n_test))
            out[name][str(a)] = {"sampling_floor": floor,
                                 "abs_gap": agg(acc[a]["gap"]),
                                 "signed_loss": agg(acc[a]["loss"]),
                                 "gap_over_floor": agg(acc[a]["gap"])["mean"] / floor,
                                 "churn": agg(acc[a]["churn"]),
                                 "churn_over_floor": agg(acc[a]["churn"])["mean"] / floor}
    return out

results = {"n_seeds": N_SEEDS, "alphas": ALPHAS, "regimes": {}}
for tag, cs in [("saturated", 1.0), ("moderate", 0.40)]:
    b = build(cs)
    results["regimes"][tag] = {"center_scale": cs, "accuracy": b["acc"], "mean_top_softmax": b["mtp"],
                               "n_cal": b["n_cal"], "n_test": b["n_test"],
                               "kernel_envelope": b["eps"], "top1_flips": b["flips"],
                               "scenarios": scenarios(b)}

json.dump(results, open(os.path.join(OUT, "results_realkernel.json"), "w"), indent=2)
print("PASTE_BACK_BEGIN")
print(json.dumps(results, indent=2))
print("PASTE_BACK_END")
for tag in results["regimes"]:
    r = results["regimes"][tag]
    print("DONE {}: mtp={:.3f} f16_native eps_max={:.3e} blocked-vs-native gap/floor(a=.1)={:.3f}".format(
        tag, r["mean_top_softmax"], r["kernel_envelope"]["f16_native"]["eps_max"],
        r["scenarios"]["crossbackend_native_vs_blocked"]["0.1"]["gap_over_floor"]))
