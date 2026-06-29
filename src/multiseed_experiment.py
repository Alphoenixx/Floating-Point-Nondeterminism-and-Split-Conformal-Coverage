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

N_SEEDS = 50
ALPHAS = [0.05, 0.10, 0.20]

def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)

def train_softmax(Phi, yy, K, iters=300, lr=0.5, l2=0.05):
    n, D = Phi.shape
    W = np.zeros((D, K))
    Y = np.eye(K)[yy]
    for _ in range(iters):
        P = softmax(Phi @ W)
        W -= lr * (Phi.T @ (P - Y) / n + l2 * W)
    return W

def conformal_qhat(cal, alpha):
    n = len(cal)
    k = min(int(np.ceil((n + 1) * (1 - alpha))), n)
    return np.sort(cal)[k - 1]

def scores(logits, labels):
    P = softmax(logits)
    return 1.0 - P[np.arange(len(labels)), labels]

def sup_density(s, lo, hi, nbins=50):
    counts, edges = np.histogram(s, bins=nbins, range=(s.min(), s.max()))
    w = edges[1] - edges[0]
    dens = counts / (len(s) * w)
    m = (edges[:-1] <= hi + 1e-12) & (edges[1:] >= lo - 1e-12)
    return float(dens[m].max()) if m.any() else float(dens.max())

def logits_fp16_chunked(Phi, W, chunk=64, order_seed=None):
    n, D = Phi.shape
    Kk = W.shape[1]
    bnds = [(i, min(i + chunk, D)) for i in range(0, D, chunk)]
    if order_seed is not None:
        np.random.default_rng(order_seed).shuffle(bnds)
    acc = np.zeros((n, Kk), dtype=np.float16)
    P64, W64 = Phi.astype(np.float64), W.astype(np.float64)
    for a, b in bnds:
        partial = (P64[:, a:b] @ W64[a:b, :]).astype(np.float32)
        acc = (acc.astype(np.float32) + partial).astype(np.float16)
    return acc.astype(np.float64)

def quant_logits(Phi, W, chunk=64, mbits=None, mode="round", order_seed=None):
    n, D = Phi.shape
    Kk = W.shape[1]
    bnds = [(i, min(i + chunk, D)) for i in range(0, D, chunk)]
    if order_seed is not None:
        np.random.default_rng(order_seed).shuffle(bnds)
    P64, W64 = Phi.astype(np.float64), W.astype(np.float64)
    acc = np.zeros((n, Kk), dtype=np.float64)
    if mbits is None:
        for a, b in bnds:
            acc = acc + P64[:, a:b] @ W64[a:b, :]
        return acc
    def q(x):
        x = np.asarray(x, dtype=np.float64)
        out = np.zeros_like(x)
        nz = x != 0
        ax = np.abs(x[nz])
        e = np.floor(np.log2(ax))
        scale = 2.0 ** (e - mbits)
        r = ax / scale
        qq = np.floor(r) if mode == "trunc" else np.round(r)
        out[nz] = np.sign(x[nz]) * qq * scale
        return out
    for a, b in bnds:
        acc = q(acc + q(P64[:, a:b] @ W64[a:b, :]))
    return acc

def agg(vals):
    a = np.asarray(vals, dtype=np.float64)
    m = float(a.mean())
    sd = float(a.std(ddof=1)) if len(a) > 1 else 0.0
    se = sd / np.sqrt(len(a))
    return {"mean": m, "sd": sd, "se": se,
            "ci95_lo": m - 1.96 * se, "ci95_hi": m + 1.96 * se,
            "frac_pos": float(np.mean(a > 0)), "n": int(len(a))}

def build_main():
    g = np.random.default_rng(1234)
    K, d, D = 10, 100, 512
    Ntr, n_cal, n_test = 4000, 8000, 10000
    Npool = n_cal + n_test
    centers = g.normal(0, 1.0, size=(K, d)) * 0.55
    Ntot = Ntr + Npool
    y = g.integers(0, K, size=Ntot)
    X = centers[y] + g.normal(0, 1.0, size=(Ntot, d))
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    rr = np.random.default_rng(7)
    Rmat = rr.normal(0, 1.0 / np.sqrt(d), size=(d, D))
    b = rr.normal(0, 0.1, size=(D,))
    Phi = np.maximum(0.0, X @ Rmat + b)
    Phi = Phi / (Phi.std(0) + 1e-8)
    W = train_softmax(Phi[:Ntr], y[:Ntr], K, iters=200, lr=0.4, l2=0.2)
    Pp, yp = Phi[Ntr:], y[Ntr:]
    sA = scores(logits_fp16_chunked(Pp, W, order_seed=None), yp)
    sB = scores(logits_fp16_chunked(Pp, W, order_seed=7), yp)
    eps = np.abs(sA - sB)
    return {"sref": sA, "sdep": sB, "eps_max": float(eps.max()),
            "n_cal": n_cal, "n_test": n_test, "pool": len(yp)}

def build_stress():
    g = np.random.default_rng(20240517)
    K, d, D = 10, 100, 512
    Ntr, n_cal, n_test = 8000, 40000, 40000
    Npool = n_cal + n_test
    centers = g.normal(0, 1.0, size=(K, d)) * 1.0
    Ntot = Ntr + Npool
    y = g.integers(0, K, size=Ntot)
    X = centers[y] + g.normal(0, 1.0, size=(Ntot, d))
    mu, sd = X[:Ntr].mean(0), X[:Ntr].std(0) + 1e-8
    X = (X - mu) / sd
    rr = np.random.default_rng(7)
    Rmat = rr.normal(0, 1.0 / np.sqrt(d), size=(d, D))
    b = rr.normal(0, 0.1, size=(D,))
    Ph = np.maximum(0.0, X @ Rmat + b)
    pmu, psd = Ph[:Ntr].mean(0), Ph[:Ntr].std(0) + 1e-8
    Ph = (Ph - pmu) / psd
    W = train_softmax(Ph[:Ntr], y[:Ntr], K, iters=300, lr=0.5, l2=0.05)
    Pp, yp = Ph[Ntr:], y[Ntr:]
    sref = scores(quant_logits(Pp, W), yp)
    regimes = {}
    for name, kw in [("fp16_round", dict(mbits=10, mode="round", order_seed=7)),
                     ("fp8_trunc", dict(mbits=7, mode="trunc", order_seed=7))]:
        dep = quant_logits(Pp, W, **kw)
        sdep = scores(dep, yp)
        regimes[name] = {"sdep": sdep, "Sall": 1.0 - softmax(dep),
                         "eps_max": float(np.abs(sdep - sref).max())}
    return {"sref": sref, "regimes": regimes, "n_cal": n_cal, "n_test": n_test, "pool": len(yp)}

def run_abs_gap(block):
    sref, sdep = block["sref"], block["sdep"]
    n_cal, n_test, pool = block["n_cal"], block["n_test"], block["pool"]
    per = {a: [] for a in ALPHAS}
    cov_calib = {a: [] for a in ALPHAS}
    cov_deploy = {a: [] for a in ALPHAS}
    for s in range(N_SEEDS):
        rng = np.random.default_rng(10000 + s)
        idx = rng.permutation(pool)
        cal, te = idx[:n_cal], idx[n_cal:n_cal + n_test]
        for a in ALPHAS:
            qh = conformal_qhat(sref[cal], a)
            covA = float(np.mean(sref[te] <= qh))
            covB = float(np.mean(sdep[te] <= qh))
            cov_calib[a].append(covA)
            cov_deploy[a].append(covB)
            per[a].append(abs(covA - covB))
    return {str(a): {"abs_cov_gap": agg(per[a]),
                     "cov_calib": agg(cov_calib[a]),
                     "cov_deploy": agg(cov_deploy[a]),
                     "sampling_floor": float(np.sqrt(a * (1 - a) / n_test))} for a in ALPHAS}

def run_stress(block):
    sref = block["sref"]
    n_cal, n_test, pool = block["n_cal"], block["n_test"], block["pool"]
    out = {}
    for name, rg in block["regimes"].items():
        sdep, Sall, eps_max = rg["sdep"], rg["Sall"], rg["eps_max"]
        acc = {a: {"loss": [], "cov_ref": [], "cov_dep": [], "cov_robust": [],
                   "set_naive": [], "set_robust": []} for a in ALPHAS}
        for s in range(N_SEEDS):
            rng = np.random.default_rng(20000 + s)
            idx = rng.permutation(pool)
            cal, te = idx[:n_cal], idx[n_cal:n_cal + n_test]
            for a in ALPHAS:
                qh = conformal_qhat(sref[cal], a)
                cov_ref = float(np.mean(sref[te] <= qh))
                cov_dep = float(np.mean(sdep[te] <= qh))
                acc[a]["cov_ref"].append(cov_ref)
                acc[a]["cov_dep"].append(cov_dep)
                acc[a]["loss"].append(cov_ref - cov_dep)
                acc[a]["cov_robust"].append(float(np.mean(sdep[te] <= qh + eps_max)))
                acc[a]["set_naive"].append(float(np.mean((Sall[te] <= qh).sum(1))))
                acc[a]["set_robust"].append(float(np.mean((Sall[te] <= qh + eps_max).sum(1))))
        rec = {"eps_max": eps_max, "per_alpha": {}}
        for a in ALPHAS:
            floor = float(np.sqrt(a * (1 - a) / n_test))
            la = agg(acc[a]["loss"])
            tstat = la["mean"] / la["se"] if la["se"] > 0 else float("inf")
            rec["per_alpha"][str(a)] = {
                "sampling_floor": floor,
                "cov_ref": agg(acc[a]["cov_ref"]),
                "cov_dep": agg(acc[a]["cov_dep"]),
                "cov_loss": la,
                "loss_over_floor": la["mean"] / floor,
                "t_stat_from_zero": tstat,
                "indistinguishable_from_zero": bool(abs(la["mean"]) <= 1.96 * la["se"]),
                "cov_robust": agg(acc[a]["cov_robust"]),
                "set_naive": agg(acc[a]["set_naive"]),
                "set_robust": agg(acc[a]["set_robust"]),
                "set_increase": agg([r - n for r, n in zip(acc[a]["set_robust"], acc[a]["set_naive"])])}
        out[name] = rec
    return out

main_block = build_main()
stress_block = build_stress()
results = {"n_seeds": N_SEEDS, "alphas": ALPHAS,
           "main_fp16_reorder": {"eps_max": main_block["eps_max"],
                                 "n_cal": main_block["n_cal"], "n_test": main_block["n_test"],
                                 "per_alpha": run_abs_gap(main_block)},
           "stress": {"n_cal": stress_block["n_cal"], "n_test": stress_block["n_test"],
                      "regimes": run_stress(stress_block)}}

json.dump(results, open(os.path.join(OUT, "results_multiseed.json"), "w"), indent=2)
print("PASTE_BACK_BEGIN")
print(json.dumps(results, indent=2))
print("PASTE_BACK_END")
print("DONE multiseed: seeds={} main_eps_max={:.4e} fp16_eps={:.4e} fp8_eps={:.4e}".format(
    N_SEEDS, main_block["eps_max"],
    stress_block["regimes"]["fp16_round"]["eps_max"],
    stress_block["regimes"]["fp8_trunc"]["eps_max"]))
