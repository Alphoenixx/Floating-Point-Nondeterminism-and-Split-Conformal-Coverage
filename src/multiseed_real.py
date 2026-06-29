import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
import json
import csv
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)
CSV = os.path.join(HERE, "data", "gapminder.csv")

N_SEEDS = 50
ALPHAS = [0.05, 0.10, 0.20]
n_tr, n_cal = 450, 600
D = 256

def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)

def train(Phi, yy, K, iters=400, lr=0.3, l2=0.05):
    n, Dd = Phi.shape
    W = np.zeros((Dd, K))
    Y = np.eye(K)[yy]
    for _ in range(iters):
        Pr = softmax(Phi @ W)
        W -= lr * (Phi.T @ (Pr - Y) / n + l2 * W)
    return W

def qhat_of(cal, a):
    n = len(cal)
    k = min(int(np.ceil((n + 1) * (1 - a))), n)
    return np.sort(cal)[k - 1]

def scores(logits, lab, T):
    Pr = softmax(logits / T)
    return 1.0 - Pr[np.arange(len(lab)), lab]

def fp16_chunked(Phi, W, K, chunk=16, order_seed=None):
    n, Dd = Phi.shape
    acc = np.zeros((n, K), np.float16)
    bnds = [(i, min(i + chunk, Dd)) for i in range(0, Dd, chunk)]
    if order_seed is not None:
        np.random.default_rng(order_seed).shuffle(bnds)
    P64, W64 = Phi.astype(np.float64), W.astype(np.float64)
    for a, b in bnds:
        partial = (P64[:, a:b] @ W64[a:b, :]).astype(np.float32)
        acc = (acc.astype(np.float32) + partial).astype(np.float16)
    return acc.astype(np.float64)

def sup_density(s, lo, hi, nbins=40):
    cnt, ed = np.histogram(s, bins=nbins, range=(s.min(), s.max()))
    w = ed[1] - ed[0]
    dens = cnt / (len(s) * w)
    m = (ed[:-1] <= hi + 1e-12) & (ed[1:] >= lo - 1e-12)
    return float(dens[m].max()) if m.any() else float(dens.max())

def agg(vals):
    a = np.asarray(vals, dtype=np.float64)
    m = float(a.mean())
    sd = float(a.std(ddof=1)) if len(a) > 1 else 0.0
    se = sd / np.sqrt(len(a))
    return {"mean": m, "sd": sd, "se": se,
            "ci95_lo": m - 1.96 * se, "ci95_hi": m + 1.96 * se,
            "min": float(a.min()), "max": float(a.max()), "n": int(len(a))}

rows = list(csv.DictReader(open(CSV)))
conts = ["Africa", "Asia", "Europe", "Americas"]
ci = {c: i for i, c in enumerate(conts)}
Xl, yl = [], []
for r in rows:
    if r["continent"] not in ci:
        continue
    Xl.append([float(r["year"]), float(r["lifeExp"]),
               np.log10(float(r["pop"])), np.log10(float(r["gdpPercap"]))])
    yl.append(ci[r["continent"]])
X = np.array(Xl, float)
y = np.array(yl, int)
X = (X - X.mean(0)) / (X.std(0) + 1e-8)
K = len(conts)
d = X.shape[1]
N = len(X)
n_test = N - n_tr - n_cal

rf = np.random.default_rng(7)
Rmat = rf.normal(0, 1.0, size=(d, D))
bvec = rf.normal(0, 0.5, size=(D,))

def feats(A):
    P = np.maximum(0.0, A @ Rmat + bvec)
    return P / (P.std(0) + 1e-8)

Ts = np.linspace(0.5, 12, 116)
acc_l, eps_max_l, eps_mean_l, eps_99_l, flip_l, temp_l = [], [], [], [], [], []
per = {a: {"gap": [], "churn": [], "cert": [], "within": []} for a in ALPHAS}

for s in range(N_SEEDS):
    rng = np.random.default_rng(30000 + s)
    perm = rng.permutation(N)
    Xs, ys = X[perm], y[perm]
    Xtr, Xcal, Xte = Xs[:n_tr], Xs[n_tr:n_tr + n_cal], Xs[n_tr + n_cal:]
    ytr, ycal, yte = ys[:n_tr], ys[n_tr:n_tr + n_cal], ys[n_tr + n_cal:]
    Ptr, Pcal, Pte = feats(Xtr), feats(Xcal), feats(Xte)
    W = train(Ptr, ytr, K)
    LA_cal = fp16_chunked(Pcal, W, K)
    nll = [(-np.mean(np.log(softmax(LA_cal / t)[np.arange(len(ycal)), ycal] + 1e-12))) for t in Ts]
    T = float(Ts[int(np.argmin(nll))])
    temp_l.append(T)
    sA_cal = scores(fp16_chunked(Pcal, W, K), ycal, T)
    LA = fp16_chunked(Pte, W, K, order_seed=None)
    LB = fp16_chunked(Pte, W, K, order_seed=7)
    sA_te = scores(LA, yte, T)
    sB_te = scores(LB, yte, T)
    jit = np.abs(sA_te - sB_te)
    eps_max = float(jit.max())
    eps_max_l.append(eps_max)
    eps_mean_l.append(float(jit.mean()))
    eps_99_l.append(float(np.percentile(jit, 99)))
    acc_l.append(float((LA.argmax(1) == yte).mean()))
    flip_l.append(float((LA.argmax(1) != LB.argmax(1)).mean()))
    for a in ALPHAS:
        qh = qhat_of(sA_cal, a)
        cA = float(np.mean(sA_te <= qh))
        cB = float(np.mean(sB_te <= qh))
        gap = abs(cA - cB)
        churn = float(np.mean((sA_te <= qh) != (sB_te <= qh)))
        fsup = sup_density(sA_cal, qh - eps_max, qh + eps_max)
        cert = fsup * eps_max
        per[a]["gap"].append(gap)
        per[a]["churn"].append(churn)
        per[a]["cert"].append(cert)
        per[a]["within"].append(1.0 if gap <= cert + 1e-12 else 0.0)

results = {"dataset": "Gapminder (4 continents)", "n_seeds": N_SEEDS, "N": int(N),
           "n_train": n_tr, "n_cal": n_cal, "n_test": int(n_test), "K": K, "D": D,
           "accuracy": agg(acc_l), "temperature": agg(temp_l),
           "eps_max": agg(eps_max_l), "eps_mean": agg(eps_mean_l), "eps_99": agg(eps_99_l),
           "top1_flip_rate_max": float(np.max(flip_l)), "per_alpha": {}}
for a in ALPHAS:
    floor = float(np.sqrt(a * (1 - a) / n_test))
    results["per_alpha"][str(a)] = {
        "sampling_floor": floor,
        "abs_cov_gap": agg(per[a]["gap"]),
        "gap_over_floor": agg(per[a]["gap"])["mean"] / floor,
        "churn": agg(per[a]["churn"]),
        "certificate": agg(per[a]["cert"]),
        "frac_seeds_within_cert": float(np.mean(per[a]["within"]))}

json.dump(results, open(os.path.join(OUT, "results_multiseed_real.json"), "w"), indent=2)
print("PASTE_BACK_BEGIN")
print(json.dumps(results, indent=2))
print("PASTE_BACK_END")
print("DONE real-multiseed: seeds={} N={} n_test={} eps_max_mean={:.4e} acc_mean={:.4f}".format(
    N_SEEDS, N, n_test, results["eps_max"]["mean"], results["accuracy"]["mean"]))
