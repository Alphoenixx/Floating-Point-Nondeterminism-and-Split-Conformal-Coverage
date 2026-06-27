import json
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)
CSV = os.path.join(HERE, "data", "gapminder.csv")

RNG = np.random.default_rng(1234)

rows = list(csv.DictReader(open(CSV)))

conts = ["Africa", "Asia", "Europe", "Americas"]
ci = {c: i for i, c in enumerate(conts)}
X, y = [], []
for r in rows:
    if r["continent"] not in ci:
        continue
    X.append([float(r["year"]), float(r["lifeExp"]),
              np.log10(float(r["pop"])), np.log10(float(r["gdpPercap"]))])
    y.append(ci[r["continent"]])
X = np.array(X, float)
y = np.array(y, int)
X = (X - X.mean(0)) / (X.std(0) + 1e-8)
K = len(conts)

perm0 = RNG.permutation(len(X))
X, y = X[perm0], y[perm0]
n_tr, n_cal = 450, 600
X_tr, X_cal, X_te = X[:n_tr], X[n_tr:n_tr + n_cal], X[n_tr + n_cal:]
y_tr, y_cal, y_te = y[:n_tr], y[n_tr:n_tr + n_cal], y[n_tr + n_cal:]

d = X.shape[1]
D = 256
rf = np.random.default_rng(7)
Rmat = rf.normal(0, 1.0, size=(d, D))
bvec = rf.normal(0, 0.5, size=(D,))


def feats(A):
    P = np.maximum(0.0, A @ Rmat + bvec)
    return (P / (P.std(0) + 1e-8))


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def train(Phi, yy, iters=400, lr=0.3, l2=0.05):
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


TEMP = 1.0


def scores(logits, lab):
    Pr = softmax(logits / TEMP)
    return 1.0 - Pr[np.arange(len(lab)), lab]


def fp16_chunked(Phi, W, chunk=16, order_seed=None):
    n, Dd = Phi.shape
    acc = np.zeros((n, K), np.float16)
    bnds = [(i, min(i + chunk, Dd)) for i in range(0, Dd, chunk)]
    if order_seed is not None:
        np.random.default_rng(order_seed).shuffle(bnds)
    Pm, Wm = Phi.astype(np.float32), W.astype(np.float32)
    for a, b in bnds:
        acc = (acc.astype(np.float32) + Pm[:, a:b] @ Wm[a:b, :]).astype(np.float16)
    return acc.astype(np.float64)


def sup_density(s, lo, hi, nbins=40):
    cnt, ed = np.histogram(s, bins=nbins, range=(s.min(), s.max()))
    w = ed[1] - ed[0]
    dens = cnt / (len(s) * w)
    m = (ed[:-1] <= hi + 1e-12) & (ed[1:] >= lo - 1e-12)
    return float(dens[m].max()) if m.any() else float(dens.max())


Phi_tr, Phi_cal, Phi_te = feats(X_tr), feats(X_cal), feats(X_te)
W = train(Phi_tr, y_tr)

LA_cal_raw = fp16_chunked(Phi_cal, W)


def _nll(T):
    P = softmax(LA_cal_raw / T)
    return -np.mean(np.log(P[np.arange(len(y_cal)), y_cal] + 1e-12))


_Ts = np.linspace(0.5, 12, 116)
TEMP = float(_Ts[int(np.argmin([_nll(t) for t in _Ts]))])

acc = float((fp16_chunked(Phi_te, W).argmax(1) == y_te).mean())
mtp = float(softmax(fp16_chunked(Phi_te, W) / TEMP).max(1).mean())

sA_cal = scores(fp16_chunked(Phi_cal, W), y_cal)
sA_te = scores(fp16_chunked(Phi_te, W, order_seed=None), y_te)
sB_te = scores(fp16_chunked(Phi_te, W, order_seed=7), y_te)
jit = np.abs(sA_te - sB_te)
eps_max = float(jit.max())
eps_99 = float(np.percentile(jit, 99))
eps_mean = float(jit.mean())
flip = float((fp16_chunked(Phi_te, W, order_seed=None).argmax(1)
              != fp16_chunked(Phi_te, W, order_seed=7).argmax(1)).mean())

res = {"dataset": "Gapminder (4 continents)", "n_train": n_tr, "n_cal": n_cal,
       "n_test": int(len(X_te)), "K": K, "D": D, "accuracy": acc, "mean_top_softmax": mtp,
       "eps_max": eps_max, "eps_99": eps_99, "eps_mean": eps_mean, "top1_flip_rate": flip,
       "per_alpha": []}
for a in [0.05, 0.10, 0.20]:
    qh = qhat_of(sA_cal, a)
    cA = float(np.mean(sA_te <= qh))
    cB = float(np.mean(sB_te <= qh))
    churn = float(np.mean((sA_te <= qh) != (sB_te <= qh)))
    fsup = sup_density(sA_cal, qh - eps_max, qh + eps_max)
    res["per_alpha"].append({"alpha": a, "qhat": float(qh), "cov_A": cA, "cov_B": cB,
        "abs_cov_loss": abs(cA - cB), "churn": churn, "f_sup": fsup,
        "cert_cov_loss": fsup * eps_max, "cert_churn": 2 * fsup * eps_max,
        "cov_within_cert": bool(abs(cA - cB) <= fsup * eps_max + 1e-12),
        "churn_within_cert": bool(churn <= 2 * fsup * eps_max + 1e-12)})

qh0 = qhat_of(sA_cal, 0.10)
cA0 = np.mean(sA_te <= qh0)
E = max(8 * eps_max, 0.12 * float(sA_te.std()))
fsup0 = sup_density(sA_cal, qh0 - E, qh0 + E)
eg = np.linspace(0, E, 25)
rad = RNG.choice([-1.0, 1.0], size=len(sA_te))
mw = [float(cA0 - np.mean((sA_te + e) <= qh0)) for e in eg]
cm = [float(np.mean((sA_te <= qh0) != ((sA_te + e * rad) <= qh0))) for e in eg]
res["cert_sweep"] = {"qhat": float(qh0), "f_sup": float(fsup0), "E": float(E),
    "eps_real": eps_max, "e_grid": eg.tolist(), "meas_worst": mw, "churn_meas": cm}

plt.rcParams.update({"font.size": 11, "figure.dpi": 150})
fig, ax = plt.subplots(figsize=(5.2, 3.6))
ax.plot(eg, mw, "o", ms=4, label="measured coverage loss (worst sign)")
ax.plot(eg, fsup0 * eg, "-", label=r"certificate $\hat f_{\sup}\,e$")
ax.plot(eg, cm, "^", ms=4, label="measured churn (random sign)")
ax.plot(eg, 2 * fsup0 * eg, "--", label=r"certificate $2\hat f_{\sup}\,e$")
ax.axvline(eps_max, color="red", ls=":", lw=1, label=r"real fp16 $\epsilon$")
ax.set_xlabel("perturbation magnitude $e$")
ax.set_ylabel("coverage loss / churn")
ax.set_title("Gapminder (real data)", fontsize=10)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/fig4_realdata_certificate.pdf")
plt.close(fig)

json.dump(res, open(f"{OUT}/results_real.json", "w"), indent=2)
print("SUMMARY", json.dumps({k: res[k] for k in ["accuracy", "mean_top_softmax", "eps_max", "eps_99", "eps_mean", "top1_flip_rate"]}, indent=2))
print("PER_ALPHA", json.dumps(res["per_alpha"], indent=2))
