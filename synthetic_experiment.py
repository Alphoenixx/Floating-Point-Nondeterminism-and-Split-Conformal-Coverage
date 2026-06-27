import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)

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


def make_features(X, D, seed=7):
    r = np.random.default_rng(seed)
    Rmat = r.normal(0, 1.0 / np.sqrt(d), size=(d, D))
    b = r.normal(0, 0.1, size=(D,))
    Phi = np.maximum(0.0, X @ Rmat + b)
    return (Phi / (Phi.std(0) + 1e-8)).astype(np.float64)


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


def conformal_qhat(cal, alpha):
    n = len(cal)
    k = min(int(np.ceil((n + 1) * (1 - alpha))), n)
    return np.sort(cal)[k - 1]


def scores(logits, labels):
    P = softmax(logits)
    return 1.0 - P[np.arange(len(labels)), labels]


def logits_fp32(Phi, W, perm=None):
    P32, W32 = Phi.astype(np.float32), W.astype(np.float32)
    if perm is None:
        return (P32 @ W32).astype(np.float64)
    return (P32[:, perm] @ W32[perm, :]).astype(np.float64)


def logits_fp16_chunked(Phi, W, chunk=64, order_seed=None):
    n, D = Phi.shape
    Kk = W.shape[1]
    bnds = [(i, min(i + chunk, D)) for i in range(0, D, chunk)]
    if order_seed is not None:
        np.random.default_rng(order_seed).shuffle(bnds)
    acc = np.zeros((n, Kk), dtype=np.float16)
    Pm, Wm = Phi.astype(np.float32), W.astype(np.float32)
    for a, b in bnds:
        partial = Pm[:, a:b] @ Wm[a:b, :]
        acc = (acc.astype(np.float32) + partial).astype(np.float16)
    return acc.astype(np.float64)


def sup_density(s, lo, hi, nbins=50):
    counts, edges = np.histogram(s, bins=nbins, range=(s.min(), s.max()))
    w = edges[1] - edges[0]
    dens = counts / (len(s) * w)
    m = (edges[:-1] <= hi + 1e-12) & (edges[1:] >= lo - 1e-12)
    return float(dens[m].max()) if m.any() else float(dens.max())


Phi = make_features(X, D_main)
Phi_tr, Phi_cal, Phi_te = Phi[:N_train], Phi[N_train:N_train + N_cal], Phi[N_train + N_cal:]
y_tr, y_cal, y_te = y[:N_train], y[N_train:N_train + N_cal], y[N_train + N_cal:]
W = train_softmax(Phi_tr, y_tr, K)
acc = float((softmax(Phi_te @ W).argmax(1) == y_te).mean())
mtp = float(softmax(Phi_te @ W).max(1).mean())

perm = RNG.permutation(D_main)
logA_cal = logits_fp16_chunked(Phi_cal, W, order_seed=None)
logA_te = logits_fp16_chunked(Phi_te, W, order_seed=None)
logB_te = logits_fp16_chunked(Phi_te, W, order_seed=7)
sA_cal, sA_te, sB_te = scores(logA_cal, y_cal), scores(logA_te, y_te), scores(logB_te, y_te)

fp32_jit = float(np.abs(scores(logits_fp32(Phi_te, W), y_te)
                        - scores(logits_fp32(Phi_te, W, perm), y_te)).max())

jit = np.abs(sA_te - sB_te)
eps_max, eps_99, eps_mean = float(jit.max()), float(np.percentile(jit, 99)), float(jit.mean())
flip_top1 = float((logA_te.argmax(1) != logB_te.argmax(1)).mean())

results = {"K": K, "d": d, "D_main": D_main, "chunk": 64,
           "N_train": N_train, "N_cal": N_cal, "N_test": N_test,
           "test_accuracy": acc, "mean_top_softmax": mtp,
           "eps_fp32_max": fp32_jit,
           "eps_max": eps_max, "eps_99": eps_99, "eps_mean": eps_mean,
           "top1_flip_rate": flip_top1, "per_alpha": []}

for alpha in [0.05, 0.10, 0.20]:
    qhat = conformal_qhat(sA_cal, alpha)
    covA = float(np.mean(sA_te <= qhat))
    covB = float(np.mean(sB_te <= qhat))
    abs_gap = abs(covA - covB)
    churn = float(np.mean((sA_te <= qhat) != (sB_te <= qhat)))
    f_sup = sup_density(sA_cal, qhat - eps_max, qhat + eps_max)
    results["per_alpha"].append({"alpha": alpha, "qhat": float(qhat),
        "cov_A": covA, "cov_B": covB, "abs_cov_loss": abs_gap, "churn": churn,
        "f_sup": f_sup, "cert_cov_loss": f_sup * eps_max, "cert_churn": 2 * f_sup * eps_max,
        "cov_within_cert": bool(abs_gap <= f_sup * eps_max + 1e-12),
        "churn_within_cert": bool(churn <= 2 * f_sup * eps_max + 1e-12)})

sweep = []
for D in [128, 256, 512, 1024, 2048, 4096]:
    rr = np.random.default_rng(2024)
    Phn = np.maximum(0.0, rr.normal(0, 1, size=(3000, D)))
    Wn = rr.normal(0, 1.0 / np.sqrt(D), size=(D, K))
    base = logits_fp16_chunked(Phn, Wn, order_seed=None)
    mx = [float(np.abs(base - logits_fp16_chunked(Phn, Wn, order_seed=100 + sd)).max()) for sd in range(10)]
    sweep.append({"D": D, "logit_jit_mean": float(np.mean(mx)), "logit_jit_worst": float(np.max(mx))})
results["width_sweep"] = sweep

alpha0 = 0.10
qhat0 = conformal_qhat(sA_cal, alpha0)
covA0 = np.mean(sA_te <= qhat0)
E = max(8 * eps_max, 0.12 * float(sA_te.std()))
f_sup0 = sup_density(sA_cal, qhat0 - E, qhat0 + E)
e_grid = np.linspace(0, E, 25)
rad = RNG.choice([-1.0, 1.0], size=len(sA_te))
mw, mr, cm = [], [], []
for e in e_grid:
    mw.append(float(covA0 - np.mean((sA_te + e) <= qhat0)))
    mr.append(float(abs(covA0 - np.mean((sA_te + e * rad) <= qhat0))))
    cm.append(float(np.mean((sA_te <= qhat0) != ((sA_te + e * rad) <= qhat0))))
results["cert_sweep"] = {"alpha": alpha0, "qhat": float(qhat0), "f_sup": float(f_sup0), "E": float(E),
    "e_grid": e_grid.tolist(), "meas_worst": mw, "meas_rand": mr, "churn_meas": cm,
    "cert_line": (f_sup0 * e_grid).tolist(), "churn_line": (2 * f_sup0 * e_grid).tolist()}

plt.rcParams.update({"font.size": 11, "figure.dpi": 150})
Ds = np.array([s["D"] for s in sweep], float)
emax = np.array([s["logit_jit_mean"] for s in sweep])
ewor = np.array([s["logit_jit_worst"] for s in sweep])
fig, ax = plt.subplots(figsize=(5.2, 3.6))
ax.loglog(Ds, emax, "o-", label="logit jitter (mean over schedules)")
ax.loglog(Ds, ewor, "s--", label="logit jitter (worst)")
ax.loglog(Ds, emax[0] * np.sqrt(Ds / Ds[0]), ":", color="gray", label=r"$\propto\sqrt{D}$")
ax.set_xlabel("reduction length (feature width $D$)")
ax.set_ylabel(r"logit jitter")
ax.legend()
ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/fig1_jitter_vs_width.pdf")
plt.close(fig)

eg = np.array(e_grid)
fig, ax = plt.subplots(figsize=(5.2, 3.6))
ax.plot(eg, mw, "o", ms=4, label="measured coverage loss (worst sign)")
ax.plot(eg, f_sup0 * eg, "-", label=r"certificate $\hat f_{\sup}\,e$")
ax.plot(eg, cm, "^", ms=4, label="measured churn (random sign)")
ax.plot(eg, 2 * f_sup0 * eg, "--", label=r"certificate $2\hat f_{\sup}\,e$")
ax.axvline(eps_max, color="red", ls=":", lw=1, label=r"real fp16 $\epsilon$")
ax.set_xlabel("perturbation magnitude $e$")
ax.set_ylabel("coverage loss / churn")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/fig2_certificate.pdf")
plt.close(fig)

fig, ax = plt.subplots(figsize=(5.2, 3.6))
ax.hist(sA_cal, bins=60, color="steelblue", alpha=0.75)
ax.axvline(qhat0, color="black", lw=1.5, label=r"$\hat q$")
ax.axvspan(qhat0 - eps_max, qhat0 + eps_max, color="red", alpha=0.4, label=r"$[\hat q-\epsilon,\hat q+\epsilon]$")
ax.set_xlabel(r"nonconformity score $s$")
ax.set_ylabel("calibration count")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/fig3_score_band.pdf")
plt.close(fig)

json.dump(results, open(f"{OUT}/results.json", "w"), indent=2)
print("SUMMARY", json.dumps({k: results[k] for k in ["test_accuracy", "mean_top_softmax", "eps_fp32_max", "eps_max", "eps_99", "eps_mean", "top1_flip_rate"]}, indent=2))
print("PER_ALPHA", json.dumps(results["per_alpha"], indent=2))
print("WIDTH", json.dumps(sweep, indent=2))
