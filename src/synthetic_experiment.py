import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
import json
import environment
import numpy as np
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_PLT = True
except Exception as _plt_err:
    print("WARN matplotlib unavailable, skipping figures:", repr(_plt_err))
    HAVE_PLT = False

    class _NoPlot:
        def __getattr__(self, _):
            return self

        def __call__(self, *a, **k):
            return self

        def __iter__(self):
            return iter((self, self))

    plt = _NoPlot()

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)

ENV = environment.collect()
for _line in environment.summary_lines(ENV):
    print("ENV " + _line)

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
    P32 = Phi.astype(np.float32)
    W32 = W.astype(np.float32)
    n, D = P32.shape
    Kk = W32.shape[1]
    order = np.arange(D) if perm is None else np.asarray(perm)
    acc = np.zeros((n, Kk), dtype=np.float32)
    for j in order:
        acc = (acc + P32[:, int(j):int(j) + 1] * W32[int(j):int(j) + 1, :]).astype(np.float32)
    return acc.astype(np.float64)


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

results = {"environment": ENV, "K": K, "d": d, "D_main": D_main, "chunk": 64,
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


SRNG = np.random.default_rng(20240517)
K_s, d_s, D_s = 10, 100, 512
Ntr_s, Ncal_s, Nte_s = 8000, 40000, 40000
Ntot_s = Ntr_s + Ncal_s + Nte_s
CS_s = 1.0
cen_s = SRNG.normal(0, 1.0, size=(K_s, d_s)) * CS_s
ys = SRNG.integers(0, K_s, size=Ntot_s)
Xs = cen_s[ys] + SRNG.normal(0, 1.0, size=(Ntot_s, d_s))
mu_s, sd_s = Xs[:Ntr_s].mean(0), Xs[:Ntr_s].std(0) + 1e-8
Xs = (Xs - mu_s) / sd_s
rs = np.random.default_rng(7)
Rs = rs.normal(0, 1.0 / np.sqrt(d_s), size=(d_s, D_s))
bs = rs.normal(0, 0.1, size=(D_s,))
Phs = np.maximum(0.0, Xs @ Rs + bs)
pmu_s, psd_s = Phs[:Ntr_s].mean(0), Phs[:Ntr_s].std(0) + 1e-8
Phs = (Phs - pmu_s) / psd_s
Phtr_s, Phcal_s, Phte_s = Phs[:Ntr_s], Phs[Ntr_s:Ntr_s + Ncal_s], Phs[Ntr_s + Ncal_s:]
ytr_s, ycal_s, yte_s = ys[:Ntr_s], ys[Ntr_s:Ntr_s + Ncal_s], ys[Ntr_s + Ncal_s:]
Ws = train_softmax(Phtr_s, ytr_s, K_s, iters=300, lr=0.5, l2=0.05)
ref_cal_s = quant_logits(Phcal_s, Ws)
ref_te_s = quant_logits(Phte_s, Ws)
acc_s = float((softmax(ref_te_s).argmax(1) == yte_s).mean())
mtp_s = float(softmax(ref_te_s).max(1).mean())
scal_s = scores(ref_cal_s, ycal_s)
sref_s = scores(ref_te_s, yte_s)
stress = {"K": K_s, "d": d_s, "D": D_s, "N_train": Ntr_s, "N_cal": Ncal_s, "N_test": Nte_s,
          "center_scale": CS_s, "test_accuracy": acc_s, "mean_top_softmax": mtp_s, "regimes": []}
stress_alphas = [0.05, 0.10, 0.20]
stress_plot = {}
for name, kw in [("fp16_round", dict(mbits=10, mode="round", order_seed=7)),
                 ("fp8_trunc", dict(mbits=7, mode="trunc", order_seed=7))]:
    dep_s = quant_logits(Phte_s, Ws, **kw)
    sdep_s = scores(dep_s, yte_s)
    Sall_dep = 1.0 - softmax(dep_s)
    eps_s = np.abs(sdep_s - sref_s)
    epsmax_s, epsmean_s = float(eps_s.max()), float(eps_s.mean())
    flips_s = float((dep_s.argmax(1) != ref_te_s.argmax(1)).mean())
    rec = {"name": name, "eps_max": epsmax_s, "eps_mean": epsmean_s,
           "top1_flip_rate": flips_s, "per_alpha": []}
    losses = []
    for alpha in stress_alphas:
        floor = float(np.sqrt(alpha * (1 - alpha) / Nte_s))
        qh = conformal_qhat(scal_s, alpha)
        cov_ref = float(np.mean(sref_s <= qh))
        cov_dep = float(np.mean(sdep_s <= qh))
        loss = cov_ref - cov_dep
        fsup = sup_density(scal_s, qh - epsmax_s, qh + epsmax_s)
        cov_dep_robust = float(np.mean(sdep_s <= qh + epsmax_s))
        setsize_naive = float(np.mean((Sall_dep <= qh).sum(1)))
        setsize_robust = float(np.mean((Sall_dep <= qh + epsmax_s).sum(1)))
        robust_valid = bool(cov_dep_robust >= 1.0 - alpha - 1e-9)
        rec["per_alpha"].append({"alpha": alpha, "qhat": float(qh), "sampling_floor": floor,
            "cov_ref": cov_ref, "cov_dep": cov_dep, "cov_loss": loss,
            "loss_over_floor": loss / floor, "f_sup": fsup,
            "cert_cov_loss": fsup * epsmax_s,
            "loss_within_cert": bool(loss <= fsup * epsmax_s + 1e-12),
            "cov_dep_robust": cov_dep_robust, "cov_loss_robust": cov_ref - cov_dep_robust,
            "set_size_naive": setsize_naive, "set_size_robust": setsize_robust,
            "set_size_increase": setsize_robust - setsize_naive,
            "robust_restores_coverage": robust_valid})
        losses.append(loss)
    stress["regimes"].append(rec)
    stress_plot[name] = losses
results["stress_regime"] = stress

xpos = np.arange(len(stress_alphas))
bw = 0.36
floor_s = [np.sqrt(a * (1 - a) / Nte_s) for a in stress_alphas]
fig, ax = plt.subplots(figsize=(5.6, 3.8))
ax.bar(xpos - bw / 2, [abs(v) for v in stress_plot["fp16_round"]], bw,
       color="steelblue", label="fp16 round-to-nearest (unbiased)")
ax.bar(xpos + bw / 2, [abs(v) for v in stress_plot["fp8_trunc"]], bw,
       color="indianred", label="fp8 truncating accumulator (biased)")
for i, fl in enumerate(floor_s):
    ax.hlines(fl, xpos[i] - 0.46, xpos[i] + 0.46, color="black", ls="--", lw=1.3)
ax.plot([], [], "k--", lw=1.3, label=r"sampling floor $\sqrt{\alpha(1-\alpha)/n}$")
ax.set_yscale("log")
ax.set_xticks(xpos)
ax.set_xticklabels([rf"$\alpha={a}$" for a in stress_alphas])
ax.set_ylabel("measured coverage loss")
ax.set_title(rf"Large-$n$ stress regime ($n={Nte_s}$)")
ax.legend(fontsize=8, loc="upper left")
ax.grid(True, which="both", axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/fig5_above_floor.pdf")
plt.close(fig)

syn_alpha_pass = sum(1 for p in results["per_alpha"] if p["cov_within_cert"] and p["churn_within_cert"])
syn_alpha_total = len(results["per_alpha"])
syn_stress_pass = all(pa["loss_within_cert"] for rg in results["stress_regime"]["regimes"] for pa in rg["per_alpha"])
syn_robust_pass = all(pa["robust_restores_coverage"] for rg in results["stress_regime"]["regimes"] for pa in rg["per_alpha"])
syn_flip_pass = results["top1_flip_rate"] == 0.0
syn_all_pass = (syn_alpha_pass == syn_alpha_total) and syn_stress_pass and syn_robust_pass and syn_flip_pass
results["self_check"] = {"alpha_within_cert": syn_alpha_pass, "alpha_total": syn_alpha_total, "stress_within_cert": bool(syn_stress_pass), "robust_restores_coverage": bool(syn_robust_pass), "top1_flips_zero": bool(syn_flip_pass), "all_passed": bool(syn_all_pass)}
json.dump(results, open(f"{OUT}/results.json", "w"), indent=2)
print("SUMMARY", json.dumps({k: results[k] for k in ["test_accuracy", "mean_top_softmax", "eps_fp32_max", "eps_max", "eps_99", "eps_mean", "top1_flip_rate"]}, indent=2))
print("PER_ALPHA", json.dumps(results["per_alpha"], indent=2))
print("WIDTH", json.dumps(sweep, indent=2))
print("STRESS", json.dumps(stress, indent=2))
print("SELFCHECK synthetic: {}/{} alpha within certificate, stress {}, robust-restore {}, top1_flips={:.0f}, eps_max(fp16)={:.1e}".format(syn_alpha_pass, syn_alpha_total, "PASS" if syn_stress_pass else "FAIL", "PASS" if syn_robust_pass else "FAIL", results["top1_flip_rate"], results["eps_max"]))
print("RESULT synthetic: " + ("ALL CERTIFICATE CHECKS PASSED" if syn_all_pass else "SOME CHECKS FAILED"))
