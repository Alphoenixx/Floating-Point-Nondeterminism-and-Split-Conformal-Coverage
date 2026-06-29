import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(HERE, "results")


def load(name):
    path = os.path.join(RESULTS, name)
    if not os.path.exists(path):
        return None
    try:
        return json.load(open(path))
    except Exception:
        return None


def fig_dichotomy():
    ms = load("results_multiseed.json")
    iq = load("results_int8_ptq.json")
    if ms is None and iq is None:
        print("fig6 skipped (no data)")
        return
    a = "0.1"
    bars = []
    if ms is not None:
        try:
            m = ms["main_fp16_reorder"]["per_alpha"][a]
            bars.append(("fp16\nreorder", abs(m["abs_cov_gap"]["mean"]) / m["sampling_floor"], False))
        except Exception:
            pass
        try:
            r = ms["stress"]["regimes"]["fp16_round"]["per_alpha"][a]
            bars.append(("fp16\nround", abs(r["loss_over_floor"]), False))
        except Exception:
            pass
    if iq is not None:
        try:
            r = iq["regimes"]["saturated"]["kernels"]["int8_round"]["per_alpha"][a]
            bars.append(("int8\nRTN", abs(r["loss_over_floor"]), False))
        except Exception:
            pass
        try:
            r = iq["regimes"]["saturated"]["kernels"]["int8_trunc"]["per_alpha"][a]
            bars.append(("int8\ntrunc", r["loss_over_floor"], True))
        except Exception:
            pass
    if ms is not None:
        try:
            r = ms["stress"]["regimes"]["fp8_trunc"]["per_alpha"][a]
            bars.append(("fp8\ntrunc", r["loss_over_floor"], True))
        except Exception:
            pass
    if not bars:
        print("fig6 skipped (no series)")
        return
    labels = [b[0] for b in bars]
    vals = [max(float(b[1]), 1e-3) for b in bars]
    colors = ["#c0392b" if b[2] else "#2e6db0" for b in bars]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    x = np.arange(len(bars))
    ax.bar(x, vals, color=colors, width=0.62)
    ax.axhline(1.0, ls="--", color="black", lw=1)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"coverage loss / sampling floor ($\alpha=0.1$)")
    ax.text(len(bars) - 0.5, 1.15, "sampling floor", ha="right", va="bottom", fontsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, color="#2e6db0"),
               plt.Rectangle((0, 0), 1, 1, color="#c0392b")]
    ax.legend(handles, ["unbiased kernel", "biased kernel"], frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "fig6_dichotomy.pdf"))
    plt.close(fig)
    print("wrote fig6_dichotomy.pdf")


def fig_vit_sqrtd():
    vit = load("results_vit.json")
    points = None
    if vit is not None:
        points = vit.get("reduction_length_sweep", {}).get("points")
    if not points:
        print("fig7 skipped (no ViT reduction-length sweep; needs results_vit.json)")
        return
    D = np.array([p["reduction_length"] for p in points], dtype=float)
    eps = np.array([p["eps_max"] for p in points], dtype=float)
    order = np.argsort(D)
    D = D[order]
    eps = eps[order]
    ref = eps[0] * np.sqrt(D / D[0])
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.loglog(D, eps, "o-", color="#2e6db0", label=r"measured $\varepsilon_{\max}$ (ViT-B/16)")
    ax.loglog(D, ref, "--", color="black", label=r"$\propto\sqrt{D}$ reference")
    ax.set_xlabel("reduction length $D$")
    ax.set_ylabel(r"logit envelope $\varepsilon_{\max}$")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "fig7_vit_sqrtD.pdf"))
    plt.close(fig)
    print("wrote fig7_vit_sqrtD.pdf")


def main():
    os.makedirs(RESULTS, exist_ok=True)
    fig_dichotomy()
    fig_vit_sqrtd()


if __name__ == "__main__":
    main()
