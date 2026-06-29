import os
import sys
import json
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ALPHAS = [0.05, 0.10, 0.20]


def conformal_qhat(cal, alpha):
    n = len(cal)
    k = min(int(np.ceil((n + 1) * (1 - alpha))), n)
    return np.sort(cal)[k - 1]


def sup_density(s, lo, hi, nbins=50):
    counts, edges = np.histogram(s, bins=nbins, range=(s.min(), s.max()))
    w = edges[1] - edges[0]
    dens = counts / (len(s) * w)
    m = (edges[:-1] <= hi + 1e-12) & (edges[1:] >= lo - 1e-12)
    return float(dens[m].max()) if m.any() else float(dens.max())


def load_dev(path):
    z = np.load(path, allow_pickle=False)
    label = str(z["__label"][0])
    device = str(z["__device"][0])
    sha = str(z["__artifact_sha256"][0])
    variants = {}
    for k in z.files:
        if k.startswith("scores_cal__"):
            name = k.split("__", 1)[1]
            variants[name] = {"scores_cal": z["scores_cal__" + name],
                              "scores_te": z["scores_te__" + name],
                              "argmax_te": z["argmax_te__" + name]}
    return {"label": label, "device": device, "sha": sha, "variants": variants, "path": path}


def pair_row(src, dst, src_name, dst_name):
    ssc = src["variants"][src_name]["scores_cal"]
    sst = src["variants"][src_name]["scores_te"]
    dst_te = dst["variants"][dst_name]["scores_te"]
    eps = np.abs(sst - dst_te)
    eps_max = float(eps.max())
    eps_99 = float(np.percentile(eps, 99))
    eps_mean = float(eps.mean())
    flip = float((src["variants"][src_name]["argmax_te"] != dst["variants"][dst_name]["argmax_te"]).mean())
    per = []
    for alpha in ALPHAS:
        qh = conformal_qhat(ssc, alpha)
        cov_src = float(np.mean(sst <= qh))
        cov_dst = float(np.mean(dst_te <= qh))
        churn = float(np.mean((sst <= qh) != (dst_te <= qh)))
        fsup = sup_density(ssc, qh - eps_max, qh + eps_max)
        gap = abs(cov_src - cov_dst)
        per.append({"alpha": alpha, "qhat": float(qh),
                    "cov_source": cov_src, "cov_target": cov_dst,
                    "cov_loss": cov_src - cov_dst, "abs_cov_loss": gap,
                    "churn": churn, "f_sup": fsup,
                    "cert_cov_loss": fsup * eps_max, "cert_churn": 2 * fsup * eps_max,
                    "cov_within_cert": bool(gap <= fsup * eps_max + 1e-12),
                    "churn_within_cert": bool(churn <= 2 * fsup * eps_max + 1e-12)})
    return {"eps_max": eps_max, "eps_99": eps_99, "eps_mean": eps_mean,
            "top1_flip_rate": flip, "per_alpha": per}


def parse_pair(spec):
    left, right = spec.split("->")
    sl, sn = left.split(":")
    dl, dn = right.split(":")
    return sl.strip(), sn.strip(), dl.strip(), dn.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--pairs", nargs="+", required=True)
    ap.add_argument("--out", default=os.path.join(HERE, "crossdevice_results.json"))
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    devs = {}
    shas = set()
    for f in args.files:
        d = load_dev(f)
        devs[d["label"]] = d
        shas.add(d["sha"])
        print("LOADED {} device={} variants={} sha={}".format(d["label"], d["device"], sorted(d["variants"].keys()), d["sha"][:12]), flush=True)
    print("ARTIFACT consensus = {} ({} distinct sha)".format(len(shas) == 1, len(shas)), flush=True)
    if len(shas) != 1:
        print("WARN device files were produced from different artifacts; cross-device epsilon is not isolated", flush=True)

    rows = []
    for spec in args.pairs:
        sl, sn, dl, dn = parse_pair(spec)
        if sl not in devs or dl not in devs:
            print("SKIP " + spec + " (missing device file)", flush=True)
            continue
        if sn not in devs[sl]["variants"] or dn not in devs[dl]["variants"]:
            print("SKIP " + spec + " (missing variant)", flush=True)
            continue
        r = pair_row(devs[sl], devs[dl], sn, dn)
        r["pair"] = spec
        r["source"] = sl + ":" + sn
        r["target"] = dl + ":" + dn
        rows.append(r)
        print("", flush=True)
        print("PAIR {}  ->  {}".format(r["source"], r["target"]), flush=True)
        print("  eps_max={:.3e}  eps_99={:.3e}  eps_mean={:.3e}  top1_flip={:.4%}".format(r["eps_max"], r["eps_99"], r["eps_mean"], r["top1_flip_rate"]), flush=True)
        for p in r["per_alpha"]:
            print("  alpha={:.2f}  qhat={:.6f}  cov_src={:.4f}  cov_tgt={:.4f}  loss={:+.4f}  churn={:.4f}  cert_loss={:.3e}  within_cert={} churn_within={}".format(
                p["alpha"], p["qhat"], p["cov_source"], p["cov_target"], p["cov_loss"], p["churn"], p["cert_cov_loss"], p["cov_within_cert"], p["churn_within_cert"]), flush=True)

    all_within = all(p["cov_within_cert"] and p["churn_within_cert"] for r in rows for p in r["per_alpha"])
    out = {"artifact_consensus": bool(len(shas) == 1), "n_distinct_artifacts": len(shas),
           "all_within_cert": bool(all_within), "rows": rows}
    json.dump(out, open(args.out, "w"), indent=2)
    print("", flush=True)
    print("WROTE " + args.out, flush=True)
    print("RESULT cross-device: " + ("ALL PAIRS WITHIN CERTIFICATE" if all_within else "SOME PAIRS EXCEED CERTIFICATE"), flush=True)


if __name__ == "__main__":
    main()
