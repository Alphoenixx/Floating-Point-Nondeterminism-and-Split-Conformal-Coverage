import os
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "vit_crossdevice_results.json")
    try:
        r = json.load(open(path))
    except Exception as exc:
        print("VIT VERIFY could not read " + path + ": " + repr(exc))
        return 1
    rows = r.get("rows", [])
    ok = bool(r.get("all_within_cert", False)) and len(rows) > 0
    for row in rows:
        print("VIT pair {} eps_max={:.3e} top1_flip={:.4%}".format(row.get("pair", ""), row.get("eps_max", float("nan")), row.get("top1_flip_rate", float("nan"))))
        for p in row.get("per_alpha", []):
            print("  alpha={:.2f} cov_src={:.4f} cov_tgt={:.4f} loss={:+.4f} cert_loss={:.3e} within_cert={}".format(
                p["alpha"], p["cov_source"], p["cov_target"], p["cov_loss"], p["cert_cov_loss"], p["cov_within_cert"]))
    print("VIT REPRODUCTION CHECK: " + ("PASS (certificate held on saved ViT outputs)" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
