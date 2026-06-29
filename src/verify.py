import json
import sys

def _load(path):
    try:
        return json.load(open(path))
    except Exception as exc:
        print("VERIFY could not read " + path + ": " + repr(exc))
        return None

def main():
    syn = _load("results/results.json")
    real = _load("results/results_real.json")
    if syn is None or real is None:
        print("REPRODUCTION CHECK: FAIL (missing results; run the experiments first)")
        return 1
    a = syn.get("self_check", {})
    b = real.get("self_check", {})
    syn_a, syn_t = a.get("alpha_within_cert", 0), a.get("alpha_total", 0)
    real_a, real_t = b.get("alpha_within_cert", 0), b.get("alpha_total", 0)
    stress_ok = bool(a.get("stress_within_cert", False))
    robust_ok = bool(a.get("robust_restores_coverage", False))
    flips_ok = bool(a.get("top1_flips_zero", False)) and bool(b.get("top1_flips_zero", False))
    all_pass = bool(a.get("all_passed", False)) and bool(b.get("all_passed", False)) and stress_ok and robust_ok and flips_ok
    eps = syn.get("eps_max", float("nan"))
    print("ALL CERTIFICATE CHECKS {} ({}/{} alpha synthetic + {}/{} alpha real, stress {}, robust-restore {}) | top-1 flips = {} | eps ~ {:.0e} (fp16)".format(
        "PASSED" if all_pass else "NOT PASSED", syn_a, syn_t, real_a, real_t,
        "ok" if stress_ok else "FAIL", "ok" if robust_ok else "FAIL", "0" if flips_ok else "NONZERO", eps))
    print("REPRODUCTION CHECK: " + ("PASS  (confirm this line, not the exact digits)" if all_pass else "FAIL"))
    return 0 if all_pass else 1

if __name__ == "__main__":
    sys.exit(main())
