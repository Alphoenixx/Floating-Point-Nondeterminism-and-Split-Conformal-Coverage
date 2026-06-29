import glob
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
RESULTS = os.path.join(ROOT, "results")
sys.path.insert(0, SRC)

import environment

REQUIRED_STEPS = [
    ("Synthetic coverage experiment", "src/synthetic_experiment.py"),
    ("Real-data coverage experiment", "src/real_experiment.py"),
    ("Multi-seed synthetic stress", "src/multiseed_experiment.py"),
    ("Multi-seed real-data stress", "src/multiseed_real.py"),
    ("Stock-kernel envelope (float32/float16)", "src/realkernel_experiment.py"),
    ("int8 PTQ envelope", "src/int8_ptq_experiment.py"),
]

OPTIONAL_STEPS = [
    ("ViT envelope (needs PyTorch)", "extras/vit_envelope_experiment.py"),
]

FIGURE_STEP = ("Generating figures", "src/make_figures.py")

BAR_WIDTH = 28


def bar(done, total, label):
    done = max(0, min(done, total))
    filled = int(round(BAR_WIDTH * float(done) / total))
    rendered = "[" + "#" * filled + "-" * (BAR_WIDTH - filled) + "]"
    pct = int(round(100.0 * done / total))
    print("\n{} {:3d}%  ({}/{})  {}".format(rendered, pct, done, total, label), flush=True)


def run(rel, required):
    path = os.path.join(ROOT, rel)
    if not os.path.exists(path):
        print("  SKIP (not found): " + rel, flush=True)
        return
    result = subprocess.run([sys.executable, "-u", path], cwd=ROOT)
    if result.returncode != 0:
        message = "  FAILED: " + rel + " (exit code " + str(result.returncode) + ")"
        if required:
            print(message, flush=True)
            sys.exit(result.returncode)
        print(message + "  --  optional step (needs PyTorch), continuing", flush=True)


def stamp_provenance(info):
    for path in sorted(glob.glob(os.path.join(RESULTS, "*.json"))):
        try:
            data = json.load(open(path))
        except Exception as exc:
            print("  could not stamp " + path + ": " + repr(exc), flush=True)
            continue
        if isinstance(data, dict) and "environment" not in data:
            data["environment"] = info
            json.dump(data, open(path, "w"), indent=2)
            print("  recorded environment in " + os.path.basename(path), flush=True)


def main():
    if os.path.isdir(RESULTS):
        for path in glob.glob(os.path.join(RESULTS, "*")):
            try:
                os.remove(path)
            except OSError:
                pass
    os.makedirs(RESULTS, exist_ok=True)

    steps = [(label, rel, True) for label, rel in REQUIRED_STEPS]
    steps += [(label, rel, False) for label, rel in OPTIONAL_STEPS]
    steps += [(FIGURE_STEP[0], FIGURE_STEP[1], False)]

    total = len(steps) + 2
    done = 0

    for label, rel, required in steps:
        bar(done, total, "Running: " + label)
        run(rel, required)
        done += 1

    bar(done, total, "Recording execution environment")
    info = environment.collect()
    stamp_provenance(info)
    for line in environment.summary_lines(info):
        print("  " + line, flush=True)
    done += 1

    bar(done, total, "Verifying certificate reproduction")
    subprocess.run([sys.executable, "-u", os.path.join(SRC, "verify.py")], cwd=ROOT)
    done += 1

    bar(done, total, "Complete")
    print("\nAll JSON results and figures are in the results/ folder.", flush=True)


if __name__ == "__main__":
    main()
