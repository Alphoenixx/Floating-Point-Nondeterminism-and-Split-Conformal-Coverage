import os
import sys
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))

for script in ["synthetic_experiment.py", "real_experiment.py"]:
    subprocess.run([sys.executable, os.path.join(HERE, script)], check=True)
