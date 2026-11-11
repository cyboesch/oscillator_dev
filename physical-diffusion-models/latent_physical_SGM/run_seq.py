#!/usr/bin/env python3
"""Run several scripts of this folder one after another in one job (one queue wait), each in its own process.

Usage (from latent_physical_SGM/):
  sbatch --time=00:45:00 jobs/run_gpu_neuronic.sh run_seq.py "speed_check.py 6" "train_checks.py" ...
Every argument is one command: a script of this folder and its arguments (split like a shell would). Each script
writes its own outputs (under out/latent_physical_SGM/<its folder>/<job>/). Exit code: 1 if any command failed.
"""

import os
import shlex
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    failed = []
    for cmd in sys.argv[1:]:
        parts = shlex.split(cmd)
        t = time.time()
        print(f"\n######## run_seq: {cmd}  ({time.strftime('%H:%M:%S')})", flush=True)
        r = subprocess.run([sys.executable, "-u", os.path.join(HERE, parts[0])] + parts[1:])
        print(f"######## run_seq: {cmd}: exit code {r.returncode} after {time.time() - t:.0f} s", flush=True)
        if r.returncode:
            failed.append(cmd)
    print(f"\nrun_seq: {len(sys.argv) - 1} commands, failed: {failed}", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
