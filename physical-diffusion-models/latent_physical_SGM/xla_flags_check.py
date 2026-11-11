#!/usr/bin/env python3
"""Step time of the fast training step under several XLA_FLAGS settings (command buffers = CUDA graphs); a GPU job.

XLA reads XLA_FLAGS once per process, so every setting runs speed_check.py (section 1, which logs the flags, and
section 5 for the selected rows) in a subprocess of its own. All runs append to the same report:
out/latent_physical_SGM/speed_check/<job>_xla/report.txt.

Usage (from latent_physical_SGM/):  sbatch --time=00:30:00 jobs/run_gpu_neuronic.sh xla_flags_check.py [rows]
rows: section-5 row filter of speed_check.py (default: the nobest and lean rows at K = 0 and 784).
"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FLAGSETS = [
    "",                                                     # XLA's defaults
    "--xla_gpu_enable_command_buffer=",                     # no command buffers (every kernel launched alone)
    "--xla_gpu_enable_command_buffer=FUSION,CUBLAS,CUBLASLT,CUSTOM_CALL,CUDNN,CONDITIONALS",   # + while loops
    "--xla_gpu_graph_min_graph_size=1",
]


def main():
    rows = sys.argv[1] if len(sys.argv) > 1 else \
        "K0 stencil mixed nobest,K784 stencil mixed nobest,K0 stencil mixed lean,K784 stencil mixed lean"
    for flags in FLAGSETS:
        env = dict(os.environ, XLA_FLAGS=flags, SPEED_CHECK_SUFFIX="_xla")
        print(f"===== XLA_FLAGS={flags!r}", flush=True)
        r = subprocess.run([sys.executable, "-u", os.path.join(HERE, "speed_check.py"), "1,5", rows], env=env)
        print(f"===== exit code {r.returncode}", flush=True)


if __name__ == "__main__":
    main()
