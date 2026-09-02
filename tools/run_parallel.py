#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run N swarms of an LKlight docking task in parallel.

Swarms of the GSO are fully independent, so a whole docking job (setup already
done: setup.json + initial_positions_0..S-1.dat present) can be run with P
concurrent engine processes. Measured on a 12-core server (GPU engine):
4 swarms serial ~35 s -> P4 parallel ~8.5 s (~4x, near-linear).

Usage:
  python3 run_parallel.py <engine> <steps> <method> [swarms] [parallel]
    engine   path to the LKlight binary
    steps    GSO steps per swarm (e.g. 100)
    method   scoring function (dna / pydock / vdw / ...)
    swarms   total number of swarms to run (default: count initial_positions_*.dat)
    parallel max concurrent engine processes (default: min(cpu_count, swarms))
Run in the directory that holds setup.json and initial_positions_*.dat.
After all swarms finish, run `rank`/`generate` as usual (engine does not).
"""
import glob
import multiprocessing as mp
import os
import subprocess
import sys
import time


def main() -> None:
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(2)
    engine, steps, method = sys.argv[1], sys.argv[2], sys.argv[3]
    swarm_files = sorted(glob.glob("initial_positions_*.dat"),
                         key=lambda p: int(p.split("_")[-1].split(".")[0]))
    if not swarm_files:
        print("No initial_positions_*.dat found in the current directory.")
        sys.exit(1)
    total = int(sys.argv[4]) if len(sys.argv) > 4 else len(swarm_files)
    total = min(total, len(swarm_files))
    p_max = int(sys.argv[5]) if len(sys.argv) > 5 else min(
        os.cpu_count() or 4, total)
    p_max = max(1, min(p_max, total))

    print(f"Running {total} swarms x {steps} steps ({method}) with "
          f"parallelism {p_max}...")
    t0 = time.time()
    pool = mp.Pool(p_max)
    args = [(i, engine, steps, method) for i in range(total)]
    results = []
    for i, rc in pool.imap_unordered(_run_one, args):
        results.append((i, rc))
    pool.close()
    pool.join()
    wall = time.time() - t0
    failed = [i for i, rc in results if rc != 0]
    ok = total - len(failed)
    print(f"Done: {ok}/{total} swarms ok in {wall:.1f}s "
          f"({wall / max(1, total):.2f}s/swarm avg).")
    if failed:
        print("Failed swarms:", failed)
        sys.exit(1)


def _run_one(job):
    i, engine, steps, method = job
    cmd = [engine, "run", "setup.json", f"initial_positions_{i}.dat",
           str(steps), method]
    rc = subprocess.call(cmd, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
    return i, rc


if __name__ == "__main__":
    main()
