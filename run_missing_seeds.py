"""Run the missing hardened seeds (3, 4) to bring every method to five seeds.

Writes to ./results_newseeds (isolated; your committed ./results and the
store are untouched until you explicitly ingest). Reproduction of seed 0 was
verified bit-exact, so seeds 3, 4 are valid members of the same experiment.

Skips any (name, seed) that is already finished, so re-running resumes where
it left off. Run this from the project root with the project venv as the
interpreter.

    python run_missing_seeds.py
"""
import os
import subprocess
import sys
import time

PY = sys.executable
OUT = "./results_newseeds"
SEEDS = [3, 4]

# (config, run-name, method, expected #rounds) — only methods/datasets below 5 seeds
JOBS = [
    ("configs/exp_a_hardened.yaml",   "exp_h_adsys_trustfl",   "trustfl",   100),
    ("configs/femnist_hardened.yaml", "femh_adsys_trustfl",    "trustfl",    60),
    ("configs/femnist_hardened.yaml", "femh_adsys_fedavg",     "fedavg",     60),
    ("configs/femnist_hardened.yaml", "femh_adsys_multikrum",  "multikrum",  60),
    ("configs/femnist_hardened.yaml", "femh_adsys_fltrust",    "fltrust",    60),
    ("configs/femnist_hardened.yaml", "femh_adsys_fedgreed",   "fedgreed",   60),
]


def done(name, seed, rounds):
    f = os.path.join(OUT, name, f"seed{seed}", "metrics.jsonl")
    if not os.path.exists(f):
        return False
    return sum(1 for _ in open(f)) >= rounds


def main():
    t0 = time.time()
    for i, (cfg, name, method, rounds) in enumerate(JOBS, 1):
        todo = [s for s in SEEDS if not done(name, s, rounds)]
        if not todo:
            print(f"[{i}/{len(JOBS)}] {name}: seeds {SEEDS} already done, skip", flush=True)
            continue
        print(f"\n===== JOB {i}/{len(JOBS)}: {name} seeds {todo} =====", flush=True)
        cmd = [PY, "-m", "trustfl.experiments.run", "--config", cfg,
               "--override", f"name={name}", f"method.name={method}",
               f"seeds={todo}".replace(" ", ""), f"out_dir={OUT}"]
        r = subprocess.run(cmd)
        print(f"[job {i}] exit={r.returncode}  elapsed={(time.time()-t0)/60:.1f} min", flush=True)
    print(f"\nALL DONE in {(time.time()-t0)/60:.1f} min", flush=True)
    print("Next: python -m trustfl.store ingest --results ./results_newseeds  (into a store copy)")


if __name__ == "__main__":
    main()
