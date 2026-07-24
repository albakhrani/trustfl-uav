"""Phase 0 results infrastructure: a single queryable source of truth.

Every experiment run — past or future — is registered in one SQLite index
(one row per run: provenance + headline metrics) with the full per-round
time series stored alongside as Parquet. This is what makes results
(a) reproducible by construction (every number traces to a config hash,
seed, and git commit) and (b) trivial to turn into publication figures
(one deterministic read, no re-running).

Layout produced under <store_root> (default: ./results_store):
    results.db              SQLite index (runs, configs tables)
    rounds/<run_id>.parquet per-round metrics, one file per run
    configs/<hash>.yaml     content-addressed resolved configs

The store INGESTS the existing ./results tree (per-seed metrics.jsonl +
run_meta.json + config_resolved.yaml + groups.json) without modification, so
nothing needs re-running. New runs can also register directly.

CLI:
    python -m trustfl.store ingest --results ./results
    python -m trustfl.store list --dataset femnist --method krum
    python -m trustfl.store show <run_id>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import yaml

DEFAULT_STORE = "./results_store"

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id        TEXT PRIMARY KEY,   -- <name>/<seed> (stable, human-readable)
    name          TEXT NOT NULL,      -- experiment name (e.g. exp_a_krum)
    seed          INTEGER NOT NULL,
    config_hash   TEXT NOT NULL,      -- content hash of resolved config
    dataset       TEXT,
    model         TEXT,
    method        TEXT,
    attack_type   TEXT,
    malicious_frac REAL,
    num_uavs      INTEGER,
    rounds        INTEGER,
    high_mob_frac REAL,
    -- headline metrics (filled from the final eval round)
    final_acc     REAL,
    final_loss    REAL,
    best_acc      REAL,               -- max over eval rounds (robust to drift)
    acc_last10    REAL,               -- mean of last 10 evals (stable summary)
    rounds_to_95  INTEGER,            -- first round reaching 0.95 (NULL if never)
    -- system-attack summaries (NULL when not applicable)
    mal_sel_mean  REAL,               -- mean malicious selection fraction
    mal_infl_mean REAL,               -- mean malicious influence share
    deliv_eff     REAL,               -- mean delivered/invited
    agg_time_ms   REAL,               -- mean per-round aggregation wall-clock
    -- provenance
    git_commit    TEXT,
    torch_version TEXT,
    gpu_name      TEXT,
    started_utc   TEXT,
    ingested_utc  TEXT,
    n_rounds_logged INTEGER
);
CREATE TABLE IF NOT EXISTS configs (
    config_hash TEXT PRIMARY KEY,
    yaml_text   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_runs_group ON runs(dataset, method, attack_type);
CREATE INDEX IF NOT EXISTS idx_runs_name  ON runs(name);
"""


def _hash_config(cfg: dict) -> str:
    """Content hash of a resolved config, excluding volatile fields so that
    two runs differing only by seed/output path map to the same experiment."""
    volatile = {"seed", "seeds", "out_dir", "name"}
    stripped = {k: v for k, v in cfg.items() if k not in volatile}
    blob = json.dumps(stripped, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


@dataclass
class Store:
    root: str = DEFAULT_STORE

    def __post_init__(self):
        os.makedirs(self.root, exist_ok=True)
        os.makedirs(os.path.join(self.root, "rounds"), exist_ok=True)
        os.makedirs(os.path.join(self.root, "configs"), exist_ok=True)
        self.db_path = os.path.join(self.root, "results.db")
        with self._conn() as c:
            c.executescript(SCHEMA)

    def _conn(self):
        return sqlite3.connect(self.db_path)

    # ---------------- ingestion ----------------
    def ingest_tree(self, results_dir: str, verbose: bool = True) -> int:
        """Walk a results/ tree; register every seed dir that has metrics."""
        n = 0
        if not os.path.isdir(results_dir):
            print(f"no such dir: {results_dir}", file=sys.stderr)
            return 0
        for name in sorted(os.listdir(results_dir)):
            exp_dir = os.path.join(results_dir, name)
            if not os.path.isdir(exp_dir):
                continue
            for seed_dir in sorted(os.listdir(exp_dir)):
                sd = os.path.join(exp_dir, seed_dir)
                mpath = os.path.join(sd, "metrics.jsonl")
                if os.path.isdir(sd) and os.path.exists(mpath):
                    try:
                        self._ingest_run(name, sd, verbose)
                        n += 1
                    except Exception as e:
                        print(f"  skip {name}/{seed_dir}: {e}", file=sys.stderr)
        if verbose:
            print(f"ingested {n} run(s) into {self.root}")
        return n

    def _ingest_run(self, name: str, seed_dir: str, verbose: bool) -> None:
        import time
        rows = [json.loads(l) for l in open(os.path.join(seed_dir, "metrics.jsonl")) if l.strip()]
        if not rows:
            raise ValueError("empty metrics")
        meta = {}
        mp = os.path.join(seed_dir, "run_meta.json")
        if os.path.exists(mp):
            meta = json.load(open(mp))
        cfg = meta.get("config_dump")
        if cfg is None:
            cp = os.path.join(seed_dir, "config_resolved.yaml")
            cfg = yaml.safe_load(open(cp)) if os.path.exists(cp) else {}
        seed = int(meta.get("seed", cfg.get("seed", 0)))
        run_id = f"{name}/seed{seed}"
        chash = _hash_config(cfg)

        # store config (dedup)
        with self._conn() as c:
            c.execute("INSERT OR IGNORE INTO configs(config_hash, yaml_text) VALUES (?,?)",
                      (chash, yaml.safe_dump(cfg, sort_keys=False)))
            cfg_path = os.path.join(self.root, "configs", f"{chash}.yaml")
            if not os.path.exists(cfg_path):
                open(cfg_path, "w").write(yaml.safe_dump(cfg, sort_keys=False))

        # build per-round dataframe -> parquet
        df = pd.DataFrame(rows)
        df["run_id"] = run_id
        pq_path = os.path.join(self.root, "rounds", f"{run_id.replace('/', '__')}.parquet")
        # keep list/dict columns as JSON strings so parquet stays flat+typed
        for col in df.columns:
            if df[col].apply(lambda v: isinstance(v, (list, dict))).any():
                df[col] = df[col].apply(lambda v: json.dumps(v) if isinstance(v, (list, dict)) else v)
        df.to_parquet(pq_path, index=False)

        # headline metrics
        evals = [(r["round"], r.get("test_acc"), r.get("test_loss"))
                 for r in rows if "test_acc" in r]
        final_acc = evals[-1][1] if evals else None
        final_loss = evals[-1][2] if evals else None
        accs = [a for _, a, _ in evals if a is not None]
        best_acc = max(accs) if accs else None
        acc_last10 = sum(accs[-10:]) / len(accs[-10:]) if accs else None
        r95 = next((rd for rd, a, _ in evals if a is not None and a >= 0.95), None)

        def _mean(key):
            vals = [r[key] for r in rows if r.get(key) is not None]
            return sum(vals) / len(vals) if vals else None

        def _sel_frac():
            fr = []
            for r in rows:
                ms, hs = r.get("malicious_selected"), r.get("honest_selected")
                if ms is not None and hs is not None and (ms + hs) > 0:
                    fr.append(ms / (ms + hs))
            return sum(fr) / len(fr) if fr else None

        def _deliv():
            fr = []
            for r in rows:
                inv, dlv = len(r.get("invited", [])), len(r.get("delivered", []))
                if inv:
                    fr.append(dlv / inv)
            return sum(fr) / len(fr) if fr else None

        agg = _mean("agg_time_s")
        rec = {
            "run_id": run_id, "name": name, "seed": seed, "config_hash": chash,
            "dataset": cfg.get("data", {}).get("dataset"),
            "model": cfg.get("model", {}).get("name"),
            "method": cfg.get("method", {}).get("name"),
            "attack_type": cfg.get("attack", {}).get("attack_type"),
            "malicious_frac": cfg.get("attack", {}).get("malicious_fraction"),
            "num_uavs": cfg.get("uav", {}).get("num_uavs"),
            "rounds": cfg.get("train", {}).get("rounds"),
            "high_mob_frac": cfg.get("uav", {}).get("high_mobility_fraction"),
            "final_acc": final_acc, "final_loss": final_loss,
            "best_acc": best_acc, "acc_last10": acc_last10, "rounds_to_95": r95,
            "mal_sel_mean": _sel_frac(),
            "mal_infl_mean": _mean("malicious_influence_share"),
            "deliv_eff": _deliv(),
            "agg_time_ms": agg * 1000 if agg is not None else None,
            "git_commit": meta.get("git_commit"),
            "torch_version": meta.get("torch_version"),
            "gpu_name": meta.get("gpu_name"),
            "started_utc": meta.get("started_utc"),
            "ingested_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "n_rounds_logged": len(rows),
        }
        cols = ",".join(rec.keys())
        qs = ",".join("?" * len(rec))
        with self._conn() as c:
            c.execute(f"INSERT OR REPLACE INTO runs({cols}) VALUES ({qs})", list(rec.values()))
        if verbose:
            fa = f"{final_acc:.4f}" if final_acc is not None else "n/a"
            print(f"  + {run_id:42s} {rec['method'] or '?':10s} "
                  f"{rec['dataset'] or '?':9s} {rec['attack_type'] or 'none':22s} acc={fa}")

    # ---------------- query ----------------
    def query(self, where: str = "", params: tuple = ()) -> pd.DataFrame:
        sql = "SELECT * FROM runs"
        if where:
            sql += f" WHERE {where}"
        sql += " ORDER BY name, seed"
        with self._conn() as c:
            return pd.read_sql_query(sql, c, params=params)

    def rounds(self, run_id: str) -> pd.DataFrame:
        pq = os.path.join(self.root, "rounds", f"{run_id.replace('/', '__')}.parquet")
        return pd.read_parquet(pq)

    def config_of(self, run_id: str) -> dict:
        df = self.query("run_id = ?", (run_id,))
        if df.empty:
            raise KeyError(run_id)
        chash = df.iloc[0]["config_hash"]
        return yaml.safe_load(open(os.path.join(self.root, "configs", f"{chash}.yaml")))

    def groupby_experiment(self, where: str = "", params: tuple = ()) -> pd.DataFrame:
        """Aggregate seeds -> mean/std per experiment name. The paper-table view."""
        df = self.query(where, params)
        if df.empty:
            return df
        g = df.groupby("name")
        out = g.agg(
            dataset=("dataset", "first"), method=("method", "first"),
            attack=("attack_type", "first"), n_seeds=("seed", "count"),
            final_acc_mean=("final_acc", "mean"), final_acc_std=("final_acc", "std"),
            best_acc_mean=("best_acc", "mean"),
            acc_last10_mean=("acc_last10", "mean"),
            rounds_to_95_mean=("rounds_to_95", "mean"),
            mal_sel=("mal_sel_mean", "mean"), mal_infl=("mal_infl_mean", "mean"),
            deliv_eff=("deliv_eff", "mean"), agg_ms=("agg_time_ms", "mean"),
        ).reset_index()
        return out.round(4)


def main():
    ap = argparse.ArgumentParser(description="TrustFL-UAV results store")
    ap.add_argument("--store", default=DEFAULT_STORE)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_ing = sub.add_parser("ingest"); p_ing.add_argument("--results", default="./results")
    p_ls = sub.add_parser("list")
    p_ls.add_argument("--dataset"); p_ls.add_argument("--method"); p_ls.add_argument("--attack")
    p_grp = sub.add_parser("groups")
    p_grp.add_argument("--dataset"); p_grp.add_argument("--method"); p_grp.add_argument("--attack")
    p_show = sub.add_parser("show"); p_show.add_argument("run_id")
    args = ap.parse_args()

    store = Store(args.store)
    if args.cmd == "ingest":
        store.ingest_tree(args.results)
    elif args.cmd in ("list", "groups"):
        conds, params = [], []
        for f in ("dataset", "method", "attack"):
            v = getattr(args, f, None)
            if v:
                col = "attack_type" if f == "attack" else f
                conds.append(f"{col} = ?"); params.append(v)
        where = " AND ".join(conds)
        if args.cmd == "list":
            df = store.query(where, tuple(params))
            cols = ["run_id", "method", "dataset", "attack_type", "final_acc",
                    "best_acc", "acc_last10", "mal_infl_mean", "agg_time_ms"]
            print(df[cols].to_string(index=False) if not df.empty else "no runs")
        else:
            print(store.groupby_experiment(where, tuple(params)).to_string(index=False))
    elif args.cmd == "show":
        df = store.query("run_id = ?", (args.run_id,))
        if df.empty:
            print("not found"); return
        for k, v in df.iloc[0].items():
            print(f"  {k:18s} {v}")


if __name__ == "__main__":
    main()
