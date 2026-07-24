"""Apply paper-provenance tags to the results store.

Reads configs/canonical_runs.yaml (the auditable blessed-run manifest) and
writes a `paper_tag` column into the store's runs table: either the claim
group an experiment belongs to (e.g. 'claim1_honest_outlier') or
'exploratory'. Paper tables then filter to paper_tag != 'exploratory', so
diagnostics, superseded runs, and smoke tests can never leak into the paper.

CLI:
    python -m trustfl.provenance apply        # tag the store
    python -m trustfl.provenance audit        # show tag assignment + gaps
"""
from __future__ import annotations

import argparse
import os
import sqlite3

import yaml

from .store import Store, DEFAULT_STORE

MANIFEST = "configs/canonical_runs.yaml"


def _flatten_canonical(manifest: dict) -> dict:
    """name -> claim_group, walking the nested canonical section."""
    tags = {}
    canon = manifest.get("canonical", {})
    for claim_group, body in canon.items():
        _collect(body, claim_group, tags)
    return tags


def _collect(node, claim_group, tags):
    if isinstance(node, dict):
        for k, v in node.items():
            if isinstance(v, str):
                tags[k] = claim_group          # leaf: name -> justification
            else:
                _collect(v, claim_group, tags)  # nested subgroup (dataset/attack)


def apply_tags(store_root: str, manifest_path: str) -> None:
    store = Store(store_root)
    manifest = yaml.safe_load(open(manifest_path))
    canonical = _flatten_canonical(manifest)

    with sqlite3.connect(store.db_path) as c:
        cols = [r[1] for r in c.execute("PRAGMA table_info(runs)").fetchall()]
        if "paper_tag" not in cols:
            c.execute("ALTER TABLE runs ADD COLUMN paper_tag TEXT")
        names = [r[0] for r in c.execute("SELECT DISTINCT name FROM runs").fetchall()]
        tagged, exploratory = 0, 0
        for name in names:
            tag = canonical.get(name, "exploratory")
            c.execute("UPDATE runs SET paper_tag=? WHERE name=?", (tag, name))
            if tag == "exploratory":
                exploratory += 1
            else:
                tagged += 1
    print(f"tagged {tagged} canonical experiment name(s), "
          f"{exploratory} exploratory, across {len(names)} total")


def audit(store_root: str, manifest_path: str) -> None:
    store = Store(store_root)
    manifest = yaml.safe_load(open(manifest_path))
    canonical = _flatten_canonical(manifest)
    df = store.query()
    if df.empty:
        print("store empty"); return
    if "paper_tag" not in df.columns:
        print("no paper_tag yet — run `apply` first"); return

    print("=== CANONICAL (in paper) ===")
    canon_df = df[df["paper_tag"] != "exploratory"]
    for tag in sorted(canon_df["paper_tag"].dropna().unique()):
        sub = canon_df[canon_df["paper_tag"] == tag]
        names = sub["name"].nunique()
        seeds = len(sub)
        print(f"  {tag:28s} {names:2d} experiments, {seeds:2d} runs")
    print(f"\n  total canonical: {canon_df['name'].nunique()} experiments, "
          f"{len(canon_df)} runs")

    print("\n=== EXPLORATORY (excluded) ===")
    exp_df = df[df["paper_tag"] == "exploratory"]
    print(f"  {exp_df['name'].nunique()} experiments, {len(exp_df)} runs excluded")

    # gap check: manifest names that aren't in the store
    store_names = set(df["name"].unique())
    missing = [n for n in canonical if n not in store_names]
    if missing:
        print("\n=== MANIFEST GAPS (blessed but not in store — not yet run) ===")
        for n in missing:
            print(f"  MISSING: {n}  ({canonical[n]})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default=DEFAULT_STORE)
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("cmd", choices=["apply", "audit"])
    args = ap.parse_args()
    if args.cmd == "apply":
        apply_tags(args.store, args.manifest)
    else:
        audit(args.store, args.manifest)


if __name__ == "__main__":
    main()
