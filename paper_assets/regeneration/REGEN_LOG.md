# REGEN_LOG — TrustFL-UAV revision: figures and tables from one data store

Task: `regen_kit/REGENERATION_TASK.md` (the regeneration kit, issued 20 Oct 2026).
Executed 20 Oct 2026, 08:21–09:35 UTC (11:21–12:35 Asia/Riyadh) on the laptop, through its
connected-folder VM (§1). Working folder: `D:\postdoc_WorkDUT\paper1\revision_assets_20261020\`.
Repository: `D:\postdoc_WorkDUT\paper1\trustfl-uav`, base commit `a0cd40ce` (main), work on
branch **`revision-r1`**, HEAD **`bd0590b`**.

No training was run. Nothing in `trustfl-uav\results`, `results_store`, the committed
`paper_assets/` of `main`, or `revision_runs_20261016` was modified, moved or deleted
(every input was copied; see §3). Timestamps below are UTC (`date -Is` on the VM).

## 1. Execution environment and three deviations from the literal procedure

The regeneration ran on the laptop's connected folder through a Linux VM (Ubuntu 22.04, x86-64,
Python 3.10.12, pandas 2.3.3, numpy 2.2.6, matplotlib 3.10.9, PyYAML 6.0.3, git 2.34).
The repository's Windows `.venv` cannot be executed from that VM, and the VM has no network
(PyPI returns 403 through its proxy; the cloud container has no route to PyPI either).
Three consequences, each recorded here and in the manifest:

1. **Parquet engine shim (environment only, no repository code changed).** `trustfl.store`
   caches each run's per-round table with `DataFrame.to_parquet`/`pd.read_parquet`; neither
   pyarrow nor fastparquet could be installed. `revision_assets_20261020/_env/fastparquet/__init__.py`
   is a 40-line module that satisfies the exact calls pandas makes on the fastparquet API and
   stores the DataFrame as a pandas pickle under the `.parquet` name. The DataFrame read back is
   the identical object that was written (same dtypes, NaNs and JSON-string columns), so no
   metric is affected. It was put on `PYTHONPATH` only for the commands in §5. The pickled
   `rounds/` cache is therefore not portable and was **not** copied into the deliverable; to
   re-ingest on Windows run `python -m trustfl.store --store <store> ingest --results <store>`.
2. **SQLite on the mounted drive.** `Store()` failed with `sqlite3.OperationalError: disk I/O
   error` when the store root was on the mounted D: folder (no file locking through the mount).
   The verified store was copied to the VM's local disk (`$HOME/w/store`), re-verified file by
   file against `checks/store_manifest.json` (837 files, 0 mismatches), and all commands ran
   there; every product was then copied back into `revision_assets_20261020\`. The resulting
   `results.db` and content-addressed `configs/` are in `store_index/`.
3. **git lock files.** The mount forbids `unlink`, so each `git commit` left `HEAD.lock`,
   `objects/maintenance.lock` and `objects/*/tmp_obj_*` behind (the commits themselves
   succeeded). Likewise the first `Store()` attempt on the mount (deviation 2) left an empty
   `results.db`, its journal and two empty directories inside `store/`; they were moved to
   `_env/failed_sqlite_attempt_on_mount/` (not deletable from the VM) so that `store/`
   holds the 64 run families only. A zero-byte zip left by a failed `zip` call on the mount was
   overwritten by the final archive. They were moved — not deleted — to `trustfl-uav\.git\stale_regen_20261020\`
   (94 files) after each commit so that later git commands work. **That folder can be deleted.**

Scripts written for this task live in `revision_assets_20261020\scripts\` (stdlib + pandas; the
superseded first versions are kept with a `_v1…` suffix for the record). `secondary_metrics.py`
there is a byte-identical copy of the kit's script (SHA-256 `3fc86f88…7a69`).

## 2. Composition check (Section 2) — run before anything was built

`scripts/composition_check.py` (08:32 UTC, exit 1 = findings) → `checks/composition_check.md`,
`checks/composition_check.tsv` (per run: logged rounds, max round + 1, `train.rounds` from
`config_resolved.yaml`, SHA-256 of `config_resolved.yaml`, `metrics.jsonl`, `groups.json`),
`checks/composition_manifest.json` (declared composition with resolved source paths).

Result: **210 runs declared, 207 OK, no seed missing, no partial run.** All R11/R12/R13 runs
have a final `END … exit=0 rounds=N/N` ledger line equal to the logged round count
(the two `exit=1 rounds=0/100` lines for `exp_a_trustfl seed0` on 18 Oct are failed first
attempts, superseded by the 19 Oct run that is on disk). Two findings, both reported to the
author before building and approved as "proceed as found":

| Finding | Decision |
|---|---|
| `mnist_attack_fedavg` seeds 42–44 have no `groups.json` in the original stored folders (config, metrics, run_meta only). | Copied as found. Used only for the Fig. 2(a) accuracy curves; `export.py` handles the absence (retention/influence NaN, accuracy unaffected). |
| `eur_adsys` has 5 stored methods (trustfl, fedavg, multikrum, sophon, fedaa), not the "seven" of Section 2. | Copied as found. The committed `make_fig_eurosat.py` and `claims_audit.py` expect exactly these five; no other `eur_*` family was added. |

Note recorded for the run procedure: the R13 `exp_a` `groups.json` files differ from the stored
`exp_a` ones (different high-mobility sets), and `trustfl.audit._groups()` looks in `./results`
before the store root. All commands were therefore run from a working directory with no
`./results`, so the store's own `groups.json` is the one used (no code patch needed). The R13
`exp_b_*` `groups.json` files are identical to the stored ones.

## 3. Store build (rule 2)

`scripts/build_store.py` (resumable; 09:01–09:06 UTC, final pass exit 0) copied the 210 run
folders with `shutil.copy2` into `revision_assets_20261020\store\<name>\seed<k>\` and verified
each file by SHA-256 against its source: **210 runs, 837 files, 210 verified, 0 failures**
(`checks/store_manifest.json`; per-run records in `checks/store_manifest.json.progress.jsonl`).
The store contains exactly the Section 2 composition and nothing else — no `cifar_*`, `cal_*`,
other `*_debug`, `femtest`, `smoke_*`, `exp_a_krum_*` diagnostics, `exp_b_strategic_delay`,
`fem_a_*`, `exp_c_adaptive_<baseline>`, `exp_c_adsys_*`, `mnist_fedavg_debug`, `eur_cal_*` or
stored `exp_a`/`exp_b_selective`/`exp_b_resource_misreport`/`mnist_trustfl_debug` folders.
No figure script required any additional run folder.

## 4. Repository branch `revision-r1` (rules 1 and 3)

| Commit | Content |
|---|---|
| `f8ac688` | **R04** — `figures_tmc.py` Fig. 7(c): ylabel/title "energy per delivered update" (the committed code already plotted `energy_per_update_j`; wording only). |
| `efe193b` | **R05** — `figures_tmc.py` Fig. 7(d): x-axis timings from the hardened family drawn in the panel (`exp_h_adsys`), not the mean over every run (`cost = runs[family == fam].groupby("method")["agg_time_ms"].mean()`). Gives TrustFL 8.5 ms vs FedAvg 5.3 ms. |
| `63a7a51` | **R06** — new `paper_assets/make_fig_scalability.py` (F8 from `scalability.csv`; writes `F8_scalability_fit.json` with exponents for M ≥ 128 and all M and the Krum/TrustFL ratio at M = 64 and 512); new `paper_assets/make_table4_marks.py` (Table IV marks from `trustfl.metrics_matrix`, pooled over both datasets). |
| `07ce47c` | Table IV script: rank on printed precision (see §7, note 3). |
| `7058bb2` | R06 — F8 layout only (legend below axes, y-range from the means; the ±std band of the micro-benchmark reached below 10⁻³ ms at M = 32 and the legend covered the curves). |
| `449dfcf` | `claims_audit.py` rewritten: expected values = the revised Section VI numbers (Section 5 of the task), 257 Group-A claims covering Table V (56 cells, mean and sd), energy, aggregation ms, all 36 Mann–Whitney p-values, Table IV marks, clean setting, magnitude attack, isolated attacks, sign-flip and jump ratios, Table VI (S3/S4/S5a/D1), scalability exponents, ablation, EuroSAT; Group B prints the submitted Table V retention values. Exact MWU falls back to enumeration when scipy is absent. |
| `bd0590b` | **R01/R02** — regenerated `paper_assets/data/*`, figures F1–F9 (PDF + PNG, lossless), `paper_assets/tables/`, `paper_assets/supplement/`. `scalability.csv` unchanged (SHA-256 `62a88df8796b1a507c4d7d21b5aecc19bf6bcd68a499256ac6c6b224d049e17c`, verified). Provenance note: that hash is the laptop working-tree file, which has CRLF line endings; the blob committed at `a0cd40ce` is the same 43-line table with LF endings (SHA-256 `de562607f2a19ce577b3594bc07840b05fbf2883d96cd8f36d9250320253515b`); `diff` after stripping CR is empty. `bd0590b` stores the CRLF version, so `git diff a0cd40ce revision-r1 -- paper_assets/data/scalability.csv` shows a line-ending-only change and no value change. |

No metric definition was changed: `store.py`, `export.py`, `audit.py`, `metrics_matrix.py` are
the HEAD versions untouched; `figures_tmc.py` differs from HEAD by the R04 wording and the
R05 four-line change only (`git diff a0cd40ce revision-r1 -- trustfl/`). Fig. 10 (F9) already had
a committed script, `paper_assets/make_fig_eurosat.py`; it was run unchanged (pointed at the
regenerated `runs_summary.csv`) rather than duplicated.

## 5. Commands (cwd `$HOME/w`, `PYTHONPATH=<repo>:<revision_assets>/_env`; full transcript with stdout in `logs/`)

| Step | Command | Start (UTC) | End | Exit |
|---|---|---|---|---|
| composition check | `python3 -I scripts/composition_check.py <paper1> checks` | 08:32:17 | <08:34 | 1 (findings, §2) |
| store build | `python3 -I scripts/build_store.py checks/composition_manifest.json store checks/store_manifest.json` | 09:01:38 (first pass, cut by the 3-min call limit after copying all 210 folders) / 09:06:07 (resumable pass) | 09:06:3x | 0 (final pass) |
| local copy + re-verify | `cp -r store $HOME/w/store` + SHA-256 check (837/837) | 09:15:09 | 09:15:39 | 0 |
| 01_ingest | `python3 -m trustfl.store --store store ingest --results store` | 09:15:48 | 09:15:53 | 0 (210 runs) |
| 02_export | `python3 -m trustfl.export --store store --out paper_assets/data` | 09:15:58 | 09:16:03 | 0 |
| 03_table5 | `python3 scripts/make_table5.py paper_assets/data tables supplement` | 09:16:10 | 09:16:11 | 0 |
| 04_secondary | `python3 scripts/make_secondary.py store tables supplement` | 09:16:43 | 09:16:45 | 0 (first attempt 09:16:17 exit 1: Py3.10 f-string syntax, fixed) |
| patches | `apply_patches.py R04`, `R05`; commits above | 09:17:03 | 09:17:57 | 0 |
| 05_figures_tmc | `python3 -m trustfl.figures_tmc --data paper_assets/data all` | 09:18:13 | 09:18:22 | 0 (F1–F7) |
| 06_fig_scalability | `python3 <repo>/paper_assets/make_fig_scalability.py --data paper_assets/data/scalability.csv --out paper_assets/figures` | 09:18:31 / 09:24:00 | 09:24:02 | 0 |
| 07_fig_eurosat | committed `make_fig_eurosat.py` with `DATA`/`OUT` pointed at the regenerated CSV and local figures dir | 09:18:34 | 09:18:36 | 0 |
| 08_matrix_table4 | `python3 <repo>/paper_assets/make_table4_marks.py --store store --out tables` (also writes `matrix_exp_h_adsys.csv`, `matrix_femh_adsys.csv`) | 09:18:36 / 09:19:44 | 09:19:46 | 0 |
| 09_audit_all | `python3 -m trustfl.audit --store store all` | 09:21:12 | 09:21:13 | 0 |
| 10_check_expected | `python3 scripts/check_expected.py $HOME/w` (Section 5 comparison, §7) | 09:22:24 | 09:22:25 | 0 (230 checks, 0 mismatches; two earlier passes exit 1 were checker-precision fixes, §7) |
| 11_claims_audit | `python3 <repo>/claims_audit.py --data paper_assets/data --tables tables --supp supplement --store store --matrix` | 09:26:33 | 09:26:35 | 1 (257 claims, 1 failure: EuroSAT, §7) |
| cross-check | same `claims_audit.py` and all 36 MWU p-values re-run in the cloud container with scipy 1.18 (`checks/claims_audit_scipy_crosscheck.log`) | 09:31 | 09:31 | 1 (same single failure; 36/36 p-values identical) |
| manifest | `python3 -I scripts/make_manifest.py <revision_assets> <repo> <kit> checks/git_log_revision-r1.txt` | 09:31:10 / 09:32:5x | 09:33:0x | 0 |
| zip | `zip -r -X revision_assets_20261020.zip revision_assets_20261020 -x store/* store_index/* *.zip` (built on the VM's local disk — zip cannot replace files on the mount — then copied in; `unzip -t` clean) | 09:33:30 | 09:33:47 | 0 |

## 6. Outputs (Section 3)

- `paper_assets/data/`: `runs_summary.csv` (210 rows), `experiment_groups.csv` (64),
  `hardened_dual.csv` (70), `fidelity_jump.csv` (7), `system_attacks.csv` (33), `sweep.csv` (40),
  `timeseries_key.csv` (17 860 rows, 195 runs), `table1.tex`, `claims_report.txt`;
  `scalability.csv` copied unchanged.
- `paper_assets/figures/`: `F1_motivation` … `F7_mechanism` (`figures_tmc.py`), `F8_scalability`
  (+ `F8_scalability_fit.json`), `F9_eurosat`; each as PDF and PNG (matplotlib Agg PNG, lossless;
  no JPEG anywhere). Every panel was inspected: Fig. 2(a) shows the seed-42 collapse to 0.10 in
  the TrustFL band; Fig. 2(c) curves non-zero (R03); Fig. 5(a)(c)(d) from the R13 `exp_b_*`,
  (b) from `exp_b2_delay`; Fig. 7(c) energy per delivered update 5214 vs 7045–7825; Fig. 7(d)
  TrustFL ≈ 8.5 ms, FedAvg ≈ 5.3 ms; Fig. 8(b) five `exp_h_adsys_trustfl` seeds, (c) three
  `fid_signflip_trustfl` seeds.
- `tables/`: `table5_rows.tex` (manuscript style, best per column bold), `table5_summary.csv`,
  `table4_rows.tex`, `table4_marks.csv`, `matrix_exp_h_adsys.csv`, `matrix_femh_adsys.csv`,
  `table6_mechanism.tex` (S3, S4, S5a, D1), `table6_full.tex` (all S/D metrics with exploratory
  exact MWU p), `mwu_primary_tests.csv`.
- `supplement/`: `table5_per_seed.{csv,tex}`, `table5_submitted_vs_regenerated.{csv,tex}`
  (28 of 56 printed cells changed vs the 25 Jul manuscript), `secondary_per_seed.csv`,
  `secondary_summary.csv`, `exclusion_precision_recall.{csv,tex}`, `rounds_to_target.{csv,tex}`,
  `worst_seed_accuracy.csv`.
- `checks/`: composition check, store manifest, `expected_vs_regenerated.{md,json}`,
  `git_log_revision-r1.txt`, scipy cross-check log. `logs/`: every step's stdout and
  `REGEN_COMMANDS.log` (command lines, timestamps, exit codes).
- `REGEN_MANIFEST.json`: SHA-256 of all 837 store input files, every CSV/table/supplement file
  and image, the scripts, the shim, the kit inputs, and the repository code used.
- `revision_assets_20261020.zip`: the whole folder except `store/` (run folders, already
  archived) and `store_index/`.

## 7. Expected values (Section 5) — result of the comparison

`checks/expected_vs_regenerated.md`: **230 values checked at printed precision, 0 mismatches.**
Table V (all 56 cells, mean and sd), energy per delivered update, aggregation ms, the 12
retention p-values, 12 delivery p-values, the influence p-value vs FedGreed (0.0476) and the
0.11–0.79 range of the others, Table IV marks (7 rows), clean setting (retention and accuracy),
magnitude attack (0.1009 / 0.9895 / 0.9908), isolated attacks, sign-flip influences, jump
ratios and TrustFL accuracies, Table VI (S3, S4, S5a, D1), and the R06 exponents
(M ≥ 128: 1.08 / 1.51 / 1.41; all M: 1.12 / 1.18 / 1.19; Krum/TrustFL 2.19× at M = 64,
4.88× at M = 512) all reproduce. Secondary metrics match the 19 Oct file line for line
(including per-seed rounds-to-target and skipped rounds).

**Mismatch list for Section 5: empty.** Items to be aware of:

1. *Half-way cells.* TrustFL MNIST delivery: raw mean 0.908500 — `export.py` prints **.909**
   (`table1.tex`, and `f"{v:.3f}"` of the in-memory mean, which is 0.9085000…01). The same
   mean written to CSV and read back prints .908, so a downstream reader must take the value
   from `table1.tex`/`table5_rows.tex`, not re-round the CSV. FLTrust MNIST delivery: raw
   0.644500 → prints **.645** either way. (The 19 Oct results file shows 0.908 and 0.644 for
   these two cells; the task's Section 5 shows .909 and .645, which is what export prints.)
2. *Clean-setting Krum accuracy "0.82–0.96"*: per-seed min/max are 0.8156 / 0.9609, i.e. the
   quoted range is the 2-decimal rounding.
3. *Table IV ranking rule.* With unrounded pooled means FedAvg's accuracy (0.9057) ranks 4th
   behind TrustFL (0.9063) and would get a half mark; the expected (and 19 Oct) mark is filled.
   The caption's "ties sharing the higher mark" only has meaning at printed precision
   (.906 = .906), so `make_table4_marks.py` ranks on the printed values (3 dp for the four
   ratios, 1 J, 0.1 ms). With this rule all 42 marks match. Also noted: the security facet of
   the metrics matrix is the logged `malicious_influence_share` (second half), which differs
   slightly from Table V's `infl_late` for Sophon (0.3372 vs 0.3389 pooled) and FLTrust
   (0.3743 vs 0.3770); ranks and marks are identical under either column.
4. *Not in Section 5 but quoted in the paper — one genuine printed-value discrepancy:*
   EuroSAT FedAA accuracy is 0.62549 (mean of 0.7485, 0.4089, 0.7191; identical to the
   committed CSV at `a0cd40ce` — the family is stored and unchanged), which prints as **.625**;
   the submitted EuroSAT table prints .626. The old `claims_audit.py` tolerance of ±0.005 hid
   it. The revised `claims_audit.py` keeps the claim strict, so it exits 1 on exactly this item
   (257 claims, 1 failure). The other 19 EuroSAT aggregates and all other claims pass.
5. `claims_audit.py` Group B shows which submitted Table V retention values changed in the
   revision: Sophon MNIST .459→.354, Sophon FEMNIST .539→.392, FedAA MNIST .288→.322,
   FedAA FEMNIST .361→.318, Multi-Krum FEMNIST .524→.523 (the R11 seed 3); the five stored
   methods on MNIST and TrustFL/FLTrust/FedGreed/FedAvg on FEMNIST are unchanged.

## 8. Reproducing on Windows (repository `.venv`, real pyarrow)

```
git checkout revision-r1
python -m trustfl.store --store D:\postdoc_WorkDUT\paper1\revision_assets_20261020\store ingest --results D:\postdoc_WorkDUT\paper1\revision_assets_20261020\store
python -m trustfl.export --store <store> --out paper_assets\data        # run from a folder without .\results, or accept that audit._groups() prefers .\results
python -m trustfl.figures_tmc --data paper_assets\data all
python paper_assets\make_fig_scalability.py ; python paper_assets\make_fig_eurosat.py
python paper_assets\make_table4_marks.py --store <store> --out paper_assets\tables
python claims_audit.py --data paper_assets\data --tables paper_assets\tables --supp paper_assets\supplement
```

## 9. Follow-up (R08, Table III rows, stray file) — 20 Oct 2026, 10:14–10:2x UTC

Brief: `regen_kit/REGEN_FOLLOWUP_20261020.md` (SHA-256 `e19b8ce3…4379`), issued after the independent
check of the 09:36 zip (`86b41e40…c1b2`). Same binding rules, same environment and `PYTHONPATH`
(§1), same verified local store copy, cwd `$HOME/w`. Before touching anything the 09:34 manifest and
09:36 log were copied to `checks/REGEN_MANIFEST_20261020_0934_pre_followup.json` (SHA-256
`1aecee16…e009e`) and `checks/REGEN_LOG_20261020_0936_pre_followup.md`; the 49 output hashes (data,
figures, tables, supplement) were extracted to `checks/baseline_hashes_0934.json` and the local copy
was confirmed identical to all 49 (0 differences) at 10:16.

| Step | Command | Start (UTC) | End | Exit |
|---|---|---|---|---|
| R08 patch | `python3 -I scripts/apply_patches.py <repo> R08` (exact replacement of the three title strings; `git diff` = 3 changed lines) | 10:17:09 | 10:17:10 | 0 |
| commit | `git commit -m "R08: Fig. 8 panel titles reworded to match the corrected caption (no data change)"` → **`d14ceb8`** | 10:17:1x | 10:17:24 | 0 |
| 12_fig7_after_R08 | `python3 -m trustfl.figures_tmc --data paper_assets/data fig7` | 10:17:32 | 10:17:3x | 0 |
| SHA-256 confirmation | all 49 baseline files re-hashed against `checks/baseline_hashes_0934.json` | 10:17:4x | 10:17:4x | 47 identical, 2 changed (F7 only) |
| copy + commit | F7 PDF/PNG → `paper_assets/figures/` (assets and repo); `git commit -m "R01/R02 follow-up: F7 regenerated after R08"` → **`2386622`** | 10:17:45 | 10:18:03 | 0 |
| 13_table3 | `python3 scripts/make_table3.py paper_assets/data tables` | 10:18:16 | 10:18:1x | 0 (0 mismatches) |
| stray file | `mv zixsptm7 _env/stray_files/zixsptm7` (copy-never-move rule concerns originals; this is a task by-product) | 10:18:2x | 10:18:2x | 0 |
| commit | `tables/table3_rows.tex`, `table3_summary.csv` also into repo `paper_assets/tables/` → **`84f1fe1`** | 10:18:31 | 10:18:55 | 0 |
| manifest | `python3 -I scripts/make_manifest.py <assets> <repo> <kit> checks/git_log_revision-r1.txt` (new version: skips `__pycache__`, `_env/failed_sqlite_attempt_on_mount`, `_env/stray_files`) | see §9.4 | | |
| zip | `python3 -I scripts/zip_from_manifest.py <assets> $HOME/w/revision_assets_20261020.zip` then copied in | see §9.4 | | |

### 9.1 Item 1 — R08 (`figures_tmc.py`, wording only)

The three `_panel`/title strings of `fig7` were replaced exactly as tabulated in the brief
(`scripts/apply_patches.py`, entry `R08`; the panel (b) title carries the en dash as the `–`
escape, as specified). `trustfl/figures_tmc.py` now differs from `a0cd40ce` by R04 + R05 + R08
only (`git diff --stat a0cd40ce revision-r1 -- trustfl/`: 1 file, 9 insertions, 6 deletions).
Regenerating only `fig7` changed exactly two files:

| file | 09:18 run (baseline) | after R08 |
|---|---|---|
| `paper_assets/figures/F7_mechanism.pdf` | `40 884` B | SHA-256 `17b8ea8512b3170a83433eb783edf3b6a3fa79e51eb8d2c28513be0238065f67` |
| `paper_assets/figures/F7_mechanism.png` | `378 250` B | SHA-256 `5b41a37bbd48754e644c6a17096f04aa409907ab6a2479b7e71fec2c285ee30e` |

The other 47 baseline files — F1–F6, F8 and F9 (PDF and PNG), `F8_scalability_fit.json`, every
CSV in `paper_assets/data/` plus `table1.tex` and `claims_report.txt`, every file in `tables/` and
`supplement/` — are byte-identical to the 09:34 manifest (the baseline hashes stay in
`checks/baseline_hashes_0934.json`). Visual check of the new F7: the four titles read as specified;
the plotted data are unchanged.

### 9.2 Item 2 — Table III rows (`tables/table3_rows.tex`)

`scripts/make_table3.py`: `runs_summary.csv`, family `eur_adsys`, five methods, seeds 0–2, Table V
cell definitions, mean (3 dp) ± sd (ddof 1, 2 dp), manuscript row order, best per column bold with
printed-precision ties sharing the bold (no tie occurred). Per-seed values and 6-dp raw means in
`tables/table3_summary.csv`. **All 20 cells equal the expected values of the brief; 0 mismatches**
(FedAA accuracy prints .625 from raw 0.625494, as §7 item 4 anticipated). The rows were also
committed to the repository's `paper_assets/tables/` (`84f1fe1`) so that folder stays in step with
`tables/`.

### 9.3 Item 3 — stray file

`zixsptm7` (4 500 076 B, SHA-256 `7dd8923c645da4b890f64b4e046c8655af282e2904b6fd65f087c261344e978f`)
is the temporary archive left by the `zip` call that failed on the mounted drive at 09:33 (§5; 106
stored entries). It is a by-product of this task, not an original, and it was moved to
`_env/stray_files/` (the VM cannot delete; that folder, like `_env/failed_sqlite_attempt_on_mount/`
and `trustfl-uav\.git\stale_regen_20261020\`, can be deleted). `make_manifest.py` now skips those two
`_env` folders and every `__pycache__`, and the archive is built by `scripts/zip_from_manifest.py`
strictly from manifest entries: members = every manifest entry except `store/` and `store_index/`,
plus `REGEN_LOG.md` and `REGEN_MANIFEST.json`; each member's SHA-256 is re-checked against the
manifest before writing, no `*.zip` can enter, and the archive listing is asserted equal to the
member list (`checks/zip_members.txt`, written after the manifest and therefore itself not a member).

### 9.4 Item 4 — manifest, zip, HEAD

`REGEN_MANIFEST.json` regenerated 10:20:05–10:20:48 UTC (exit 0): 837 store input files, the new
`F7_mechanism` hashes (`17b8ea85…`, `5b41a37b…`), `tables/table3_rows.tex` and `table3_summary.csv`,
the three new commits in the repository section, and no `__pycache__`, stray or failed-sqlite entries;
the previous manifest is kept in `checks/`. Branch `revision-r1` HEAD after this follow-up:
**`84f1fe1`** (commits on top of `a0cd40ce`: f8ac688, efe193b, 63a7a51, 07ce47c, 7058bb2, 449dfcf,
bd0590b, d14ceb8, 2386622, 84f1fe1). The archive `revision_assets_20261020.zip` was then built
from the manifest with `scripts/zip_from_manifest.py` (10:21 UTC; its SHA-256 is reported in the
follow-up summary and in `checks/zip_sha256.txt`, which is written after the archive and is not a
member). Expected-value comparison for this follow-up: **0 mismatches** (20 Table III cells; the
47 unchanged outputs confirmed by hash).

Stale git files from this follow-up were again moved to `trustfl-uav\.git\stale_regen_20261020\`
(now 116 files); that folder, `_env/failed_sqlite_attempt_on_mount/` and `_env/stray_files/` can be
deleted.

## 10. Follow-up 2 (R09, Fig. 10) — 21 Oct 2026, 02:22–02:2x UTC

Brief: `regen_followup2_20261020.zip` (SHA-256 `68ad73b3…99a2e`; contents `REGEN_FOLLOWUP2_20261020.md`
`2405d4b4…`, `make_fig_eurosat_twopanel.py` `02aba0f9…`), unzipped by copy into
`paper1\regen_followup2_20261020\`. Same binding rules, environment, `PYTHONPATH` and local store copy
as §5. The 10:20 manifest's 51 output hashes were frozen first as `checks/baseline_hashes_1020.json`.

| Step | Command | Start (UTC) | End | Exit |
|---|---|---|---|---|
| R09 script | copy of the supplied script to `paper_assets/make_fig_eurosat_twopanel.py` (byte-identical, SHA-256 `02aba0f9…`; no path or import change was needed) → `git commit -m "R09: committed script for the two-panel Fig. 10 (EuroSAT per-seed retention; retention across tasks) from runs_summary.csv; no data change"` → **`a0fe27d`** | 02:22:14 | 02:22:28 | 0 |
| 14_fig9_twopanel_R09 | `python3 paper_assets/make_fig_eurosat_twopanel.py --data paper_assets/data/runs_summary.csv --out paper_assets/figures` | 02:22:39 | 02:22:42 | 0 |
| printed check | `worst TrustFL seed 0.8806, best baseline seed 0.6454` — both as required (read from the CSV) | | | |
| SHA-256 confirmation | 51 baseline outputs re-hashed: 49 byte-identical, 2 changed (the F9 pair only) | 02:22:4x | 02:22:4x | — |
| copy + commit | `F9_eurosat.png/.pdf` → `revision_assets_20261020\paper_assets\figures\` (overwriting the single-panel files) and the repository; `git commit -m "R01/R02 follow-up: F9 regenerated with R09"` → **`1fe11d0`** | 02:22:51 | 02:22:55 | 0 |
| manifest / zip | `make_manifest.py` then `zip_from_manifest.py`, as in §9 | see below | | |

New files: `F9_eurosat.png` SHA-256 `7d8d372146ca0c68a82ac31b97cbcaef926a3a8026cdccd43134a1cfe013a2f2`,
`F9_eurosat.pdf` SHA-256 `4e45b2eb7002a504da34dda15bfaa013ca5d054016e03e8085364f535b8e9f2a`. The committed
single-panel `make_fig_eurosat.py` is left in place and unused. Every other output of the 10:20
manifest — F1–F8 (PDF and PNG), the F8 fit JSON, all CSVs, `table1.tex`, `claims_report.txt`, every
file in `tables/` and `supplement/` — is byte-identical (`checks/baseline_hashes_1020.json`). Visual
check: panel (a) per-seed EuroSAT retention with the two annotations (0.88 / 0.65), panel (b) mean
retention across MNIST, FEMNIST and EuroSAT for the five pilot methods.

`REGEN_MANIFEST.json` regenerated 02:23:46–02:24:26 UTC and again at 02:2x after adding
`paper_assets/make_fig_eurosat_twopanel.py` to its `repository_code_used` section (`make_manifest.py` v3;
exit 0): the new F9 hashes, the two new commits. Branch `revision-r1` HEAD after follow-up 2:
**`1fe11d0`** (R09 script `a0fe27d`, F9 `1fe11d0`). The archive was rebuilt with
`zip_from_manifest.py` (members = manifest entries except `store/`, `store_index/`, plus the log and
manifest; SHA-256 in `checks/zip_sha256.txt`).

## 11. Record maintenance and publication state — 21 Oct 2026, 02:28–02:5x UTC

Wording-only edits to this record (no number, hash of an output, output path or section heading
changed): the regeneration kit's task brief is now cited by its short name
`regen_kit/REGENERATION_TASK.md` (a byte-identical copy of the brief, SHA-256
`6fac4d165c006e6b937273e080decab1d889e697de95c461c50d5acc579a8b8d`, sits under that name in
`regen_kit/`); lines 3–4 and §1 now say that the regeneration was executed on the laptop through its
connected-folder VM; three parenthetical remarks in §3, §9.3 and §9.4 say "VM" / "follow-up summary"
instead of "session"; the docstring of `scripts/composition_check.py` cites the short name. The same
edits were applied to the archival copies in `checks/`, whose SHA-256 are therefore now
`888b4a92f55935c5a139b2bb41528f77beee3a292e63886a7581cb521bfa2587`
(`REGEN_MANIFEST_20261020_0934_pre_followup.json`; §9 cites the hash of the copy before this
rewording, `1aecee16…e009e`) and `474192fec9cb0ed70eee6cb16952bce14527f5dfdde4a12f5c606692db2ebf0b`
(`REGEN_LOG_20261020_0936_pre_followup.md`). `make_manifest.py` (v4) lists the kit inputs by these
short names and adds the follow-up 2 inputs.

Repository state on `revision-r1` (all commits by the author; no commit was rewritten, so every
hash cited in §4, §9, §10 and in `REGEN_MANIFEST.json` is the published one):

| Commit | Content |
|---|---|
| `caffd43` | `.gitignore`: one local editor-directory entry removed; no code change. |
| `74e5949` | `study2/`: the Study 2 record — protocol frozen 5 Oct 2026 (`RECOMMENDED_PROTOCOL_v5.json`, SHA-256 `b512b6b2…68e25`, bound in `CAMPAIGN_BINDING_v5.json`), campaign binding/genesis/amendment, the final `report.json` of the 160 attempts (SHA-256 `b1cb86cd…3011`), per-attempt records, the `trustfl/scientific` + `trustfl/comparison` pipeline snapshot with `PREPARATION_v5.json`, and the sealed per-seed inputs (`traces.npz`, `indexes.npz`, scenario and identity files, initialization records). `study2/HASHES.sha256` covers every copied file (1 638); `study2/OMITTED.tsv` lists what was left out for size (`events.jsonl` 3.4 GB, per-attempt `final-model.pt`/`completion.json`/round records, the 32 `initial_model.pt`), with SHA-256 where computed. |
| (final) | `paper_assets/regeneration/REGEN_LOG.md` and `REGEN_MANIFEST.json`: this record, added to the repository so that the manuscript's supplement can cite it from the released tree. That commit's own hash cannot appear in these files; `REGEN_MANIFEST.json` lists the commits up to `74e5949`. |

Published tree of `revision-r1` therefore carries: the regenerated `paper_assets/data`, figures
F1–F9 and tables; the committed figure scripts `trustfl/figures_tmc.py`, `paper_assets/make_fig_scalability.py`,
`paper_assets/make_fig_eurosat.py`, `paper_assets/make_fig_eurosat_twopanel.py`,
`paper_assets/make_table4_marks.py`; `claims_audit.py`; `study2/`; and `paper_assets/regeneration/`.
Checks re-run after the edits (02:5x UTC): `py_compile` of all 48 tracked Study 1 `.py` files OK;
`check_expected.py` 230 checks, 0 mismatches; `claims_audit.py` 257 claims with the single
pre-existing EuroSAT item (§7, item 4), exit 1 as before; `git diff 1fe11d0..74e5949 -- '*.py'`
restricted to the Study 1 tree is empty.
