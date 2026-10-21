# Study 2 — paired common-policy comparison (v5 campaign, October 2026)

This directory carries the record of the second study of the revised manuscript: a paired,
equal-policy comparison of TrustFL with FedAvg, FLTrust (comparison identity
`fltrust_declared_root_fallback`), FedGreed (`fedgreed_algorithm1`) and FedAA
(`fd_distance_validation_local`) on EMNIST-byclass and EuroSAT under the common
invitation policy, 30 UAVs, 60 rounds, honest and 40 % adaptive-system conditions,
16 seeds (10 EMNIST, 6 EuroSAT), 160 attempts in all. Every attempt was executed by the
sealed-input pipeline in `code/trustfl/scientific`; nothing here was produced by the Study 1
(`trustfl/`) code path.

| What | Where | Provenance |
|---|---|---|
| Protocol, frozen 5 October 2026 before any attempt ran | `protocol/RECOMMENDED_PROTOCOL_v5.json` (SHA-256 `b512b6b21cfb8a706ba855e34cbe07a52f3f5e75e79e7b97dd5f2b7a77068e25`, the value bound in `campaign/CAMPAIGN_BINDING_v5.json`), `protocol/RECOMMENDED_PROTOCOL_v5.md` | copied from the transfer package |
| Campaign binding, genesis, resource amendment | `campaign/CAMPAIGN_BINDING_v5.json`, `campaign/genesis.json`, `campaign/resource-amendment-264h.json` | copied from the campaign root |
| Final report of the 160 attempts | `campaign/report.json` (SHA-256 `b1cb86cd8afb5237091b5cf8ddc229dd1cc010f8c9e2f241c9507affd09d3011`, the value recorded in the collection certificate `COMPLETE.json`) | copied from the final collection |
| Per-attempt records (160 × `attempt.json`, `environment.json`, `execution-summary.json`, `final-evaluation.json`, `final-model-identity.json`, `groups.json`, `import-origins.json`) | `campaign/attempts/<attempt>/` | copied from the campaign |
| Pipeline code that ran the campaign: `trustfl/scientific` (incl. `analysis.py`, the endpoint analysis, and `protocol.py`) and `trustfl/comparison` | `code/trustfl/`, with `code/PREPARATION_v5.json` (per-file SHA-256 of the prepared tree) | snapshot of the prepared v5 working tree (base commit `2b4988d` plus the staged v5 preparation patch, as recorded in `PREPARATION_v5.json`) |
| Sealed per-seed inputs: data scenario, source identity, sealed index arrays (`indexes.npz`) and sealed randomness traces (`traces.npz`) | `sealed_inputs/data/<dataset>-s<seed>/`, `sealed_inputs/DATA_MANIFEST_v5.json` | copied from the staging area |
| Initialization records per seed and condition (`environment.json`, `scenario.json`, `source_identity.json`, `traces.npz`, preparation status, `initialization_complete.json` with the bundle ids and model hashes) | `sealed_inputs/initializations/<dataset>-s<seed>/<condition>/` | copied from the staging area |

`HASHES.sha256` lists the SHA-256 of every file in this directory as copied; `OMITTED.tsv` lists
what was deliberately not committed because of size, with SHA-256 where it was computed:

- `campaign/events.jsonl` (3.4 GB, the campaign event stream);
- the 160 per-attempt `final-model.pt` (6.8 MB each), `completion.json` (hashed) and the 60
  per-round `round-000NN.json` records of each attempt (9,600 files; the three endpoints in
  `report.json` are recomputable from them);
- the 32 `initial_model.pt` files (one model initialization per seed, 6.8 MB each; their raw and
  tensor hashes are in each `initialization_complete.json`) and the duplicate `indexes.npz` under
  `initializations/` (byte-identical to `sealed_inputs/data/<seed>/indexes.npz`).

These omitted files are archived with the campaign on the authors' machine and are available on
request; everything needed to check the protocol, the sealed inputs and the reported endpoints
against the per-attempt records is in this directory.
