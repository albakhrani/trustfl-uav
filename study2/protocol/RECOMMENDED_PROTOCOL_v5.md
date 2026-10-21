# Recommended scientific protocol v5

5 October 2026 · IoT-71387-2026 · revision deadline 27 October 2026. **Decision-ready proposal, not an executable configuration or authorization to run.** The original project is `/Users/mac/Desktop/Post_docDUT/paper1/trustfl-uav`. No scientific source, manuscript, model, optimizer or historical evidence was changed/executed during this preparation.

## Decision

Recommend the existing fallback's scientific objectives in a prospectively declared **sampled-client, common-policy regime**: EMNIST-byclass with at most 512 examples/client and a 620-example balanced public root; EuroSAT with at most 256 examples/client and its existing 210-example root. Keep five methods, honest and adaptive-system 40% conditions, 60 rounds, 10 EMNIST seeds and six EuroSAT seeds. This is 160 attempts and 9,600 rounds. Retain the strong distance/finite-difference local comparator. No difficult method or seed is removed to fit a favorable outcome.

This recommendation targets roughly 7–10 continuous compute days on Ali's 24-hour/day Windows laptop. The chosen conditional planning case is **202.317 hours**, plus a separate **12-hour unmeasured lifecycle allowance**, or **214.317 hours / 8.93 days**. Maximum FedGreed prefix plus a stated FD latency sensitivity gives **235.968 hours / 9.83 days including that allowance**. No-root-saving and much wider cross-method stress cases are **262.009 hours / 10.92 days** and **427.338 hours / 17.81 days**. These are assumptions and sensitivities, not confidence intervals, upper bounds or a promise. The protocol is a plausible target, not an affordability certificate.

Only one contingency is proposed: if the fixed scope cannot fit resource/calendar criteria, retain it and obtain an author decision on an extension or additional resources. Do not silently shrink seeds, methods, rounds, root/data definitions or tests, and do not reset allowances. Larger-node experiments, observer hardware, analysis, writing and submission are separate work; the compute target does not certify completion of all 17 comments.

## Fixed cells and claims

| Dataset/role | Conditions | Methods | Seeds | Rounds | Attempts |
|---|---|---|---|---:|---:|
| EMNIST-byclass / confirmatory core | honest; adaptive40=`adaptive_system` | TrustFL, FedAvg, FLTrust declared root fallback, FedGreed Algorithm 1, FD local | 41000–41009 | 60 | 100 |
| EuroSAT / descriptive satellite-image extension | same | same | 42000–42005 | 60 | 60 |

All cells use common invitations/public resource requests, N=30, K=12. `FIXED_ATTEMPT_MANIFEST_v5.json` enumerates all 160 IDs and the exact execution order. The first prospective attempt is **EMNIST, honest, FD, seed 41000**. Its data are retained as one of the 160 observations; it is not another disposable smoke/cost campaign. No numerical command is supplied in this handoff.

The estimand is the difference between complete defenses under shared opportunities and public requests, with each method's endogenous trust, model, batteries, usable candidates and adaptive decisions retained. It is neither aggregation alone nor a new native-versus-common causal experiment. Historical native/full-data results cannot serve as a matched counterfactual to this new data/root regime.

The retained adaptive attack is **adaptive_system**: delay, selective participation and resource misreporting according to own-method trust history. Its submitted gradient is not directly poisoned. This new study does not independently reproduce the historical hardened gradient-evasive condition or establish poisoning-damage robustness. EuroSAT is satellite imagery, not UAV-acquired imagery or flight evidence; EMNIST-byclass is not LEAF FEMNIST.

## Data and root changes — explicit before implementation

1. Use the existing authenticated full-data recipe to create each prospective seed's base reserve and full Dirichlet partitions, with its **unchanged base recipe/namespace**. This preserves the specified partition pool and procedure, not a claim of recovered historical indices. Add separate v5 subset domains afterwards; changing the root count inside the base namespace would change its draws and is disallowed by this recipe.
2. EMNIST has 697,932 train and 116,323 test examples, 62 classes. The base reserve is 6,944, exactly 112/class, leaving 690,988 for full client partitions. Select 10/class from that reserve by fixed HMAC priorities: **620 active root examples**. The unused **6,324 remain excluded from every method**, rather than moving into clients.
3. EuroSAT retains the fixed seed-42 stratified split of 27,000 images into 21,600 train / 5,400 test. Keep 210 root examples, 21/class, and the 21,390-example full client pool.
4. Apply the existing Dirichlet alpha=.2 allocator and minimum-10 rule to each full non-root pool **before** subsampling. Within each original client, rank ultimate sample identities by a separate HMAC-SHA256 `v5_client_cap` event keyed by dataset identity, base partition namespace, client and sample. Ties use ultimate sample ID. Select `n_j=min(cap, original_n_j)` uniformly without replacement. Root subset priorities have their own per-class domain. Sort saved retained IDs; minibatch order uses a separate existing event domain. Neither selection depends on method, condition, results or test outcomes.
5. No refill, redistribution, forced class coverage, new seed or partition retry is introduced. Unexpected invalid base partitions fail preparation and remain recorded. Smaller clients keep all their examples. Save original and retained histograms, missing classes and all excluded IDs. Class coverage is reported, not repaired after inspection.
6. All identity/disjointness checks use namespace-qualified identities. EMNIST train index 0 and test index 0 are different records. Root, clients and tests must be disjoint, with full reserve/retained/excluded unions proven in the index manifest.

The exact root/test counts above are fixed. Actual client totals are not available from supplied diagnostic records for these new seeds: they must be computed and sealed in later **data-only preparation**, before any scientific attempt. Maxima are 15,360 EMNIST and 7,680 EuroSAT client samples; they are not claimed observed totals. A smaller client total can make root information proportionally larger, so publish the actual ratio and class coverage.

Use actual retained `n_j` and the existing label-variance calculation in training, sample weighting, utility, cycles, compute energy and resource metadata. Do not train on a cap while weighting/charging the full partition. All five methods see the same active root. Equal examples do not imply equal query work. Reducing the root changes TrustFL rewards, baseline shadow trust/adaptive feedback, FLTrust training and fallback, FedGreed ranking/prefixes and FD rewards; its scientific consequences must accompany the runtime rationale.

## Settings and information budgets retained

Keep the existing CNNMnist(62) and CNNEuroSAT(10), models/transform/operator identities, strict deterministic float32/no-TF32 policy, local SGD lr=.01/momentum=.9/batch64/weight decay0, two EMNIST epochs and four EuroSAT epochs, root batch256, test batch512, sequential CUDA and workers0. No equation, attack rule, baseline algorithm, tolerance, precision setting, deadline or resource rule changes. The full literal settings, methods, memberships and trace rules appear in the JSON.

For `n<=12` finite usable clients, TrustFL's learner costs `n+1` root queries if n>=2; every baseline retains the corresponding shadow feedback, counted separately from its intrinsic work. TrustFL is never counted twice. At n=12:

| Method | Root-validation passes/round, maximum | Separate root SGD |
|---|---:|---|
| TrustFL | 13 | none |
| FedAvg | 13 shadow | none |
| FLTrust declared fallback | 13 shadow | EMNIST 6 steps; EuroSAT 4 steps, if candidates exist |
| FedGreed Algorithm 1 | 23 intrinsic +13 shadow =36 | none |
| FD local | 64 intrinsic +13 shadow =77 | none |

FedGreed keeps first-nonimprovement prefixes, UID ties and cached singleton behavior, not fixed top five. FD keeps distance screening, k=round(.6n), eight finite-difference steps, lr=.8/epsilon=.15 and persistent UID logits; it is not published DDPG. FLTrust keeps ReLU cosine, norm scaling and the <=1e-12 root-only fallback. Its EMNIST root workload changes explicitly from 56 to six SGD steps because the root changes. All methods retain the same 13 scheduled root/test evaluations at model boundaries **0,5,10,…,60**; the final evaluation is reused, not repeated. Test results never feed selection, attacks, trust or tuning.

Krum/Multi-Krum with K=12 and Byzantine bound12 do not meet their stated candidate regimes; no silent bound/neighbor adjustment admits them. Sophon fidelity and a published DDPG comparator remain separate unresolved gates. These exclusions already belong to the five-method fallback; they are disclosed, not presented as proof those algorithms lose.

## Endpoints and inference

Keep the three fixed confirmatory endpoints under EMNIST adaptive40/common: final fixed-boundary accuracy composite, pooled honest-high-mobility opportunity retention across rounds0–59, and mean unconditional applied malicious coefficient share across rounds30–59. There are four TrustFL-minus-comparator contrasts × three endpoints = **12**, with Holm family .05. Honest cells and EuroSAT's six-seed extension are descriptive.

The unit is the independent paired scenario seed, not a client, round, test image or second execution of one seed. Report all paired differences, means, seed points and two-sided exact sign-flip p-values (all2^10 flips, zeros retained, equally extreme included). Sign symmetry/exchangeability is an assumption, not proved by pairing. Use the preserved 50,000-resample percentile paired-seed bootstrap/seed51001 for approximate marginal 95% intervals; these are not simultaneous intervals. No significance or power is promised. Non-rejection is not equality. Historical Holm analyses remain non-rejecting, including retention.

Algorithmic numerical failure or method-caused OOM has a labeled accuracy-composite zero, never a claimed measured failed-model accuracy. External time/storage cuts, interruptions and corrupt/incomplete evidence are unobserved. Mechanistic endpoints require their full valid window. Any unavailable pair makes that confirmatory contrast unavailable; available pairs are descriptive only, and p=1 is merely its preserved Holm bookkeeping slot. No replacement seeds or retry observations.

Log scheduled/executable invitations, participation, physical arrival, quarantine, usable candidates, exact same-round unrounded coefficients, root coefficient, whether application occurred and reconstruction checks. Valid no-application/zero-client/root-only events differ from absent/stale/nonfinite weights. A nonfinite guarded aggregate is a failure, not a valid zero. Retention counts positive applied honest-HM client opportunities, not data volume; show denominators. Coefficient share is representation, not attack damage. Report conditional shares, raw masses/norms, exposure, no-application/root-only/zero/singleton frequencies and accuracy alongside it.

## Physics and energy

Common requests remain 2GHz and .525W, with simulator caps enforced and oracle computation declared. External schedule/trace identity does not force identical realized participation. No silent replacement of unavailable scheduled clients and no scripted identical adaptive strikes.

At the selected caps both datasets have at most1,024 example presentations/client-round. Modeled cycles are at most2.048e9; compute time at true .5–2GHz is1.024–4.096s and the modeled deadline is2.4576s. For positive n the compute condition cancels workload: true frequency must be >=.833333GHz, analytically7/9 under the stated initial uniform range. Caps reduce time/energy but **do not relax this gate**. Full model uploads remain roughly53–54Mbit, independent of the data cap.

The specialist's distance-conditioned channel calculations indicate multi-client arrival is plausible, not certified. The prior singleton fixture does not determine scientific coverage. Mobility energy can dominate capped compute energy. Failed transmissions still consume modeled duration; position excursions from the existing single-reflection altitude implementation preclude a claimed universal battery/distance bound. Preserve the current physical equations and report actual extremes and excluded clients. These uncalibrated predictions are not measured UAV energy or deployment evidence.

Record initial batteries and each round's compute/communication/mobility demand, clipped depletion and overdraw. Numerator and denominator span pre-round0 through post-round59. A zero useful-update denominator is NA. Low delivery and absent application are retained outcomes: no favorable seed search, longer trajectories or relaxed deadlines. If coverage is weak, narrow the interpretation rather than rerunning.

## Work, disk and calendar gates

New scientific maximum allowances: **1,853,280 SGD steps** (1,843,200 client +10,080 root), **293,920 root-validation passes**, **2,080 full-test passes**. The exact per-method and per-attempt ceilings are serialized. These are new proposed allowances, separate from the exhausted cost budget; historical38 complete+1censored and the old outstanding debit are immutable.

Storage must cover approximately8,005,120 basic reserve/ack events, not just the7.714GiB artifact proxy. A new compact event schema is explicitly required, retaining separate batch/step/query accounting with <=4KiB records and an8.2M-event allowance. Old full-snapshot records exceed that limit and cannot simply be reused. Use a64GiB new-output cap, separate64GiB staging allowance and fresh free-space>=160GiB while reserving20% of initial free space. No runtime saving is credited for serialization changes. Final checkpoints only; do not silently retain all evaluation-boundary models. No historical file may be deleted to meet capacity.

The first fixed honest-FD attempt later requires separate execution authorization. A manual, outcome-blind cost review uses actual work counts, sustained component timing, source/storage/error status and a conservative remaining-work forecast. It cannot infer whole-grid affordability from lower arrival/query counts or a successful return code. No automatic grid follows. Proposed safety limits are6h/owned attempt and240h cumulative new scientific worker time; external budget cuts remain missingness. Preparation and collection overhead must also fit the calendar.

Conditional schedule: review/implementation and data-only freezing by8October; if separately accepted, scientific compute around9–19October, leaving20–25October for analysis and author-approved edits,26October for clean/marked/response checks and submission buffer before27October. These are planning milestones, not promises. Review deadline readiness by12October and again19October; if the fixed scope or remaining reviewer work cannot fit, the submitting author decides on an extension. No message to the journal is sent by this task.

The next implementation task is precisely specified in `FUTURE_IMPLEMENTATION_AND_EXECUTION_SPEC_v5.md`. All17 comments remain open; the separate action matrix includes hardware, scaling, terminology, theory mapping, the18/17-page build mismatch and administrative requirements.
