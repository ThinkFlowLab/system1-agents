# Public231 accuracy leaderboard

This board ranks five deployed decision models on 231 public JevBench tasks:
easy48, original72 and hard111. It is a fixed evaluation group, not the article's
248-question leaderboard or a claim about general model quality.

## H800 group: Slurm 415076, 2026-10-07 UTC

Rank uses correct / 231 in the first measured round. All three rounds
produced identical predictions and probability distributions for each deployment.
The repeated rounds add timing samples, not unique quality tasks. Equal correct
counts share a rank; calibration does not break ties.

| Rank | Deployment | Correct / planned | Accuracy | Validity | Brier | ECE |
|---:|---|---:|---:|---:|---:|---:|
| 1 | Intern-Decision-4B, official HF | 200 / 231 | 86.5801% | 100% | 0.189601 | 0.067214 |
| 2 | Open-Jev-27B-v1.1, Omni native | 198 / 231 | 85.7143% | 100% | 0.240493 | 0.130333 |
| 3 | Intern-Decision-2B, official HF | 181 / 231 | 78.3550% | 100% | 0.306049 | 0.115420 |
| 4 | Open-Jev-9B, Omni native | 179 / 231 | 77.4892% | 100% | 0.320752 | 0.086786 |
| 5 | Intern-Decision-0.8B, official HF | 164 / 231 | 70.9957% | 100% | 0.369906 | 0.111077 |

Each deployment completed 693 / 693 measured requests without failed or invalid
responses. Validity, Brier and 10-bin top-label ECE use those 693 distributions;
their repetition does not increase the independent sample size. Brier and ECE
are secondary calibration measures, with lower values better.

## Source and controls

- Dataset and scorer: [JevBench 7ce310c7](https://github.com/fstandhartinger/jevbench/tree/7ce310c7262ed49cc85853339a8a42459298e3f3).
  The collector checks the pinned dataset and imported scorer file hashes.
- Canonical dataset SHA-256 (`dataset_hash` of the pinned tasks, also recorded
  as `canonical_dataset_sha256` in the numeric attachment's source manifest):
  `dc3995d8ae1e2fc8e81ce38431add509eb8bb39b85aadfd0c7c32079382dde51`.
- One H800, BF16, concurrency 1, thinking off, no retries, fixed task and option
  order; five client warmups per deployment followed by three measured traversals.
- Intern used its official HF service at T=1, with optional fast paths unavailable.
  Open-Jev used native Rust/CUDA, merged LoRA/head and its checkpoint temperature;
  graph and prefix cache were disabled. These are different service configurations.
- The [immutable run report](https://github.com/linear3735/system1-agents/blob/40bf742e2b84ad6157bf2990dd5e4bab73571702/benchmarks/results/public231-h800.md)
  records exact checkpoint, backend and dependency revisions, all round results
  and limitations. The companion [latency contribution](https://github.com/ThinkFlowLab/system1-omni/pull/131)
  keeps request timing separate from quality.

The published report is available now. The numeric audit attachment for this run
is prepared but not yet published; complete original responses are not included
in the repository. The attachment removes question/option text and retains
probability vectors, label indices, timings and original record hashes. It supports
numeric recomputation; full response replay additionally requires the original raw.

## Add a result

Use the [collection and aggregation commands](README.md#run) against an existing
decision endpoint. Keep the dataset/scorer version, task order, precision and
sampling policy fixed; record exact model/backend revisions and any required
configuration differences. Labels and provenance must never enter model requests.

Publish planned, attempted, valid, failed, incomplete and unattempted counts.
Failures and missing tasks stay in the planned accuracy denominator. Score the
original probabilities with the pinned scorer; do not replace them with one-hot
labels. Missing calibration inputs remain missing, rather than becoming zero error.

Predeclare the primary round and repeat plan. This group uses the first measured
round for ranking; publish every round and any prediction disagreement. Never
choose the best round after seeing results. A partial run remains an unranked
observation with its coverage shown.

Add a dated campaign with a linked report and auditable numeric evidence; preserve
prior campaigns. Changed datasets, scorer rules or sampling policies need a separate
group. Compare new deployments with a reference in the same campaign. Ticket-routing
completion, screenshot/VL tasks and other suites also need separate groups and
their own fixed inputs; they cannot inherit this text board's rank.
