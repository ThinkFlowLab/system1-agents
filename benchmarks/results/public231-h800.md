# Public231 on one H800

Five open checkpoints, one NVIDIA H800, one HTTP request at a time. Each service
received five warmups followed by the same 231 questions in the same order for three
rounds. Latency covers the complete HTTP request and response. No profiler was active.

Open-Jev used System1-Omni's native Rust/CUDA service; Intern-Decision used its
upstream HF service. These results compare deployed configurations. Their ratios
include different architectures, templates and execution paths, not just model size
or kernel speed.

## Results

| Model | Service | Correct / 231 | Accuracy | Validity | Brier | ECE | p50 / p95 ms |
|---|---|---:|---:|---:|---:|---:|---:|
| Intern-Decision-0.8B | HF / T=1 | 164/231 | 70.9957% | 100% | 0.369906 | 0.111077 | 80.493 / 175.237 |
| Intern-Decision-2B | HF / T=1 | 181/231 | 78.3550% | 100% | 0.306049 | 0.115420 | 79.054 / 175.276 |
| Intern-Decision-4B | HF / T=1 | 200/231 | 86.5801% | 100% | 0.189601 | 0.067214 | 104.552 / 235.291 |
| Open-Jev-9B | Omni native | 179/231 | 77.4892% | 100% | 0.320752 | 0.086786 | 42.688 / 327.057 |
| Open-Jev-27B-v1.1 | Omni native | 198/231 | 85.7143% | 100% | 0.240493 | 0.130333 | 68.877 / 1068.817 |

All five configurations completed 693/693 measured requests, with no failed or invalid
responses. Predictions and probabilities matched across the three rounds. Accuracy
therefore has 231 unique questions; the repeated rounds are timing repetitions.
Brier/ECE use all 693 valid distributions. Failure latency has n=0 and null quantiles.

Open-Jev 9B had the lowest median latency. Intern 4B answered 200/231 correctly versus
Open-Jev 27B's 198/231, and had a lower p95; Open-Jev 27B had a lower median. This small
set does not establish a general quality ranking or a uniform 2–3× speed advantage.

## Per-round latency

| Model | Round 1 p50 / p95 ms | Round 2 p50 / p95 ms | Round 3 p50 / p95 ms |
|---|---:|---:|---:|
| Intern-Decision-0.8B | 84.534 / 180.725 | 73.301 / 173.430 | 73.485 / 172.817 |
| Intern-Decision-2B | 81.454 / 180.338 | 74.251 / 173.638 | 74.154 / 173.411 |
| Intern-Decision-4B | 107.711 / 240.246 | 98.628 / 232.829 | 98.065 / 232.709 |
| Open-Jev-9B | 43.166 / 317.867 | 42.624 / 317.644 | 42.525 / 317.579 |
| Open-Jev-27B-v1.1 | 68.925 / 1030.605 | 68.892 / 1034.555 | 68.821 / 1033.953 |

Pooled quantiles above are computed from all raw timings, not from these three quantiles.

## Configuration

- BF16; client concurrency 1; no retries; thinking disabled.
- Intern: official source `3572c8a68b5df5dafe02d0e093989ba8ec0183bc`,
  PyTorch 2.9.1+cu128, torchvision 0.24.1+cu128, Transformers 5.14.1, SDPA,
  temperature 1, no calibration artifact, no KV cache. Optional FLA/causal-conv1d
  fast paths were unavailable, so the service used the upstream Torch fallback.
- Open-Jev: Omni `4a79980d8a75fd063cb3f8247e06215e288b18ac`, CUDA 13,
  merged LoRA and decision head, Graph and prefix cache disabled. 9B evaluates
  candidates sequentially; 27B packs candidates. A single service request is not
  necessarily one candidate sequence.
- Context limits: Intern 8192 tokens for the complete chat; Open-Jev 16384 per
  candidate. Native usage counts tokens summed over candidates, so the two services'
  token counts are not directly comparable.

| Checkpoint | Revision | Temperature | Spawn-to-ready s |
|---|---|---:|---:|
| [Intern 0.8B](https://huggingface.co/internlm/Intern-Decision-0.8B) | `85a0cc5a99d67ea8d56dfe98115689212867171d` | 1 | 16.084 |
| [Intern 2B](https://huggingface.co/internlm/Intern-Decision-2B) | `8797836c65fc91a2435b1fb6850b5f0aabd75cc3` | 1 | 14.045 |
| [Intern 4B](https://huggingface.co/internlm/Intern-Decision-4B) | `0e5e6aa7d6d750e2b1504ba11a8136cb58aeb3cd` | 1 | 16.048 |
| [Open-Jev 9B](https://huggingface.co/ZefanCai/Open-Jev-9B) | `47e966881e489511c0c7f5633a9e1960a676a551` | 1.8969118766 | 8.015 |
| [Open-Jev 27B v1.1](https://huggingface.co/ZefanCai/Open-Jev-27B-v1.1) | `28cf73067d5b337860bbef3c85b8b82ba8730956` | 2.5343690298 | 20.029 |

Cold startup is excluded from request latency and includes native internal warmup.
Open 9B's base is `c202236235762e1c871ad0ccb60c8ee5ba337b9a`; Open 27B's base is
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.

## Inputs and verification

[JevBench](https://github.com/fstandhartinger/jevbench/tree/7ce310c7262ed49cc85853339a8a42459298e3f3)
provides easy48, original72 and hard111 and the scorer. Source and data hashes are
pinned in `benchmarks/evidence.py`. The article's complete 248-question set was not
available; this is not a reproduction of its 23-model leaderboard.

The run used Agents base `9790b52ef4d78a2215e8dd5a526d3a8daea9875a`, dated
2026-10-07 UTC, Slurm job 415076. All 3490 raw requests, including warmups, were
checked for source identity, order, response hash and absence of target leakage.
A non-author independently re-scored the saved responses with the pinned scorer
and recomputed latency from raw samples. The final collector's offline aggregation
matched counts exactly and floating-point metrics within 1e-12.

Collection preceded fixes to metadata validation, URL validation before file writes,
and JSONL physical-line reading. These fixes preserve the HTTP and scoring paths.
Verification replays the saved responses; it is not a second model run.

Actual loaded-library maps and the weight hashes were recorded. The native binary
SHA-256 was `a0694633308ea0ad0ca2a302bfb16cbb73fd3367cd36cd142596761a71d23afc`;
CUDA library SHA-256 was `a0db4905df4fe2fb5d006acaaa09b5b51dbd3933cc8e9b20ec8114f23def8b20`.
Use the [collection and replay commands](../README.md#run) with the fixed inputs and
record the actual serving environment. Large raw traces and weights stay outside Git.
