# OpenChoice-9B results

**Runtime labels matter.** The public package now serves merged GGUF Q4_K_M through llama.cpp. Older benchmark scores below were measured using PyTorch NF4/BF16 on an RTX 4060 and do not automatically transfer to the GGUF release. Comparator scores marked “published” are author-reported, not local reruns. No Jev parity or affiliation is claimed.

## GGUF runtime and quality

Pinned llama.cpp: `4364bf7232e65c34eca8d9500c5464389662de6b`. Text artifact SHA-256: `6d8eef23223c2f53a772fa5a945842569e2cd646261bcb0a354e6c0f4e5ee238`; Q8_0 projector: `44c7ddca015a856956bbf2ff4a88049a4bc1dcff47fabd93602f23554d307e0c`. Linux x86-64, Ryzen 9 7900X3D, 12 inference threads, 4,096-token context, 512-token microbatches, CPU repacking disabled. Measurements were made on a shared desktop, not an isolated machine.

| GGUF measurement | Result | Public evidence |
|---|---|---|
| Text weights | 5,629,108,704 bytes / 5.24 GiB | [CPU runtime record](results/runtime/gguf-q4-norepack.json) |
| Text peak process RSS | About 6.10 GiB | Same record; mapped weights included |
| Warm Choice/Noul at 300–600 tokens | About 1.7–3.3 s per decision | Same record; median of two independent uncached forwards per shape |
| Frozen feasibility panel | 178/200 correct (83/100 decision-panel + 95/100 SciQ); NF4 reference 174/200; 96% decision agreement | [Parity excerpt](results/runtime/frozen-parity.json), [raw 200-row predictions](results/runtime/gguf-q4-norepack-predictions.jsonl) |
| Q8_0 vision projector | 595.31 MiB; CPU text+vision peak RSS about 6.71 GiB | [Vision runtime record](results/runtime/vision-q4-q8_0.json) |
| Chart development set | 49/60 (Choice 30/30, Score 19/30), versus historical NF4 52/60 | [Raw 60-row predictions](results/runtime/vision-q4-q8_0-predictions.jsonl) |
| Four-layer optional CUDA profile | 1,926 MiB observed peak VRAM; CPU remains loaded; total peak RSS about 12.35 GiB | [Production GPU policy verification](results/runtime/production-gpu-verification.json) |

Vision timings in the lab: mean 4.16 s per Choice and 25.74 s per six-level Score, including repeated image encoding. The production bridge reuses image embeddings within a request; the [two-row integration verification](results/runtime/production-vision-verification.json) observed a 13.22 s standalone Score. These are different workloads/cache boundaries, not competing headline speeds. CPU/GPU probabilities differ. The 200-row panel is small and reused: its higher accuracy does not establish an improvement.

These runtime evidence files predate the public torch-free packaging cutover; the package uses identical GGUF/projector bytes and the pinned bridge. New release HTTP results are separately labeled below. Full benchmark dataset text is not redistributed; raw response ledgers retain IDs, request hashes, and outputs. No new training, calibration, or model selection is performed during release benchmarking.

**Public-package CPU verification:** a fresh copied source tree and newly compiled pinned runtime ran with `CUDA_VISIBLE_DEVICES=`, no Torch installed, and no GPU backend. The README Choice/Score/Noul request returned probabilities. [Clean CPU vision smoke](results/gguf-clean-vision-smoke.json) exercises Choice plus six-level Score images, matches archived GGUF probabilities to a maximum 1.25e-8 difference, and rejects a wrong asset hash. It used two low-priority threads under concurrent desktop work; its elapsed time is not a latency benchmark. HTTP remains text-only.

## GGUF release re-benchmarks

Frozen suites run against the public torch-free GGUF server, CPU-only at nice 10 with 12 threads. No training, retry, truncation, or outcome-based filtering. Loopback response ledgers omit dataset input text and retain request hashes. Errors count incorrect. Background desktop work, release checks, and low-priority rendering share the machine; recorded wall times are not isolated latency benchmarks.

| Suite | GGUF CPU (new) | Historical OpenChoice NF4 | Comparator | Evidence |
|---|---:|---:|---|---|
| SciQ official test, 1,000 | **95.3% (953/1000)**, zero errors | 95.6% (956/1000) | CLM NF4 58.7% (587/1000), locally measured previously | [Summary](results/gguf-sciq.json), [raw](results/gguf-sciq-predictions.jsonl) |
| Frozen 400-task decision panel | **80.25% (321/400)**, zero errors | 79.5% (318/400) | CLM NF4 44.5% (178/400), locally measured previously | [Summary](results/gguf-panel400.json), [raw](results/gguf-panel400-predictions.jsonl) |
| JevBench **legacy public 231**, easy + original + hard | **81.39% (188/231)**, zero errors | 77.49% (179/231), three errors | Published hosted Jev 1.13.0 86.6% (200/231) on those legacy tasks; not a paired local run | [Summary](results/gguf-jevbench.json), [raw](results/gguf-jevbench-predictions.jsonl) |
| Nimble public, 13 frozen subsets / 3,880 rows | **75.41% micro (2926/3880)**; **74.72% macro**; one runner-boundary error counted wrong | 74.97% micro (2909/3880), zero errors | Author-published Nimble 75.9% micro / 74.8% macro and Jev 77.3% / 76.0%; not locally remeasured | [Summary](results/gguf-nimble-public.json), [raw](results/gguf-nimble-public-predictions.jsonl) |

SciQ uses the identical frozen closed-book test rendering and answer order, without support passages. Its dataset hash matches the historical protocol. The new GGUF–historical CLM gap is 36.6 points; the old paired NF4 confidence interval does not apply. The benchmark process started before the final metadata-only model-identity change, so its returned name uses the old configuration fingerprint; every result also records the exact released GGUF hash. No inference arithmetic or model bytes changed.

Nimble uses the 13 exact public subset manifests (all 3,880 frozen payloads hash-checked). **Micro** is total correct / 3,880; **macro** is the unweighted mean of the 13 subset accuracies. The serving API uses native Score, unlike Nimble's enum/Choice rating path. The first runner was terminated by the tool's one-hour execution cap after 425 recorded Nimble rows. All recorded attempts were preserved; the first unrecorded boundary case is conservatively counted wrong without resubmission, and remaining rows continued under a no-timeout process. There were no HTTP inference errors among the 3,879 recorded responses. This disclosed infrastructure error is included in both accuracy denominators, not filtered out. [Ledger consistency checks](results/release-benchmark-checks.json).

The [release protocol](results/benchmark-protocol.json) records frozen input and scorer hashes. The panel has 100 examples per family: Aegis **88/100**, BoolQ **90/100**, HelpSteer2 **53/100**, and SNLI **90/100**. Those source families were seen during training; this is a release regression panel, not a new sealed holdout. JevBench reuses the upstream TypeSafe request construction and scorer, with no retries. Its public-task accuracy is not JevBench's broader composite ranking and does not establish Jev parity.

The historical public-task comparisons below are retained only as version-labeled archives. They do not establish current JevBench standing. The obsolete v1.4.2.2 field-ranking table has been removed; OpenChoice has no official v1.5.1 composite or sealed result.

**JevBench version boundary:** this is **legacy public 231**, with the harness byte-identical to the v1.4.2.2 legacy comparison / upstream `bb05a33`. A fresh upstream checkout at [`bb05a335bc809e61b20c0f745d25499a82b326fc`](https://github.com/fstandhartinger/jevbench/tree/bb05a335bc809e61b20c0f745d25499a82b326fc) has identical hashes for all three task files, the request adapter, scorer, and task loader; see the [compatibility receipt](results/jevbench-method-compatibility.json). **This is not a v1.5 measurement.** The [v1.5 method](https://github.com/fstandhartinger/jevbench/blob/bb05a335bc809e61b20c0f745d25499a82b326fc/docs/METHOD-v1.5-README.md) expands to 904 open decisions (601 published) plus 720 sealed, changes Noul/Score competence scoring, and uses different composite weights. Its scores cannot be substituted into this legacy-task comparison.

## Current official benchmark status

**OpenChoice: pending (sealed split requires maintainer submission).** JevBench v1.5.1 covers **904 open + 720 sealed decisions**; its official score is a four-axis composite, not accuracy. The current published comparator figures below were re-read from the [versioned aggregate](https://benchmarkheaven.com/api/jevbench/v1.5.1); [retained values and source hash](results/benchmarkheaven-v1.5.1.json). No OpenChoice legacy accuracy is plotted alongside them.

| Benchmark-pinned system | Official composite | Intelligence | I-open (904) | I-sealed (720) |
|---|---:|---:|---:|---:|
| Cygnet (blockbrain, frozen Gemma-4-12B-it) | 73.7 | 71.1 | 72.5 | 69.7 |
| Winnow-12B Q8 | 73.2 | 74.4 | 72.5 | 76.4 |
| Jev 1.13.0 (TypeSafe AI) | 72.1 | 72.0 | 71.5 | 72.5 |
| JevK5 v0.3 (4B) | 71.9 | 56.3 | 61.6 | 50.9 |
| Plumb-4B (crh225, JevK5 v0.2 + LoRA) | 71.6 | 55.8 | 60.9 | 50.8 |
| Jev-Omni (akhilaaa3, Gemma-4-12B merged) | 71.5 | 70.5 | 70.5 | 70.5 |
| decider-4b v2 (Mapika) | 71.3 | 55.8 | 61.7 | 49.8 |
| Imajev-4B (RTX 5090) | 70.4 | 53.5 | 56.2 | 50.7 |

These are maintainer-published results, not our reproductions. The benchmark row for decider is v2, not current-card v2.1; a score does not automatically carry over to a smaller GGUF variant. GLiNER2.5-Decide has no exact-artifact row and must not inherit GLiNER2.5-Multi’s result. See [the artifact/capability comparison](README.md#current-alternatives-artifacts-not-a-quality-ranking).

**New public rerun access:** the full upstream checkout, release assets, Benchmark Heaven pages, and public API exposed no runnable frozen v1.5 public task/scorer bundle or ImageJevBench v0.1.3 public image-task/assets/scorer bundle. The image release contains aggregates and explicitly excludes item text, images, answer keys and per-item outputs. The [access receipt](results/current-benchmark-access.json) records the checked sources and missing prerequisites. No local v1.5 open-Intelligence or ImageJevBench score is invented from unrelated tasks. ImageJevBench v0.1.3 has 228 public + 456 sealed items; local chart-development accuracy is not its score. The [release procedure](PUBLISHING.md#official-benchmark-heaven-evaluation) includes maintainer submission/access.

## Historical NF4 configuration

OpenChoice-9B is Qwen/Qwen3.5-9B (revision c202236235762e1c871ad0ccb60c8ee5ba337b9a, Apache-2.0) plus rank-16 LoRA (alpha 32), trained for 100 updates. Training processed 3,200 rows / 3,234,666 tokens with seed 20260921. Evaluation used NF4 weights, BF16 compute, CPU token-table offload, and one RTX 4060 8 GB. It directly predicts decision probabilities without text generation.

The five historical headline totals are retained in a [sanitized report excerpt](results/historical-nf4.json), with hashes of the full archived source reports. Every OpenChoice measurement in the remaining historical tables uses NF4 unless explicitly labeled otherwise.

## Head-to-head vs CLM-v0.1-8B

Both run NF4 on the same RTX 4060; protocols frozen before inference.

| Benchmark | OpenChoice-9B | CLM-v0.1-8B | Difference |
|---|---:|---:|---:|
| SciQ official test, 1,000 questions (held out from tuning/selection; exact/containment check found no overlap) | 95.6% (956/1000) | 58.7% (587/1000) | +36.9 pts; paired bootstrap 95% CI [+33.7, +40.2] |
| 400-task decision panel: 100 each Aegis 2.0 safety, BoolQ, HelpSteer2 score, SNLI; source families seen during training, panel previously used for final evaluation | 79.5% (318/400) | 44.5% (178/400) | +35.0 pts; 95% CI [+29.4, +40.9] |

## Legacy JevBench 231 public tasks: historical NF4

**Legacy JevBench 231 public tasks (harness pinned to v1.4.2.2), not a v1.5 score.** The tasks (easy 48 / original 72 / hard 111) have been frozen since v1.2; exact IDs matched the v1.2.8 published per-task outcomes.

| System | Accuracy | Status |
|---|---:|---|
| Jev 1.13.0 (TypeSafe hosted) | 86.6% (200/231) | published |
| SemIf (Qwen3.5-4B, formerly OpenJev) | 81.0% (187/231) | published |
| Bespoke-Nimble-9B (Qwen3.5-9B LoRA) | 79.7% (184/231) | published |
| OpenChoice-9B | 77.5% (179/231) | measured; 3 early configuration failures counted wrong |
| OpenChoice-4B (previous) | 63.2% (146/231) | measured |

OpenChoice-9B minus Jev: −9.1 pts, 95% CI [−13.3, −4.8]. Versus Nimble-9B: −2.2 pts, CI [−7.4, +3.1] (not significant). 4B→9B: +14.3 pts, CI [+9.4, +19.3]. Warm loopback HTTP median latency 0.25 s/decision (p95 1.95 s); no latency/cost comparison with Jev is made.

## Nimble public suite

3,880 rows across 13 public human-labeled datasets: Aegis2, BoolQ, Civil Comments, HelpSteer2, MASSIVE de/en, MultiNLI, PAWS, PubMedQA, SQuAD2, SummEval x2, VitaminC.

| System | Accuracy | Status |
|---|---:|---|
| Jev 1.13.0 | 77.3% | published; derived from rounded per-subset percentages |
| Bespoke-Nimble-9B | 75.9% | published; same derivation |
| OpenChoice-9B | 75.0% (2,909/3,880) | measured |
| OpenChoice-4B | 71.0% (2,753/3,880) | measured |

4B→9B: +4.0 pts, CI [+2.0, +5.6].

## Nimble evidence-324

Bespoke's held-out split, synthetic model-checked labels; in-domain for Nimble's 2,676-example training data.

| System | Accuracy | Status |
|---|---:|---|
| Jev 1.13.0 | 93.2% (302/324) | published |
| Jev 1.13.0 | 88.9% (288/324; 15 API errors counted wrong) | live rerun by us |
| Bespoke-Nimble-9B | 90.1% (292/324) | published |
| Qwen3.8-27B | 84.9% (275/324) | published |
| OpenChoice-9B | 73.1% (237/324) | measured |
| Qwen3.5-9B base, untuned | 66.4% (215/324) | published |
| Qwen3.5-4B | 61.4% (199/324) | published |

## Capabilities

### Per-task results on the Nimble public suite

OpenChoice measured; Jev and Nimble per-subset figures are author-published. (T) marks a task family that appears in OpenChoice training.

| Capability | Dataset (rows) | OpenChoice-9B | Jev 1.13 | Nimble-9B |
|---|---|---:|---:|---:|
| Safety moderation (T) | Aegis 2.0 (250) | 82.0% | 80.4% | 81.2% |
| Toxicity | Civil Comments (300) | 78.7% | 81.0% | 70.3% |
| Intent, English | MASSIVE en-US (350) | 86.0% | 87.4% | 86.9% |
| Intent, German | MASSIVE de-DE (350) | 84.0% | 86.9% | 83.4% |
| Natural-language inference | MultiNLI (299) | 83.9% | 82.9% | 85.3% |
| Fact verification | VitaminC (599) | 75.5% | 80.1% | 76.6% |
| Biomedical QA | PubMedQA (250) | 76.8% | 77.2% | 75.6% |
| Reading-comprehension yes/no (T) | BoolQ (300) | 86.7% | 89.7% | 86.0% |
| Answerability | SQuAD2 (299) | 77.9% | 82.9% | 80.6% |
| Paraphrase detection | PAWS (250) | 81.6% | 89.2% | 82.8% |
| Summary consistency score | SummEval consistency (144) | 80.6% | 81.2% | 75.7% |
| Helpfulness rating, exact 5-level score (T) | HelpSteer2 (249) | 41.4% | 34.1% | 39.0% |
| Summary relevance score | SummEval relevance (240) | 25.8% | 35.0% | 49.2% |

OpenChoice is above published Jev on three subsets: Aegis 2.0 and HelpSteer2, both trained families, and MultiNLI, which is not a training dataset but shares SNLI's task type.

### JevBench public task families

OpenChoice-9B only; samples are small. Perfect: tool selection 12/12, routing 12/12 (hard routing 5/5), extraction 24/24, fact 12/12, ordinal 12/12, adversarial 6/6, trap 8/8. Strong: policy 11/12, adequacy 11/12, intent 20/24, hard judging 14/17. Weak: temporal/numeric 4/15, ambiguity 2/7, trade-offs 2/6, long policies 9/19, probability 5/10, multi-hop 10/18.

### Development-set evidence

These rows were used for checkpoint selection, so they are not a sealed test.

- Historical NF4 chart reading: 52/60 synthetic chart questions (Choice 30/30, Score 22/30), versus 50/60 for the untuned Qwen3.5-9B read the same way. Charts share a generator with the 48 image rows in training. The new Q4_K_M + Q8_0 projector result is 49/60, above. Images work through the Python GGUF runtime on CPU; HTTP remains text-only. This is not broad photo/OCR/video evaluation.
- 20-way intent classification (CLINC, not a training source): 59/60. Emotion yes/no (GoEmotions, not trained): 51/60. Commonsense (WinoGrande, not trained): 46/60.

## Efficiency and cost

- Peak PyTorch GPU allocation: 3.7 GiB (NF4 + CPU token tables, RTX 4060 8 GB).
- Adapter download size: **160 MB (160,492,456 bytes; 153 MiB)**; Qwen3.5-9B base is separately required.
- **$3.82 training pilot on one A100**; the full pilot that produced this checkpoint, excluding prior research spending.
- Local inference has no per-call API fee; electricity and hardware are excluded.

## Methodology and caveats

These task-specific decision/classification results are not broad intelligence scores. Some comparison numbers are author-published, not independently rerun. The 400-task decision panel's source families appeared in training and the panel was previously used for final evaluation. SciQ was held out from tuning/selection and exact/containment overlap check found none, but upstream pretraining contamination was not assessed. Nimble evidence-324 uses synthetic model-checked labels and is in-domain for Nimble's training data. The exploratory upstream human labels did not receive new independent human review. Exact-text/group checks cannot prove absence of paraphrases or pretraining contamination. HelpSteer2 responses are model-generated and human-rated; integer ratings are rounded means, not unanimous ground truth or calibrated probabilities. Empty calibration/probe files provide no calibration evidence.

**Honesty:** Historical NF4 OpenChoice-9B trails Jev on every aggregate benchmark compared; it is ahead of published Jev on only 3 of 13 Nimble subsets. No claim of beating, matching, or parity with Jev, nor best-open-model status. Limits: uncalibrated confidence; English-centric training; text-only HTTP (CPU image input in Python); 4,096 tokens per forward; Choice at most 26 options; no sealed JevBench result or coding/tool-trajectory evaluation.