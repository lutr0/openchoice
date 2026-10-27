# Fine-tune your own decisions

OpenChoice returns probabilities, not generated text. Adapt the released rank-16 LoRA to your own Choice, Score or Noul labels, compare it with the original, then export a portable Q4_K_M package for the same CPU server. Training uses PyTorch/PEFT; serving does not.

## Four commands, then serve

Training needs a **24 GB CUDA GPU** for this short-text recipe; serving the result needs only a CPU. Install [uv](https://docs.astral.sh/uv/getting-started/installation/), git and a C++ compiler. Download the released package named in the repository README into `OpenChoice-9B-GGUF`, including its adapter, heads and projector: a standalone GGUF cannot be fine-tuned.

```bash
uv sync --group train --group build
bash scripts/build_llamacpp.sh cpu
```

```bash
uv run --group train python -m openchoice.finetune validate \
  --data examples/support-tickets.jsonl

uv run --group train python -m openchoice.finetune train \
  --data examples/support-tickets.jsonl --init OpenChoice-9B-GGUF \
  --output runs/policy --steps 300 --batch-size 8 --lr 0.00005 \
  --replay-fraction 0.2 --checkpoint-every 50

uv run --group train python -m openchoice.finetune eval \
  --data examples/support-tickets.jsonl --before OpenChoice-9B-GGUF \
  --after runs/policy/model --output runs/policy-eval.json

uv run --group train python -m openchoice.finetune export \
  --data examples/support-tickets.jsonl --checkpoint runs/policy/model \
  --projector-from OpenChoice-9B-GGUF --output Policy-GGUF --parity-rows 64

uv run python -m openchoice.server --model ./Policy-GGUF --port 8769
```

The explicit passing recipe above uses seed 42, 300 BF16 LoRA updates, learning rate 5e-5, logical batch 8 and 20% replay sampling. Microbatch is always one decision. CLI defaults remain the more conservative learning rate 1e-5 and 100 updates; those defaults are not the measured recipe. Use `--init base` for a fresh rank-16 adapter on the pinned Qwen3.5-9B base. `eval` compares adapters in **BF16 PyTorch**, even when `--before` points at a GGUF package containing an adapter; it does not silently benchmark that package's GGUF. Export parity measures the actual CPU serving artifact. Always measure a baseline and keep train/dev groups separate.

`export` runs real BF16 PyTorch reference predictions, moves the unquantized backbone to CPU FP32 before merging LoRA deltas, serializes F16, and quantizes to Q4_K_M with pinned llama.cpp. The finer merge precision avoids dropping small deltas through BF16 rounding; it does not make Q4 lossless. Export strips the untouched visual tower from the text conversion and copies the base projector. The manifest contains relative filenames and SHA-256 hashes, not build-host paths. A sibling parity JSON records CPU GGUF decision/probability drift. Default `--min-agreement 0.9` refuses promotion below 90% decision agreement. Check all dev rows with `--parity-rows 64`, rather than the default first 16. Inspect probability changes as well as top-label agreement, then evaluate your complete deployment distribution.

The vision projector remains valid because training modifies only language-model LoRA projections: the visual tower is frozen. Image-conditioned examples can train language decisions, but do not train a new vision encoder.
## What the example learns

The bundled **Cedar Cloud** policy improves **CPU GGUF accuracy from 28/64 to 61/64** on separately worded dev messages ([measured results](#measured-policy-learning-outcome)). Its 256 decisions comprise **192 train / 64 dev**, built from **96 training messages and 32 separately authored dev messages**. Each message appears twice in the same split/group, with only the contract code changed. Menu, urgency and wording stay identical within each pair. S1 routes topics naturally; C7 rotates ownership: revenue→identity, identity→reliability, reliability→developer, developer→revenue. This mapping is in the labels, **not printed in the model's prompt**.

Menus rotate across messages and labels are balanced. Hard negatives distinguish a resolved outage followed by a credit request from an active outage, or an invoice-schema integration question from a billing dispute. Dev wording is not produced by substituting names/numbers into training templates. The [split audit](examples/policy-results/split-audit.json) verifies no message/group overlap and code-only counterfactual pairs. These are authored teaching data, not an independent real-world benchmark. A fixed known mapping can often be handled more simply with ordinary code or an explicit prompt; this demonstrates learning private decision labels, not superiority over those alternatives.

## Hardware and dependencies

- **Serving:** CPU only; the training GPU is no longer needed after export.
- **Physically tested:** 24 GB RTX A5000; 300 BF16 LoRA updates in **35.31 minutes** across two resumed invocations; peak **18.45 GiB allocated / 18.61 GiB reserved**. Inputs are at most **133 tokens**. Sources: [hardware](examples/policy-results/hardware.csv), [summary](examples/policy-results/final-summary.json), [validation](examples/policy-results/policy-validate.json). These are short-text measurements, not a promise that all workloads fit 24 GB.
- `--qlora` optionally trains FP32 LoRA on a frozen NF4 backbone; it is **not** an NF4 serving path. The 24 GB QLoRA trial used **13.04 / 15.77 GiB** ([report](examples/policy-results/train-report.json)), but its learned policy failed the Q4 gate. The passing recipe above therefore uses BF16 LoRA. Export always reloads the unquantized base. An earlier 48 GB RTX 6000 Ada check exercised a **3,641-token** QLoRA forward/backward/optimizer step and a 128px image-conditioned backward ([boundary smoke](examples/results/memory-boundary.json)); those do not qualify every 4,096-token/image workload or establish that it fits 24 GB. Use 48 GB for extra headroom.
- Budget **64 GB host RAM and 100 GB free disk** for dependencies, the approximately 19 GB base cache, checkpoints, CPU FP32 merge and F16/GGUF conversion. These are planning recommendations, not tested minima. The shared 8 GB desktop GPU is not a supported training target.
- Continuation requires `adapter/`, `heads.safetensors` and `model.json`; export additionally needs the base package's `manifest.json` and `mmproj-q8_0.gguf`. A local adapter checkpoint with that layout is also accepted.


## Your JSONL

One line is one decision. State, instructions and criteria follow the HTTP API's text/JSON rules; JSON objects are canonically rendered exactly as in serving. Question IDs and dataset IDs do not enter the prompt.

```json
{"id":"ticket-1","group":"customer-thread-8","split":"train","state":{"message":"We were charged twice."},"question":{"type":"choice","instructions":"Choose the responsible team.","criteria":{"billing":"Invoices and refunds","access":"Login and passwords"}},"label":"billing"}
{"id":"review-2","group":"review-2","split":"dev","state":"Arrived in three days.","question":{"type":"score","instructions":"Classify delivery speed.","criteria":["0–4 days","5–14 days","15+ days"]},"label":0}
{"id":"claim-3","group":"claim-3","split":"train","state":"The signed order contains five items.","question":{"type":"noul","instructions":"The order contains five items.","criteria":{"true":"The count is five.","false":"The count differs from five."}},"label":true,"probabilities":[0.95,0.05]}
```

- Choice labels are option **keys**; 1–26 options. Score labels are **zero-based integer levels** in criteria order; 1–10 levels. Noul labels are JSON booleans, never the strings `"true"` or `"false"`.
- Optional `probabilities` are a finite nonnegative list summing to one, in criteria order; Noul order is **[true, false]**. They replace the hard label in the training loss. Accuracy still uses `label`. Score trains independent Yes/No support for each criterion with BCE, then normalizes supported levels for the returned distribution—matching serving. Overlapping, unnormalized Score marginals are not this user format.
- Optional `image` is a relative PNG/JPEG path inside the JSONL directory. Images are decoded, hashed and checked; paths/symlinks cannot escape that directory. Training and validation account for expanded image tokens, not only text.
- Set `split` on **every** row, or omit it on **every** row. Without explicit splits, a seeded 20% group-held-out split is produced (`--dev-fraction`, `--seed`). Put an entire conversation/document/customer episode in one `group`; near-duplicate detection is your responsibility. Exact duplicate decisions and group overlaps across train/dev are rejected.
- `validate` reports class counts, missing classes, large imbalances and exact processor token lengths. Any individual forward over **4,096 tokens**, or Score decision over **32,768 total processed tokens**, fails; nothing is silently truncated. Fix the data before paying for training. A valid schema does not establish correct labels or rights.

## Replay, retention and calibration

`examples/general-decisions.jsonl` supplies 48 general training/replay rows and a **disjoint 24-row held-out panel** across Choice, Score and Noul. All examples and the reproducible `examples/make_examples.py` generator were authored for this repository and are offered under its Apache-2.0 license. No third-party training corpus is bundled. The released model's training-data and weight licensing is described in [PUBLISHING.md](PUBLISHING.md#licensing-and-rights).

`--replay-fraction 0.2` samples a general row with 20% probability for each microbatch. `--replay PATH` uses only that file's train split; `eval --retention PATH` uses only its dev split. Keep replay and held-out general examples separate, and never include user dev rows in replay. `--replay-fraction 0` disables replay but raises forgetting risk; a small authored sample cannot preserve every original capability.

Evaluation reports before/after **accuracy, Brier score and log-loss**, plus deltas, per-row probabilities and a retention warning (accuracy drops by more than 2 points or log-loss rises by more than 0.02). Brier is the sum of squared errors over the categorical response vector, including both Noul outcomes. Losses use soft targets when supplied. These are descriptive thresholds on a small panel, not statistical certification. The output is uncalibrated; no temperature is fitted on the dev set or copied from a different runtime. Fit calibration separately on fresh held-out data if your application needs reliable probabilities, and revalidate after GGUF quantization.

## Resume and reproducibility

Each committed `checkpoint-NNNNNN` contains the adapter model, optimizer/RNG state and data/hyperparameter contract. `latest.json` points at the last complete checkpoint. Ctrl-C during an update leaves that checkpoint usable. Resume with the same arguments; `--steps` is the desired **total**, not additional steps:

```bash
uv run --group train python -m openchoice.finetune train \
  --data examples/support-tickets.jsonl --init OpenChoice-9B-GGUF \
  --output runs/policy-resumed --steps 300 --batch-size 8 --lr 0.00005 \
  --replay-fraction 0.2 --checkpoint-every 50 \
  --resume runs/policy/checkpoint-000150
```

Row sampling is keyed by seed and absolute update. Data hashes, replay hashes and hyperparameters must match to resume. PyTorch deterministic algorithms are enabled, but different GPUs, library versions and BF16/NF4 modes are not promised bitwise equivalent. Checkpoint files must be trusted; optimizer deserialization uses `weights_only=True`.

## Rent a GPU without leaving a bill running

Vast.ai and RunPod are optional providers, not dependencies or endorsements. Upload only data you are permitted to send to a third party. Choose one modern 24 GB GPU for this short-text recipe (48 GB for headroom), at least 64 GB host RAM and 120 GB disk. Inspect the actual instance rate after renting: disk and transfer charges are additional, and quoted rates can differ.

A concrete Vast CLI workflow (replace the offer and SSH endpoint with those actually returned):

```bash
vastai search offers 'num_gpus=1 gpu_ram>=24 cpu_ram>=64 disk_space>=120 reliability>=0.98' -o dph_total
vastai create instance OFFER_ID --image pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel \
  --disk 120 --ssh --direct --label openchoice-finetune
vastai show instances
# Use the host/port returned by Vast; do not expose an unauthenticated model API.
rsync -az --exclude .git --exclude .venv --exclude .runtime \
  -e 'ssh -p SSH_PORT' ./ root@SSH_HOST:/workspace/openchoice/
rsync -az -e 'ssh -p SSH_PORT' OpenChoice-9B-GGUF/ \
  root@SSH_HOST:/workspace/openchoice/OpenChoice-9B-GGUF/
ssh -p SSH_PORT root@SSH_HOST
```

Inside the rented host, `cd /workspace/openchoice`, install uv (`python -m pip install uv`), then execute the same install/build/four commands above. RunPod's SSH terminal works identically after copying these directories; select a persistent workspace disk large enough for the export. Do not run the HTTP server on `0.0.0.0` without authentication/firewalling.

Retrieve results **before** destroying the instance:

```bash
rsync -az -e 'ssh -p SSH_PORT' root@SSH_HOST:/workspace/openchoice/runs/ ./runs/
rsync -az -e 'ssh -p SSH_PORT' root@SSH_HOST:/workspace/openchoice/Policy-GGUF/ ./Policy-GGUF/
rsync -az -e 'ssh -p SSH_PORT' root@SSH_HOST:/workspace/openchoice/Policy-GGUF.parity.json ./
# Verify the copied package by starting the local server, then destroy the rental.
vastai destroy instance INSTANCE_ID --yes
vastai show instances
```

On RunPod, terminate the pod and any separately billable volume once results are copied. Set a provider spending limit; do not rent a GPU merely to wait for code or labels. The measured outcome below concerns this small policy task, not an estimate for an arbitrary private dataset.

## Measured policy-learning outcome

The released model was measured **before adaptation** at 28/64 on this separately worded, group-held-out dev split. The final recipe uses seed 42, 300 BF16 LoRA updates, batch 8, learning rate 5e-5 and 20% replay. The actual run resumed at update 150. The served GGUF improves by **33 correct decisions** (34 wrong→right, one right→wrong).

| Panel / metric | Released | Final adapted |
|---|---:|---:|
| **CPU GGUF policy accuracy** | **28/64 (43.75%)** | **61/64 (95.31%)** |
| CPU GGUF policy log-loss | 1.557323 | 0.353121 |
| CPU GGUF policy Brier | 0.839295 | 0.093291 |
| BF16 policy accuracy | 28/64 | 62/64 |
| BF16 policy log-loss | 1.547931 | 0.262049 |
| BF16 authored retention accuracy | 24/24 | 24/24 |
| BF16 authored retention log-loss | 0.0431644 | 0.00001612 |
| CPU GGUF authored retention accuracy | 24/24 | 24/24 |
| CPU GGUF authored retention log-loss | 0.0542042 | 0.00001899 |

Sources: [final summary](examples/policy-results/final-summary.json), [released GGUF predictions](examples/policy-results/released-gguf-baseline.json), [adapted GGUF predictions on the same local host and 4-thread settings](examples/policy-results/local-gguf-policy-after.json), [final BF16 evaluation](examples/policy-results/policy-bf16-300-eval.json), and [all-64-row export-host parity](examples/policy-results/Policy-300-GGUF.parity.json). Both CPU hosts measured 61/64 adapted decisions correct; probabilities differ slightly by hardware/runtime arithmetic. No retention warning was triggered on the small authored panel. This is not evidence that broad original capabilities are preserved.

The real FP32 merge→F16 conversion→Q4_K_M quantization→CPU parity run took **388.6 seconds**. Agreement with the BF16 endpoint is **63/64 (98.44%)**. **One quantization flip changed a probability by 0.9983:** passing the top-label gate does not imply calibrated or numerically interchangeable probabilities. Three CPU GGUF dev errors remain. Evaluate your actual domains and calibrate on fresh data before using probabilities as risk estimates.

The exported model served through the public CPU HTTP server on a nonproduction port. Two real `curl` requests changed only the contract code: S1 selected **revenue**, while C7 selected **identity**, on the identical message and menu. Both generated zero tokens; the artifact identity is `openchoice-gguf-6b9b4d1ff8c1002d-q4km`. See [S1 request](examples/policy-results/http-request-S1.json), [S1 response](examples/policy-results/http-response-S1.json), [C7 request](examples/policy-results/http-request-C7.json), [C7 response](examples/policy-results/http-response-C7.json).

### What was tried, and what this does not establish

This is an **exploratory authored DEV tutorial, not a blind final benchmark**. One company policy was authored; no train/dev message text or labels changed after endpoint evaluation. Before the final baseline, the option-order confound in an initial counterfactual generator was corrected and an initial nine-update run was discarded ([correction](examples/policy-results/menu-correction.json)). The initial 150-update QLoRA endpoint reached 60/64 in BF16 but failed the Q4 gate: 47/64 agreement and 44/64 correct. Higher-precision merging improved that to 51/64 agreement and 47/64 correct, still a failure. BF16 LoRA at 150 updates remained underfit (38/64; recent train loss approximately 0.42–0.62), and was continued to the final 300-update endpoint. The **90% gate was never lowered**, and every failed result is retained in `examples/policy-results/` alongside the [precision comparison protocol](examples/policy-results/precision-comparison-protocol.json) and [continuation rationale](examples/policy-results/bf16-continuation-protocol.json).

This demonstrates learning a private ownership rule and deploying the measured result, not a general capability gain, an independent customer benchmark, or superiority over ordinary code/explicit prompting. Earlier mechanics-only smoke receipts in `examples/results/` used a different templated dataset and are **not scores for this policy dataset**. No weights, model card or repository were published by these runs.

**Rental closure:** The policy-training Vast rental was destroyed after all package manifest hashes and the retrieved checkpoint were verified. The provider then listed **zero instances**. Currently displayed charges are **$0.545** (GPU $0.476 + storage $0.069), covering the full policy experiment, failed trials, export/debugging and retrieval—not only final training. Provider accounting may post later charges; this is the displayed amount, not a guaranteed final invoice. Private rental IDs are redacted from the public receipts: [billing](examples/policy-results/billing.json), [destruction](examples/policy-results/destroy-instance.json), [empty instance list](examples/policy-results/instances-after.json), [retrieval](examples/policy-results/retrieval.json). The earlier, separate mechanics-only rental cost $0.619 ([historical billing](examples/results/billing.json)).

CPU GGUF retention was measured separately on the same host on all 24 authored Choice/Score/Noul dev rows: [before](examples/policy-results/gguf-retention-before.json), [after](examples/policy-results/local-gguf-retention-after.json). It is a small smoke panel, not broad retention certification.
