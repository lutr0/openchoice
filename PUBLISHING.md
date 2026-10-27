# Publishing

OpenChoice-9B 0.2.0 is published at:

- Source: https://github.com/lutr0/openchoice (single initial commit; the private training/research repository is not published)
- Model package: https://huggingface.co/lutr0/OpenChoice-9B
- CPU container: `ghcr.io/lutr0/openchoice:0.2.0-cpu` (linux/amd64 and linux/arm64)
- Release and launch video: https://github.com/lutr0/openchoice/releases/tag/v0.2.0

The steps below are the maintainer procedure for future releases.

## Before publishing

1. Review README.md, RESULTS.md, FINETUNING.md, LAUNCH.md, the model card, and a secret/privacy scan. Keep GGUF-measured, historical NF4-measured, and comparator-published numbers distinct. Do not carry the old NF4 scores onto an unlabeled GGUF scoreboard.
2. Re-run the CPU quickstart and SDK smoke from a clean checkout against the final model package. Verify the exact package hashes, not merely its directory name.
3. Re-render the launch assets if scenes or accounts changed (below).

## GitHub

Publish this source checkout, not the private training/research repository. The public source includes the GGUF bridge, a pinned CPU/CUDA build script, a portable CPU Dockerfile, the fine-tuning kit, raw result ledgers, and launch assets. `.runtime/`, `.venv/`, model weights, caches, and `media/launch/out/` are intentionally ignored. No binary runtime libraries belong in the source release; the container builds and packages them separately.

## Container images

The Dockerfile builds a portable CPU runtime from the same pinned llama.cpp source and exact-logit bridge. It excludes model weights, host runtimes/virtualenvs, results and secrets from the build context. Runtime layers contain only Python, uv-synced serving dependencies, runtime shared libraries and source/license notices, not build or training dependencies. Models are downloaded separately into a writable volume and mounted read-only for serving. No CUDA image target is provided.

Local build and validation (does not publish):

```sh
docker buildx build --platform linux/amd64 --load -t openchoice:cpu .
docker buildx build --platform linux/arm64 --load -t openchoice:cpu-arm64 .
```

The second command needs a native arm64 builder or registered QEMU/binfmt emulation; it may be much slower. Run each image against the final package using the README mount/authentication commands; use `--platform linux/arm64` and the arm64 tag for an emulated run. Check `/health`, the README Choice/Score/Noul request, and `uv run --group dev python scripts/check_api.py --base-url http://127.0.0.1:8768 --api-key "$OPENCHOICE_API_KEY" --output /tmp/container-sdk-smoke.json` (the output file must not already exist). Compare with native probabilities; CPU variants/compiler/architecture can change float results. Record identity, size, startup and tested platform scope as in `results/container-verification.json`. Do not label build-only platforms as runtime-tested or QEMU as Apple Silicon hardware.

Release with an immutable tag:

```sh
gh auth token | docker login ghcr.io -u lutr0 --password-stdin
docker buildx create --name openchoice-release --driver docker-container --use
docker buildx build --platform linux/amd64,linux/arm64 -t ghcr.io/lutr0/openchoice:<version>-cpu -t ghcr.io/lutr0/openchoice:cpu --push .
```

The multi-platform push needs a buildx builder that supports both architectures (Docker's containerd image store or a docker-container builder) and a token with `write:packages`. Keep registry credentials out of shell history/logs. New GHCR packages start private: set visibility to public in the package settings. Confirm the published manifest with `docker buildx imagetools inspect ghcr.io/lutr0/openchoice:<version>-cpu`, record its digest, then test a fresh pull on each advertised platform. Preserve runtime and Python dependency license notices when redistributing binaries; model licensing is separate.

## Hugging Face package

The staged **OpenChoice-9B** package retains the same repository name so both serving and fine-tuning artifacts have one source of truth. Users can download it into a directory named `OpenChoice-9B-GGUF`.

Required serving contents:

```text
manifest.json
openchoice-Q4_K_M.gguf
mmproj-q8_0.gguf
processor/
model.json
```

Fine-tuning and attribution contents remain alongside them: `adapter/`, `heads.safetensors`, `README.md`, `LICENSE`, `LICENSE-QWEN`, `NOTICE`. The slot model reads final-token logits; `heads.safetensors` contains no trained classifier tensors. Retaining it preserves the fine-tuning checkpoint contract, not an obsolete serving path. No BF16 base weights are bundled; fine-tuning downloads the pinned upstream base separately.

`manifest.json` uses schema `openchoice-gguf-v1`. Fields: `quantization`, `model`, `projector`, `processor`, `checkpoint_config`, `base_model`, `base_revision`, `llama_commit`, and `files` (relative POSIX file paths mapped to SHA-256). Hash all serving files and every processor file; the staged release also hashes adapter/checkpoint files. There are no build-host names or absolute paths. The manifest itself is not self-hashed. Documentation/license text does not affect model identity. Local compiled libraries live outside the package in `.runtime/`.

Upload (write token via `uv run hf auth login`):

```sh
uv run hf repos create lutr0/OpenChoice-9B --repo-type model
uv run hf upload lutr0/OpenChoice-9B <model-package-dir> .
```

Keep write tokens out of files and shell logs. After upload, download into a fresh directory, verify hashes via startup, and check card metadata, all artifacts, and final links in the Hub UI. The public source quickstart uses `uv run hf download ...`, which requires no write credential.

## Launch assets

- Video: `media/launch/out/openchoice-launch.mp4` (1920×1080, 60 fps, 54 s; render output is gitignored). Upload it separately or attach it as a GitHub release asset.
- Stills: `assets/openchoice-scoreboard.png`, `assets/openchoice-capabilities.png`, `assets/openchoice-cpu.png`, `assets/openchoice-finetune.png`, and `assets/openchoice-social.png` (1200×630).
- Launch copy: [LAUNCH.md](LAUNCH.md).
- Re-render: `node media/launch/render.mjs` (video) and `node media/launch/render.mjs --assets` (stills); accounts default to `lutr0`. See [media/launch/README.md](media/launch/README.md) for renderer prerequisites and verification.

Post only once both public repositories and their links work. Keep **~5.8 GiB serving weights / ~6.0 GiB complete download**, CPU vision's Python-only surface, and measured-versus-published/runtime labels. Lead with the CPU-first multimodal deployment plus integrated tuning/export path, not a claim of being first, only, smallest, fastest, or best.

## Official Benchmark Heaven evaluation

After publication, [request evaluation from Benchmark Heaven](https://benchmarkheaven.com/jev-models/custom-evaluation) / the [JevBench maintainer](https://github.com/fstandhartinger/jevbench). Supply the public model URL, exact GGUF/projector hashes, pinned runtime, CPU/GPU settings, request contract, and reproduction instructions. Request both **JevBench v1.5.1** and **ImageJevBench v0.1.3**, agreeing scope/cost with the maintainer before any paid run.

JevBench v1.5.1 uses **904 open + 720 sealed decisions**; its official score combines Intelligence, Calibration, Speed, and Cost. ImageJevBench v0.1.3 uses **228 public + 456 sealed items**. A local public-only result cannot establish either official composite or sealed standing. Until maintainer evaluation, label OpenChoice **pending (sealed split requires maintainer submission)**. Do not turn legacy raw accuracy into a current leaderboard score, and do not infer new scores for changed GGUF exports.

## Licensing and rights

- **Weights license:** CC-BY-SA-4.0, a conservative declaration because BoolQ is CC-BY-SA-3.0 and SNLI is CC-BY-SA-4.0. Code is Apache-2.0 and the Qwen base is Apache-2.0.
- **Share-alike interpretation and data rights:** the declaration is not a legal determination of how source-data terms apply to trained and redistributed weights.
- **Training-source attribution** is listed in NOTICE and the model card, including internally authored replay material.
- **Evaluation data:** raw result ledgers omit benchmark input text. SciQ is noncommercial evaluation data, not training data in this project; do not interpret published metrics as a grant to redistribute its dataset.
- **Names and comparisons:** Jev, TypeSafe, Laya, and other model names are nominative references only. Preserve distinctions between local measurements and third-party reports.

If an actual credential was exposed, revoke it promptly. Do not rotate credentials merely because a scan was run.
