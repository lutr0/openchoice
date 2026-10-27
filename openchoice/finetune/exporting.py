"""Merge, convert and quantize a trained adapter, then measure GGUF parity."""
import gc
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

from openchoice.integrity import sha256_file
from .data import BASE, REVISION, load_data, require
from .training import load_model, predictions, save_json, setup

LLAMA_COMMIT = "4364bf7232e65c34eca8d9500c5464389662de6b"


def run_export(args):
    torch = setup(args.seed, args.threads)
    output = Path(args.output).resolve()
    runtime = Path(args.runtime).resolve()
    llama = runtime / "llama.cpp"
    require(not output.exists(), f"{output} exists; export to a new directory")
    require((llama / "convert_hf_to_gguf.py").is_file(), "run bash scripts/build_llamacpp.sh cpu first; --runtime must contain llama.cpp")
    revision = subprocess.check_output(["git", "-C", str(llama), "rev-parse", "HEAD"], text=True).strip()
    require(revision == LLAMA_COMMIT, f"llama.cpp must be pinned to {LLAMA_COMMIT}")
    source = Path(args.projector_from).resolve()
    manifest = json.loads((source / "manifest.json").read_text())
    require(manifest["base_model"] == BASE and manifest["base_revision"] == REVISION, "projector package must use the pinned Qwen3.5-9B base")
    projector = (source / manifest["projector"]).resolve()
    require(projector.is_relative_to(source) and sha256_file(projector) == manifest["files"][manifest["projector"]], "base projector integrity mismatch")
    examples = [e for e in load_data(args.data, seed=args.seed, dev_fraction=args.dev_fraction) if e.row["split"] == "dev"]
    require(args.parity_rows > 0, "parity rows must be positive")
    examples = examples[:args.parity_rows]
    started = time.monotonic()
    model = load_model(args.checkpoint, args.device)
    require(model.config["model_id"] == BASE and model.config["revision"] == REVISION,
            "export requires an adapter trained on the pinned Qwen3.5-9B base")
    reference = predictions(model, examples)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Work and final package are siblings: only a complete package is promoted.
    work = Path(tempfile.mkdtemp(prefix=output.name + ".export-", dir=output.parent))
    package = work / "package"
    package.mkdir()
    try:
        # Small LoRA deltas can vanish in a BF16 merge. Accumulate them in
        # FP32 on the CPU, then retain F16 mantissa precision for quantization.
        model.backbone.to(device="cpu", dtype=torch.float32)
        merged = model.backbone.merge_and_unload().to(dtype=torch.float16)
        # The vision tower is frozen; reuse its mmproj and convert text only.
        text_weights = {key: value for key, value in merged.state_dict().items()
                        if not key.startswith("model.visual.")}
        merged.save_pretrained(work / "f16", state_dict=text_weights, safe_serialization=True, max_shard_size="4GB")
        model.processor.save_pretrained(work / "f16")
        model.processor.save_pretrained(package / "processor")
        merged.config.save_pretrained(package / "processor")
        del text_weights, merged, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        subprocess.run([sys.executable, str(llama / "convert_hf_to_gguf.py"), str(work / "f16"), "--outfile", str(work / "model-f16.gguf"), "--outtype", "f16", "--no-mtp"], check=True)
        quantizer = llama / "build-cpu/bin/llama-quantize"
        if not quantizer.is_file():
            subprocess.run(["cmake", "--build", str(llama / "build-cpu"), "--target", "llama-quantize", "--parallel", str(min(args.threads, 4))], check=True)
        subprocess.run([str(quantizer), str(work / "model-f16.gguf"), str(package / "openchoice-Q4_K_M.gguf"), "Q4_K_M", str(args.threads)], check=True)
        checkpoint = Path(args.checkpoint)
        shutil.copytree(checkpoint / "adapter", package / "adapter")
        shutil.copyfile(checkpoint / "heads.safetensors", package / "heads.safetensors")
        shutil.copyfile(checkpoint / "model.json", package / "model.json")
        shutil.copyfile(projector, package / "mmproj-q8_0.gguf")
        manifest = dict(schema_version="openchoice-gguf-v1", quantization="Q4_K_M",
            model="openchoice-Q4_K_M.gguf", projector="mmproj-q8_0.gguf", processor="processor",
            checkpoint_config="model.json", base_model=BASE, base_revision=REVISION,
            llama_commit=LLAMA_COMMIT,
            files={str(p.relative_to(package)): sha256_file(p) for p in sorted(package.rglob("*")) if p.is_file()})
        save_json(package / "manifest.json", manifest)
        from openchoice.llamacpp import LlamaCppEngine
        records = []
        with LlamaCppEngine(package, threads=args.threads, runtime=runtime) as engine:
            for e, ref in zip(examples, reference, strict=True):
                result = engine.predict_rows([e.row], e.root)
                logits = result.logits[0]
                maximum = max(logits)
                exp = [math.exp(x - maximum) for x in logits]
                probs = [x / sum(exp) for x in exp]
                pred = max(range(len(probs)), key=probs.__getitem__)
                records.append(dict(id=e.row["id"], pytorch=ref["probabilities"], gguf=probs,
                    gold=e.gold, agrees=pred == ref["prediction"], gguf_correct=pred == e.gold,
                    max_probability_error=max(abs(a - b) for a, b in zip(probs, ref["probabilities"]))))
        agreement = sum(r["agrees"] for r in records) / len(records)
        report = dict(schema_version="openchoice-export-parity-v1", rows=len(records),
            decision_agreement=agreement, gguf_accuracy=sum(r["gguf_correct"] for r in records) / len(records),
            max_probability_error=max(r["max_probability_error"] for r in records),
            elapsed_seconds=time.monotonic() - started, reference_runtime="pytorch-bfloat16",
            output_runtime="llama.cpp-Q4_K_M-cpu", predictions=records,
            warning="Q4 is lossy: parity is measured, not exact equality; reevaluate your full held-out data before deployment.",
            checkpoint_sha256=sha256_file(checkpoint / "model.json"), model_sha256=manifest["files"]["openchoice-Q4_K_M.gguf"])
        save_json(output.with_name(output.name + ".parity.json"), report)
        require(agreement >= args.min_agreement, f"parity agreement {agreement:.1%} below --min-agreement {args.min_agreement:.1%}; report saved, package not promoted")
        package.rename(output)
        return {k: v for k, v in report.items() if k != "predictions"}
    finally:
        shutil.rmtree(work)
