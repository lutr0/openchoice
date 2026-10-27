"""Small deterministic LoRA trainer and paired user/retention evaluation."""
import gc
import json
import math
import os
from pathlib import Path
import random
import time

from openchoice.integrity import sha256_file
from .data import BASE, REVISION, EXAMPLES, dataset_hash, inspect_data, load_data, require, check_replay_leakage


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def setup(seed, threads):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    return torch


def load_model(path, device, qlora=False, seed=42):
    from decision_training.model import DecisionModel
    if str(path) == "base":
        return DecisionModel(dict(
            model_id=BASE, revision=REVISION, arm="slot", dtype="bfloat16",
            initialization="pretrained", lora_rank=16, lora_alpha=32,
            lora_dropout=0, slot_loss="candidate", score_loss="independent_bce",
            max_sequence_tokens=4096, max_decision_tokens=32768,
            gradient_checkpointing=True, seed=seed,
        ), device, quantization="nf4" if qlora else None)
    return DecisionModel.load_pretrained(Path(path), device, quantization="nf4" if qlora else None)


def predictions(model, examples):
    import torch
    model.eval()
    records = []
    with torch.inference_mode():
        for e in examples:
            out = model(e.row, e.root)
            logits = out.logits.double()
            if e.row["primitive"] == "noul":
                logits = torch.stack((logits, torch.zeros_like(logits)))
            probs = logits.softmax(-1).cpu().tolist()
            target = e.row["target"]
            if e.row["primitive"] == "noul":
                truth = [float(target["value"]), 1 - float(target["value"])]
            elif target["kind"] == "class":
                truth = [float(i == e.gold) for i in range(len(probs))]
            else:
                truth = [target["values"][str(i)] for i in range(len(probs))]
            pred = max(range(len(probs)), key=probs.__getitem__)
            records.append(dict(id=e.row["id"], primitive=e.row["primitive"], gold=e.gold,
                probabilities=probs, prediction=pred, accuracy=float(pred == e.gold),
                brier=sum((p - t) ** 2 for p, t in zip(probs, truth)),
                log_loss=-sum(t * math.log(max(p, 1e-15)) for p, t in zip(probs, truth))))
    return records


def summarize(records):
    require(records, "evaluation split is empty")
    metrics = {key: sum(r[key] for r in records) / len(records) for key in ("accuracy", "brier", "log_loss")}
    return dict(rows=len(records), **metrics)


def retention_data(path=None):
    return load_data(path or EXAMPLES / "general-decisions.jsonl", require_split=True)


def run_train(args):
    torch = setup(args.seed, args.threads)
    require(args.device.startswith("cuda") and torch.cuda.is_available(), "train requires a CUDA GPU; use a rented 24/48 GB GPU (CPU serving needs no GPU)")
    require(args.steps > 0 and args.batch_size > 0 and args.checkpoint_every > 0, "steps, batch size and checkpoint interval must be positive")
    require(math.isfinite(args.lr) and args.lr > 0 and 0 <= args.replay_fraction < 1, "lr must be positive; replay fraction must be in [0,1)")
    output = Path(args.output)
    require(args.resume or not output.exists(), f"{output} exists; choose a new output or --resume a checkpoint")
    user = load_data(args.data, seed=args.seed, dev_fraction=args.dev_fraction)
    replay = retention_data(args.replay)
    check_replay_leakage(user, replay)
    # General DEV is never admitted to training, even when replay is enabled.
    train = [e for e in user if e.row["split"] == "train"]
    general = [e for e in replay if e.row["split"] == "train"]
    contract = dict(data_sha256=dataset_hash(user), replay_sha256=dataset_hash(replay),
        seed=args.seed, batch_size=args.batch_size, lr=args.lr,
        replay_fraction=args.replay_fraction, qlora=args.qlora, initialize=str(args.init))
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    resume = Path(args.resume) if args.resume else None
    model = load_model(resume / "model" if resume else args.init, args.device, args.qlora, args.seed)
    validation = inspect_data(user + replay, model.processor)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
    step = 0
    if resume:
        require(json.loads((resume / "contract.json").read_text()) == contract, "resume data/hyperparameters differ; restore original arguments or start a new run")
        state = torch.load(resume / "optimizer.pt", map_location=args.device, weights_only=True)
        optimizer.load_state_dict(state["optimizer"])
        step = state["step"]
        torch.set_rng_state(state["rng_cpu"].cpu())
        torch.cuda.set_rng_state_all([s.cpu() for s in state["rng_cuda"]])
        require(args.steps > step, f"checkpoint already completed {step} steps; --steps is the desired total")
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "validation.json", validation)
    save_json(output / "contract.json", contract)
    model.train()
    history = []
    def checkpoint(number):
        directory = output / f"checkpoint-{number:06d}"
        directory.mkdir(exist_ok=False)
        model.save_pretrained(directory / "model")
        torch.save(dict(step=number, optimizer=optimizer.state_dict(), rng_cpu=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state_all()), directory / "optimizer.pt")
        save_json(directory / "contract.json", contract)
        save_json(output / "latest.json", {"checkpoint": directory.name})
    try:
        for step in range(step + 1, args.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            # Sampling is keyed by absolute update, independent of resumes.
            rng = random.Random(f"{args.seed}/{step}")
            loss_sum = tokens = 0
            for _ in range(args.batch_size):
                source = general if rng.random() < args.replay_fraction else train
                e = source[rng.randrange(len(source))]
                result = model(e.row, e.root)
                require(bool(torch.isfinite(result.loss)), f"nonfinite loss at {e.row['id']}; lower learning rate")
                (result.loss / args.batch_size).backward()
                loss_sum += result.loss.detach().item() / args.batch_size
                tokens += result.processed_tokens
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            record = dict(step=step, loss=loss_sum, tokens=tokens, elapsed_seconds=time.monotonic() - started)
            history.append(record)
            print(json.dumps(record), flush=True)
            if step % args.checkpoint_every == 0 or step == args.steps:
                checkpoint(step)
    except KeyboardInterrupt:
        # Do not save a partially accumulated update. The last committed
        # checkpoint remains independently resumable.
        print("Interrupted; resume from the checkpoint in latest.json", flush=True)
        raise
    model.save_pretrained(output / "model")
    report = dict(schema_version="openchoice-finetune-run-v1", steps=args.steps, updates_this_invocation=len(history),
        elapsed_seconds=time.monotonic() - started, gpu=torch.cuda.get_device_name(),
        peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
        peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
        mode="qlora-nf4" if args.qlora else "lora-bfloat16", contract=contract, history=history,
        model_sha256=sha256_file(output / "model/model.json"))
    save_json(output / "train-report.json", report)
    return report


def run_eval(args):
    torch = setup(args.seed, args.threads)
    user = load_data(args.data, seed=args.seed, dev_fraction=args.dev_fraction)
    panels = {"dev": [e for e in user if e.row["split"] == "dev"],
        "retention": [e for e in retention_data(args.retention) if e.row["split"] == "dev"]}
    result = dict(schema_version="openchoice-finetune-eval-v1", data_sha256=dataset_hash(user),
        retention_sha256=dataset_hash(panels["retention"]), runtime="nf4" if args.qlora else "bfloat16",
        before={}, after={}, calibration="uncalibrated; not an accuracy guarantee")
    for key, path in (("before", args.before), ("after", args.after)):
        model = load_model(path, args.device, args.qlora, args.seed)
        inspect_data(user, model.processor)
        for name, panel in panels.items():
            records = predictions(model, panel)
            result[key][name] = dict(summary=summarize(records), predictions=records)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    result["delta"] = {name: {metric: result['after'][name]['summary'][metric] - result['before'][name]['summary'][metric]
        for metric in ('accuracy', 'brier', 'log_loss')} for name in panels}
    result["retention_warning"] = result["delta"]["retention"]["accuracy"] < -0.02 or result["delta"]["retention"]["log_loss"] > 0.02
    result["retention_note"] = "Small authored smoke panel, not a general capability guarantee; evaluate your own held-out domains."
    save_json(args.output, result)
    return {key: value for key, value in result.items() if key not in {'before', 'after'}} | {key: {name: panel['summary'] for name, panel in result[key].items()} for key in ('before', 'after')}
