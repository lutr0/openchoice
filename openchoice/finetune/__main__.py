"""Validate, adapt, evaluate and export your own probability-only decisions."""
import argparse
import json
from pathlib import Path

from .data import BASE, REVISION, inspect_data, load_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "train", "eval", "export"):
        p = commands.add_parser(name)
        p.add_argument("--data", type=Path, required=True, help="User JSONL with state/question/label and optional split/image/probabilities")
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--dev-fraction", type=float, default=0.2, help="Used only when split is omitted on all rows")
        p.add_argument("--threads", type=int, default=8)
        if name != "validate":
            p.add_argument("--device", default="cuda")
            p.add_argument("--output", type=Path, required=True)
        if name in {"train", "eval"}:
            p.add_argument("--qlora", action="store_true", help="NF4 frozen backbone, trainable LoRA (training/evaluation only)")
        if name == "train":
            p.add_argument("--init", default="OpenChoice-9B-GGUF", help="Released package/adapter directory, or base for a fresh LoRA")
            p.add_argument("--steps", type=int, default=100)
            p.add_argument("--batch-size", type=int, default=8, help="Logical batch; microbatch is one decision")
            p.add_argument("--lr", type=float, default=1e-5)
            p.add_argument("--replay", type=Path, help="Optional replacement general JSONL; only its train split is used")
            p.add_argument("--replay-fraction", type=float, default=0.2)
            p.add_argument("--checkpoint-every", type=int, default=25)
            p.add_argument("--resume", type=Path, help="Committed checkpoint directory; keep original arguments")
        elif name == "eval":
            p.add_argument("--before", default="OpenChoice-9B-GGUF")
            p.add_argument("--after", type=Path, required=True)
            p.add_argument("--retention", type=Path, help="Optional replacement general panel; only dev is evaluated")
        elif name == "export":
            p.add_argument("--checkpoint", type=Path, required=True)
            p.add_argument("--projector-from", type=Path, default=Path("OpenChoice-9B-GGUF"))
            p.add_argument("--runtime", type=Path, default=Path(".runtime"))
            p.add_argument("--parity-rows", type=int, default=16)
            p.add_argument("--min-agreement", type=float, default=0.9)
    args = parser.parse_args()
    try:
        if args.command == "validate":
            from transformers import AutoProcessor
            processor = AutoProcessor.from_pretrained(BASE, revision=REVISION, trust_remote_code=False, token=False)
            result = inspect_data(load_data(args.data, seed=args.seed, dev_fraction=args.dev_fraction), processor)
        elif args.command == "train":
            from .training import run_train
            result = run_train(args)
        elif args.command == "eval":
            from .training import run_eval
            result = run_eval(args)
        else:
            from .exporting import run_export
            result = run_export(args)
        print(json.dumps(result, indent=2, allow_nan=False))
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(2, f"{exc}\n")


if __name__ == "__main__":
    main()
