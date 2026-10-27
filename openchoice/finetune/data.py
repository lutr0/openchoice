"""Strict, serving-compatible user JSONL and grouped train/dev splitting."""
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random

from openchoice.integrity import content_hash, sha256_file
from openchoice.inference import Decision
from openchoice.native import decision_row, render_decision_prompt
from openchoice.server import SystemOneRequest, _text, _unique_object, _invalid_constant

BASE = "Qwen/Qwen3.5-9B"
REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


@dataclass
class Example:
    row: dict
    gold: int
    labels: list
    root: Path
    request: dict


def require(condition, message):
    if not condition:
        raise ValueError(message)


def convert(value, root, index):
    require(isinstance(value, dict), "each line must be a JSON object")
    require(not set(value) - {"id", "group", "split", "state", "image", "question", "label", "probabilities"}, "unknown row fields")
    require({"state", "question", "label"} <= value.keys(), "required fields: state, question, label")
    question = value["question"]
    require(isinstance(question, dict) and not set(question) - {"type", "instructions", "criteria"}, "question fields: type, instructions, criteria")
    if question.get("type") == "noul" and isinstance(question.get("criteria"), dict):
        require(not set(question["criteria"]) - {"true", "false"}, "Noul criteria keys must be true and false")
    request = SystemOneRequest.model_validate(dict(state=value["state"], model="openchoice-latest", questions={"decision": question}))
    q = request.questions["decision"]
    true = false = None
    if q.type == "choice":
        labels = list(q.criteria)
        options = [key if val is None else _text(val) for key, val in q.criteria.items()]
        require(isinstance(value["label"], str) and value["label"] in labels, "Choice label must be a criteria key")
    elif q.type == "score":
        labels = list(range(len(q.criteria)))
        options = [_text(c) for c in q.criteria]
        require(type(value["label"]) is int and value["label"] in labels, "Score label must be a zero-based integer level")
    else:
        require(type(value["label"]) is bool, "Noul label must be JSON true or false, not a string")
        labels = [True, False]
        options = ["Yes", "No"]
        if q.criteria:
            true = _text(q.criteria.true) if q.criteria.true is not None else None
            false = _text(q.criteria.false) if q.criteria.false is not None else None
    row = decision_row(_text(request.state), Decision(_text(q.instructions), options, primitive=q.type, true_criterion=true, false_criterion=false))
    identifier = value.get("id", f"row-{index}")
    group = value.get("group", identifier)
    require(isinstance(identifier, str) and identifier.strip() and isinstance(group, str) and group.strip(), "id and group must be nonblank strings")
    split = value.get("split")
    require(split in {None, "train", "dev"}, "split must be train or dev")
    row.update(id=identifier, group_id=group, source_id="user", split=split, target_mask=True)
    gold = labels.index(value["label"])
    probs = value.get("probabilities")
    if probs is not None:
        require(isinstance(probs, list) and len(probs) == len(labels), "probabilities must be a list in criteria order (Noul: true, false)")
        require(all(type(p) in {int, float} and math.isfinite(p) and 0 <= p <= 1 for p in probs) and math.isclose(sum(probs), 1, abs_tol=1e-7), "probabilities must be finite, nonnegative and sum to one")
    if q.type == "noul":
        row["target"] = {"kind": "binary", "value": probs[0] if probs is not None else value["label"]}
    elif probs is None:
        row["target"] = {"kind": "class", "id": str(gold)}
    else:
        row["target"] = {"kind": "distribution", "values": {str(i): p for i, p in enumerate(probs)}}
    if "image" in value:
        name = value["image"]
        require(isinstance(name, str) and not Path(name).is_absolute() and ".." not in Path(name).parts, "image must be a relative path inside the JSONL directory")
        path = (root / name).resolve()
        require(path.is_relative_to(root.resolve()) and path.is_file(), f"image not found inside dataset: {name}")
        from PIL import Image
        with Image.open(path) as image:
            require(image.format in {"PNG", "JPEG"}, "image must be PNG or JPEG")
            media = Image.MIME[image.format]
            image.verify()
        row["assets"] = [{"path": name, "sha256": sha256_file(path), "media_type": media}]
    return Example(row, gold, labels, root, value)


def load_data(path, *, seed=42, dev_fraction=0.2, require_split=True):
    path = Path(path)
    examples, errors = [], []
    with path.open() as stream:
        for line, text in enumerate(stream, 1):
            if not text.strip():
                continue
            try:
                value = json.loads(text, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
                examples.append(convert(value, path.parent, line))
            except (ValueError, TypeError, OSError) as exc:
                errors.append(f"{path}:{line}: {exc}")
    if errors:
        raise ValueError("\n".join(errors))
    require(examples, f"{path}: no examples")
    require(len({e.row['id'] for e in examples}) == len(examples), "duplicate ids: assign a unique id to every decision")
    explicit = [e.row["split"] is not None for e in examples]
    require(all(explicit) or not any(explicit), "set split on every row or omit it on every row")
    if not any(explicit):
        require(0 < dev_fraction < 1, "dev fraction must lie between zero and one")
        groups = sorted({e.row["group_id"] for e in examples})
        random.Random(seed).shuffle(groups)
        heldout = set(groups[:max(1, round(len(groups) * dev_fraction))])
        for e in examples:
            e.row["split"] = "dev" if e.row["group_id"] in heldout else "train"
    group_splits, fingerprints = {}, {}
    for e in examples:
        split = e.row["split"]
        require(group_splits.setdefault(e.row["group_id"], split) == split, f"group {e.row['group_id']} leaks across train/dev")
        fingerprint = decision_fingerprint(e)
        require(fingerprints.setdefault(fingerprint, split) == split, f"duplicate decision {e.row['id']} leaks across train/dev")
    if require_split:
        require({e.row["split"] for e in examples} == {"train", "dev"}, "need nonempty train and dev splits; add examples or adjust dev fraction")
    return examples


def inspect_data(examples, processor):
    from PIL import Image
    counts = defaultdict(Counter)
    max_tokens = 0
    errors = []
    for e in examples:
        row = e.row
        counts[f"{row['split']}/{row['primitive']}"][str(e.labels[e.gold])] += 1
        options = [o["text"] for o in row.get("options", [])]
        candidates = [options] if row["primitive"] == "choice" else ([[o] for o in options] if row["primitive"] == "score" else [[]])
        images = []
        for asset in row.get("assets", []):
            with Image.open(e.root / asset["path"]) as image:
                images.append(image.convert("RGB"))
        decision_tokens = 0
        for candidate in candidates:
            prompt, _ = render_decision_prompt(processor, row, candidate, row["primitive"] == "choice")
            encoded = processor(text=[prompt], images=images or None, return_tensors="pt", truncation=False)
            tokens = encoded["input_ids"].shape[1]
            decision_tokens += tokens
            max_tokens = max(max_tokens, tokens)
            if tokens > 4096:
                errors.append(f"{row['id']}: {tokens} tokens >4096; shorten state/criteria or downsize image (no truncation)")
        if decision_tokens > 32768:
            errors.append(f"{row['id']}: {decision_tokens} total decision tokens >32768; shorten Score criteria/state")
    if errors:
        raise ValueError("\n".join(errors))
    warnings = []
    for name, balance in counts.items():
        if max(balance.values()) > 4 * min(balance.values()):
            warnings.append(f"{name}: >4:1 class imbalance; collect minority labels or assess per-class metrics")
    for e in examples:
        key = f"{e.row['split']}/{e.row['primitive']}"
        missing = sorted(set(map(str, e.labels)) - counts[key].keys())
        if missing:
            warnings.append(f"{key}: missing labels {missing}; add coverage before trusting accuracy")
    return dict(rows=len(examples), max_sequence_tokens=max_tokens, class_balance={k: dict(v) for k, v in counts.items()}, warnings=sorted(set(warnings)), splits={s: sum(e.row['split'] == s for e in examples) for s in ('train', 'dev')})


def dataset_hash(examples):
    return content_hash([dict(row=e.row, gold=e.gold) for e in examples])


def decision_fingerprint(example):
    row = example.row
    return content_hash({**{k: row.get(k) for k in ("context", "question", "options")},
                         "images": [a["sha256"] for a in row.get("assets", [])]})


def check_replay_leakage(user, replay):
    """Protect user DEV and general DEV across both input files."""
    trained = {decision_fingerprint(e) for e in user + replay if e.row["split"] == "train"}
    for e in user + replay:
        if e.row["split"] == "dev":
            require(decision_fingerprint(e) not in trained,
                    f"{e.row['id']}: held-out decision appears in user/replay training; remove the overlap")
