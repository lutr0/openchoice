"""Torch-free native prompt and row contract shared by serving and training."""

import json
import string

from .inference import Decision, UnsupportedDecision


def quote(value: str) -> str:
    """JSON strings with literal native control-token delimiters disabled."""
    return json.dumps(value, ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e")


def render_decision_prompt(processor, row, candidates, aliases=False):
    """The native prompt contract, shared by training and exact-logit GGUF readers."""
    body = "Evidence: " + quote(row["context"]) + "\nQuestion: " + quote(row["question"]) + "\n"
    endpoints = []
    if candidates:
        body += "Options:\n" if aliases or len(candidates) > 1 else "Candidate:\n"
        for i, candidate in enumerate(candidates):
            if aliases:
                body += string.ascii_uppercase[i] + ": "
            body += quote(candidate)
            endpoints.append(len(body))
            body += "\n"
    body += "Choose the option alias." if aliases else "Does the evidence support this decision? Answer Yes or No."
    content = [{"type": "image"} for _ in row.get("assets", [])]
    content.append({"type": "text", "text": body})
    rendered = processor.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    if rendered.count(body) != 1:
        raise RuntimeError("Native template did not preserve the constructed input")
    start = rendered.index(body)
    return rendered, [start + end for end in endpoints]


def native_answer_ids(tokenizer, rendered, answers):
    """Require distinct single-token answers in their actual rendered context."""
    prefix = tokenizer.encode(rendered, add_special_tokens=False)
    ids = []
    for answer in answers:
        completed = tokenizer.encode(rendered + answer, add_special_tokens=False)
        if completed[:-1] != prefix or len(completed) != len(prefix) + 1:
            raise ValueError(f"Native answer {answer!r} is not one token in actual prompt context")
        ids.append(completed[-1])
    if len(set(ids)) != len(ids):
        raise ValueError("Native answer aliases are not distinct")
    return ids


def decision_row(state: str, decision: Decision) -> dict:
    primitive = decision.primitive
    if primitive not in {"choice", "score", "noul"}:
        raise UnsupportedDecision("Unknown native decision primitive")
    count = len(decision.options)
    if primitive == "choice" and not 1 <= count <= 26:
        raise UnsupportedDecision("Native backend supports 1–26 Choice options; menus are never cropped or sharded")
    if primitive == "score" and not 1 <= count <= 10:
        raise UnsupportedDecision("Native backend supports 1–10 Score levels")
    if primitive == "noul" and count != 2:
        raise UnsupportedDecision("Native Noul requires true/false response slots")
    if primitive != "noul" and any(not text.strip() for text in decision.options):
        raise UnsupportedDecision("Native backend requires nonblank option descriptions")
    question = decision.instructions or {
        "choice": "Choose the matching option.",
        "score": "Evaluate the evidence against the criterion.",
        "noul": "The statement in the state is true.",
    }[primitive]
    if not question.strip():
        raise UnsupportedDecision("Native backend requires nonblank instructions")
    if primitive == "noul":
        for label, criterion in (("True", decision.true_criterion), ("False", decision.false_criterion)):
            if criterion is not None:
                question += f"\n{label} criterion: {quote(criterion)}"
    return {
        "schema_version": "openchoice-decision-v1", "id": "request", "group_id": "request",
        "source_id": "http", "split": "probe", "primitive": primitive, "family": "serving", "view": "base",
        "context": state, "question": question,
        "options": [] if primitive == "noul" else [
            {"id": str(index), "text": text} for index, text in enumerate(decision.options)
        ],
        "target": None, "target_mask": False, "provenance": {"source": "http-request"},
    }
