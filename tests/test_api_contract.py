"""Consumer-visible HTTP boundaries with a tiny deterministic scorer, not model proof.

Run with python -m unittest discover -s tests -p test_api_contract.py.
Real GGUF/official-SDK evidence belongs to scripts/check_api.py.
"""

import json
import math
from pathlib import Path
import tempfile
import unittest

from fastapi.testclient import TestClient

from openchoice.inference import BatchPrediction, InputTooLong
from openchoice.server import create_app


class TinyEngine:
    """Content-dependent toy logits make position/ID leakage observable."""

    models = {"openchoice-tiny": {"kind": "test-only", "temperature": 1.0}}
    default_model = "openchoice-tiny"

    def resolve(self, model):
        if model in (self.default_model, "openchoice-latest", "jev-latest"):
            return self.default_model
        raise KeyError(model)

    def predict(self, model, state, questions, cancel=None):
        if len(state) > 10000 or any(
            len(option) > 10000 for q in questions for option in q.options
        ):
            raise InputTooLong("Toy engine input budget exceeded")
        rows = []
        for question in questions:
            # Stable semantic markers, deliberately independent of row/option position.
            rows.append(
                [
                    math.log(
                        4
                        if "refund" in option.lower()
                        else 2
                        if "invoice" in option.lower()
                        else 1
                    )
                    for option in question.options
                ]
            )
        return BatchPrediction(self.resolve(model), rows, 7)


CHOICE = {
    "type": "choice",
    "instructions": "Classify the message",
    "criteria": {"a": "weather", "b": "invoice", "c": "refund"},
}


class APIContractTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(
            create_app(engine=TinyEngine(), api_key="test-local")
        )
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        self.headers = {"Authorization": "Bearer test-local"}

    def post(self, questions=None, **updates):
        payload = {
            "model": "openchoice-latest",
            "state": "Please refund my invoice",
            "questions": {"topic": CHOICE} if questions is None else questions,
        }
        payload.update(updates)
        return self.client.post("/v1/systemone", json=payload, headers=self.headers)

    def answers(self, questions, **updates):
        response = self.post(questions, **updates)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["model"], "openchoice-tiny")
        self.assertEqual(set(response.json()["answers"]), set(questions))
        return response.json()["answers"]

    def error(self, response, status=422):
        self.assertEqual(response.status_code, status, response.text)
        self.assertIn("detail", response.json())

    def test_choice_score_noul_formulas_and_nested_legend(self):
        rubric = [
            {"description": ["weather", {"nested": [None, True, 3]}]},
            ["invoice"],
            {"description": "refund"},
        ]
        answers = self.answers(
            {
                "choice": CHOICE,
                "score": {"type": "score", "criteria": rubric},
                "noul": {
                    "type": "noul",
                    "criteria": {"true": "refund", "false": "weather"},
                },
            }
        )
        choice = answers["choice"]
        self.assertEqual(choice["choice"], "c")
        for label, weight in zip(("a", "b", "c"), (1, 2, 4)):
            self.assertAlmostEqual(choice["probabilities"][label], weight / 7)
        score = answers["score"]
        self.assertEqual(
            score["legend"], {str(i): value for i, value in enumerate(rubric)}
        )
        self.assertAlmostEqual(score["score"], 10 / 7)
        for index, weight in enumerate((1, 2, 4)):
            self.assertAlmostEqual(score["probabilities"][str(index)], weight / 7)
        self.assertAlmostEqual(answers["noul"]["noul"], 4 / 5)
        self.assertAlmostEqual(choice["confidence"], 5 / 14)
        self.assertAlmostEqual(score["confidence"], 1 / 7)

    def test_permutation_renaming_and_batch_isolation(self):
        original = self.answers({"topic": CHOICE})["topic"]
        reversed_question = {
            **CHOICE,
            "criteria": dict(reversed(list(CHOICE["criteria"].items()))),
        }
        batch = self.answers(
            {
                "unrelated": {"type": "noul", "instructions": "Is it weather?"},
                "renamed/雪": reversed_question,
            }
        )
        self.assertEqual(original, batch["renamed/雪"])
        # Renaming labels with descriptions must remap probability keys, not alter semantics.
        relabeled = {
            **CHOICE,
            "criteria": {"x": "refund", "y": "weather", "z": "invoice"},
        }
        changed = self.answers({"x": relabeled})["x"]
        self.assertEqual(changed["choice"], "x")
        for old, new in (("a", "y"), ("b", "z"), ("c", "x")):
            self.assertAlmostEqual(
                original["probabilities"][old], changed["probabilities"][new]
            )

    def test_structured_state_and_criteria_preserve_json(self):
        question = {
            "type": "choice",
            "instructions": {"task": ["classify", {"nested": True}]},
            "criteria": {
                "selected": {"text": "refund", "nested": [None, False, 2]},
                "other": ["weather", {"nested": True}],
            },
        }
        for state in (
            {"text": "refund", "details": [None, True, 2]},
            ["refund", {"paid": False}],
        ):
            with self.subTest(state=state):
                self.assertEqual(
                    self.answers({"q": question}, state=state)["q"]["choice"],
                    "selected",
                )
        # Null descriptions mean the label itself is the semantic description.
        answer = self.answers(
            {"q": {"type": "choice", "criteria": {"weather": None, "refund": None}}}
        )["q"]
        self.assertEqual(answer["choice"], "refund")

    def test_singleton_and_maximum_question_sizes(self):
        answer = self.answers(
            {"q": {"type": "choice", "criteria": {"only": "refund"}}}
        )["q"]
        self.assertEqual(answer["probabilities"], {"only": 1.0})
        self.assertEqual(answer["choice"], "only")
        wide = {str(i): "refund" if i == 254 else "weather" for i in range(255)}
        answers = self.answers(
            {
                "wide": {"type": "choice", "criteria": wide},
                "score": {"type": "score", "criteria": ["weather"] * 9 + ["refund"]},
            }
        )
        self.assertEqual(answers["wide"]["choice"], "254")
        self.assertAlmostEqual(answers["wide"]["probabilities"]["254"], 4 / 258)
        self.assertAlmostEqual(answers["score"]["score"], (sum(range(9)) + 36) / 13)
        self.error(
            self.post(
                {
                    "q": {
                        "type": "choice",
                        "criteria": {str(i): "weather" for i in range(256)},
                    }
                }
            )
        )
        self.error(self.post({"q": {"type": "score", "criteria": ["weather"] * 11}}))

    def test_empty_question_map(self):
        self.error(self.post({}))

    def test_singleton_score(self):
        rubric = {"nested": ["refund", {"allowed": True}]}
        answer = self.answers({"q": {"type": "score", "criteria": [rubric]}})["q"]
        self.assertEqual(answer["score"], 0)
        self.assertEqual(answer["legend"], {"0": rubric})
        self.assertEqual(answer["probabilities"], {"0": 1.0})
        self.assertEqual(answer["confidence"], 1)

    def test_malformed_payloads_and_strict_types(self):
        for state in (False, True, None, 42):
            with self.subTest(state=state):
                self.error(self.post(state=state))
        for question in (
            {"type": "unknown"},
            {"type": "choice", "criteria": {}},
            {"type": "choice", "criteria": []},
            {"type": "choice", "criteria": {"x": 1}},
            {"type": "choice", "criteria": {"x": "ok"}, "instructions": False},
            {"type": "score", "criteria": []},
            {"type": "score", "criteria": ["ok", None]},
            {"type": "noul", "criteria": {"true": False}},
        ):
            with self.subTest(question=question):
                self.error(self.post({"q": question}))
        for payload in (
            [],
            None,
            {},
            {"state": "hello", "questions": {}},
            {"state": "hello", "model": 1, "questions": {}},
        ):
            with self.subTest(payload=payload):
                self.error(
                    self.client.post(
                        "/v1/systemone",
                        content=json.dumps(payload),
                        headers={**self.headers, "Content-Type": "application/json"},
                    )
                )

    def test_nonfinite_duplicate_and_malformed_json(self):
        valid = json.dumps(
            {
                "state": "refund",
                "model": "openchoice-latest",
                "questions": {"q": CHOICE},
            }
        )
        bodies = ["{"]
        for number in ("NaN", "Infinity", "-Infinity", "1e999"):
            bodies.append(
                valid.replace('"state": ', '"extra": ' + number + ', "state": ', 1)
            )
        bodies.append(valid.replace('"state": ', '"state": "first", "state": ', 1))
        bodies.append(
            valid.replace('"state": "refund"', '"state": {"nested": 1, "nested": 2}', 1)
        )
        for body in bodies:
            with self.subTest(body=body):
                self.error(
                    self.client.post(
                        "/v1/systemone",
                        content=body,
                        headers={**self.headers, "Content-Type": "application/json"},
                    )
                )

    def test_auth_unknown_model_body_and_token_errors(self):
        self.error(self.post(model="unrecognized-model"), 404)
        for headers in ({}, {"Authorization": "Bearer wrong"}):
            self.error(self.client.get("/v1/models", headers=headers), 401)
            self.error(self.client.post("/v1/systemone", json={}, headers=headers), 401)
        self.error(self.post(state="x" * (2 * 1024 * 1024)), 413)
        self.error(self.post(state="x" * 10001))
        self.error(
            self.post({"q": {"type": "choice", "criteria": {"long": "x" * 10001}}})
        )
        self.error(
            self.client.post("/v1/systemone", content="not-json", headers=self.headers),
            415,
        )


class PackageIntegrityTests(unittest.TestCase):
    def test_rejects_corrupt_or_escaping_artifacts_before_loading(self):
        import hashlib
        from openchoice.llamacpp import LLAMA_COMMIT, LlamaCppEngine

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "package"
            package.mkdir()
            data = b"not a model: integrity checks run before model loading"
            (package / "model.gguf").write_bytes(data)
            (root / "outside.gguf").write_bytes(data)
            (package / "linked.gguf").symlink_to(root / "outside.gguf")
            cases = [
                {"model.gguf": "0" * 64},
                {"../outside.gguf": hashlib.sha256(data).hexdigest()},
                {str(root / "outside.gguf"): hashlib.sha256(data).hexdigest()},
                {"linked.gguf": hashlib.sha256(data).hexdigest()},
            ]
            for files in cases:
                with self.subTest(files=files):
                    (package / "manifest.json").write_text(json.dumps({
                        "schema_version": "openchoice-gguf-v1", "quantization": "Q4_K_M",
                        "llama_commit": LLAMA_COMMIT, "files": files,
                    }))
                    with self.assertRaises(ValueError):
                        LlamaCppEngine(package)


if __name__ == "__main__":
    unittest.main()
