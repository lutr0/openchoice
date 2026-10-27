"""Live GGUF CPU/GPU SDK contract smoke; not a Jev benchmark or quality evaluation."""

import argparse
import asyncio
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from typesafe_sdk import (
    AsyncTypeSafeClient,
    Choice,
    Noul,
    RetryPolicy,
    Score,
    TypeSafeClient,
    __version__,
)


TOPICS = {
    "sports": "Football teams scoring goals in a championship match",
    "weather": "Rain, thunderstorms, wind and a weather forecast",
    "billing": "Invoices, credit card charges, payments and refunds",
}
STATES = {
    "sports": "The football team scored three goals and won the championship match.",
    "weather": "The forecast warns of heavy rain, strong winds and thunderstorms tomorrow.",
    "billing": "My credit card was charged twice for this invoice. Please refund the duplicate payment.",
}
RUBRIC = [
    "No payment problem; discussing football or weather",
    "A payment question without an incorrect charge",
    "An incorrect or duplicate credit card charge requiring a refund",
]
TOLERANCE = 2e-5


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def validate(body, questions):
    require(
        body["model"].startswith("openchoice-") and body["model"].endswith("-q4km"),
        "Response must identify the real local Q4_K_M model",
    )
    require(set(body["answers"]) == set(questions), "Answer IDs differ from request")
    for name, question in questions.items():
        answer = body["answers"][name]
        kind = (
            question["type"]
            if isinstance(question, dict)
            else (
                "choice"
                if isinstance(question, Choice)
                else "score"
                if isinstance(question, Score)
                else "noul"
            )
        )
        require(answer["type"] == kind, "Answer discriminator changed")
        if kind == "noul":
            require(
                math.isfinite(answer["noul"]) and 0 <= answer["noul"] <= 1,
                "Invalid noul",
            )
            continue
        criteria = (
            question["criteria"] if isinstance(question, dict) else question.criteria
        )
        expected = (
            set(criteria)
            if kind == "choice"
            else {str(i) for i in range(len(criteria))}
        )
        probabilities = answer["probabilities"]
        require(
            set(probabilities) == expected, "Probability labels differ from criteria"
        )
        require(
            all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities.values()),
            "Invalid probability",
        )
        require(
            math.isclose(sum(probabilities.values()), 1, abs_tol=1e-6),
            "Probabilities do not normalize",
        )
        n = len(probabilities)
        if n == 1:
            confidence = 1.0
        elif kind == "choice":
            confidence = (max(probabilities.values()) - 1 / n) / (1 - 1 / n)
        else:
            values = [probabilities[str(i)] for i in range(n)]
            mode = max(range(n), key=values.__getitem__)
            distance = sum(p * abs(i - mode) for i, p in enumerate(values))
            uniform_distance = sum(abs(i - (n - 1) / 2) for i in range(n)) / n
            confidence = max(0.0, 1 - distance / uniform_distance)
        require(
            math.isclose(answer["confidence"], confidence, abs_tol=1e-6),
            "Invalid confidence",
        )
        if kind == "choice":
            require(
                probabilities[answer["choice"]] == max(probabilities.values()),
                "Choice is not argmax",
            )
        else:
            require(
                answer["legend"] == {str(i): value for i, value in enumerate(criteria)},
                "Legend changed JSON values",
            )
            require(
                math.isclose(
                    answer["score"],
                    sum(int(i) * p for i, p in probabilities.items()),
                    abs_tol=1e-6,
                ),
                "Score is not expected index",
            )


def compare_answers(left, right):
    require(left["type"] == right["type"], "Answer type changed")
    if left["type"] == "noul":
        require(
            math.isclose(left["noul"], right["noul"], abs_tol=TOLERANCE),
            "Noul changed under isolation",
        )
    else:
        require(
            set(left["probabilities"]) == set(right["probabilities"]),
            "Labels changed under isolation",
        )
        for label, probability in left["probabilities"].items():
            require(
                math.isclose(
                    probability, right["probabilities"][label], abs_tol=TOLERANCE
                ),
                "Probabilities changed under isolation",
            )


class Checker:
    def __init__(self, args, report):
        self.args = args
        self.report = report
        self.options = dict(
            base_url=args.base_url,
            api_key=args.api_key or "local",
            timeout=180,
            retry=RetryPolicy(max_retries=0),
        )
        if args.model is not None:
            self.options["model"] = args.model

    def record(self, name, started, body, request=None):
        self.report["cases"].append(
            {
                "name": name,
                "seconds": time.perf_counter() - started,
                "request": request,
                "response": body,
            }
        )

    def call(self, client, name, state, questions, model=None):
        started = time.perf_counter()
        kwargs = {} if model is None else {"model": model}
        result = client.system_one(state=state, questions=questions, **kwargs)
        body = result.raw_http_response.json()
        self.record(
            name,
            started,
            body,
            {
                "state": state,
                "model": model or self.args.model or "SDK default",
                "question_ids": list(questions),
            },
        )
        validate(body, questions)
        return body

    def raw(self, name, payload, expected, key=None, encoded=False):
        data = payload.encode() if encoded else json.dumps(payload).encode()
        headers = {"Content-Type": "application/json"}
        if key is not None:
            headers["Authorization"] = "Bearer " + key
        elif self.args.api_key:
            headers["Authorization"] = "Bearer " + self.args.api_key
        request = Request(
            self.args.base_url.rstrip("/") + "/v1/systemone", data=data, headers=headers
        )
        started = time.perf_counter()
        try:
            with urlopen(request, timeout=180) as response:
                status, text = response.status, response.read().decode()
        except HTTPError as error:
            status, text = error.code, error.read().decode()
        try:
            body = json.loads(text)
        except ValueError:
            body = text
        self.record(
            name,
            started,
            {"status": status, "body": body},
            {"bytes": len(data), "expected_status": expected},
        )
        require(status in expected, f"{name}: expected {expected}, got {status}")
        require(
            isinstance(body, dict) and "detail" in body,
            f"{name}: missing structured error",
        )

    def sync(self):
        questions = {
            "topic": Choice(
                instructions="Choose the main topic of this message.", criteria=TOPICS
            ),
            "payment_severity": Score(
                instructions="How severe is the payment problem?", criteria=RUBRIC
            ),
            "needs_refund": Noul(
                instructions="Does this message concern a duplicate payment requiring a refund?",
                criteria={
                    "true": "Duplicate invoice charge requiring a payment refund",
                    "false": "Football match or weather forecast, no payment issue",
                },
            ),
        }
        with TypeSafeClient(**self.options) as client:
            started = time.perf_counter()
            models = client.models.list()
            self.record("sdk_sync_models", started, models.raw_http_response.json())
            names = {model.name for model in models.models}
            require(len(names) == 1, "Native service must expose exactly one model")
            require(
                all(name.startswith("openchoice-") for name in names),
                "Catalog must identify local models",
            )
            baseline = self.call(
                client,
                "typed_requested_model" if self.args.model else "typed_default_alias",
                STATES["billing"],
                questions,
            )
            actual = baseline["model"]
            require(actual in names, "Resolved model absent from catalog")
            for name, question in questions.items():
                alone = self.call(
                    client, "alone_" + name, STATES["billing"], {name: question}, actual
                )
                compare_answers(baseline["answers"][name], alone["answers"][name])
            renamed = self.call(
                client,
                "question_id_rename",
                STATES["billing"],
                {"renamed_" + name: q for name, q in questions.items()},
                actual,
            )
            for name in questions:
                compare_answers(
                    baseline["answers"][name], renamed["answers"]["renamed_" + name]
                )
            permuted = self.call(
                client,
                "option_permutation",
                STATES["billing"],
                {
                    "topic": Choice(
                        instructions=questions["topic"].instructions,
                        criteria=dict(reversed(list(TOPICS.items()))),
                    )
                },
                actual,
            )
            # The native slot model reads options in listed order, so permutation
            # sensitivity is reported as an observation rather than asserted away.
            before = baseline["answers"]["topic"]["probabilities"]
            after = permuted["answers"]["topic"]["probabilities"]
            require(set(before) == set(after), "Permutation changed option keys")
            self.report["option_permutation_observation"] = {
                "max_absolute_probability_change": max(
                    abs(before[key] - after[key]) for key in before
                ),
                "same_choice": baseline["answers"]["topic"]["choice"]
                == permuted["answers"]["topic"]["choice"],
            }
            observations = {}
            for topic, state in STATES.items():
                body = self.call(
                    client, "semantic_contrast_" + topic, state, questions, actual
                )
                observations[topic] = {
                    "selected_topic": body["answers"]["topic"]["choice"],
                    "expected_topic_probability": body["answers"]["topic"][
                        "probabilities"
                    ][topic],
                    "payment_score": body["answers"]["payment_severity"]["score"],
                    "refund_noul": body["answers"]["needs_refund"]["noul"],
                }
            self.report["semantic_observations"] = observations
            self.report["semantic_deltas"] = {
                "billing_minus_sports_payment_score": observations["billing"][
                    "payment_score"
                ]
                - observations["sports"]["payment_score"],
                "billing_minus_weather_refund_noul": observations["billing"][
                    "refund_noul"
                ]
                - observations["weather"]["refund_noul"],
            }
            nested = {
                "structured": {
                    "type": "choice",
                    "instructions": {"task": ["Classify the message"]},
                    "criteria": {
                        "billing": {
                            "description": [TOPICS["billing"], {"refund": True}]
                        },
                        "weather": [TOPICS["weather"]],
                    },
                },
                "rubric": {
                    "type": "score",
                    "criteria": [
                        {"topic": ["weather", {"payment": False}]},
                        ["duplicate charge", {"refund": True}],
                    ],
                },
            }
            self.call(
                client,
                "dictionary_nested_json",
                {"message": STATES["billing"], "metadata": [None, True, 3]},
                nested,
                actual,
            )
            self.call(
                client,
                "array_state",
                [STATES["weather"], {"source": "forecast"}],
                {"topic": questions["topic"]},
                actual,
            )
            self.call(
                client,
                "singleton_choice",
                STATES["sports"],
                {"only": Choice(criteria={"sports": TOPICS["sports"]})},
                actual,
            )
            self.call(
                client,
                "singleton_score",
                STATES["billing"],
                {"only": Score(criteria=[{"message": "refund"}])},
                actual,
            )
            self.call(
                client,
                "native_26_choice_10_score_capacity_not_quality",
                STATES["billing"],
                {
                    "wide": Choice(
                        criteria={str(i): f"Invoice category {i}" for i in range(26)}
                    ),
                    "levels": Score(
                        criteria=[f"Payment problem level {i}" for i in range(10)]
                    ),
                },
                actual,
            )
        return actual, questions

    async def concurrent(self, model, questions):
        async with AsyncTypeSafeClient(**self.options) as client:
            started = time.perf_counter()
            models = await client.models.list()
            self.record("sdk_async_models", started, models.raw_http_response.json())

            async def one(index):
                started = time.perf_counter()
                result = await client.system_one(
                    state=STATES["billing"], questions=questions, model=model
                )
                body = result.raw_http_response.json()
                self.record(f"async_concurrent_{index}", started, body)
                validate(body, questions)
                return body

            started = time.perf_counter()
            bodies = await asyncio.gather(*(one(i) for i in range(4)))
            self.report["four_async_calls_seconds"] = time.perf_counter() - started
            for body in bodies[1:]:
                for name in questions:
                    compare_answers(bodies[0]["answers"][name], body["answers"][name])

    def errors(self, model):
        base = {
            "model": model,
            "state": STATES["billing"],
            "questions": {"topic": {"type": "choice", "criteria": TOPICS}},
        }
        for name, updates in [
            ("false_state", {"state": False}),
            ("empty_questions", {"questions": {}}),
            ("unknown_model", {"model": "not-a-real-model"}),
            ("invalid_question_type", {"questions": {"bad": {"type": "mystery"}}}),
            ("state_overflow", {"state": "invoice " * 5000}),
            (
                "schema_overflow",
                {
                    "questions": {
                        "long": {"type": "choice", "criteria": {"a": "invoice " * 5000}}
                    }
                },
            ),
            (
                "native_choice_27_options_rejected",
                {
                    "questions": {
                        "wide": {
                            "type": "choice",
                            "criteria": {str(i): f"Invoice category {i}" for i in range(27)},
                        }
                    }
                },
            ),
        ]:
            self.raw(
                name, {**base, **updates}, [404] if name == "unknown_model" else [422]
            )
        for name, text in [
            ("malformed_json", "{"),
            (
                "nan_json",
                json.dumps(base).replace('"state": ', '"extra": NaN, "state": ', 1),
            ),
            (
                "infinity_json",
                json.dumps(base).replace(
                    '"state": ', '"extra": Infinity, "state": ', 1
                ),
            ),
            (
                "duplicate_keys",
                json.dumps(base).replace('"state": ', '"state": "first", "state": ', 1),
            ),
        ]:
            self.raw(name, text, [422], encoded=True)
        self.raw(
            "combined_input_overflow",
            {
                **base,
                "state": "invoice " * 2500,
                "questions": {
                    "long": {"type": "choice", "criteria": {"a": "invoice " * 2500}}
                },
            },
            [422],
        )
        self.raw("body_limit", {**base, "state": "x" * (2 * 1024 * 1024)}, [413])
        if self.args.api_key:
            self.raw("bad_auth", base, [401], key=self.args.api_key + "-invalid")
        else:
            self.report["unchecked"].append(
                "Authentication: no --api-key supplied; service assumed unauthenticated"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8768")
    parser.add_argument(
        "--model", help="Explicit local model; omit to exercise the SDK default alias"
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Configured server key; also enables wrong-key rejection check",
    )
    parser.add_argument("--output", type=Path, default=Path("api-smoke.json"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation refuses any existing report, including one from a failed run.
    with args.output.open("x") as output:
        report = {
            "sdk_version": __version__,
            "base_url": args.base_url,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "cases": [],
            "unchecked": [],
            "claims": [
                "Live HTTP requests using official sync and async SDK clients; no mocks.",
                "Checks wire formulas, local model identity, isolation, capacity and error behavior.",
                "Semantic contrasts are observations, not a quality threshold or Jev benchmark.",
                "26-option and ten-level fixture measures native runtime capacity only.",
            ],
            "probability_isolation_absolute_tolerance": TOLERANCE,
            "passed": False,
        }
        started = time.perf_counter()
        try:
            checker = Checker(args, report)
            model, questions = checker.sync()
            asyncio.run(checker.concurrent(model, questions))
            checker.errors(model)
            report["passed"] = True
        except Exception as error:
            report["failure"] = {"type": type(error).__name__, "message": str(error)}
            raise
        finally:
            report["total_seconds"] = time.perf_counter() - started
            json.dump(report, output, indent=2, allow_nan=False)
            output.write("\n")
    print(f"SDK contract smoke passed: {args.output}")


if __name__ == "__main__":
    main()
