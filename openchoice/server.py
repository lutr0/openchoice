"""Local TypeSafe SDK 0.6-compatible HTTP service (not a Jev implementation).

Compatibility choices: question maps must be nonempty. Singleton choice and
score criteria are accepted as in the current Python SDK/live OpenAPI, despite
older prose/JavaScript score guidance requiring two levels.
Unknown object fields are ignored as in the SDK. Described choices
use only their description, while null descriptions use the choice key. Question
IDs never enter inference. Structured values use canonical JSON, preserving
nested scalar values but rejecting scalar state/instructions/criteria. Score
legends retain the original JSON values and use zero-based string indices.

Confidence follows public adapter commit 0bb819b85d67a98c736d7c3004eae95f49f3daa3:
choice uses rescaled maximum probability; score uses modal distance versus
uniform dispersion. Neither is calibrated. Noul is P(true), no confidence. Billing is
always zero. Input token counts include every processed prompt token, including
evidence repeated across Score levels. --model selects a verified GGUF package.
Choice is limited to 26 options, Score to 10 levels, with no truncation.
CPU is the default; --gpu auto enables optional four-layer CUDA offload.
Requests run one at a time and cancellation stops decoding between microbatches.
When --max-queued requests are already waiting, new ones get 503.
"""

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import hmac
import ipaddress
import json
import math
import os
import threading
from typing import Annotated, Literal
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError

from .inference import (
    Cancelled, Decision, InputTooLong, ResourceExhausted, UnsupportedDecision,
)

MAX_BODY_BYTES = 2 * 1024 * 1024
Content = str | dict[str, JsonValue] | list[JsonValue]


class WireModel(BaseModel):
    model_config = ConfigDict(strict=True, extra="ignore")


class ChoiceQuestion(WireModel):
    type: Literal["choice"]
    instructions: Content | None = None
    criteria: Annotated[dict[str, Content | None], Field(min_length=1, max_length=255)]


class ScoreQuestion(WireModel):
    type: Literal["score"]
    instructions: Content | None = None
    criteria: Annotated[list[Content], Field(min_length=1, max_length=10)]


class NoulCriteria(WireModel):
    true: Content | None = None
    false: Content | None = None


class NoulQuestion(WireModel):
    type: Literal["noul"]
    instructions: Content | None = None
    criteria: NoulCriteria | None = None


Question = Annotated[
    ChoiceQuestion | ScoreQuestion | NoulQuestion, Field(discriminator="type")
]


class SystemOneRequest(WireModel):
    state: Content
    model: str
    questions: Annotated[dict[str, Question], Field(min_length=1, max_length=64)]


def _text(value: Content | None) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON object key")
        result[key] = value
    return result


def _finite_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Nonfinite JSON number")
    return number


def _invalid_constant(value):
    raise ValueError("Nonfinite JSON number")


def _probabilities(logits):
    if not logits or not all(math.isfinite(value) for value in logits):
        raise RuntimeError("Invalid model output")
    peak = max(logits)
    weights = [math.exp(value - peak) for value in logits]
    total = math.fsum(weights)
    return [value / total for value in weights]


def _confidence(probabilities, *, score=False):
    if len(probabilities) == 1:
        return 1.0
    count = len(probabilities)
    if score:
        mode = max(range(count), key=probabilities.__getitem__)
        dispersion = math.fsum(p * abs(i - mode) for i, p in enumerate(probabilities))
        center = (count - 1) / 2
        uniform_dispersion = math.fsum(abs(i - center) for i in range(count)) / count
        return max(0.0, min(1.0, 1.0 - dispersion / uniform_dispersion))
    baseline = 1.0 / count
    return max(0.0, min(1.0, (max(probabilities) - baseline) / (1.0 - baseline)))


def _consume(future):
    # An abandoned job may still end with Cancelled; retrieve it so asyncio does not
    # report an unhandled exception for a result nobody will read.
    if not future.cancelled():
        future.exception()


async def _disconnect(request):
    # The body is already read, so the next ASGI message is the client going away.
    # Awaiting receive (rather than polling is_disconnected) also works behind
    # Starlette's BaseHTTPMiddleware, which only forwards a disconnect on await.
    while (await request.receive())["type"] != "http.disconnect":
        pass


def create_app(
    *,
    model=None,
    runtime=None,
    threads=12,
    gpu="off",
    engine=None,
    api_key=None,
    max_queued=8,
):
    """Build an app; lifespan owns one serialized GGUF engine."""
    if (model is None) == (engine is None):
        raise ValueError("pass exactly one of model or engine")
    api_key = os.environ.get("OPENCHOICE_API_KEY") if api_key is None else api_key
    if max_queued < 1:
        raise ValueError("max_queued must be at least 1")

    @asynccontextmanager
    async def lifespan(app):
        app.state.ready = False
        # Startup and inference share one worker, not merely an ASGI-thread lock.
        worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="openchoice-inference")
        app.state.inference_worker = worker
        try:
            def startup():
                if engine is not None:
                    owned_engine = engine
                else:
                    from .llamacpp import LlamaCppEngine

                    owned_engine = LlamaCppEngine(model, runtime=runtime, threads=threads, gpu=gpu)
                if engine is None:
                    owned_engine.predict(
                        owned_engine.default_model,
                        "Service readiness warmup.",
                        [
                            Decision(
                                "Choose the matching statement.",
                                ["Service readiness warmup.", "Unrelated statement."],
                                primitive="choice",
                            )
                        ],
                    )
                return owned_engine

            app.state.engine = await asyncio.get_running_loop().run_in_executor(worker, startup)
            app.state.ready = True
            yield
        finally:
            app.state.ready = False
            # Stop the request on the worker at its next decision boundary rather than
            # letting a long batch hold shutdown past the service stop timeout.
            for pending in list(app.state.cancels):
                pending.set()
            worker.shutdown(wait=True, cancel_futures=True)
            del app.state.inference_worker
            if hasattr(app.state, "engine"):
                if engine is None:
                    app.state.engine.close()
                del app.state.engine

    app = FastAPI(
        title="OpenChoice local TypeSafe-compatible API",
        version="1.0",
        lifespan=lifespan,
    )
    app.state.ready = False
    app.state.pending = 0
    app.state.cancels = set()

    @app.middleware("http")
    async def request_identity(request: Request, call_next):
        request_id = str(uuid4())
        try:
            response = await call_next(request)
        except Exception:
            response = JSONResponse(
                status_code=500, content={"detail": "Internal server error"}
            )
        response.headers["x-typesafe-request-id"] = request_id
        return response

    async def authenticate(request: Request):
        if api_key:
            header = request.headers.get("authorization", "")
            scheme, _, token = header.partition(" ")
            if scheme.lower() != "bearer" or not hmac.compare_digest(
                token.encode("utf-8"), api_key.encode("utf-8")
            ):
                raise HTTPException(
                    401,
                    "Missing or invalid bearer token",
                    headers={"WWW-Authenticate": "Bearer"},
                )

    async def parse_request(request: Request, _=Depends(authenticate)):
        media_type = (
            request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        )
        if media_type != "application/json":
            raise HTTPException(415, "Content-Type must be application/json")
        body = bytearray()
        async for chunk in request.stream():
            if len(body) + len(chunk) > MAX_BODY_BYTES:
                raise HTTPException(413, "Request body exceeds 2 MiB")
            body.extend(chunk)
        try:
            raw = json.loads(
                body.decode("utf-8"),
                object_pairs_hook=_unique_object,
                parse_float=_finite_float,
                parse_constant=_invalid_constant,
            )
            # Reject unpaired surrogates, including in ignored fields, before any
            # model work or response serialization. Valid UTF-8 keys round-trip.
            json.dumps(raw, ensure_ascii=False, allow_nan=False).encode("utf-8")
            payload = SystemOneRequest.model_validate(raw)
        except (ValueError, UnicodeError, RecursionError, ValidationError):
            raise HTTPException(422, "Invalid JSON or request schema") from None
        count = sum(
            len(q.criteria) if isinstance(q, (ChoiceQuestion, ScoreQuestion)) else 2
            for q in payload.questions.values()
        )
        if count > 2048:
            raise HTTPException(422, "Request exceeds 2048 total options")
        return payload

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        return JSONResponse(
            status_code=422, content={"detail": "Invalid request schema"}
        )

    @app.get("/health")
    async def health():
        if not app.state.ready:
            raise HTTPException(503, "Service not ready")
        return {"status": "ok"}

    @app.get("/v1/models", dependencies=[Depends(authenticate)])
    def models():
        return {
            "models": [
                {
                    "name": name,
                    "description": (
                        f"Local OpenChoice {spec['kind']} scorer; not a Jev model. "
                        "Request-only aliases jev-latest and openchoice-latest route to the selected "
                        f"local default {app.state.engine.default_model}, never a Jev model."
                    ),
                    "release_date": "2026-09-17",
                }
                for name, spec in app.state.engine.models.items()
            ]
        }

    request_schema = SystemOneRequest.model_json_schema(
        ref_template="#/components/schemas/{model}"
    )
    definitions = request_schema.pop("$defs", {})

    @app.post(
        "/v1/systemone",
        openapi_extra={
            "requestBody": {
                "required": True,
                "content": {"application/json": {"schema": request_schema}},
            }
        },
    )
    async def systemone(request: Request, payload: SystemOneRequest = Depends(parse_request)):
        engine = app.state.engine
        try:
            resolved = engine.resolve(payload.model)
        except KeyError:
            raise HTTPException(404, "Unknown model") from None
        decisions = []
        for question in payload.questions.values():
            instructions = _text(question.instructions)
            true_criterion = false_criterion = None
            if isinstance(question, ChoiceQuestion):
                options = [
                    key if value is None else _text(value)
                    for key, value in question.criteria.items()
                ]
            elif isinstance(question, ScoreQuestion):
                options = [_text(value) for value in question.criteria]
            else:
                criteria = question.criteria or NoulCriteria()
                true_criterion = (
                    _text(criteria.true) if criteria.true is not None else None
                )
                false_criterion = (
                    _text(criteria.false) if criteria.false is not None else None
                )
                # Historical engines use a two-option comparison. Native Noul
                # receives explicit criteria separately and reads a single scalar.
                statement = instructions or "The statement in the state is true."
                options = [
                    true_criterion if true_criterion is not None else f"Yes. {statement}",
                    false_criterion
                    if false_criterion is not None
                    else f"No. It is not true that: {statement}",
                ]
            decisions.append(
                Decision(
                    instructions,
                    options,
                    primitive=question.type,
                    true_criterion=true_criterion,
                    false_criterion=false_criterion,
                )
            )
        if app.state.pending >= max_queued:
            raise HTTPException(
                503, "Inference queue is full; retry later", headers={"Retry-After": "5"}
            )
        cancel = threading.Event()
        loop = asyncio.get_running_loop()

        def finished(_):
            # Runs when the worker finishes the job or drops it unstarted.
            loop.call_soon_threadsafe(release)

        def release():
            app.state.pending -= 1
            app.state.cancels.discard(cancel)

        app.state.pending += 1
        app.state.cancels.add(cancel)
        future = app.state.inference_worker.submit(
            engine.predict, resolved, _text(payload.state), decisions, cancel
        )
        future.add_done_callback(finished)
        waiter = asyncio.wrap_future(future)
        waiter.add_done_callback(_consume)
        disconnected = asyncio.ensure_future(_disconnect(request))
        try:
            await asyncio.wait({waiter, disconnected}, return_when=asyncio.FIRST_COMPLETED)
            if not waiter.done():
                # Nobody is waiting for this answer any more: drop it if it has not
                # started, otherwise stop it at the next decision boundary.
                cancel.set()
                future.cancel()
                return JSONResponse(status_code=499, content={"detail": "Client closed request"})
        finally:
            disconnected.cancel()
        try:
            prediction = waiter.result()
            if len(prediction.logits) != len(decisions):
                raise RuntimeError("Invalid model output count")
            answers = {}
            for (key, question), decision, row in zip(
                payload.questions.items(), decisions, prediction.logits, strict=True
            ):
                if len(row) != len(decision.options):
                    raise RuntimeError("Invalid model option count")
                probabilities = _probabilities(row)
                if isinstance(question, NoulQuestion):
                    answers[key] = {"type": "noul", "noul": probabilities[0]}
                elif isinstance(question, ChoiceQuestion):
                    keys = list(question.criteria)
                    answers[key] = {
                        "type": "choice",
                        "choice": keys[
                            max(range(len(row)), key=probabilities.__getitem__)
                        ],
                        "confidence": _confidence(probabilities),
                        "probabilities": dict(zip(keys, probabilities, strict=True)),
                    }
                else:
                    keys = [str(index) for index in range(len(row))]
                    answers[key] = {
                        "type": "score",
                        "score": math.fsum(i * p for i, p in enumerate(probabilities)),
                        "confidence": _confidence(probabilities, score=True),
                        "legend": dict(zip(keys, question.criteria, strict=True)),
                        "probabilities": dict(zip(keys, probabilities, strict=True)),
                    }
            return {
                "model": prediction.model,
                "answers": answers,
                "usage": {
                    "billing_units": 0,
                    "input_tokens": prediction.input_tokens,
                    "output_tokens": 0,
                },
            }
        except InputTooLong:
            raise HTTPException(422, "Input exceeds the model token limit") from None
        except Cancelled:
            return JSONResponse(status_code=499, content={"detail": "Client closed request"})
        except UnsupportedDecision as exc:
            raise HTTPException(422, str(exc)) from None
        except ResourceExhausted:
            raise HTTPException(
                503,
                "Inference memory is temporarily unavailable; retry later",
                headers={"Retry-After": "30"},
            ) from None
        except Exception:
            # Handled here so inference failures never escape into server logs
            # with request-derived model exception messages.
            raise HTTPException(500, "Internal server error") from None

    def openapi():
        if app.openapi_schema is None:
            schema = get_openapi(
                title=app.title, version=app.version, routes=app.routes
            )
            schema.setdefault("components", {}).setdefault("schemas", {}).update(
                definitions
            )
            schema["components"]["securitySchemes"] = {
                "BearerAuth": {"type": "http", "scheme": "bearer"}
            }
            for path in ("/v1/models", "/v1/systemone"):
                for operation in schema["paths"][path].values():
                    operation["security"] = [{"BearerAuth": []}] if api_key else []
            app.openapi_schema = schema
        return app.openapi_schema

    app.openapi = openapi
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8768)
    parser.add_argument("--model", required=True, help="Verified OpenChoice GGUF package directory")
    parser.add_argument("--runtime", help="Pinned local build directory (default: .runtime)")
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--gpu", choices=["off", "auto"], default="off")
    parser.add_argument(
        "--max-queued", type=int, default=8,
        help="Requests allowed to wait or run at once; more get 503 with Retry-After",
    )
    args = parser.parse_args()
    try:
        loopback = ipaddress.ip_address(args.host).is_loopback
    except ValueError:
        loopback = args.host.lower() == "localhost"
    if not loopback and not os.environ.get("OPENCHOICE_API_KEY"):
        parser.error("Non-loopback binding requires OPENCHOICE_API_KEY")
    import uvicorn

    uvicorn.run(
        create_app(
            model=args.model,
            runtime=args.runtime,
            threads=args.threads,
            gpu=args.gpu,
            max_queued=args.max_queued,
        ),
        host=args.host,
        port=args.port,
        workers=1,
        # Stop waiting for open requests after 5 s; the lifespan then cancels queued
        # jobs and stops the running one at its next decision boundary.
        timeout_graceful_shutdown=5,
    )


if __name__ == "__main__":
    main()
