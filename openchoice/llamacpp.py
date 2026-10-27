"""Exact native slot readout on a pinned, locally verified llama.cpp runtime.

No generation, remote tokenizer loading, truncation, or cross-request prefix/image
cache. The optional GPU instance is disposable; the CPU instance stays resident.
"""

import ctypes as C
from dataclasses import dataclass
import hashlib
import json
import logging
import math
from pathlib import Path
import string
import subprocess
import threading
import time

import numpy as np
from transformers import AutoTokenizer

from .native import decision_row, native_answer_ids, render_decision_prompt
from .inference import BatchPrediction, Cancelled, Decision, InputTooLong, ResourceExhausted, UnsupportedDecision

MAX_SEQUENCE_TOKENS = 4096
MAX_DECISION_TOKENS = 32768
MICROBATCH_TOKENS = 512
GPU_POLL_SECONDS = 2.0
GPU_QUIET_SECONDS = 60.0
GPU_ADMISSION_RESERVE_MIB = 2048
GPU_RETENTION_RESERVE_MIB = 1024
LLAMA_COMMIT = "4364bf7232e65c34eca8d9500c5464389662de6b"
_LOG = logging.getLogger(__name__)
_ABORT = C.CFUNCTYPE(C.c_bool, C.c_void_p)
_INTS = C.POINTER(C.c_int32)
_FLOATS = C.POINTER(C.c_float)


def _sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _cancelled(cancel):
    if cancel is not None and cancel.is_set():
        raise Cancelled("Client disconnected")


@dataclass(frozen=True)
class _Prepared:
    rendered: str
    tokens: list[int]
    answer_ids: list[int]
    assets: tuple[tuple[str, str], ...] = ()


class _Runtime:
    def __init__(self, library, model, projector, threads, gpu_layers=0):
        self.lib = library
        self.cancel = None
        self.callback = _ABORT(lambda _: self.cancel is not None and self.cancel.is_set())
        self.handle = self.lib.oc_open(
            str(model).encode(), str(projector).encode(), threads, gpu_layers, self.callback, None,
        )
        if not self.handle:
            raise ResourceExhausted("Unable to load the verified llama.cpp model/context")

    def close(self):
        if self.handle:
            self.lib.oc_close(self.handle)
            self.handle = None

    def clear(self):
        if self.handle:
            self.lib.oc_clear(self.handle)
        self.cancel = None

    def _check(self, status):
        _cancelled(self.cancel)
        if status == -200:
            raise Cancelled("Client disconnected")
        if status in {-2, -3}:
            raise ResourceExhausted("llama.cpp memory is temporarily unavailable")
        if status:
            raise RuntimeError(f"llama.cpp evaluation failed (status {status})")

    def eval(self, tokens, answers, *, position=0, reset=True):
        _cancelled(self.cancel)
        output = (C.c_float * len(answers))()
        self._check(self.lib.oc_eval(
            self.handle, (C.c_int32 * len(tokens))(*tokens), len(tokens), position, int(reset),
            (C.c_int32 * len(answers))(*answers), len(answers), output,
        ))
        return list(output)

    def image_tokens(self, prompt, image_token):
        _cancelled(self.cancel)
        wrapped = "<|vision_start|><|image_pad|><|vision_end|>"
        if prompt.rendered.count(wrapped) != len(prompt.assets):
            raise RuntimeError("Native image placeholder contract changed")
        text = prompt.rendered.replace(wrapped, "<__media__>")
        paths = (C.c_char_p * len(prompt.assets))(*(path.encode() for path, _ in prompt.assets))
        hashes = (C.c_char_p * len(prompt.assets))(*(digest.encode() for _, digest in prompt.assets))
        tokens = (C.c_int32 * MAX_SEQUENCE_TOKENS)()
        count = self.lib.oc_media_prepare(
            self.handle, text.encode(), paths, hashes, len(prompt.assets), image_token,
            tokens, MAX_SEQUENCE_TOKENS,
        )
        if count == -106:
            raise InputTooLong("Image and text exceed the model token limit")
        if count < 0:
            self._check(count)
        return list(tokens[:count])

    def image_eval(self, prompt, image_token):
        if self.image_tokens(prompt, image_token) != prompt.tokens:
            raise RuntimeError("Image-token alignment changed within a request")
        answers = (C.c_int32 * len(prompt.answer_ids))(*prompt.answer_ids)
        output = (C.c_float * len(answers))()
        self._check(self.lib.oc_media_eval(self.handle, answers, len(answers), output))
        return list(output)


def _library(path):
    lib = C.CDLL(str(path))
    signatures = {
        "oc_abi_version": ([], C.c_int),
        "oc_has_gpu": ([], C.c_int),
        "oc_open": ([C.c_char_p, C.c_char_p, C.c_int, C.c_int, _ABORT, C.c_void_p], C.c_void_p),
        "oc_close": ([C.c_void_p], None),
        "oc_clear": ([C.c_void_p], None),
        "oc_eval": ([C.c_void_p, _INTS, C.c_int, C.c_int, C.c_int, _INTS, C.c_int, _FLOATS], C.c_int),
        "oc_save_prefix": ([C.c_void_p, C.c_int], C.c_size_t),
        "oc_restore_prefix": ([C.c_void_p], C.c_int),
        "oc_media_prepare": ([C.c_void_p, C.c_char_p, C.POINTER(C.c_char_p), C.POINTER(C.c_char_p), C.c_int, C.c_int, _INTS, C.c_int], C.c_int),
        "oc_media_eval": ([C.c_void_p, _INTS, C.c_int, _FLOATS], C.c_int),
    }
    for name, (arguments, result) in signatures.items():
        function = getattr(lib, name)
        function.argtypes, function.restype = arguments, result
    if lib.oc_abi_version() != 1:
        raise ValueError("Unsupported OpenChoice llama.cpp bridge ABI")
    return lib


def _game_running():
    """Linux Steam app IDs and Wine/Proton runtime processes, not the Steam client."""
    for process in Path("/proc").iterdir():
        if not process.name.isdecimal():
            continue
        try:
            name = (process / "comm").read_text().strip().lower()
            if name.startswith(("wine", "proton")):
                return True
            environment = (process / "environ").read_bytes()
            for item in environment.split(b"\0"):
                key, _, value = item.partition(b"=")
                if key in {b"SteamAppId", b"SteamGameId"} and value not in {b"", b"0"}:
                    return True
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    return False


def _free_vram_mib():
    try:
        value = subprocess.check_output(
            ["nvidia-smi", "--id=0", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL, timeout=1.5,
        )
        return int(value.strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


class LlamaCppEngine:
    """Exact native slot logits from a verified GGUF package; no answer generation.

    ``gpu="off"`` loads a CPU-only library. ``gpu="auto"`` keeps that model
    resident and admits four GPU layers only after 60 game-free seconds
    with 2 GiB beyond the conservative 2 GiB profile allowance available. Game detection or less than
    1 GiB free releases GPU weights at a request boundary; readmission waits
    another 60 seconds. The CUDA library retains a small driver context even
    while only the CPU model is active. CPU/GPU probabilities may differ.

    All prompts are checked against strict per-forward and per-decision limits
    before decoding. Prefix state is reused only on identical 512-token
    boundaries within one request, then cleared. Verified local image assets
    are supported by ``predict_rows`` on CPU; the HTTP surface stays text-only.
    """

    _row = staticmethod(decision_row)

    def __init__(self, model, threads=12, gpu="off", runtime=None):
        if not isinstance(threads, int) or threads < 1:
            raise ValueError("threads must be a positive integer")
        if gpu not in {"off", "auto"}:
            raise ValueError("gpu must be off or auto")
        self.manifest_path = Path(model).resolve() / "manifest.json"
        self.root = self.manifest_path.parent
        self.manifest = json.loads(self.manifest_path.read_text())
        if self.manifest.get("schema_version") != "openchoice-gguf-v1" or self.manifest.get("quantization") != "Q4_K_M":
            raise ValueError("Expected an openchoice-gguf-v1 Q4_K_M package")
        if self.manifest.get("llama_commit") != LLAMA_COMMIT:
            raise ValueError("Unverified llama.cpp revision")
        for name, digest in self.manifest["files"].items():
            path = self.root / name
            if Path(name).is_absolute() or ".." in Path(name).parts or not path.resolve().is_relative_to(self.root) or _sha256(path) != digest:
                raise ValueError(f"Runtime artifact integrity mismatch: {name}")
        def artifact(name):
            value = self.manifest[name]
            if value not in self.manifest["files"]:
                raise ValueError(f"Unverified runtime artifact: {name}")
            return self.root / value
        self.model_path = artifact("model")
        self.projector_path = artifact("projector")
        self.checkpoint = self.root
        config_path = artifact("checkpoint_config")
        config = json.loads(config_path.read_text())
        if config["config"]["arm"] != "slot":
            raise ValueError("The exact-logit GGUF backend requires a slot checkpoint")
        if config["config"]["revision"] != self.manifest["base_revision"]:
            raise ValueError("Checkpoint and GGUF base revisions differ")
        processor_root = (self.root / self.manifest["processor"]).resolve()
        if Path(self.manifest["processor"]).is_absolute() or not processor_root.is_relative_to(self.root):
            raise ValueError("Processor must be stored within the verified runtime")
        # Every local processor file, including the chat template, must be hashed.
        for path in processor_root.rglob("*"):
            if path.is_file() and str(path.relative_to(self.root)) not in self.manifest["files"]:
                raise ValueError("Unverified processor file")
        self.processor = AutoTokenizer.from_pretrained(processor_root, local_files_only=True)
        self.image_token = self.processor.convert_tokens_to_ids("<|image_pad|>")
        self.default_model = f"openchoice-gguf-{self.manifest['files'][self.manifest['model']][:16]}-q4km"
        self.models = {self.default_model: {"kind": "native-slot GGUF Q4_K_M; Choice 1–26, Score 1–10, Noul scalar"}}
        self.threads, self.gpu_mode = threads, gpu
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._monitor = None
        self._gpu = None
        # Four layers measured near 1.9 GiB on RTX 4060; round up for admission.
        self._gpu_config = {"layers": 4, "peak_vram_mib": 2048}
        runtime_root = Path(runtime) if runtime else Path(__file__).resolve().parents[1] / ".runtime"
        build = json.loads((runtime_root / "build.json").read_text())
        if build["llama_commit"] != LLAMA_COMMIT or build["bridge_source_sha256"] != _sha256(Path(__file__).with_name("llamacpp_bridge.cpp")):
            raise ValueError("Rebuild the pinned runtime with scripts/build_llamacpp.sh")
        kind = "cuda" if gpu == "auto" else "cpu"
        self.lib = _library(runtime_root / f"llamacpp-bridge-{kind}.so")
        if gpu == "auto" and not self.lib.oc_has_gpu():
            raise ValueError("GPU auto requires a CUDA-enabled bridge")
        self.last_device = "cpu"
        self._cpu = _Runtime(self.lib, self.model_path, self.projector_path, threads)
        self._quiet_since = None
        self._last_release = -math.inf
        if gpu == "auto":
            self._monitor = threading.Thread(target=self._monitor_gpu, name="openchoice-gpu-policy", daemon=True)
            self._monitor.start()

    def resolve(self, model):
        if model in {self.default_model, "openchoice-latest", "jev-latest"}:
            return self.default_model
        raise KeyError(model)

    def close(self):
        self._stop.set()
        if self._monitor is not None:
            self._monitor.join()
            self._monitor = None
        with self._lock:
            if self._gpu is not None:
                self._gpu.close()
                self._gpu = None
            self._cpu.close()

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        self.close()

    def _monitor_gpu(self):
        while not self._stop.wait(GPU_POLL_SECONDS):
            if self._lock.acquire(blocking=False):
                try:
                    self._poll_gpu(admit=True)
                finally:
                    self._lock.release()

    def _poll_gpu(self, *, admit=False):
        if self.gpu_mode != "auto":
            return
        game, free = _game_running(), _free_vram_mib()
        now = time.monotonic()
        if game or free is None:
            self._quiet_since = None
        elif self._quiet_since is None:
            # Start at the first observed game-free instant, not the last game
            # observation: the latter could shorten quiet time by one poll period.
            self._quiet_since = now
        reason = "game" if game else "unavailable" if free is None else "low-vram"
        if self._gpu is not None and (game or free is None or free < GPU_RETENTION_RESERVE_MIB):
            self._gpu.close()
            self._gpu = None
            self._last_release = time.monotonic()
            _LOG.info("GPU released: reason=%s free_mib=%s", reason, free)
        elif (
            admit and self._gpu is None and not game and free is not None
            and self._quiet_since is not None
            and now - max(self._quiet_since, self._last_release) >= GPU_QUIET_SECONDS
            and free >= self._gpu_config["peak_vram_mib"] + GPU_ADMISSION_RESERVE_MIB
        ):
            try:
                self._gpu = _Runtime(
                    self.lib, self.model_path, self.projector_path, self.threads,
                    self._gpu_config["layers"],
                )
            except ResourceExhausted:
                self._last_release = time.monotonic()
                _LOG.info("GPU released: reason=allocation-failed free_mib=%s", free)
                return
            after = _free_vram_mib()
            _LOG.info("GPU admitted: reason=quiet-and-headroom layers=%s free_before_mib=%s free_after_mib=%s", self._gpu_config["layers"], free, after)
            # A game may have launched while weights were loading.
            game_after = _game_running()
            if game_after or after is None:
                self._quiet_since = None
            if game_after or after is None or after < GPU_RETENTION_RESERVE_MIB:
                self._gpu.close()
                self._gpu = None
                self._last_release = time.monotonic()
                _LOG.info("GPU released: reason=gate-changed-during-load free_mib=%s", after)

    def _prepare_row(self, row, asset_root):
        primitive = row["primitive"]
        options = [option["text"] for option in row.get("options", [])]
        # Reuse the exact serving capability checks; never reconstruct its prompt.
        self._row(row["context"], Decision(row["question"], ["True", "False"] if primitive == "noul" else options, primitive))
        if primitive == "noul" and options:
            raise UnsupportedDecision("Native Noul rows do not contain a Choice menu")
        menus = [options] if primitive == "choice" else [[option] for option in options] if primitive == "score" else [[]]
        answers = string.ascii_uppercase[:len(options)] if primitive == "choice" else ["Yes", "No"]
        root = Path(asset_root).resolve()
        assets = []
        for asset in row.get("assets", []):
            path = (root / asset["path"]).resolve()
            if not path.is_relative_to(root) or _sha256(path) != asset["sha256"]:
                raise ValueError("Image asset path/hash mismatch")
            assets.append((str(path), asset["sha256"]))
        prepared = []
        total = 0
        for menu in menus:
            rendered, _ = render_decision_prompt(self.processor, row, menu, primitive == "choice")
            ids = native_answer_ids(self.processor, rendered, answers)
            if assets:
                tokens = self._cpu.image_tokens(_Prepared(rendered, [], ids, tuple(assets)), self.image_token)
            else:
                tokens = self.processor.encode(rendered, add_special_tokens=False)
            if len(tokens) > MAX_SEQUENCE_TOKENS:
                raise InputTooLong("Native sequence exceeds max_sequence_tokens; no truncation allowed")
            total += len(tokens)
            if total > MAX_DECISION_TOKENS:
                raise InputTooLong("Decision exceeds max_decision_tokens; no partial menu allowed")
            prepared.append(_Prepared(rendered, tokens, ids, tuple(assets)))
        return prepared

    def predict(self, model, state, questions, cancel=None):
        return self.predict_rows(
            [self._row(state, question) for question in questions], self.checkpoint,
            model=model, cancel=cancel,
        )

    def predict_rows(self, rows, asset_root, *, model="openchoice-latest", cancel=None, reuse_prefix=True):
        resolved = self.resolve(model)
        with self._lock:
            if self._stop.is_set():
                raise RuntimeError("The llama.cpp engine is closed")
            _cancelled(cancel)
            self._poll_gpu()
            # Offline multimodal requests remain CPU-only, including the projector.
            runtime = self._cpu if any(row.get("assets") for row in rows) else self._gpu or self._cpu
            self.last_device = "gpu" if runtime is self._gpu else "cpu"
            runtime.cancel = cancel
            try:
                prepared = []
                for row in rows:
                    _cancelled(cancel)
                    prepared.append(self._prepare_row(row, asset_root))
                flat = [prompt for group in prepared for prompt in group]
                groups = {}
                for index, prompt in enumerate(flat):
                    key = tuple(prompt.tokens[:MICROBATCH_TOKENS]) if reuse_prefix and not prompt.assets and len(prompt.tokens) > MICROBATCH_TOKENS else index
                    groups.setdefault(key, []).append(index)
                outputs = [None] * len(flat)
                for indices in groups.values():
                    _cancelled(cancel)
                    prefix = 0
                    first = flat[indices[0]]
                    if len(indices) > 1:
                        common = min(len(flat[index].tokens) for index in indices) - 1
                        for index in indices[1:]:
                            tokens = flat[index].tokens
                            common = next((i for i in range(common) if first.tokens[i] != tokens[i]), common)
                        prefix = common // MICROBATCH_TOKENS * MICROBATCH_TOKENS
                    if prefix:
                        runtime.eval(first.tokens[:prefix], [])
                        if not self.lib.oc_save_prefix(runtime.handle, prefix):
                            raise ResourceExhausted("Unable to save request prefix state")
                    for order, index in enumerate(indices):
                        _cancelled(cancel)
                        prompt = flat[index]
                        if prompt.assets:
                            outputs[index] = runtime.image_eval(prompt, self.image_token)
                        else:
                            if prefix and order and self.lib.oc_restore_prefix(runtime.handle) != prefix:
                                raise RuntimeError("Unable to restore request prefix state")
                            outputs[index] = runtime.eval(prompt.tokens[prefix:], prompt.answer_ids, position=prefix, reset=not prefix)
                logits, offset = [], 0
                for row, prompts in zip(rows, prepared):
                    results = outputs[offset:offset + len(prompts)]
                    offset += len(prompts)
                    if row["primitive"] == "choice":
                        logits.append(results[0])
                    else:
                        differences = [np.float32(pair[0]) - np.float32(pair[1]) for pair in results]
                        if row["primitive"] == "noul":
                            logits.append([float(differences[0]), 0.0])
                        else:
                            logits.append([float(-np.logaddexp(np.float32(0), -value)) for value in differences])
                return BatchPrediction(resolved, logits, sum(len(prompt.tokens) for prompt in flat))
            except MemoryError:
                raise ResourceExhausted("CPU memory is temporarily unavailable") from None
            finally:
                runtime.clear()
                self._poll_gpu()
