"""PyTorch LoRA training reader for the released native-answer model.

Serving uses llama.cpp. Quantization here is exclusively a training/evaluation
choice; saved adapters are always portable BF16/FP32 safetensors.
"""
from dataclasses import dataclass
import json
from pathlib import Path
import string

import torch
from torch import nn
from torch.nn import functional as F

from openchoice.integrity import content_hash, sha256_file
from openchoice.native import render_decision_prompt, native_answer_ids


@dataclass
class DecisionOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None
    processed_tokens: int
    support_logits: torch.Tensor | None = None


def target_loss(logits, row):
    target = row["target"]
    if target is None:
        return None
    if row["primitive"] == "noul":
        return F.binary_cross_entropy_with_logits(logits, logits.new_tensor(float(target["value"])))
    ids = [option["id"] for option in row["options"]]
    probabilities = logits.new_tensor(
        [float(key == target["id"]) for key in ids]
        if target["kind"] == "class" else [target["values"][key] for key in ids]
    )
    if row["primitive"] == "score":
        return F.binary_cross_entropy_with_logits(logits, probabilities)
    return -(probabilities * F.log_softmax(logits, dim=-1)).sum()


class DecisionModel(nn.Module):
    """Frozen Qwen3.5 backbone plus the released rank-16 LoRA target inventory."""

    def __init__(self, config, device, *, quantization=None):
        super().__init__()
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3_5ForConditionalGeneration

        self.config = dict(config)
        self.device = torch.device(device)
        self.quantization = quantization
        self.arm = "slot"
        if config.get("arm") != "slot" or config.get("initialization", "pretrained") != "pretrained":
            raise ValueError("Fine-tuning supports only the released pretrained native slot reader")
        if quantization not in {None, "nf4"}:
            raise ValueError("Training quantization must be nf4 or omitted")
        if quantization and self.device.type != "cuda":
            raise ValueError("QLoRA requires a CUDA GPU")
        pinned = dict(revision=config["revision"], trust_remote_code=False, token=False)
        self.processor = AutoProcessor.from_pretrained(config["model_id"], **pinned)
        loading = {}
        if quantization:
            loading = dict(
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
                ), device_map={"": str(self.device)},
            )
        base = Qwen3_5ForConditionalGeneration.from_pretrained(
            config["model_id"], dtype=torch.bfloat16, **pinned, **loading,
        )
        base.config.use_cache = False
        base.config.text_config.use_cache = False
        if quantization:
            base = prepare_model_for_kbit_training(
                base, use_gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )
        else:
            base.to(self.device)
            base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        targets = []
        for i, layer_type in enumerate(base.config.text_config.layer_types):
            prefix = f"model.language_model.layers.{i}."
            projections = (
                ["self_attn." + p for p in ("q_proj", "k_proj", "v_proj", "o_proj")]
                if layer_type == "full_attention" else
                ["linear_attn." + p for p in ("in_proj_qkv", "in_proj_z", "out_proj")]
            )
            targets.extend(prefix + p for p in projections)
            targets.extend(prefix + "mlp." + p for p in ("gate_proj", "up_proj", "down_proj"))
        if not set(targets) <= dict(base.named_modules()).keys():
            raise ValueError("Base model does not match released LoRA target inventory")
        self.backbone = get_peft_model(base, LoraConfig(
            r=config.get("lora_rank", 16), lora_alpha=config.get("lora_alpha", 32),
            lora_dropout=config.get("lora_dropout", 0), target_modules=targets, bias="none",
        ))
        # Native slot decisions have no learned auxiliary head; keep the released
        # empty heads artifact for checkpoint integrity and interchange.
        self.heads = nn.ModuleDict()

    def _encode(self, row, candidates, asset_root, aliases=False):
        from PIL import Image
        rendered, _ = render_decision_prompt(self.processor, row, candidates, aliases)
        images = []
        root = Path(asset_root).resolve()
        for asset in row.get("assets", []):
            path = (root / asset["path"]).resolve()
            if not path.is_relative_to(root) or sha256_file(path) != asset["sha256"]:
                raise ValueError("Image asset path/hash mismatch")
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        encoded = self.processor(text=[rendered], images=images or None, return_tensors="pt", truncation=False)
        count = encoded["input_ids"].shape[1]
        if count > self.config.get("max_sequence_tokens", 4096):
            raise ValueError("Native sequence exceeds 4096 tokens; shorten the evidence or image")
        return {k: v.to(self.device) for k, v in encoded.items() if k != "token_type_ids"}, rendered, count

    def _native_logits(self, encoded, rendered, answers):
        ids = native_answer_ids(self.processor.tokenizer, rendered, answers)
        native = self.backbone.get_base_model()
        output = native.model(**encoded, use_cache=False, return_dict=True, output_hidden_states=False)
        state = output.last_hidden_state[0, -1]
        # Only selected vocabulary rows are needed: avoid a full-vocabulary
        # projection and its backward buffer for a probability-only task.
        return F.linear(state.to(native.lm_head.weight.dtype), native.lm_head.weight[ids]).float()

    def forward(self, row, asset_root):
        from decision_training.data import validate_row
        validate_row(row)
        primitive = row["primitive"]
        options = [option["text"] for option in row.get("options", [])]
        candidates = [options] if primitive == "choice" else ([[o] for o in options] if primitive == "score" else [[]])
        prepared = [self._encode(row, c, asset_root, primitive == "choice") for c in candidates]
        tokens = sum(count for _, _, count in prepared)
        if tokens > self.config.get("max_decision_tokens", 32768):
            raise ValueError("Decision exceeds total token budget")
        if primitive == "choice":
            encoded, rendered, _ = prepared[0]
            logits = self._native_logits(encoded, rendered, string.ascii_uppercase[:len(options)])
        else:
            differences = []
            for encoded, rendered, _ in prepared:
                yes_no = self._native_logits(encoded, rendered, ["Yes", "No"])
                differences.append(yes_no[0] - yes_no[1])
            logits = torch.stack(differences) if primitive == "score" else differences[0]
        loss = target_loss(logits, row)
        support = logits if primitive == "score" else None
        return DecisionOutput(F.logsigmoid(logits) if primitive == "score" else logits, loss, tokens, support)

    def save_pretrained(self, path):
        from safetensors.torch import save_file
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.backbone.save_pretrained(path / "adapter", safe_serialization=True, save_embedding_layers=False)
        # PEFT emits a template card with unfilled sections; checkpoints are
        # machine artifacts, not auto-published model cards.
        (path / "adapter/README.md").unlink(missing_ok=True)
        save_file({}, path / "heads.safetensors")
        metadata = dict(config=self.config, phase="joint", processor_contract={
            "model_id": self.config["model_id"], "revision": self.config["revision"],
            "tokenizer_hash": content_hash(self.processor.tokenizer.get_vocab()),
        }, files={str(p.relative_to(path)): sha256_file(p) for p in sorted(path.rglob("*")) if p.is_file() and p.name != "model.json"})
        (path / "model.json").write_text(json.dumps(metadata, indent=2) + "\n")

    @classmethod
    def load_pretrained(cls, path, device, *, quantization=None):
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        path = Path(path)
        metadata = json.loads((path / "model.json").read_text())
        for name, expected in metadata["files"].items():
            file = (path / name).resolve()
            if not file.is_relative_to(path.resolve()) or sha256_file(file) != expected:
                raise ValueError(f"Checkpoint hash mismatch: {name}")
        model = cls(metadata["config"], device, quantization=quantization)
        if content_hash(model.processor.tokenizer.get_vocab()) != metadata["processor_contract"]["tokenizer_hash"]:
            raise ValueError("Pinned processor vocabulary changed")
        set_peft_model_state_dict(model.backbone, load_file(path / "adapter/adapter_model.safetensors"))
        model.heads.load_state_dict(load_file(path / "heads.safetensors"), strict=True)
        return model
