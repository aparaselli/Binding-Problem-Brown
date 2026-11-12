"""
Model, hook and scoring helpers for ablate_feature_heads.py.

Works with Hugging Face vision-language models whose decoder layers expose
`self_attn.o_proj` (tested: Qwen2.5-VL-7B-Instruct, InternVL3.5-8B-HF,
Gemma-3-12B-it).  A "head output" is the slice of the o_proj input that
belongs to one attention head, i.e. the head's contribution before it is
mixed into the residual stream.
"""

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn
from transformers import AutoProcessor

try:
    from transformers import AutoModelForImageTextToText
except ImportError:  # older transformers
    AutoModelForImageTextToText = None
try:
    from transformers import AutoModelForVision2Seq
except ImportError:  # newer transformers
    AutoModelForVision2Seq = None


# --------------------------------------------------------------------------
# Model loading and inputs
# --------------------------------------------------------------------------
def load_vlm(model_name: str, cache_dir: Optional[str] = None):
    """Load a VLM in bf16 (fp16 if bf16 is unsupported), sharded with device_map=auto."""
    cache_dir = cache_dir or os.environ.get("HF_HOME")
    dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    name = model_name.lower()
    processor_kwargs = {"fix_mistral_regex": True} if "mistral" in name else {}
    processor = AutoProcessor.from_pretrained(model_name, cache_dir=cache_dir, **processor_kwargs)

    use_image_text = any(k in name for k in ("gemma", "internvl", "mistral"))
    loader = (
        AutoModelForImageTextToText
        if use_image_text or AutoModelForVision2Seq is None
        else AutoModelForVision2Seq
    )
    if loader is None:
        raise RuntimeError("No compatible transformers model loader found.")
    model = loader.from_pretrained(
        model_name, torch_dtype=dtype, device_map="auto",
        low_cpu_mem_usage=True, cache_dir=cache_dir,
    )
    model.eval()
    return model, processor


def get_tokenizer(processor):
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise AttributeError("The processor does not expose processor.tokenizer.")
    return tokenizer


def build_inputs(processor, device, image, prompt: str):
    """Single-turn chat input: one image followed by the text prompt."""
    messages = [{
        "role": "user",
        "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
    for key, value in inputs.items():
        if torch.is_tensor(value):
            inputs[key] = value.to(device)
    return inputs


def last_input_position(inputs) -> int:
    input_ids = inputs["input_ids"]
    if input_ids.shape[0] != 1:
        raise ValueError(f"Expected batch size 1, got {input_ids.shape[0]}.")
    mask = inputs.get("attention_mask")
    if mask is None:
        return int(input_ids.shape[1] - 1)
    return int(torch.nonzero(mask[0], as_tuple=False).flatten()[-1].item())


def find_word_position(input_ids: torch.Tensor, tokenizer, word: str) -> int:
    """Position of the LAST token of the LAST occurrence of `word` in input_ids[0].

    Tries the word with and without a leading space, since most tokenizers
    encode a mid-sentence word with its preceding space.
    """
    ids = input_ids[0].tolist()
    best = -1
    for variant in (f" {word}", word):
        pattern = tokenizer.encode(variant, add_special_tokens=False)
        if not pattern:
            continue
        n = len(pattern)
        for start in range(len(ids) - n, -1, -1):
            if ids[start:start + n] == pattern:
                best = max(best, start + n - 1)
                break
    if best < 0:
        raise ValueError(f"Could not find the word {word!r} in the tokenized prompt.")
    return best


def decode_token(tokenizer, token_id: int) -> str:
    return tokenizer.decode([int(token_id)], skip_special_tokens=False,
                            clean_up_tokenization_spaces=False)


# --------------------------------------------------------------------------
# Yes/no scoring
# --------------------------------------------------------------------------
def _single_token(tokenizer, label: str) -> int:
    for candidate in (label, f" {label}", f"\n{label}"):
        ids = tokenizer.encode(candidate, add_special_tokens=False)
        if len(ids) == 1 and decode_token(tokenizer, ids[0]).strip().lower() == label.lower():
            return int(ids[0])
    raise ValueError(f"Could not find a single token for {label!r}.")


def resolve_answer_tokens(tokenizer, yes: str = "True", no: str = "False") -> Dict[str, int]:
    yes_id, no_id = _single_token(tokenizer, yes), _single_token(tokenizer, no)
    if yes_id == no_id:
        raise ValueError("Yes and no answers resolved to the same token.")
    return {"yes_id": yes_id, "no_id": no_id}


def score_answer(logits: torch.Tensor, position: int, answer_tokens: Dict[str, int],
                 target_present: bool) -> Dict[str, Any]:
    """Prediction = yes-logit > no-logit; margin is signed toward the correct answer."""
    next_logits = logits[0, position, :].float()
    yes_minus_no = float(next_logits[answer_tokens["yes_id"]] - next_logits[answer_tokens["no_id"]])
    prediction = yes_minus_no > 0
    return {
        "prediction": bool(prediction),
        "correct": bool(prediction == target_present),
        "yes_minus_no_logit": yes_minus_no,
        "correct_logit_margin": yes_minus_no if target_present else -yes_minus_no,
    }


# --------------------------------------------------------------------------
# Decoder discovery and head layout
# --------------------------------------------------------------------------
@dataclass
class DecoderStack:
    path: str
    layers: List[nn.Module]


def discover_decoder_stack(model: nn.Module) -> DecoderStack:
    """The largest nn.ModuleList whose layers all expose self_attn.o_proj (the LM decoder)."""
    candidates = []
    for name, module in model.named_modules():
        if isinstance(module, nn.ModuleList):
            layers = list(module)
            if layers and all(
                isinstance(getattr(getattr(l, "self_attn", None), "o_proj", None), nn.Module)
                for l in layers
            ):
                candidates.append((name, layers))
    if not candidates:
        raise RuntimeError("Could not locate decoder layers exposing self_attn.o_proj.")
    name, layers = max(candidates, key=lambda c: len(c[1]))
    return DecoderStack(path=name, layers=layers)


def _config_value(model, names: Sequence[str]) -> Optional[int]:
    root = getattr(model, "config", None)
    for config in (root, getattr(root, "text_config", None), getattr(root, "llm_config", None)):
        for n in names:
            value = getattr(config, n, None) if config is not None else None
            if value is not None:
                return int(value)
    return None


def resolve_num_heads(model, layer: nn.Module) -> int:
    for attr in ("num_heads", "num_attention_heads", "n_heads"):
        value = getattr(layer.self_attn, attr, None)
        if value is not None:
            return int(value)
    value = _config_value(model, ["num_attention_heads", "n_head", "num_heads"])
    if value is None:
        raise RuntimeError("Could not determine the number of attention heads.")
    return value


def resolve_head_dim(model, layer: nn.Module, num_heads: int) -> Optional[int]:
    for attr in ("head_dim", "attention_head_size"):
        value = getattr(layer.self_attn, attr, None)
        if value is not None:
            return int(value)
    hidden = _config_value(model, ["hidden_size", "d_model", "n_embd"])
    return hidden // num_heads if hidden and hidden % num_heads == 0 else None


def to_heads(tensor: torch.Tensor, num_heads: int, head_dim: Optional[int]) -> Tuple[torch.Tensor, str]:
    """o_proj input -> [batch, seq, heads, head_dim], plus the original layout."""
    if tensor.ndim == 3:
        b, s, hidden = tensor.shape
        d = hidden // num_heads
        if hidden % num_heads or (head_dim is not None and d != head_dim):
            raise ValueError(f"Cannot split hidden size {hidden} into {num_heads} heads.")
        return tensor.reshape(b, s, num_heads, d), "BSH"
    if tensor.ndim == 4:
        if tensor.shape[1] == num_heads:
            return tensor.permute(0, 2, 1, 3), "BHSD"
        if tensor.shape[2] == num_heads:
            return tensor, "BSHD"
    raise ValueError(f"Unexpected o_proj input shape {tuple(tensor.shape)}.")


def from_heads(heads: torch.Tensor, layout: str) -> torch.Tensor:
    if layout == "BSH":
        b, s, h, d = heads.shape
        return heads.reshape(b, s, h * d)
    if layout == "BHSD":
        return heads.permute(0, 2, 1, 3)
    if layout == "BSHD":
        return heads
    raise ValueError(f"Unknown layout {layout}.")
