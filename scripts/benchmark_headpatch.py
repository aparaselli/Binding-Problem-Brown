#!/usr/bin/env python3
# 1. Runs behavioral experiment
# 2. Patches head once at a time @ Generation token
# 3. Calculates recovery 

import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from tqdm import tqdm

from search_utils import build_inputs, load_vlm
"""
Example call
python benchmark_headpatch.py \
    --source_condition disjunctive \
    --target_condition conjunctive \
    --source_distractors 40 \
    --target_distractors 40 \
    --token_offset -9 \
    --out_path ../Results/disj40_to_conj40_headpatch_qwen_-9.csv
"""

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_DATASET_DIR = os.path.abspath(
    os.path.join(
        SCRIPT_DIR,
        "..",
        "search-dataset",
        "visual_search_dataset_counterfactual",
    )
)

BINARY_PROMPT = """
Answer the following question with exactly one word:

True

or

False

Do not provide any explanation.

Question:
You are presented with an image containing letters.

Is there a green letter L in the image?
""".strip()

# NOTE: "disjunctive_T" is listed before "disjunctive" so the alternation
# prefers the longer match; with re.fullmatch this isn't strictly required
# (the trailing "_distractors_..." literal disambiguates them either way),
# but keeping the longer alternative first is safer.
FILENAME_PATTERN = re.compile(
    r"(disjunctive_T|disjunctive|conjunctive)"
    r"_distractors_(\d+)"
    r"_trial_(\d+)"
    r"_target_(true|false)\.png"
)

def parse_filename(filename: str) -> Dict[str, Any]:
    match = FILENAME_PATTERN.fullmatch(filename)
    if match is None:
        raise ValueError(f"Could not parse filename: {filename}")

    return {
        "condition": match.group(1),
        "num_distractors": int(match.group(2)),
        "trial": int(match.group(3)),
        "target_present": match.group(4) == "true",
    }


def make_image_filename(
    condition: str,
    num_distractors: int,
    trial: int,
    target_present: bool,
) -> str:
    target_text = "true" if target_present else "false"
    return (
        f"{condition}"
        f"_distractors_{num_distractors}"
        f"_trial_{trial:03d}"
        f"_target_{target_text}.png"
    )

def corresponding_image_path(
    dataset_dir: str,
    condition: str,
    num_distractors: int,
    trial: int,
    target_present: bool,
) -> str:
    return os.path.join(
        dataset_dir,
        condition,
        make_image_filename(
            condition=condition,
            num_distractors=num_distractors,
            trial=trial,
            target_present=target_present,
        ),
    )


def collect_source_paths(
    dataset_dir: str,
    conditions: Sequence[str],
    source_distractors: int,
    max_pairs: Optional[int],
) -> List[str]:
    source_paths: List[str] = []

    for condition in conditions:
        condition_dir = os.path.join(dataset_dir, condition)
        if not os.path.isdir(condition_dir):
            raise FileNotFoundError(
                f"Condition directory does not exist: {condition_dir}"
            )

        for filename in os.listdir(condition_dir):
            if not filename.endswith(".png"):
                continue

            try:
                metadata = parse_filename(filename)
            except ValueError:
                continue

            if metadata["num_distractors"] != source_distractors:
                continue

            source_paths.append(os.path.join(condition_dir, filename))

    source_paths.sort(
        key=lambda path: (
            parse_filename(os.path.basename(path))["condition"],
            parse_filename(os.path.basename(path))["trial"],
            parse_filename(os.path.basename(path))["target_present"],
        )
    )

    if max_pairs is not None:
        source_paths = source_paths[:max_pairs]

    return source_paths

def print_patch_token_window(
    inputs: Dict[str, torch.Tensor],
    tokenizer,
    patch_position: int,
    window: int = 2,
    label: str = "Patch",
) -> None:
    input_ids = inputs["input_ids"][0]

    start = max(0, patch_position - window)
    end = min(len(input_ids), patch_position + window + 1)

    print(f"\n{label} token window:")
    print(f"  patch_position: {patch_position}")

    for pos in range(start, end):
        token_id = int(input_ids[pos].item())
        token = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

        marker = "  <-- PATCHED HERE" if pos == patch_position else ""

        print(
            f"  position {pos:4d} | "
            f"id {token_id:6d} | "
            f"{token!r}{marker}"
        )

def load_target_metadata(image_path: str) -> Dict[str, Optional[float]]:
    json_path = os.path.splitext(image_path)[0] + ".json"

    if not os.path.exists(json_path):
        return {"target_x": None, "target_y": None}

    with open(json_path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    targets = [
        obj
        for obj in metadata.get("objects", [])
        if obj.get("role") == "target"
    ]

    if not targets:
        return {"target_x": None, "target_y": None}

    position = targets[0].get("position_plot")
    if position is None or len(position) != 2:
        return {"target_x": None, "target_y": None}

    return {
        "target_x": position[0],
        "target_y": position[1],
    }


def compute_target_position_features(
    target_x: Optional[float],
    target_y: Optional[float],
) -> Dict[str, Optional[float]]:
    if target_x is None or target_y is None:
        return {
            "target_x_norm": None,
            "target_y_norm": None,
            "target_dist_from_center": None,
        }

    x_norm = (target_x - 0.5) / 9.0
    y_norm = (target_y - 0.5) / 9.0
    distance = float(
        np.sqrt((x_norm - 0.5) ** 2 + (y_norm - 0.5) ** 2)
    )

    return {
        "target_x_norm": float(x_norm),
        "target_y_norm": float(y_norm),
        "target_dist_from_center": distance,
    }


def get_tokenizer(processor):
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise AttributeError(
            "The processor does not expose processor.tokenizer."
        )
    return tokenizer


def resolve_single_token(tokenizer, label: str) -> Tuple[int, str, str]:
    candidates = [label, f" {label}", f"\n{label}"]
    attempted = []

    for candidate in candidates:
        token_ids = tokenizer.encode(candidate, add_special_tokens=False)
        attempted.append((candidate, token_ids))

        if len(token_ids) != 1:
            continue

        token_id = int(token_ids[0])
        decoded = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

        if decoded.strip().lower() == label.lower():
            return token_id, decoded, candidate

    raise ValueError(
        f"Could not find a single token for {label!r}. "
        f"Attempted encodings: {attempted}"
    )


def resolve_binary_tokens(tokenizer) -> Dict[str, Any]:
    true_id, true_decoded, true_candidate = resolve_single_token(
        tokenizer, "True"
    )
    false_id, false_decoded, false_candidate = resolve_single_token(
        tokenizer, "False"
    )

    if true_id == false_id:
        raise ValueError(
            "True and False unexpectedly resolved to the same token ID."
        )

    return {
        "true_id": true_id,
        "true_decoded": true_decoded,
        "true_candidate": true_candidate,
        "false_id": false_id,
        "false_decoded": false_decoded,
        "false_candidate": false_candidate,
    }


def get_last_input_position(inputs: Dict[str, torch.Tensor]) -> int:
    input_ids = inputs.get("input_ids")
    if input_ids is None:
        raise KeyError("Model inputs do not contain input_ids.")

    if input_ids.shape[0] != 1:
        raise ValueError(
            f"Expected batch size 1, got {input_ids.shape[0]}."
        )

    attention_mask = inputs.get("attention_mask")
    if attention_mask is None:
        return int(input_ids.shape[1] - 1)

    active_positions = torch.nonzero(
        attention_mask[0], as_tuple=False
    ).flatten()

    if active_positions.numel() == 0:
        raise ValueError("attention_mask contains no active positions.")

    return int(active_positions[-1].item())


def score_next_token_logits(
    logits: torch.Tensor,
    last_position: int,
    tokenizer,
    binary_tokens: Dict[str, Any],
    target_present: bool,
) -> Dict[str, Any]:
    if logits.ndim != 3:
        raise ValueError(
            f"Expected logits [batch, seq, vocab], got {tuple(logits.shape)}"
        )

    next_token_logits = logits[0, last_position, :].float()
    next_token_logprobs = torch.log_softmax(next_token_logits, dim=-1)

    first_token_id = int(torch.argmax(next_token_logits).item())
    first_token = tokenizer.decode(
        [first_token_id],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )

    true_id = binary_tokens["true_id"]
    false_id = binary_tokens["false_id"]

    true_logit = float(next_token_logits[true_id].item())
    false_logit = float(next_token_logits[false_id].item())
    true_logprob = float(next_token_logprobs[true_id].item())
    false_logprob = float(next_token_logprobs[false_id].item())

    true_minus_false_logit = true_logit - false_logit
    prediction = true_logit > false_logit
    correct_logit_margin = (
        true_minus_false_logit
        if target_present
        else -true_minus_false_logit
    )

    return {
        "response": first_token.strip(),
        "prediction": bool(prediction),
        "correct": bool(prediction == target_present),
        "first_token": first_token,
        "first_token_stripped": first_token.strip(),
        "first_token_id": first_token_id,
        "first_token_logprob": float(
            next_token_logprobs[first_token_id].item()
        ),
        "true_logit": true_logit,
        "false_logit": false_logit,
        "true_logprob": true_logprob,
        "false_logprob": false_logprob,
        "true_minus_false_logit": true_minus_false_logit,
        "correct_logit_margin": correct_logit_margin,
        "top_token_matches_binary_choice": bool(
            first_token_id == (true_id if prediction else false_id)
        ),
    }


# decoder discovery --- allows flexibility with various VLMs (sourced from chatGPT)

@dataclass
class DecoderStack:
    path: str
    layers: List[nn.Module]


def has_patchable_attention(layer: nn.Module) -> bool:
    self_attn = getattr(layer, "self_attn", None)
    return (
        self_attn is not None
        and isinstance(getattr(self_attn, "o_proj", None), nn.Module)
    )


def discover_decoder_stack(model: nn.Module) -> DecoderStack:
    candidates: List[Tuple[str, List[nn.Module]]] = []

    for module_name, module in model.named_modules():
        if not isinstance(module, nn.ModuleList):
            continue

        layers = list(module)
        if layers and all(has_patchable_attention(layer) for layer in layers):
            candidates.append((module_name, layers))

    if not candidates:
        inspected = [
            name
            for name, _ in model.named_modules()
            if name.endswith("layers")
        ]
        raise RuntimeError(
            "Could not locate decoder layers exposing self_attn.o_proj. "
            f"Names ending in 'layers': {inspected[:30]}"
        )

    candidates.sort(key=lambda item: len(item[1]), reverse=True)
    path, layers = candidates[0]
    return DecoderStack(path=path, layers=layers)


def nested_config_value(
    model,
    names: Sequence[str],
) -> Optional[int]:
    root_config = getattr(model, "config", None)
    configs = [
        root_config,
        getattr(root_config, "text_config", None),
        getattr(root_config, "llm_config", None),
    ]

    for config in configs:
        if config is None:
            continue
        for name in names:
            value = getattr(config, name, None)
            if value is not None:
                return int(value)

    return None


def resolve_num_heads(model, layer: nn.Module) -> int:
    self_attn = layer.self_attn

    for attr_name in ["num_heads", "num_attention_heads", "n_heads"]:
        value = getattr(self_attn, attr_name, None)
        if value is not None:
            return int(value)

    value = nested_config_value(
        model,
        ["num_attention_heads", "n_head", "num_heads"],
    )
    if value is None:
        raise RuntimeError("Could not determine num_attention_heads.")

    return value


def resolve_head_dim(
    model,
    layer: nn.Module,
    num_heads: int,
) -> Optional[int]:
    self_attn = layer.self_attn

    for attr_name in ["head_dim", "attention_head_size"]:
        value = getattr(self_attn, attr_name, None)
        if value is not None:
            return int(value)

    hidden_size = nested_config_value(
        model,
        ["hidden_size", "d_model", "n_embd"],
    )

    if hidden_size is not None and hidden_size % num_heads == 0:
        return hidden_size // num_heads

    return None


def pre_o_proj_to_heads(
    tensor: torch.Tensor,
    num_heads: int,
    expected_head_dim: Optional[int],
) -> Tuple[torch.Tensor, str]:
    """Return [batch, sequence, heads, head_dim] and original layout."""

    if tensor.ndim == 3:
        batch_size, sequence_length, hidden_size = tensor.shape
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"Hidden size {hidden_size} is not divisible by "
                f"{num_heads} heads."
            )

        head_dim = hidden_size // num_heads
        if expected_head_dim is not None and head_dim != expected_head_dim:
            raise ValueError(
                f"Inferred head_dim={head_dim}, expected "
                f"{expected_head_dim}."
            )

        return (
            tensor.reshape(
                batch_size,
                sequence_length,
                num_heads,
                head_dim,
            ),
            "BSH",
        )

    if tensor.ndim == 4:
        if tensor.shape[1] == num_heads:
            heads = tensor.permute(0, 2, 1, 3)
            layout = "BHSD"
        elif tensor.shape[2] == num_heads:
            heads = tensor
            layout = "BSHD"
        else:
            raise ValueError(
                "Could not identify head axis in tensor with shape "
                f"{tuple(tensor.shape)}."
            )

        if (
            expected_head_dim is not None
            and heads.shape[-1] != expected_head_dim
        ):
            raise ValueError(
                f"Observed head_dim={heads.shape[-1]}, expected "
                f"{expected_head_dim}."
            )

        return heads, layout

    raise ValueError(
        "Expected a 3D or 4D o_proj input, got "
        f"{tuple(tensor.shape)}."
    )


def heads_to_pre_o_proj(
    heads: torch.Tensor,
    layout: str,
) -> torch.Tensor:
    if layout == "BSH":
        batch_size, sequence_length, num_heads, head_dim = heads.shape
        return heads.reshape(
            batch_size,
            sequence_length,
            num_heads * head_dim,
        )
    if layout == "BHSD":
        return heads.permute(0, 2, 1, 3)
    if layout == "BSHD":
        return heads

    raise ValueError(f"Unknown pre-o_proj layout: {layout}")


# cache each attention head (O-proj)
class OProjInputCache:
    def __init__(self):
        self.values: Dict[int, torch.Tensor] = {}

    def make_hook(self, layer_index: int):
        def hook(module, args):
            if not args:
                raise RuntimeError(
                    f"Layer {layer_index} o_proj pre-hook received no inputs."
                )

            value = args[0]
            if not torch.is_tensor(value):
                raise TypeError(
                    f"Layer {layer_index} o_proj input is not a tensor: "
                    f"{type(value)}"
                )

            self.values[layer_index] = value.detach()

        return hook


def move_cache_to_storage(
    tensor: torch.Tensor,
    cache_dtype: str,
) -> torch.Tensor:
    tensor = tensor.to("cpu")

    if cache_dtype == "float16":
        return tensor.to(torch.float16)
    if cache_dtype == "bfloat16":
        return tensor.to(torch.bfloat16)
    if cache_dtype == "float32":
        return tensor.to(torch.float32)

    raise ValueError(f"Unsupported cache dtype: {cache_dtype}")


@torch.inference_mode()
def run_source_with_attention_cache(
    model,
    processor,
    image: Image.Image,
    target_present: bool,
    tokenizer,
    binary_tokens: Dict[str, Any],
    decoder_stack: DecoderStack,
    num_heads: int,
    expected_head_dim: Optional[int],
    cache_dtype: str,
    token_offset: int,
) -> Tuple[Dict[str, Any], Dict[int, torch.Tensor], Dict[str, Any]]:
    model_device = next(model.parameters()).device
    inputs = build_inputs(
        processor=processor,
        model_device=model_device,
        image=image,
        prompt=BINARY_PROMPT,
    )
    last_position = get_last_input_position(inputs)


    patch_position = last_position + token_offset

    print_patch_token_window(
        inputs=inputs,
        tokenizer=tokenizer,
        patch_position=patch_position,
        window=2,
        label="SOURCE",
    )

    if patch_position < 0:
        raise IndexError(
            f"token_offset={token_offset} gives invalid "
            f"patch_position={patch_position}."
        )

    cache = OProjInputCache()
    handles = []

    try:
        for layer_index, layer in enumerate(decoder_stack.layers):
            handles.append(
                layer.self_attn.o_proj.register_forward_pre_hook(
                    cache.make_hook(layer_index)
                )
            )

        outputs = model(
            **inputs,
            use_cache=False,
            return_dict=True,
        )
    finally:
        for handle in handles:
            handle.remove()

    expected_layers = set(range(len(decoder_stack.layers)))
    observed_layers = set(cache.values)
    if observed_layers != expected_layers:
        missing = sorted(expected_layers - observed_layers)
        raise RuntimeError(
            f"Cached {len(observed_layers)} of {len(expected_layers)} "
            f"layers. Missing: {missing}"
        )

    behavior = score_next_token_logits(
        logits=outputs.logits,
        last_position=last_position,
        tokenizer=tokenizer,
        binary_tokens=binary_tokens,
        target_present=target_present,
    )

    source_cache: Dict[int, torch.Tensor] = {}
    observed_shapes: Dict[int, Tuple[int, ...]] = {}

    for layer_index in range(len(decoder_stack.layers)):
        heads, _ = pre_o_proj_to_heads(
            tensor=cache.values[layer_index],
            num_heads=num_heads,
            expected_head_dim=expected_head_dim,
        )

        if heads.shape[0] != 1:
            raise ValueError("Caching expects batch size 1.")
        if patch_position >= heads.shape[1]:
            raise IndexError(
                f"patch_position={patch_position} exceeds sequence "
                f"length={heads.shape[1]}."
            )

        selected = heads[0, patch_position].detach()
        source_cache[layer_index] = move_cache_to_storage(
            selected,
            cache_dtype=cache_dtype,
        )
        observed_shapes[layer_index] = tuple(selected.shape)

    first_shape = observed_shapes[0]
    metadata = {
        "input_text_length": int(inputs["input_ids"].shape[-1]),
        "last_input_position": last_position,
        "patch_position": patch_position,
        "token_offset": token_offset,
        "layer_cache_shape": first_shape,
        "cache_dtype": cache_dtype,
    }

    return behavior, source_cache, metadata


@torch.inference_mode()
def run_unpatched_forward(
    model,
    processor,
    image: Image.Image,
    target_present: bool,
    tokenizer,
    binary_tokens: Dict[str, Any],
) -> Tuple[Dict[str, Any], int, int]:
    model_device = next(model.parameters()).device
    inputs = build_inputs(
        processor=processor,
        model_device=model_device,
        image=image,
        prompt=BINARY_PROMPT,
    )
    last_position = get_last_input_position(inputs)

    outputs = model(
        **inputs,
        use_cache=False,
        return_dict=True,
    )

    result = score_next_token_logits(
        logits=outputs.logits,
        last_position=last_position,
        tokenizer=tokenizer,
        binary_tokens=binary_tokens,
        target_present=target_present,
    )

    return result, int(inputs["input_ids"].shape[-1]), last_position


class SingleHeadLastTokenPatcher:
    def __init__(
        self,
        source_head: torch.Tensor,
        head_index: int,
        target_position: int,
        num_heads: int,
        expected_head_dim: Optional[int],
    ):
        self.source_head = source_head
        self.head_index = head_index
        self.target_position = target_position
        self.num_heads = num_heads
        self.expected_head_dim = expected_head_dim

        self.called = False
        self.observed_input_shape: Optional[Tuple[int, ...]] = None
        self.observed_layout: Optional[str] = None

    def hook(self, module, args):
        if not args:
            raise RuntimeError(
                "o_proj patch hook received no positional arguments."
            )

        current = args[0]
        if not torch.is_tensor(current):
            raise TypeError(
                f"o_proj input is not a tensor: {type(current)}"
            )

        heads, layout = pre_o_proj_to_heads(
            tensor=current,
            num_heads=self.num_heads,
            expected_head_dim=self.expected_head_dim,
        )

        if heads.shape[0] != 1:
            raise ValueError("Per-head patching expects batch size 1.")
        if self.target_position >= heads.shape[1]:
            raise IndexError(
                f"Target position {self.target_position} is outside "
                f"sequence length {heads.shape[1]}."
            )
        if self.head_index >= heads.shape[2]:
            raise IndexError(
                f"Head index {self.head_index} is outside "
                f"num_heads={heads.shape[2]}."
            )

        source = self.source_head.to(
            device=current.device,
            dtype=current.dtype,
        )

        if source.ndim != 1:
            raise ValueError(
                f"Expected source head shape [head_dim], got "
                f"{tuple(source.shape)}."
            )
        if source.shape[0] != heads.shape[-1]:
            raise ValueError(
                f"Source head_dim={source.shape[0]} does not match "
                f"target head_dim={heads.shape[-1]}."
            )

        patched_heads = heads.clone()
        patched_heads[
            0,
            self.target_position,
            self.head_index,
            :,
        ] = source

        patched_input = heads_to_pre_o_proj(
            heads=patched_heads,
            layout=layout,
        )

        self.called = True
        self.observed_input_shape = tuple(current.shape)
        self.observed_layout = layout

        return (patched_input,) + tuple(args[1:])


@torch.inference_mode()
def run_single_head_patch(
    model,
    processor,
    image: Image.Image,
    target_present: bool,
    tokenizer,
    binary_tokens: Dict[str, Any],
    layer: nn.Module,
    layer_index: int,
    head_index: int,
    source_head: torch.Tensor,
    num_heads: int,
    expected_head_dim: Optional[int],
    token_offset: int,
):
    model_device = next(model.parameters()).device
    inputs = build_inputs(
        processor=processor,
        model_device=model_device,
        image=image,
        prompt=BINARY_PROMPT,
    )
    last_position = get_last_input_position(inputs)

    patch_position = last_position + token_offset

    if patch_position < 0:
        raise IndexError(
            f"token_offset={token_offset} gives invalid "
            f"patch_position={patch_position}."
        )

    patcher = SingleHeadLastTokenPatcher(
        source_head=source_head,
        head_index=head_index,
        target_position=patch_position,
        num_heads=num_heads,
        expected_head_dim=expected_head_dim,
    )

    handle = layer.self_attn.o_proj.register_forward_pre_hook(
        patcher.hook
    )

    try:
        outputs = model(
            **inputs,
            use_cache=False,
            return_dict=True,
        )
    finally:
        handle.remove()

    if not patcher.called:
        raise RuntimeError(
            f"Patch hook was not called for layer {layer_index}."
        )

    result = score_next_token_logits(
        logits=outputs.logits,
        last_position=last_position,
        tokenizer=tokenizer,
        binary_tokens=binary_tokens,
        target_present=target_present,
    )

    intervention_metadata = {
        "target_input_text_length": int(inputs["input_ids"].shape[-1]),
        "target_last_position": last_position,
        "target_patch_position": patch_position,
        "token_offset": token_offset,
        "observed_o_proj_input_shape": str(
            patcher.observed_input_shape
        ),
        "observed_o_proj_layout": patcher.observed_layout,
    }

    return result, intervention_metadata


def compute_recovery(
    source_margin: float,
    target_margin: float,
    patched_margin: float,
    min_denominator: float,
) -> Dict[str, Any]:
    margin_drop = source_margin - target_margin
    delta_margin = patched_margin - target_margin

    recovery_valid = (
        np.isfinite(margin_drop)
        and margin_drop > min_denominator
    )

    recovery = (
        delta_margin / margin_drop
        if recovery_valid
        else np.nan
    )

    return {
        "margin_drop": margin_drop,
        "delta_margin": delta_margin,
        "recovery": recovery,
        "recovery_valid": bool(recovery_valid),
    }


def make_head_summary(dataframe: pd.DataFrame) -> pd.DataFrame:
    if dataframe.empty or "error" not in dataframe.columns:
        return pd.DataFrame()

    successful = dataframe[dataframe["error"].isna()].copy()
    if successful.empty:
        return pd.DataFrame()

    def valid_recovery_mean(series: pd.Series) -> float:
        values = pd.to_numeric(series, errors="coerce").dropna()
        return float(values.mean()) if len(values) else np.nan

    return (
        successful.groupby(["layer", "head"], dropna=False)
        .agg(
            n=("delta_margin", "size"),
            mean_delta_margin=("delta_margin", "mean"),
            median_delta_margin=("delta_margin", "median"),
            positive_delta_rate=(
                "delta_margin",
                lambda values: float((values > 0).mean()),
            ),
            mean_recovery=("recovery", valid_recovery_mean),
            median_recovery=("recovery", "median"),
            valid_recovery_n=(
                "recovery_valid",
                lambda values: int(values.sum()),
            ),
            patched_correct_rate=("patched_correct", "mean"),
            patched_flip_to_correct_rate=(
                "patched_flip_to_correct",
                "mean",
            ),
        )
        .reset_index()
        .sort_values(
            ["mean_delta_margin", "mean_recovery"],
            ascending=False,
        )
    )


def make_pair_summary(dataframe: pd.DataFrame) -> pd.DataFrame:
    if dataframe.empty:
        return pd.DataFrame()

    successful = dataframe[dataframe["error"].isna()].copy()
    if successful.empty:
        return pd.DataFrame()

    pair_columns = [
        "source_condition",
        "target_condition",
        "trial",
        "target_present",
        "source_distractors",
        "target_distractors",
    ]

    return (
        successful.groupby(pair_columns, dropna=False)
        .agg(
            source_margin=("source_margin", "first"),
            target_margin=("target_margin", "first"),
            margin_drop=("margin_drop", "first"),
            source_correct=("source_correct", "first"),
            target_correct=("target_correct", "first"),
            best_delta_margin=("delta_margin", "max"),
            mean_delta_margin=("delta_margin", "mean"),
            best_recovery=("recovery", "max"),
            any_flip_to_correct=("patched_flip_to_correct", "max"),
        )
        .reset_index()
    )


def save_progress(rows: List[Dict[str, Any]], out_path: str) -> None:
    pd.DataFrame(rows).to_csv(out_path, index=False)


def main(args) -> None:
    output_directory = os.path.dirname(os.path.abspath(args.out_path))
    os.makedirs(output_directory, exist_ok=True)

    model, processor = load_vlm(args.model_name)
    tokenizer = get_tokenizer(processor)
    binary_tokens = resolve_binary_tokens(tokenizer)

    decoder_stack = discover_decoder_stack(model)
    num_layers = len(decoder_stack.layers)
    num_heads = resolve_num_heads(model, decoder_stack.layers[0])
    expected_head_dim = resolve_head_dim(
        model,
        decoder_stack.layers[0],
        num_heads,
    )

    print("\nResolved binary answer tokens:")
    print(
        f"  True:  {binary_tokens['true_id']} "
        f"{binary_tokens['true_decoded']!r}"
    )
    print(
        f"  False: {binary_tokens['false_id']} "
        f"{binary_tokens['false_decoded']!r}"
    )

    print("\nDiscovered language decoder:")
    print(f"  stack path: {decoder_stack.path}")
    print(f"  layers:     {num_layers}")
    print(f"  heads:      {num_heads}")
    print(f"  head dim:   {expected_head_dim}")

    source_paths = collect_source_paths(
        dataset_dir=args.dataset_dir,
        conditions=[args.source_condition],
        source_distractors=args.source_distractors,
        max_pairs=args.max_pairs,
    )
    if args.token_offset > 0:
        raise ValueError(
            "--token_offset must be <= 0. "
            "Use 0 for the final input token, -1 for one token before, etc."
        )
    print(
        f"\nFound {len(source_paths)} matched-source candidates with "
        f"{args.source_distractors} distractors"
    )
    print(
        f"Patching into matched {args.target_distractors}-distractor examples"
    )
    print(f"Source cache dtype: {args.cache_dtype}")

    rows: List[Dict[str, Any]] = []
    completed_patches = 0

    pair_iterator = tqdm(source_paths, desc="Trial pairs")

    for source_image_path in pair_iterator:
        source_metadata = parse_filename(
            os.path.basename(source_image_path)
        )
        source_condition = source_metadata["condition"]
        trial = source_metadata["trial"]
        target_present = source_metadata["target_present"]

        target_condition = args.target_condition

        target_image_path = corresponding_image_path(
            dataset_dir=args.dataset_dir,
            condition=target_condition,
            num_distractors=args.target_distractors,
            trial=trial,
            target_present=target_present,
        )

        if not os.path.exists(target_image_path):
            print(f"\nSkipping missing target image: {target_image_path}")
            continue

        try:
            with Image.open(source_image_path) as image_handle:
                source_image = image_handle.convert("RGB")

            (
                source_result,
                source_cache,
                source_cache_metadata,
            ) = run_source_with_attention_cache(
                model=model,
                processor=processor,
                image=source_image,
                target_present=target_present,
                tokenizer=tokenizer,
                binary_tokens=binary_tokens,
                decoder_stack=decoder_stack,
                num_heads=num_heads,
                expected_head_dim=expected_head_dim,
                cache_dtype=args.cache_dtype,
                token_offset=args.token_offset,
            )

            source_margin = float(
                source_result["correct_logit_margin"]
            )
            source_correct = bool(source_result["correct"])

            if args.require_source_correct and not source_correct:
                continue

            with Image.open(target_image_path) as image_handle:
                target_image = image_handle.convert("RGB")

            (
                target_result,
                target_input_length,
                target_last_position,
            ) = run_unpatched_forward(
                model=model,
                processor=processor,
                image=target_image,
                target_present=target_present,
                tokenizer=tokenizer,
                binary_tokens=binary_tokens,
            )

            target_margin = float(
                target_result["correct_logit_margin"]
            )
            margin_drop = source_margin - target_margin

            if (
                args.require_margin_drop
                and margin_drop <= args.min_recovery_denominator
            ):
                continue

            source_position_metadata = load_target_metadata(
                source_image_path
            )
            source_position_features = compute_target_position_features(
                source_position_metadata["target_x"],
                source_position_metadata["target_y"],
            )

            for layer_index, layer in enumerate(decoder_stack.layers):
                source_layer = source_cache[layer_index]

                for head_index in range(num_heads):
                    try:
                        (
                            patched_result,
                            intervention_metadata,
                        ) = run_single_head_patch(
                            model=model,
                            processor=processor,
                            image=target_image,
                            target_present=target_present,
                            tokenizer=tokenizer,
                            binary_tokens=binary_tokens,
                            layer=layer,
                            layer_index=layer_index,
                            head_index=head_index,
                            source_head=source_layer[head_index],
                            num_heads=num_heads,
                            expected_head_dim=expected_head_dim,
                            token_offset=args.token_offset,
                        )

                        patched_margin = float(
                            patched_result["correct_logit_margin"]
                        )
                        recovery_metrics = compute_recovery(
                            source_margin=source_margin,
                            target_margin=target_margin,
                            patched_margin=patched_margin,
                            min_denominator=args.min_recovery_denominator,
                        )

                        patched_flip_to_correct = bool(
                            (not target_result["correct"])
                            and patched_result["correct"]
                        )

                        row = {
                            "source_condition": source_condition,
                            "target_condition": target_condition,
                            "trial": trial,
                            "target_present": target_present,
                            "source_distractors": args.source_distractors,
                            "target_distractors": args.target_distractors,
                            "source_image_path": source_image_path,
                            "target_image_path": target_image_path,
                            "target_x": source_position_metadata["target_x"],
                            "target_y": source_position_metadata["target_y"],
                            "target_x_norm": source_position_features[
                                "target_x_norm"
                            ],
                            "target_y_norm": source_position_features[
                                "target_y_norm"
                            ],
                            "target_dist_from_center": source_position_features[
                                "target_dist_from_center"
                            ],
                            "layer": layer_index,
                            "head": head_index,
                            "source_prediction": source_result["prediction"],
                            "source_correct": source_correct,
                            "source_true_logit": source_result["true_logit"],
                            "source_false_logit": source_result["false_logit"],
                            "source_true_logprob": source_result["true_logprob"],
                            "source_false_logprob": source_result["false_logprob"],
                            "source_margin": source_margin,
                            "target_prediction": target_result["prediction"],
                            "target_correct": target_result["correct"],
                            "target_true_logit": target_result["true_logit"],
                            "target_false_logit": target_result["false_logit"],
                            "target_true_logprob": target_result["true_logprob"],
                            "target_false_logprob": target_result["false_logprob"],
                            "target_margin": target_margin,
                            "patched_prediction": patched_result["prediction"],
                            "patched_correct": patched_result["correct"],
                            "patched_true_logit": patched_result["true_logit"],
                            "patched_false_logit": patched_result["false_logit"],
                            "patched_true_logprob": patched_result["true_logprob"],
                            "patched_false_logprob": patched_result["false_logprob"],
                            "patched_margin": patched_margin,
                            **recovery_metrics,
                            "patched_flip_to_correct": patched_flip_to_correct,
                            "source_input_text_length": source_cache_metadata[
                                "input_text_length"
                            ],
                            "source_last_position": source_cache_metadata[
                                "last_input_position"
                            ],
                            "source_layer_cache_shape": str(
                                source_cache_metadata["layer_cache_shape"]
                            ),
                            "source_cache_dtype": args.cache_dtype,
                            "target_input_text_length": target_input_length,
                            "target_last_position": target_last_position,
                            "observed_o_proj_input_shape": intervention_metadata[
                                "observed_o_proj_input_shape"
                            ],
                            "observed_o_proj_layout": intervention_metadata[
                                "observed_o_proj_layout"
                            ],
                            "token_offset": args.token_offset,
                            "source_patch_position": source_cache_metadata["patch_position"],
                            "target_patch_position": intervention_metadata["target_patch_position"],
                            "decoder_stack_path": decoder_stack.path,
                            "num_layers": num_layers,
                            "num_heads": num_heads,
                            "head_dim": expected_head_dim,
                            "error": None,
                        }

                    except Exception as patch_error:
                        row = {
                            "source_condition": source_condition,
                            "target_condition": target_condition,
                            "trial": trial,
                            "target_present": target_present,
                            "source_distractors": args.source_distractors,
                            "target_distractors": args.target_distractors,
                            "source_image_path": source_image_path,
                            "target_image_path": target_image_path,
                            "target_x": source_position_metadata["target_x"],
                            "target_y": source_position_metadata["target_y"],
                            "target_x_norm": source_position_features[
                                "target_x_norm"
                            ],
                            "target_y_norm": source_position_features[
                                "target_y_norm"
                            ],
                            "target_dist_from_center": source_position_features[
                                "target_dist_from_center"
                            ],
                            "layer": layer_index,
                            "head": head_index,
                            "source_prediction": source_result["prediction"],
                            "source_correct": source_correct,
                            "source_true_logit": source_result["true_logit"],
                            "source_false_logit": source_result["false_logit"],
                            "source_true_logprob": source_result["true_logprob"],
                            "source_false_logprob": source_result["false_logprob"],
                            "source_margin": source_margin,
                            "target_prediction": target_result["prediction"],
                            "target_correct": target_result["correct"],
                            "target_true_logit": target_result["true_logit"],
                            "target_false_logit": target_result["false_logit"],
                            "target_true_logprob": target_result["true_logprob"],
                            "target_false_logprob": target_result["false_logprob"],
                            "target_margin": target_margin,
                            "patched_prediction": None,
                            "patched_correct": False,
                            "patched_true_logit": None,
                            "patched_false_logit": None,
                            "patched_true_logprob": None,
                            "patched_false_logprob": None,
                            "patched_margin": None,
                            "margin_drop": margin_drop,
                            "delta_margin": None,
                            "recovery": None,
                            "recovery_valid": False,
                            "patched_flip_to_correct": False,
                            "source_input_text_length": source_cache_metadata[
                                "input_text_length"
                            ],
                            "source_last_position": source_cache_metadata[
                                "last_input_position"
                            ],
                            "source_layer_cache_shape": str(
                                source_cache_metadata["layer_cache_shape"]
                            ),
                            "source_cache_dtype": args.cache_dtype,
                            "target_input_text_length": target_input_length,
                            "target_last_position": target_last_position,
                            "observed_o_proj_input_shape": None,
                            "observed_o_proj_layout": None,
                            "decoder_stack_path": decoder_stack.path,
                            "num_layers": num_layers,
                            "num_heads": num_heads,
                            "head_dim": expected_head_dim,
                            "error": repr(patch_error),
                        }

                    rows.append(row)
                    completed_patches += 1

                    if (
                        args.save_every > 0
                        and completed_patches % args.save_every == 0
                    ):
                        save_progress(rows=rows, out_path=args.out_path)

            # Explicitly release the current pair's source cache before moving on.
            del source_cache

        except Exception as pair_error:
            print(
                f"\nPAIR ERROR "
                f"source={source_condition} "
                f"target={target_condition} "
                f"trial={trial} "
                f"target_present={target_present}: "
                f"{pair_error}"
            )

    results = pd.DataFrame(rows)
    results.to_csv(args.out_path, index=False)

    head_summary = make_head_summary(results)
    head_summary_path = (
        os.path.splitext(args.out_path)[0] + "_head_summary.csv"
    )
    head_summary.to_csv(head_summary_path, index=False)

    pair_summary = make_pair_summary(results)
    pair_summary_path = (
        os.path.splitext(args.out_path)[0] + "_pair_summary.csv"
    )
    pair_summary.to_csv(pair_summary_path, index=False)

    print("\nSaved per-head patch results:")
    print(args.out_path)

    print("\nSaved head summary:")
    print(head_summary_path)

    print("\nSaved pair summary:")
    print(pair_summary_path)

    if not head_summary.empty:
        print("\nTop heads by mean delta margin:")
        print(head_summary.head(20).to_string(index=False))

    if not results.empty:
        print("\nPatch errors:")
        print(results["error"].notna().sum())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Run behavioral baselines, cache source attention heads in RAM, "
            "and patch each source head into matched target runs."
        )
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=DEFAULT_DATASET_DIR,
    )
    parser.add_argument(
        "--out_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--source_condition",
        choices=["disjunctive", "disjunctive_T", "conjunctive"],
        default="disjunctive",
    )

    parser.add_argument(
        "--target_condition",
        choices=["disjunctive", "disjunctive_T", "conjunctive"],
        default="conjunctive",
    )
    parser.add_argument(
        "--source_distractors",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--target_distractors",
        type=int,
        default=40,
    )
    parser.add_argument(
        "--cache_dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
        help="CPU dtype for the one-example in-memory source cache.",
    )
    parser.add_argument(
        "--max_pairs",
        type=int,
        default=None,
        help="Optional number of source/target pairs for debugging.",
    )
    parser.add_argument(
        "--require_source_correct",
        action="store_true",
        help="Only patch pairs for which the source run was correct.",
    )
    parser.add_argument(
        "--require_margin_drop",
        action="store_true",
        help=(
            "Only patch pairs where source_margin - target_margin exceeds "
            "--min_recovery_denominator."
        ),
    )
    parser.add_argument(
        "--token_offset",
        type=int,
        default=0,
        help=(
            "Token position relative to the final input token for head patching. "
            "0 = final input token, -1 = one token before, -2 = two before, etc."
        ),
    )
    parser.add_argument(
        "--min_recovery_denominator",
        type=float,
        default=0.25,
        help=(
            "Minimum positive source-to-target margin drop required for "
            "normalized recovery to be valid."
        ),
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=100,
        help="Save partial results every N patched forward passes.",
    )

    main(parser.parse_args())