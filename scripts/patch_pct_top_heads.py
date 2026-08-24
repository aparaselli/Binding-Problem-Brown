#!/usr/bin/env python3
# 1. Runs behavioral experiment
# 2. Patches cumulative ranked head sets at one or more selectable input tokens
# 3. Calculates recovery 
"""
Cumulative multi-head activation patching at one or more selectable input tokens.

For each matched source/target image pair, this script:
  1. Runs one source image per requested patch position and caches every
     attention head's output immediately before o_proj at that position.
  2. Runs the target condition without intervention to establish its baseline.
  3. Patches cumulative subsets of ranked source attention heads into the
     target simultaneously at all requested positions. Each position has its
     own source dataset and ranking, and selects its own top N% of heads.
  4. Measures accuracy, correct-answer margin change, and normalized recovery.

Token-offset convention:
    --token_offset 0   -> final active input token
    --token_offset -1  -> second-to-last active input token
    --token_offset -2  -> third-to-last active input token
    ...

The selected source and target tokens are printed at the start of the run and
recorded in the output CSV. Regardless of the intervention position, the answer
is always scored using the logits at the final active input position.

Example: at offset -9, use heads ranked on target-present trials for present
examples and heads ranked on target-absent trials for absent examples. This
mode requires the raw per-trial/per-head CSV, not the aggregated summary CSV:

python patch_pct_top_heads.py \
    --source_condition disjunctive \
    --target_condition conjunctive \
    --source_distractors 40 \
    --target_distractors 40 \
    --token_offset -9 \
    --head_ranking_csv ../Results/disj40_to_conj40_headpatch_qwen_-9.csv \
    --ranking_metric delta_margin \
    --rank_by_target_presence \
    --patch_percentages 0.1,0.5,1,3,5,10,20,50,100 \
    --out_path ../Results/disj40_to_conj40_cumulative_offset_full_-9.csv

Two-position example (the top N% is selected independently for -11 and -9):

Two-position example (the top N% is selected independently for -11 and -9):

python patch_pct_top_heads.py \
  --target_condition conjunctive \
  --source_condition disjunctive \
  --source_distractors 40 \
  --target_distractors 40 \
  --patch_config='-11|/oscar/data/epavlick/aparasel/Binding-Problem-Brown/search-dataset/visual_search_dataset_counterfactual/src_green_mask|../Results/disj35_T_to_conj35_headpatch_qwen_-11.csv' \
  --patch_config='-9|/oscar/data/epavlick/aparasel/Binding-Problem-Brown/search-dataset/visual_search_dataset_counterfactual/src_L_mask|../Results/disj40_to_conj40_headpatch_qwen_-9.csv' \
  --ranking_metric delta_margin \
  --always_use_target_present_ranking \
  --rank_by_target_presence \
  --patch_percentages 0.1,0.5,1,3,5,10,20,50,100 \
  --out_path ../Results/green_L_mask_to_conj40.csv

#L mask only
python patch_pct_top_heads.py \
  --target_condition conjunctive \
  --source_condition disjunctive \
  --source_distractors 40 \
  --target_distractors 40 \
  --patch_config='-9|/oscar/data/epavlick/aparasel/Binding-Problem-Brown/search-dataset/visual_search_dataset_counterfactual/src_L_mask|../Results/disj40_to_conj40_headpatch_qwen_-9.csv' \
  --ranking_metric delta_margin \
  --always_use_target_present_ranking \
  --rank_by_target_presence \
  --patch_percentages 0.1,0.5,1,3,5,10,20,50,100 \
  --out_path ../Results/L_mask_to_conj40.csv

#Green mask only
python patch_pct_top_heads.py \
  --target_condition conjunctive \
  --source_condition disjunctive \
  --source_distractors 40 \
  --target_distractors 40 \
  --patch_config='-11|/oscar/data/epavlick/aparasel/Binding-Problem-Brown/search-dataset/visual_search_dataset_counterfactual/src_green_mask|../Results/disj35_T_to_conj35_headpatch_qwen_-11.csv' \
  --ranking_metric delta_margin \
  --always_use_target_present_ranking \
  --rank_by_target_presence \
  --patch_percentages 0.1,0.5,1,3,5,10,20,50,100 \
  --out_path ../Results/green_mask_to_conj40.csv

Each --patch_config is OFFSET|SOURCE_DATASET_DIR|HEAD_RANKING_CSV.  SOURCE_DATASET_DIR
may be the ordinary dataset root (containing disjunctive/ and conjunctive/) or
a leaf directory such as src_green_mask/ or src_L_mask/.
"""


import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import random
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from tqdm import tqdm

from search_utils import build_inputs, load_vlm


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

FILENAME_PATTERN = re.compile(
    r"(disjunctive|conjunctive|src_green_mask|src_L_mask)"
    r"_distractors_(\d+)"
    r"_trial_(\d+)"
    r"_target_(true|false)\.png"
)

def parse_filename(filename: str) -> Dict[str, Any]:
    match = FILENAME_PATTERN.fullmatch(filename)
    if match is None:
        raise ValueError(f"Could not parse filename: {filename}")

    return {
        "file_prefix": match.group(1),
        "condition": match.group(1),
        "num_distractors": int(match.group(2)),
        "trial": int(match.group(3)),
        "target_present": match.group(4) == "true",
    }


def make_image_filename(
    file_prefix: str,
    num_distractors: int,
    trial: int,
    target_present: bool,
) -> str:
    target_text = "true" if target_present else "false"
    return (
        f"{file_prefix}"
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
    """Resolve either a normal condition directory or a mask leaf directory."""
    leaf_prefix = os.path.basename(os.path.normpath(dataset_dir))
    prefixes = [leaf_prefix, condition]
    candidates = []
    for prefix in dict.fromkeys(prefixes):
        filename = make_image_filename(
            file_prefix=prefix,
            num_distractors=num_distractors,
            trial=trial,
            target_present=target_present,
        )
        candidates.extend([
            os.path.join(dataset_dir, filename),
            os.path.join(dataset_dir, condition, filename),
        ])

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    # Return the conventional path for useful missing-file diagnostics.
    return candidates[-1]


def collect_source_paths(
    dataset_dir: str,
    conditions: Sequence[str],
    source_distractors: int,
    max_pairs: Optional[int],
) -> List[str]:
    source_paths: List[str] = []

    for condition in conditions:
        direct_dir = dataset_dir
        nested_dir = os.path.join(dataset_dir, condition)
        condition_dir = nested_dir if os.path.isdir(nested_dir) else direct_dir
        if not os.path.isdir(condition_dir):
            raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

        for filename in os.listdir(condition_dir):
            if not filename.endswith(".png"):
                continue

            try:
                metadata = parse_filename(filename)
            except ValueError:
                continue

            if (
                metadata["num_distractors"] != source_distractors
                or (metadata["file_prefix"] in {"disjunctive", "conjunctive"}
                    and metadata["file_prefix"] != condition)
            ):
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


def resolve_patch_position(
    inputs: Dict[str, torch.Tensor],
    token_offset: int,
) -> Tuple[int, int]:
    """Return (final active position, selected patch position)."""
    if token_offset > 0:
        raise ValueError(
            "--token_offset must be <= 0. "
            "Use 0 for the final input token, -1 for one token before, etc."
        )

    last_position = get_last_input_position(inputs)
    patch_position = last_position + token_offset

    if patch_position < 0:
        raise IndexError(
            f"token_offset={token_offset} gives invalid "
            f"patch_position={patch_position}."
        )

    return last_position, patch_position


def selected_token_metadata(
    inputs: Dict[str, torch.Tensor],
    tokenizer,
    token_offset: int,
) -> Dict[str, Any]:
    last_position, patch_position = resolve_patch_position(
        inputs=inputs,
        token_offset=token_offset,
    )
    token_id = int(inputs["input_ids"][0, patch_position].item())
    token = tokenizer.decode(
        [token_id],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    return {
        "last_position": last_position,
        "patch_position": patch_position,
        "token_offset": token_offset,
        "patch_token_id": token_id,
        "patch_token": token,
    }


def print_selected_token(
    label: str,
    token_metadata: Dict[str, Any],
) -> None:
    print(f"\n{label} selected patch token:")
    print(f"  token offset:    {token_metadata['token_offset']}")
    print(f"  final position:  {token_metadata['last_position']}")
    print(f"  patch position:  {token_metadata['patch_position']}")
    print(f"  token id:        {token_metadata['patch_token_id']}")
    print(f"  decoded token:   {token_metadata['patch_token']!r}")


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
    token_metadata = selected_token_metadata(
        inputs=inputs,
        tokenizer=tokenizer,
        token_offset=token_offset,
    )
    last_position = token_metadata["last_position"]
    patch_position = token_metadata["patch_position"]

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
        "patch_token_id": token_metadata["patch_token_id"],
        "patch_token": token_metadata["patch_token"],
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
    token_offset: int,
) -> Tuple[Dict[str, Any], int, Dict[str, Any]]:
    model_device = next(model.parameters()).device
    inputs = build_inputs(
        processor=processor,
        model_device=model_device,
        image=image,
        prompt=BINARY_PROMPT,
    )
    token_metadata = selected_token_metadata(
        inputs=inputs,
        tokenizer=tokenizer,
        token_offset=token_offset,
    )
    last_position = token_metadata["last_position"]

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

    return result, int(inputs["input_ids"].shape[-1]), token_metadata



class MultiHeadTokenPatcher:
    """Patch a selected set of heads at one layer and one token position."""

    def __init__(
        self,
        source_layer_heads: torch.Tensor,
        head_indices: Sequence[int],
        target_position: int,
        num_heads: int,
        expected_head_dim: Optional[int],
    ):
        self.source_layer_heads = source_layer_heads
        self.head_indices = [int(head) for head in head_indices]
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
            raise ValueError("Multi-head patching expects batch size 1.")
        if self.target_position >= heads.shape[1]:
            raise IndexError(
                f"Target position {self.target_position} is outside "
                f"sequence length {heads.shape[1]}."
            )

        source = self.source_layer_heads.to(
            device=current.device,
            dtype=current.dtype,
        )

        if source.ndim != 2:
            raise ValueError(
                "Expected source layer cache shape [num_heads, head_dim], "
                f"got {tuple(source.shape)}."
            )
        if source.shape[0] != self.num_heads:
            raise ValueError(
                f"Source cache has {source.shape[0]} heads, expected "
                f"{self.num_heads}."
            )
        if source.shape[1] != heads.shape[-1]:
            raise ValueError(
                f"Source head_dim={source.shape[1]} does not match "
                f"target head_dim={heads.shape[-1]}."
            )

        patched_heads = heads.clone()
        for head_index in self.head_indices:
            if head_index < 0 or head_index >= heads.shape[2]:
                raise IndexError(
                    f"Head index {head_index} is outside "
                    f"num_heads={heads.shape[2]}."
                )
            patched_heads[
                0,
                self.target_position,
                head_index,
                :,
            ] = source[head_index]

        patched_input = heads_to_pre_o_proj(
            heads=patched_heads,
            layout=layout,
        )

        self.called = True
        self.observed_input_shape = tuple(current.shape)
        self.observed_layout = layout

        return (patched_input,) + tuple(args[1:])


@dataclass(frozen=True)
class PatchConfig:
    token_offset: int
    source_dataset_dir: str
    head_ranking_csv: str

    @property
    def key(self) -> str:
        return str(self.token_offset)


def parse_patch_config(text: str) -> PatchConfig:
    """Parse OFFSET|SOURCE_DATASET_DIR|HEAD_RANKING_CSV."""
    parts = [part.strip() for part in text.split("|")]
    if len(parts) != 3 or not all(parts):
        raise ValueError(
            "Each --patch_config must be exactly "
            "OFFSET|SOURCE_DATASET_DIR|HEAD_RANKING_CSV."
        )
    token_offset = int(parts[0])
    if token_offset > 0:
        raise ValueError("Each patch-config OFFSET must be <= 0.")
    return PatchConfig(
        token_offset=token_offset,
        source_dataset_dir=os.path.abspath(parts[1]),
        head_ranking_csv=parts[2],
    )


class MultiPositionHeadPatcher:
    """Patch head sets at multiple token positions in one layer hook."""

    def __init__(
        self,
        assignments: Sequence[Tuple[int, Sequence[int], torch.Tensor]],
        num_heads: int,
        expected_head_dim: Optional[int],
    ):
        self.assignments = [
            (int(position), [int(head) for head in heads], source)
            for position, heads, source in assignments
        ]
        self.num_heads = num_heads
        self.expected_head_dim = expected_head_dim
        self.called = False
        self.observed_input_shape: Optional[Tuple[int, ...]] = None
        self.observed_layout: Optional[str] = None

    def hook(self, module, args):
        if not args or not torch.is_tensor(args[0]):
            raise RuntimeError("o_proj multi-position patch hook received no tensor input.")
        current = args[0]
        heads, layout = pre_o_proj_to_heads(
            tensor=current,
            num_heads=self.num_heads,
            expected_head_dim=self.expected_head_dim,
        )
        if heads.shape[0] != 1:
            raise ValueError("Multi-position patching expects batch size 1.")

        patched_heads = heads.clone()
        for position, head_indices, source_cache in self.assignments:
            if position < 0 or position >= heads.shape[1]:
                raise IndexError(
                    f"Target position {position} is outside sequence length {heads.shape[1]}."
                )
            source = source_cache.to(device=current.device, dtype=current.dtype)
            if source.shape != (self.num_heads, heads.shape[-1]):
                raise ValueError(
                    "Expected source cache [num_heads, head_dim], got "
                    f"{tuple(source.shape)}; target shape is "
                    f"({self.num_heads}, {heads.shape[-1]})."
                )
            for head_index in head_indices:
                if head_index < 0 or head_index >= self.num_heads:
                    raise IndexError(f"Head index {head_index} is outside num_heads={self.num_heads}.")
                patched_heads[0, position, head_index, :] = source[head_index]

        self.called = True
        self.observed_input_shape = tuple(current.shape)
        self.observed_layout = layout
        return (heads_to_pre_o_proj(patched_heads, layout),) + tuple(args[1:])


@torch.inference_mode()
def run_multi_head_patch(
    model,
    processor,
    image: Image.Image,
    target_present: bool,
    tokenizer,
    binary_tokens: Dict[str, Any],
    decoder_stack: DecoderStack,
    patch_sources: Sequence[Tuple[int, Sequence[Tuple[int, int]], Dict[int, torch.Tensor]]],
    num_heads: int,
    expected_head_dim: Optional[int],
    token_offsets: Sequence[int],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Patch all selected (layer, head) pairs simultaneously. Every element of
    patch_sources supplies (token_offset, selected_heads, source_cache).
    """
    model_device = next(model.parameters()).device
    inputs = build_inputs(
        processor=processor,
        model_device=model_device,
        image=image,
        prompt=BINARY_PROMPT,
    )
    token_metadata_by_offset = {
        int(offset): selected_token_metadata(inputs, tokenizer, int(offset))
        for offset in token_offsets
    }
    last_position = get_last_input_position(inputs)

    assignments_by_layer: Dict[int, List[Tuple[int, Sequence[int], torch.Tensor]]] = {}
    for token_offset, selected_heads, source_cache in patch_sources:
        patch_position = token_metadata_by_offset[token_offset]["patch_position"]
        for layer_index, head_index in selected_heads:
            assignments_by_layer.setdefault(int(layer_index), []).append(
                (patch_position, [int(head_index)], source_cache[int(layer_index)])
            )

    patchers: Dict[int, MultiHeadTokenPatcher] = {}
    handles = []

    try:
        for layer_index, assignments in assignments_by_layer.items():
            if layer_index < 0 or layer_index >= len(decoder_stack.layers):
                raise IndexError(
                    f"Layer index {layer_index} is outside "
                    f"num_layers={len(decoder_stack.layers)}."
                )

            patcher = MultiPositionHeadPatcher(
                assignments=assignments,
                num_heads=num_heads,
                expected_head_dim=expected_head_dim,
            )
            patchers[layer_index] = patcher
            handles.append(
                decoder_stack.layers[
                    layer_index
                ].self_attn.o_proj.register_forward_pre_hook(
                    patcher.hook
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

    uncalled_layers = [
        layer_index
        for layer_index, patcher in patchers.items()
        if not patcher.called
    ]
    if uncalled_layers:
        raise RuntimeError(
            f"Patch hooks were not called for layers: {uncalled_layers}"
        )

    result = score_next_token_logits(
        logits=outputs.logits,
        last_position=last_position,
        tokenizer=tokenizer,
        binary_tokens=binary_tokens,
        target_present=target_present,
    )

    layouts = {
        str(layer_index): patcher.observed_layout
        for layer_index, patcher in patchers.items()
    }
    input_shapes = {
        str(layer_index): str(patcher.observed_input_shape)
        for layer_index, patcher in patchers.items()
    }

    intervention_metadata = {
        "target_input_text_length": int(inputs["input_ids"].shape[-1]),
        "target_last_position": last_position,
        "target_patch_positions": json.dumps({
            str(offset): metadata["patch_position"]
            for offset, metadata in token_metadata_by_offset.items()
        }, sort_keys=True),
        "token_offsets": json.dumps(list(token_offsets)),
        "patch_tokens": json.dumps({
            str(offset): {
                "id": metadata["patch_token_id"],
                "token": metadata["patch_token"],
            }
            for offset, metadata in token_metadata_by_offset.items()
        }, sort_keys=True),
        "patched_layer_count": len(assignments_by_layer),
        "observed_o_proj_layouts": json.dumps(layouts, sort_keys=True),
        "observed_o_proj_input_shapes": json.dumps(
            input_shapes, sort_keys=True
        ),
    }

    return result, intervention_metadata


def parse_percentages(text: str) -> List[float]:
    values = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        value = float(item)
        if value <= 0 or value > 100:
            raise ValueError(
                f"Each patch percentage must be in (0, 100], got {value}."
            )
        values.append(value)

    if not values:
        raise ValueError("No patch percentages were provided.")

    return sorted(set(values))


def load_head_ranking(
    ranking_csv: str,
    ranking_metric: str,
    num_layers: int,
    num_heads: int,
    target_present_filter: Optional[bool] = None,
) -> pd.DataFrame:
    """
    Load a global head ranking.

    Accepts either:
      1. a head-summary CSV with one row per (layer, head), or
      2. a raw per-trial/per-head CSV, which is averaged by (layer, head).

    For a raw CSV, --ranking_metric can be delta_margin or recovery.
    For a summary CSV, it will typically be mean_delta_margin.
    """
    ranking = pd.read_csv(ranking_csv)

    if target_present_filter is not None:
        if "target_present" not in ranking.columns:
            raise ValueError(
                "--rank_by_target_presence requires a raw per-trial "
                "head-patching CSV containing a target_present column. "
                "A pre-aggregated head-summary CSV cannot be split by label."
            )

        target_values = ranking["target_present"]
        if target_values.dtype == object:
            target_values = (
                target_values.astype(str).str.strip().str.lower().map(
                    {"true": True, "false": False, "1": True, "0": False}
                )
            )
        else:
            target_values = target_values.astype("boolean")

        ranking = ranking[target_values == target_present_filter].copy()
        if ranking.empty:
            label = "present" if target_present_filter else "absent"
            raise ValueError(
                f"Ranking CSV contains no target-{label} rows."
            )

    required = {"layer", "head", ranking_metric}
    missing = required - set(ranking.columns)
    if missing:
        raise ValueError(
            f"Ranking CSV is missing columns {sorted(missing)}. "
            f"Available columns: {list(ranking.columns)}"
        )

    ranking = ranking.copy()
    ranking["layer"] = pd.to_numeric(
        ranking["layer"], errors="coerce"
    )
    ranking["head"] = pd.to_numeric(
        ranking["head"], errors="coerce"
    )
    ranking[ranking_metric] = pd.to_numeric(
        ranking[ranking_metric], errors="coerce"
    )
    ranking = ranking.dropna(
        subset=["layer", "head", ranking_metric]
    )

    ranking["layer"] = ranking["layer"].astype(int)
    ranking["head"] = ranking["head"].astype(int)

    invalid = ranking[
        (ranking["layer"] < 0)
        | (ranking["layer"] >= num_layers)
        | (ranking["head"] < 0)
        | (ranking["head"] >= num_heads)
    ]
    if not invalid.empty:
        raise ValueError(
            "Ranking CSV contains out-of-range layer/head indices:\n"
            + invalid[["layer", "head"]].head(20).to_string(index=False)
        )

    # This also makes raw per-trial results valid ranking inputs.
    ranking = (
        ranking.groupby(["layer", "head"], as_index=False)[ranking_metric]
        .mean()
        .sort_values(
            [ranking_metric, "layer", "head"],
            ascending=[False, True, True],
        )
        .reset_index(drop=True)
    )

    # Ensure the 100% condition always contains every model head.
    present = set(zip(ranking["layer"], ranking["head"]))
    missing_rows = []
    for layer_index in range(num_layers):
        for head_index in range(num_heads):
            if (layer_index, head_index) not in present:
                missing_rows.append(
                    {
                        "layer": layer_index,
                        "head": head_index,
                        ranking_metric: -np.inf,
                    }
                )

    if missing_rows:
        ranking = pd.concat(
            [ranking, pd.DataFrame(missing_rows)],
            ignore_index=True,
        )
        ranking = ranking.sort_values(
            [ranking_metric, "layer", "head"],
            ascending=[False, True, True],
        ).reset_index(drop=True)

    ranking["rank"] = np.arange(1, len(ranking) + 1)
    return ranking


def select_top_heads(
    ranking: pd.DataFrame,
    percentage: float,
) -> List[Tuple[int, int]]:
    # ceil means 1% of 784 heads becomes 8 rather than 7.
    count = max(
        1,
        int(np.ceil(len(ranking) * percentage / 100.0)),
    )
    selected = ranking.head(count)
    return [
        (int(row.layer), int(row.head))
        for row in selected.itertuples(index=False)
    ]


def make_percentage_summary(
    dataframe: pd.DataFrame,
    split_by_target_presence: bool = False,
) -> pd.DataFrame:
    if dataframe.empty:
        return pd.DataFrame()

    successful = dataframe[dataframe["error"].isna()].copy()
    if successful.empty:
        return pd.DataFrame()

    def valid_mean(series: pd.Series) -> float:
        values = pd.to_numeric(series, errors="coerce").dropna()
        return float(values.mean()) if len(values) else np.nan

    group_columns = ["patch_percentage", "selected_head_count"]
    if split_by_target_presence:
        group_columns = [
            "target_present",
            "ranking_group",
            *group_columns,
        ]

    return (
        successful.groupby(group_columns, as_index=False)
        .agg(
            n_pairs=("trial", "size"),
            source_accuracy=("source_correct", "mean"),
            target_accuracy=("target_correct", "mean"),
            patched_accuracy=("patched_correct", "mean"),
            accuracy_gain=("patched_correct", lambda x: float(x.mean())),
            mean_source_margin=("source_margin", "mean"),
            mean_target_margin=("target_margin", "mean"),
            mean_patched_margin=("patched_margin", "mean"),
            mean_delta_margin=("delta_margin", "mean"),
            median_delta_margin=("delta_margin", "median"),
            mean_recovery=("recovery", valid_mean),
            median_recovery=("recovery", "median"),
            valid_recovery_n=(
                "recovery_valid",
                lambda values: int(values.sum()),
            ),
            flip_to_correct_rate=("patched_flip_to_correct", "mean"),
        )
        .assign(
            accuracy_gain=lambda df: (
                df["patched_accuracy"] - df["target_accuracy"]
            )
        )
        .sort_values(group_columns)
        .reset_index(drop=True)
    )


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
    if args.patch_config:
        patch_configs = [parse_patch_config(item) for item in args.patch_config]
    else:
        if args.token_offset > 0:
            raise ValueError("--token_offset must be <= 0.")
        if not args.head_ranking_csv:
            raise ValueError(
                "Provide --head_ranking_csv for a legacy single-position run, "
                "or provide one or more --patch_config entries."
            )
        patch_configs = [PatchConfig(
            token_offset=args.token_offset,
            source_dataset_dir=os.path.abspath(args.dataset_dir),
            head_ranking_csv=args.head_ranking_csv,
        )]

    offsets = [config.token_offset for config in patch_configs]
    if len(set(offsets)) != len(offsets):
        raise ValueError("Each --patch_config must use a distinct token offset.")

    output_directory = os.path.dirname(os.path.abspath(args.out_path))
    os.makedirs(output_directory, exist_ok=True)

    percentages = parse_percentages(args.patch_percentages)

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
    total_heads = num_layers * num_heads

    if args.always_use_target_present_ranking and not args.rank_by_target_presence:
        raise ValueError(
            "--always_use_target_present_ranking requires "
            "--rank_by_target_presence."
        )

    if args.rank_by_target_presence and args.random_heads:
        raise ValueError(
            "--rank_by_target_presence cannot be combined with --random_heads."
        )

    rankings_by_offset = {}
    for config in patch_configs:
        if args.rank_by_target_presence:
            present_ranking = load_head_ranking(
                config.head_ranking_csv, args.ranking_metric,
                num_layers, num_heads, True,
            )
            rankings_by_offset[config.token_offset] = {
                True: present_ranking,
                False: (
                    present_ranking
                    if args.always_use_target_present_ranking
                    else load_head_ranking(
                        config.head_ranking_csv, args.ranking_metric,
                        num_layers, num_heads, False,
                    )
                ),
            }
        else:
            ranking = load_head_ranking(config.head_ranking_csv, args.ranking_metric, num_layers, num_heads)
            rankings_by_offset[config.token_offset] = {True: ranking, False: ranking}

    all_heads = [
        (layer, head)
        for layer in range(num_layers)
        for head in range(num_heads)
    ]

    rng = random.Random(args.random_seed)

    selected_by_offset_label_percentage = {}
    for config in patch_configs:
        selected_by_label = {}
        for target_present_value in [True, False]:
            selected_by_percentage = {}
            for percentage in percentages:
                count = max(1, int(np.ceil(len(all_heads) * percentage / 100.0)))
                selected_by_percentage[percentage] = (
                    rng.sample(all_heads, count)
                    if args.random_heads
                    else select_top_heads(
                        rankings_by_offset[config.token_offset][target_present_value],
                        percentage,
                    )
                )
            selected_by_label[target_present_value] = selected_by_percentage
        selected_by_offset_label_percentage[config.token_offset] = selected_by_label

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
    print(f"  heads/layer:{num_heads}")
    print(f"  total heads:{total_heads}")
    print(f"  head dim:   {expected_head_dim}")

    ranking_mode = (
        "target-present ranking used for every trial"
        if args.always_use_target_present_ranking
        else
        "target-presence-specific"
        if args.rank_by_target_presence
        else "global"
    )
    print(f"\nHead ranking mode: {ranking_mode}")
    print("Cumulative patch conditions (independently per token position):")
    for config in patch_configs:
        print(f"  offset {config.token_offset}: source={config.source_dataset_dir}")
        for percentage in percentages:
            count = len(selected_by_offset_label_percentage[config.token_offset][True][percentage])
            print(f"    {percentage:g}% -> {count} heads ({100 * count / total_heads:.2f}% actual)")

    ranking_base = os.path.splitext(args.out_path)[0]
    ranking_out_paths = []
    for config in patch_configs:
        offset_tag = f"_offset_{config.token_offset}"
        if args.rank_by_target_presence:
            ranking_exports = [(True, "target_present")]
            if not args.always_use_target_present_ranking:
                ranking_exports.append((False, "target_absent"))
            for present, label in ranking_exports:
                path = ranking_base + offset_tag + f"_{label}_ranking_used.csv"
                rankings_by_offset[config.token_offset][present].to_csv(path, index=False)
                ranking_out_paths.append(path)
        else:
            path = ranking_base + offset_tag + "_ranking_used.csv"
            rankings_by_offset[config.token_offset][True].to_csv(path, index=False)
            ranking_out_paths.append(path)

    # Target images define the trial list. This also makes a mask directory a
    # valid target dataset, rather than assuming every dataset has conditions.
    source_paths = collect_source_paths(
        dataset_dir=args.dataset_dir,
        conditions=[args.target_condition],
        source_distractors=args.target_distractors,
        max_pairs=args.max_pairs,
    )

    print(
        f"\nFound {len(source_paths)} target candidates with "
        f"{args.target_distractors} distractors"
    )
    print(
        f"Patching into matched {args.target_distractors}-distractor examples"
    )
    print(f"Source cache dtype: {args.cache_dtype}")
    print(
        "Patch token offsets: " + ", ".join(str(offset) for offset in offsets)
    )

    rows: List[Dict[str, Any]] = []
    completed_forwards = 0
    printed_token_sanity_check = False

    pair_iterator = tqdm(source_paths, desc="Trial pairs")

    for target_image_path in pair_iterator:
        target_metadata = parse_filename(os.path.basename(target_image_path))
        source_condition = "multi_source"
        trial = target_metadata["trial"]
        target_present = target_metadata["target_present"]
        target_condition = args.target_condition

        try:
            source_runs = {}
            for config in patch_configs:
                source_image_path = corresponding_image_path(
                    dataset_dir=config.source_dataset_dir,
                    condition=args.source_condition,
                    num_distractors=args.source_distractors,
                    trial=trial,
                    target_present=target_present,
                )
                if not os.path.exists(source_image_path):
                    raise FileNotFoundError(
                        f"Missing source image for offset {config.token_offset}: {source_image_path}"
                    )
                with Image.open(source_image_path) as image_handle:
                    source_image = image_handle.convert("RGB")
                source_runs[config.token_offset] = (*run_source_with_attention_cache(
                    model=model, processor=processor, image=source_image,
                    target_present=target_present, tokenizer=tokenizer,
                    binary_tokens=binary_tokens, decoder_stack=decoder_stack,
                    num_heads=num_heads, expected_head_dim=expected_head_dim,
                    cache_dtype=args.cache_dtype, token_offset=config.token_offset,
                ), source_image_path)

            # Recovery retains the legacy definition, using the first config's
            # source behavior as its reference; every source is recorded below.
            primary_result, _, primary_metadata, _ = source_runs[offsets[0]]
            source_result = primary_result
            source_cache_metadata = primary_metadata
            source_image_path = source_runs[offsets[0]][3]
            source_margin = float(primary_result["correct_logit_margin"])
            source_correct = bool(primary_result["correct"])

            if args.require_source_correct and not source_correct:
                continue

            with Image.open(target_image_path) as image_handle:
                target_image = image_handle.convert("RGB")

            (
                target_result,
                target_input_length,
                target_token_metadata,
            ) = run_unpatched_forward(
                model=model,
                processor=processor,
                image=target_image,
                target_present=target_present,
                tokenizer=tokenizer,
                binary_tokens=binary_tokens,
                token_offset=offsets[0],
            )

            if not printed_token_sanity_check:
                for config in patch_configs:
                    _, _, metadata, _ = source_runs[config.token_offset]
                    print_selected_token(label=f"SOURCE offset {config.token_offset}", token_metadata={
                        "token_offset": metadata["token_offset"], "last_position": metadata["last_input_position"],
                        "patch_position": metadata["patch_position"], "patch_token_id": metadata["patch_token_id"],
                        "patch_token": metadata["patch_token"],
                    })
                print_selected_token(
                    label="TARGET",
                    token_metadata=target_token_metadata,
                )
                printed_token_sanity_check = True

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
                target_image_path
            )
            source_position_features = compute_target_position_features(
                source_position_metadata["target_x"],
                source_position_metadata["target_y"],
            )

            for percentage in percentages:
                selected_by_offset = {
                    config.token_offset: selected_by_offset_label_percentage[
                        config.token_offset
                    ][bool(target_present)][percentage]
                    for config in patch_configs
                }
                patch_sources = [
                    (config.token_offset, selected_by_offset[config.token_offset],
                     source_runs[config.token_offset][1])
                    for config in patch_configs
                ]
                selected_heads_json = json.dumps({
                    str(offset): heads for offset, heads in selected_by_offset.items()
                }, sort_keys=True)
                source_paths_json = json.dumps({
                    str(offset): source_runs[offset][3] for offset in offsets
                }, sort_keys=True)
                config_json = json.dumps([
                    {"token_offset": config.token_offset,
                     "source_dataset_dir": config.source_dataset_dir,
                     "head_ranking_csv": config.head_ranking_csv}
                    for config in patch_configs
                ], sort_keys=True)
                total_selected_heads = sum(len(heads) for heads in selected_by_offset.values())
                ranking_group = (
                    "target_present"
                    if (
                        args.always_use_target_present_ranking
                        or (args.rank_by_target_presence and target_present)
                    )
                    else "target_absent"
                    if args.rank_by_target_presence
                    else "global"
                )

                try:
                    (
                        patched_result,
                        intervention_metadata,
                    ) = run_multi_head_patch(
                        model=model,
                        processor=processor,
                        image=target_image,
                        target_present=target_present,
                        tokenizer=tokenizer,
                        binary_tokens=binary_tokens,
                        decoder_stack=decoder_stack,
                        patch_sources=patch_sources,
                        num_heads=num_heads,
                        expected_head_dim=expected_head_dim,
                        token_offsets=offsets,
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
                        "patch_percentage": percentage,
                        "selected_head_count": total_selected_heads,
                        "actual_percentage": (
                            100.0 * total_selected_heads / total_heads
                        ),
                        "selected_heads": selected_heads_json,
                        "ranking_metric": args.ranking_metric,
                        "head_ranking_csv": config_json,
                        "source_image_paths_by_offset": source_paths_json,
                        "patch_configs": config_json,
                        "ranking_mode": ranking_mode,
                        "ranking_group": ranking_group,
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
                        "token_offset": offsets[0],
                        "token_offsets": json.dumps(offsets),
                        "source_patch_position": source_cache_metadata[
                            "patch_position"
                        ],
                        "source_patch_token_id": source_cache_metadata[
                            "patch_token_id"
                        ],
                        "source_patch_token": source_cache_metadata[
                            "patch_token"
                        ],
                        "source_layer_cache_shape": str(
                            source_cache_metadata["layer_cache_shape"]
                        ),
                        "source_cache_dtype": args.cache_dtype,
                        "target_input_text_length": target_input_length,
                        "target_last_position": target_token_metadata[
                            "last_position"
                        ],
                        "target_patch_positions": intervention_metadata["target_patch_positions"],
                        "target_patch_tokens": intervention_metadata["patch_tokens"],
                        "patched_layer_count": intervention_metadata[
                            "patched_layer_count"
                        ],
                        "observed_o_proj_input_shapes": (
                            intervention_metadata[
                                "observed_o_proj_input_shapes"
                            ]
                        ),
                        "observed_o_proj_layouts": intervention_metadata[
                            "observed_o_proj_layouts"
                        ],
                        "decoder_stack_path": decoder_stack.path,
                        "num_layers": num_layers,
                        "num_heads": num_heads,
                        "total_heads": total_heads,
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
                        "patch_percentage": percentage,
                        "selected_head_count": total_selected_heads,
                        "actual_percentage": (
                            100.0 * total_selected_heads / total_heads
                        ),
                        "selected_heads": selected_heads_json,
                        "ranking_metric": args.ranking_metric,
                        "head_ranking_csv": config_json,
                        "source_image_paths_by_offset": source_paths_json,
                        "patch_configs": config_json,
                        "ranking_mode": ranking_mode,
                        "ranking_group": ranking_group,
                        "token_offset": offsets[0],
                        "token_offsets": json.dumps(offsets),
                        "source_correct": source_correct,
                        "source_margin": source_margin,
                        "target_correct": target_result["correct"],
                        "target_margin": target_margin,
                        "patched_correct": False,
                        "patched_margin": None,
                        "margin_drop": margin_drop,
                        "delta_margin": None,
                        "recovery": None,
                        "recovery_valid": False,
                        "patched_flip_to_correct": False,
                        "num_layers": num_layers,
                        "num_heads": num_heads,
                        "total_heads": total_heads,
                        "head_dim": expected_head_dim,
                        "error": repr(patch_error),
                    }

                rows.append(row)
                completed_forwards += 1

                if (
                    args.save_every > 0
                    and completed_forwards % args.save_every == 0
                ):
                    save_progress(rows=rows, out_path=args.out_path)

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

    summary = make_percentage_summary(
        results,
        split_by_target_presence=args.rank_by_target_presence,
    )
    summary_path = (
        os.path.splitext(args.out_path)[0] + "_percentage_summary.csv"
    )
    summary.to_csv(summary_path, index=False)

    print("\nSaved cumulative multi-head patch results:")
    print(args.out_path)

    print("\nSaved percentage summary:")
    print(summary_path)

    print("\nSaved exact ranking(s) used:")
    for ranking_out_path in ranking_out_paths:
        print(ranking_out_path)

    if not summary.empty:
        print("\nRecovery by cumulative head percentage:")
        print(summary.to_string(index=False))

    if not results.empty:
        print("\nPatch errors:")
        print(results["error"].notna().sum())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Patch cumulative top-ranked attention-head subsets from matched "
            "source images into target images at a selectable input token."
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
        "--head_ranking_csv",
        type=str,
        default=None,
        help=(
            "Prior per-head result or head-summary CSV used to rank heads "
            "globally."
        ),
    )
    parser.add_argument(
        "--ranking_metric",
        type=str,
        default="mean_delta_margin",
        help=(
            "Column used for descending head ranking. Use mean_delta_margin "
            "for a head-summary CSV or delta_margin for the raw result CSV."
        ),
    )
    parser.add_argument(
        "--rank_by_target_presence",
        action="store_true",
        help=(
            "Build separate rankings from target-present and target-absent "
            "rows in the raw per-trial head-patching CSV, then use the "
            "matching ranking for each cumulative-patching example. When "
            "enabled, --head_ranking_csv must contain target_present and "
            "--ranking_metric should usually be delta_margin."
        ),
    )
    parser.add_argument(
        "--always_use_target_present_ranking",
        action="store_true",
        help=(
            "Use only target_present=True rows to rank heads for every "
            "trial, including target-absent trials. Requires "
            "--rank_by_target_presence."
        ),
    )
    parser.add_argument(
        "--patch_percentages",
        type=str,
        default="1,3,5,10,20,50,100",
        help="Comma-separated cumulative percentages of ranked heads.",
    )
    parser.add_argument(
        "--source_condition",
        choices=["disjunctive", "conjunctive"],
        default="disjunctive",
    )
    parser.add_argument(
        "--target_condition",
        choices=["disjunctive", "conjunctive"],
        default="conjunctive",
    )
    parser.add_argument(
        "--source_distractors",
        type=int,
        default=40,
    )
    parser.add_argument(
        "--target_distractors",
        type=int,
        default=40,
    )
    parser.add_argument(
        "--token_offset",
        type=int,
        default=0,
        help=(
            "Patch position relative to the final active input token. "
            "0=final token, -1=second-to-last, -2=third-to-last, etc."
        ),
    )
    parser.add_argument(
        "--patch_config",
        action="append",
        default=[],
        metavar="OFFSET|SOURCE_DATASET_DIR|HEAD_RANKING_CSV",
        help=(
            "Repeat once per patched token position. Each position receives "
            "its own source dataset and independently selected top-N%% head set. "
            "When supplied, this replaces --token_offset and --head_ranking_csv."
        ),
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
        default=25,
        help="Save partial results every N patched forward passes.",
    )

    parser.add_argument(
        "--random_heads",
        action="store_true",
        help="Use randomly selected heads instead of the ranked heads.",
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
    )

    main(parser.parse_args())
