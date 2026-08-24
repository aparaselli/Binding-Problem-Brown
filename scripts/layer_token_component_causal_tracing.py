#!/usr/bin/env python3
"""
Layer x token causal tracing for matched VLM visual-search examples.

For every matched source/target pair (normally disjunctive -> conjunctive),
this script:

  1. Runs a clean source pass and caches, at every decoder layer:
       - the decoder-layer input residual
       - the full self-attention sublayer output (before residual addition)
       - the MLP sublayer output (before residual addition)
  2. Runs a clean target pass to obtain the target baseline.
  3. For each component, token offset, and centered layer window, runs the
     target example while replacing the selected target activation with the
     corresponding CLEAN SOURCE activation.
  4. Lets every unpatched activation and all downstream computation evolve
     naturally.
  5. Measures ground-truth-aligned Correct Margin Metric (CMM), delta CMM,
     and normalized recovery.

For component=attention or component=mlp, only the chosen sublayer output at
the chosen token is replaced. For component=residual, the chosen decoder-layer
input residual is replaced. A window of size 3 centered at layer 10 patches
layers 9, 10, and 11. Windows are clipped at model boundaries.

Example:

python layer_token_component_causal_tracing.py \
    --source_condition disjunctive \
    --target_condition conjunctive \
    --source_distractors 40 \
    --target_distractors 40 \
    --components attention,mlp \
    --token_offsets=-19:0 \
    --layers all \
    --window_size 3 \
    --pair_filter all \
    --out_path ../Results/disj40_to_conj40_layer_token_components_w3.csv

This script expects to live beside search_utils.py.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

from search_utils import build_inputs, load_vlm


# ============================================================
# PATHS / PROMPT
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_DATASET_DIR = (
    SCRIPT_DIR.parent
    / "search-dataset"
    / "visual_search_dataset_counterfactual"
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


# ============================================================
# DATASET / PAIRING
# ============================================================

# NOTE: "disjunctive_T" must be listed before "disjunctive" so the
# alternation prefers the longer match; with re.fullmatch this isn't
# strictly required (the trailing "_distractors_..." literal disambiguates
# them either way), but keeping the longer alternative first is safer.
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


def collect_condition_images(
    dataset_dir: Path,
    condition: str,
    distractors: int,
) -> Dict[Tuple[int, bool], Path]:
    condition_dir = dataset_dir / condition
    if not condition_dir.is_dir():
        raise FileNotFoundError(
            f"Condition directory does not exist: {condition_dir}"
        )

    found: Dict[Tuple[int, bool], Path] = {}
    for path in sorted(condition_dir.glob("*.png")):
        try:
            metadata = parse_filename(path.name)
        except ValueError:
            continue

        if metadata["condition"] != condition:
            continue
        if metadata["num_distractors"] != distractors:
            continue

        key = (metadata["trial"], metadata["target_present"])
        if key in found:
            raise RuntimeError(f"Duplicate matched-pair key {key}: {path}")
        found[key] = path

    return found


def make_matched_pairs(
    dataset_dir: Path,
    source_condition: str,
    target_condition: str,
    source_distractors: int,
    target_distractors: int,
    target_present_only: bool,
) -> List[Dict[str, Any]]:
    source = collect_condition_images(
        dataset_dir, source_condition, source_distractors
    )
    target = collect_condition_images(
        dataset_dir, target_condition, target_distractors
    )

    shared = sorted(set(source) & set(target))
    pairs: List[Dict[str, Any]] = []
    for trial, target_present in shared:
        if target_present_only and not target_present:
            continue
        pairs.append(
            {
                "trial": trial,
                "target_present": target_present,
                "source_path": source[(trial, target_present)],
                "target_path": target[(trial, target_present)],
            }
        )
    return pairs


# ============================================================
# ARGUMENT PARSING
# ============================================================

def parse_index_spec(
    spec: str,
    *,
    minimum: int,
    maximum: int,
    name: str,
) -> List[int]:
    """Parse 'all', '-19:0', '0-27', or comma-separated integers/ranges."""
    text = str(spec).strip().lower()
    if text == "all":
        return list(range(minimum, maximum + 1))

    selected: List[int] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue

        if ":" in item:
            start_text, end_text = item.split(":", maxsplit=1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"Descending {name} range is invalid: {item}")
            selected.extend(range(start, end + 1))
        elif re.fullmatch(r"\d+-\d+", item):
            start_text, end_text = item.split("-", maxsplit=1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"Descending {name} range is invalid: {item}")
            selected.extend(range(start, end + 1))
        else:
            selected.append(int(item))

    selected = sorted(set(selected))
    if not selected:
        raise ValueError(f"{name} did not select anything.")

    invalid = [value for value in selected if not minimum <= value <= maximum]
    if invalid:
        raise ValueError(
            f"Invalid {name} values {invalid}; allowed {minimum}..{maximum}."
        )
    return selected


def parse_choice_list(
    spec: str,
    *,
    allowed: Sequence[str],
    name: str,
) -> List[str]:
    values = [v.strip().lower() for v in str(spec).split(",") if v.strip()]
    invalid = sorted(set(values) - set(allowed))
    if invalid:
        raise ValueError(f"Invalid {name} values {invalid}; allowed={allowed}")
    if not values:
        raise ValueError(f"{name} did not select anything.")
    return list(dict.fromkeys(values))


def centered_window(center: int, width: int, n_layers: int) -> List[int]:
    if width <= 0 or width % 2 == 0:
        raise ValueError("--window_size must be a positive odd integer.")
    radius = width // 2
    return list(range(max(0, center - radius), min(n_layers, center + radius + 1)))


# ============================================================
# MODEL / TOKEN HELPERS
# ============================================================

def get_model_device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


def get_tokenizer(processor: Any) -> Any:
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise AttributeError("processor.tokenizer is unavailable.")
    return tokenizer


def get_last_input_position(inputs: Mapping[str, torch.Tensor]) -> int:
    input_ids = inputs["input_ids"]
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError(f"Expected input_ids [1,T], got {tuple(input_ids.shape)}")

    mask = inputs.get("attention_mask")
    if mask is None:
        return int(input_ids.shape[-1] - 1)

    active = (mask[0] != 0).nonzero(as_tuple=False).flatten()
    if active.numel() == 0:
        raise ValueError("attention_mask contains no active positions.")
    return int(active[-1].item())


def resolve_token_positions(
    inputs: Mapping[str, torch.Tensor],
    offsets: Sequence[int],
) -> Dict[int, int]:
    last = get_last_input_position(inputs)
    output: Dict[int, int] = {}
    for offset in offsets:
        if offset > 0:
            raise ValueError("Token offsets must be <= 0.")
        position = last + offset
        if position < 0:
            raise IndexError(
                f"Offset {offset} gives position {position} for last position {last}."
            )
        output[int(offset)] = int(position)
    return output


def decode_token(tokenizer: Any, token_id: int) -> str:
    return tokenizer.decode(
        [int(token_id)],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def print_selected_tokens(
    inputs: Mapping[str, torch.Tensor],
    tokenizer: Any,
    positions: Mapping[int, int],
) -> None:
    print("\n" + "=" * 80)
    print("SELECTED TOKEN OFFSETS")
    print("=" * 80)
    for offset, position in positions.items():
        token_id = int(inputs["input_ids"][0, position].item())
        token = decode_token(tokenizer, token_id)
        print(
            f"offset={offset:4d} | position={position:4d} | "
            f"id={token_id:6d} | token={token!r}"
        )


def resolve_boolean_token_id(processor: Any, text: str) -> int:
    tokenizer = get_tokenizer(processor)
    candidates = [text, " " + text]
    resolved: List[Tuple[str, List[int]]] = []
    for candidate in candidates:
        ids = tokenizer.encode(candidate, add_special_tokens=False)
        resolved.append((candidate, [int(v) for v in ids]))
        if len(ids) == 1:
            return int(ids[0])
    raise RuntimeError(f"{text!r} is not a single token: {resolved}")


def discover_decoder_layers(model: nn.Module) -> Tuple[Sequence[nn.Module], str]:
    candidates = (
        "model.language_model.layers",
        "language_model.model.layers",
        "model.layers",
        "language_model.layers",
    )

    for path in candidates:
        owner: Any = model
        try:
            for part in path.split("."):
                owner = getattr(owner, part)
        except AttributeError:
            continue
        if isinstance(owner, (nn.ModuleList, list, tuple)) and len(owner) > 0:
            return owner, path

    discovered = []
    for name, module in model.named_modules():
        layers = getattr(module, "layers", None)
        if isinstance(layers, nn.ModuleList) and len(layers) > 0:
            discovered.append((name + ".layers", layers))

    if len(discovered) == 1:
        return discovered[0][1], discovered[0][0]
    raise RuntimeError(
        "Could not uniquely discover decoder layers. Candidates: "
        + str([name for name, _ in discovered])
    )


def resolve_sublayers(layer: nn.Module, layer_index: int) -> Tuple[nn.Module, nn.Module]:
    attention = None
    for name in ("self_attn", "attention", "attn"):
        attention = getattr(layer, name, None)
        if isinstance(attention, nn.Module):
            break

    mlp = None
    for name in ("mlp", "feed_forward", "ffn"):
        mlp = getattr(layer, name, None)
        if isinstance(mlp, nn.Module):
            break

    if attention is None or mlp is None:
        raise RuntimeError(
            f"Layer {layer_index} lacks recognizable attention/MLP sublayers."
        )
    return attention, mlp


def validate_layout(
    source_inputs: Mapping[str, torch.Tensor],
    target_inputs: Mapping[str, torch.Tensor],
    require_aligned_input_ids: bool,
) -> None:
    source_ids = source_inputs["input_ids"]
    target_ids = target_inputs["input_ids"]
    if source_ids.shape != target_ids.shape:
        raise ValueError(
            f"Source/target input shapes differ: {tuple(source_ids.shape)} vs "
            f"{tuple(target_ids.shape)}."
        )
    if require_aligned_input_ids and not torch.equal(source_ids, target_ids):
        mismatch = (source_ids != target_ids).nonzero(as_tuple=False)
        first = mismatch[0].tolist() if mismatch.numel() else None
        raise ValueError(
            "Source and target input IDs are not position-aligned. "
            f"First mismatch: {first}."
        )


# ============================================================
# HOOK HELPERS
# ============================================================

def first_tensor(value: Any) -> torch.Tensor:
    if torch.is_tensor(value):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            if torch.is_tensor(item):
                return item
    raise TypeError(f"Could not find tensor in value of type {type(value)}")


def replace_first_tensor(value: Any, replacement: torch.Tensor) -> Any:
    if torch.is_tensor(value):
        return replacement
    if isinstance(value, tuple):
        items = list(value)
        for index, item in enumerate(items):
            if torch.is_tensor(item):
                items[index] = replacement
                return tuple(items)
    if isinstance(value, list):
        items = list(value)
        for index, item in enumerate(items):
            if torch.is_tensor(item):
                items[index] = replacement
                return items
    raise TypeError(f"Could not replace tensor in value of type {type(value)}")


def score_logits(
    logits: torch.Tensor,
    *,
    last_position: int,
    true_id: int,
    false_id: int,
    target_present: bool,
) -> Dict[str, Any]:
    row = logits[0, last_position].float()
    true_logit = float(row[true_id].item())
    false_logit = float(row[false_id].item())
    raw_margin = true_logit - false_logit
    cmm = raw_margin if target_present else -raw_margin
    prediction = bool(raw_margin > 0)
    return {
        "prediction": prediction,
        "correct": bool(prediction == target_present),
        "true_logit": true_logit,
        "false_logit": false_logit,
        "raw_margin": raw_margin,
        "cmm": cmm,
    }


@torch.no_grad()
def capture_clean_activations(
    *,
    model: nn.Module,
    inputs: Mapping[str, torch.Tensor],
    layers: Sequence[nn.Module],
    sublayers: Sequence[Tuple[nn.Module, nn.Module]],
    components: Sequence[str],
    cache_device: str,
    true_id: int,
    false_id: int,
    target_present: bool,
) -> Tuple[Dict[str, Dict[int, torch.Tensor]], Dict[str, Any]]:
    cache: Dict[str, Dict[int, torch.Tensor]] = {
        component: {} for component in components
    }
    handles = []

    def cache_tensor(component: str, layer_index: int, tensor: torch.Tensor) -> None:
        if tensor.ndim != 3 or tensor.shape[0] != 1:
            raise RuntimeError(
                f"{component} L{layer_index}: expected [1,T,D], got {tuple(tensor.shape)}"
            )
        cache[component][layer_index] = tensor.detach().to(cache_device).clone()

    if "residual" in components:
        for layer_index, layer in enumerate(layers):
            def residual_hook(module, module_inputs, layer_index=layer_index):
                cache_tensor("residual", layer_index, first_tensor(module_inputs))
            handles.append(layer.register_forward_pre_hook(residual_hook))

    if "attention" in components:
        for layer_index, (attention, _) in enumerate(sublayers):
            def attention_hook(module, module_inputs, output, layer_index=layer_index):
                cache_tensor("attention", layer_index, first_tensor(output))
            handles.append(attention.register_forward_hook(attention_hook))

    if "mlp" in components:
        for layer_index, (_, mlp) in enumerate(sublayers):
            def mlp_hook(module, module_inputs, output, layer_index=layer_index):
                cache_tensor("mlp", layer_index, first_tensor(output))
            handles.append(mlp.register_forward_hook(mlp_hook))

    try:
        outputs = model(**inputs, use_cache=False, return_dict=True)
        baseline = score_logits(
            outputs.logits,
            last_position=get_last_input_position(inputs),
            true_id=true_id,
            false_id=false_id,
            target_present=target_present,
        )
    finally:
        for handle in handles:
            handle.remove()

    expected = set(range(len(layers)))
    for component in components:
        missing = expected - set(cache[component])
        if missing:
            raise RuntimeError(
                f"Failed to cache {component} at layers {sorted(missing)}"
            )
    return cache, baseline


@torch.no_grad()
def run_component_patch(
    *,
    model: nn.Module,
    inputs: Mapping[str, torch.Tensor],
    layers: Sequence[nn.Module],
    sublayers: Sequence[Tuple[nn.Module, nn.Module]],
    source_cache: Mapping[str, Mapping[int, torch.Tensor]],
    component: str,
    patch_layers: Sequence[int],
    token_position: int,
    true_id: int,
    false_id: int,
    target_present: bool,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    handles = []
    changes: Dict[int, float] = {}

    def patch_tensor(tensor: torch.Tensor, layer_index: int) -> torch.Tensor:
        if tensor.ndim != 3 or tensor.shape[0] != 1:
            raise RuntimeError(
                f"{component} L{layer_index}: expected [1,T,D], got {tuple(tensor.shape)}"
            )
        if not 0 <= token_position < tensor.shape[1]:
            raise IndexError(
                f"Token position {token_position} outside length {tensor.shape[1]}"
            )

        source_value = source_cache[component][layer_index][:, token_position, :]
        source_value = source_value.to(device=tensor.device, dtype=tensor.dtype)
        before = tensor[:, token_position, :]
        replacement = tensor.clone()
        replacement[:, token_position, :] = source_value
        changes[layer_index] = float(
            (source_value.float() - before.float()).norm(dim=-1).mean().item()
        )
        return replacement

    for layer_index in patch_layers:
        if component == "residual":
            layer = layers[layer_index]

            def residual_patch(module, module_inputs, layer_index=layer_index):
                hidden = first_tensor(module_inputs)
                replacement = patch_tensor(hidden, layer_index)
                return (replacement,) + tuple(module_inputs[1:])

            handles.append(layer.register_forward_pre_hook(residual_patch))

        elif component in {"attention", "mlp"}:
            module = (
                sublayers[layer_index][0]
                if component == "attention"
                else sublayers[layer_index][1]
            )

            def output_patch(module, module_inputs, output, layer_index=layer_index):
                tensor = first_tensor(output)
                replacement = patch_tensor(tensor, layer_index)
                return replace_first_tensor(output, replacement)

            handles.append(module.register_forward_hook(output_patch))
        else:
            raise ValueError(f"Unknown component: {component}")

    try:
        outputs = model(**inputs, use_cache=False, return_dict=True)
        result = score_logits(
            outputs.logits,
            last_position=get_last_input_position(inputs),
            true_id=true_id,
            false_id=false_id,
            target_present=target_present,
        )
    finally:
        for handle in handles:
            handle.remove()

    debug = {
        "mean_patch_l2_change": float(np.mean(list(changes.values()))),
        "patch_l2_by_layer": changes,
    }
    return result, debug


# ============================================================
# FILTERS / RESULTS
# ============================================================

def pair_passes_filter(
    pair_filter: str,
    source_correct: bool,
    target_correct: bool,
) -> bool:
    if pair_filter == "all":
        return True
    if pair_filter == "source_correct":
        return source_correct
    if pair_filter == "source_correct_target_incorrect":
        return source_correct and not target_correct
    if pair_filter == "both_correct":
        return source_correct and target_correct
    raise ValueError(f"Unknown pair filter: {pair_filter}")


def make_result_row(
    *,
    pair: Mapping[str, Any],
    source_baseline: Mapping[str, Any],
    target_baseline: Mapping[str, Any],
    patched: Mapping[str, Any],
    debug: Mapping[str, Any],
    component: str,
    center_layer: int,
    patch_layers: Sequence[int],
    requested_window_size: int,
    token_offset: int,
    token_position: int,
    token_id: int,
    token_string: str,
) -> Dict[str, Any]:
    delta_cmm = patched["cmm"] - target_baseline["cmm"]
    source_target_gap = source_baseline["cmm"] - target_baseline["cmm"]
    recovery = (
        delta_cmm / source_target_gap
        if source_target_gap != 0
        else float("nan")
    )

    return {
        "trial": int(pair["trial"]),
        "target_present": bool(pair["target_present"]),
        "source_path": str(pair["source_path"]),
        "target_path": str(pair["target_path"]),
        "component": component,
        "center_layer": int(center_layer),
        "patch_layers": ",".join(str(v) for v in patch_layers),
        "requested_window_size": int(requested_window_size),
        "effective_window_size": int(len(patch_layers)),
        "token_offset": int(token_offset),
        "token_position": int(token_position),
        "token_id": int(token_id),
        "token": token_string,
        "source_prediction": source_baseline["prediction"],
        "source_correct": source_baseline["correct"],
        "source_cmm": source_baseline["cmm"],
        "target_prediction": target_baseline["prediction"],
        "target_correct": target_baseline["correct"],
        "target_cmm": target_baseline["cmm"],
        "patched_prediction": patched["prediction"],
        "patched_correct": patched["correct"],
        "patched_cmm": patched["cmm"],
        "delta_cmm": delta_cmm,
        "source_target_cmm_gap": source_target_gap,
        "recovery": recovery,
        "mean_patch_l2_change": debug["mean_patch_l2_change"],
    }


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    safe = df.copy()
    safe["recovery"] = safe["recovery"].replace([np.inf, -np.inf], np.nan)

    return (
        safe.groupby(
            [
                "component",
                "center_layer",
                "patch_layers",
                "requested_window_size",
                "effective_window_size",
                "token_offset",
                "token_position",
                "token_id",
                "token",
            ],
            as_index=False,
            dropna=False,
        )
        .agg(
            n=("trial", "size"),
            source_accuracy=("source_correct", "mean"),
            target_accuracy=("target_correct", "mean"),
            patched_accuracy=("patched_correct", "mean"),
            mean_source_cmm=("source_cmm", "mean"),
            mean_target_cmm=("target_cmm", "mean"),
            mean_patched_cmm=("patched_cmm", "mean"),
            mean_delta_cmm=("delta_cmm", "mean"),
            median_delta_cmm=("delta_cmm", "median"),
            mean_recovery=("recovery", "mean"),
            median_recovery=("recovery", "median"),
            mean_patch_l2_change=("mean_patch_l2_change", "mean"),
        )
        .sort_values(["component", "token_offset", "center_layer"])
        .reset_index(drop=True)
    )


def save_partial(rows: Sequence[Mapping[str, Any]], out_path: Path) -> None:
    if not rows:
        return
    partial = pd.DataFrame(rows)
    partial.to_csv(
        out_path.with_name(out_path.stem + "_partial.csv"), index=False
    )
    summarize_results(partial).to_csv(
        out_path.with_name(out_path.stem + "_summary_partial.csv"), index=False
    )


# ============================================================
# MAIN
# ============================================================

def main(args: argparse.Namespace) -> None:
    torch.set_grad_enabled(False)

    if args.window_size <= 0 or args.window_size % 2 == 0:
        raise ValueError("--window_size must be a positive odd integer.")

    out_path = Path(args.out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model, processor = load_vlm(args.model_name)
    model.eval()
    tokenizer = get_tokenizer(processor)

    layers, layer_path = discover_decoder_layers(model)
    n_layers = len(layers)
    sublayers = [resolve_sublayers(layer, i) for i, layer in enumerate(layers)]

    components = parse_choice_list(
        args.components,
        allowed=("residual", "attention", "mlp"),
        name="--components",
    )
    center_layers = parse_index_spec(
        args.layers,
        minimum=0,
        maximum=n_layers - 1,
        name="--layers",
    )
    token_offsets = parse_index_spec(
        args.token_offsets,
        minimum=-100000,
        maximum=0,
        name="--token_offsets",
    )

    true_id = (
        args.true_token_id
        if args.true_token_id is not None
        else resolve_boolean_token_id(processor, "True")
    )
    false_id = (
        args.false_token_id
        if args.false_token_id is not None
        else resolve_boolean_token_id(processor, "False")
    )

    print(f"Decoder stack: {layer_path} ({n_layers} layers)")
    print(f"Components: {components}")
    print(f"Center layers: {center_layers}")
    print(f"Window size: {args.window_size} (centered, boundary-clipped)")
    print(f"Token offsets: {token_offsets}")
    print(f"True token ID: {true_id}; False token ID: {false_id}")

    pairs = make_matched_pairs(
        dataset_dir=Path(args.dataset_dir),
        source_condition=args.source_condition,
        target_condition=args.target_condition,
        source_distractors=args.source_distractors,
        target_distractors=args.target_distractors,
        target_present_only=args.target_present_only,
    )
    if args.max_pairs is not None:
        pairs = pairs[: args.max_pairs]
    if not pairs:
        raise RuntimeError("No matched source/target pairs were found.")

    print(f"Matched candidate pairs: {len(pairs)}")
    rows: List[Dict[str, Any]] = []
    kept_pairs = 0
    printed_tokens = False

    for pair in tqdm(pairs, desc="Layer x token causal tracing"):
        with Image.open(pair["source_path"]) as handle:
            source_image = handle.convert("RGB")
        with Image.open(pair["target_path"]) as handle:
            target_image = handle.convert("RGB")

        source_inputs = build_inputs(
            processor=processor,
            model_device=get_model_device(model),
            image=source_image,
            prompt=BINARY_PROMPT,
        )
        target_inputs = build_inputs(
            processor=processor,
            model_device=get_model_device(model),
            image=target_image,
            prompt=BINARY_PROMPT,
        )

        validate_layout(
            source_inputs,
            target_inputs,
            require_aligned_input_ids=args.require_aligned_input_ids,
        )
        source_positions = resolve_token_positions(source_inputs, token_offsets)
        target_positions = resolve_token_positions(target_inputs, token_offsets)
        if source_positions != target_positions:
            raise ValueError("Source/target token-offset positions differ.")

        if not printed_tokens:
            print_selected_tokens(target_inputs, tokenizer, target_positions)
            printed_tokens = True

        source_cache, source_baseline = capture_clean_activations(
            model=model,
            inputs=source_inputs,
            layers=layers,
            sublayers=sublayers,
            components=components,
            cache_device=args.cache_device,
            true_id=true_id,
            false_id=false_id,
            target_present=bool(pair["target_present"]),
        )

        # No target cache is needed: each intervention starts from a fresh clean
        # target forward and only the selected source activation is inserted.
        with torch.no_grad():
            target_outputs = model(
                **target_inputs, use_cache=False, return_dict=True
            )
        target_baseline = score_logits(
            target_outputs.logits,
            last_position=get_last_input_position(target_inputs),
            true_id=true_id,
            false_id=false_id,
            target_present=bool(pair["target_present"]),
        )
        del target_outputs

        if not pair_passes_filter(
            args.pair_filter,
            source_baseline["correct"],
            target_baseline["correct"],
        ):
            del source_cache, source_inputs, target_inputs, source_image, target_image
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        kept_pairs += 1
        if kept_pairs == 1 or args.verbose:
            print("\nKept pair:")
            print(
                f"  trial={pair['trial']} target_present={pair['target_present']}"
            )
            print(
                f"  source CMM={source_baseline['cmm']:+.4f} "
                f"correct={source_baseline['correct']}"
            )
            print(
                f"  target CMM={target_baseline['cmm']:+.4f} "
                f"correct={target_baseline['correct']}"
            )

        for component in components:
            for token_offset in token_offsets:
                token_position = target_positions[token_offset]
                token_id = int(target_inputs["input_ids"][0, token_position].item())
                token_string = decode_token(tokenizer, token_id)

                for center_layer in center_layers:
                    patch_layers = centered_window(
                        center_layer, args.window_size, n_layers
                    )
                    patched, debug = run_component_patch(
                        model=model,
                        inputs=target_inputs,
                        layers=layers,
                        sublayers=sublayers,
                        source_cache=source_cache,
                        component=component,
                        patch_layers=patch_layers,
                        token_position=token_position,
                        true_id=true_id,
                        false_id=false_id,
                        target_present=bool(pair["target_present"]),
                    )
                    row = make_result_row(
                        pair=pair,
                        source_baseline=source_baseline,
                        target_baseline=target_baseline,
                        patched=patched,
                        debug=debug,
                        component=component,
                        center_layer=center_layer,
                        patch_layers=patch_layers,
                        requested_window_size=args.window_size,
                        token_offset=token_offset,
                        token_position=token_position,
                        token_id=token_id,
                        token_string=token_string,
                    )
                    rows.append(row)

                    if args.verbose:
                        print(
                            f"  {component:9s} offset={token_offset:3d} "
                            f"center=L{center_layer:02d} layers={patch_layers} "
                            f"delta_CMM={row['delta_cmm']:+.4f} "
                            f"recovery={row['recovery']:+.4f}"
                        )

        if kept_pairs == 1 or kept_pairs % args.save_every == 0:
            save_partial(rows, out_path)

        del source_cache, source_inputs, target_inputs, source_image, target_image
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not rows:
        raise RuntimeError("No examples passed the requested pair filter.")

    detail_df = pd.DataFrame(rows)
    detail_df.to_csv(out_path, index=False)

    summary_df = summarize_results(detail_df)
    summary_path = out_path.with_name(out_path.stem + "_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    config_path = out_path.with_name(out_path.stem + "_config.json")
    with config_path.open("w") as handle:
        json.dump(vars(args), handle, indent=2)

    print("\nFinished.")
    print(f"Candidate pairs: {len(pairs)}")
    print(f"Kept pairs:      {kept_pairs}")
    print(f"Detailed CSV:    {out_path}")
    print(f"Summary CSV:     {summary_path}")
    print(f"Config:          {config_path}")
    print("\nTop 30 cells by mean delta CMM:")
    print(
        summary_df.nlargest(30, "mean_delta_cmm")[
            [
                "component",
                "center_layer",
                "patch_layers",
                "token_offset",
                "token",
                "patched_accuracy",
                "mean_delta_cmm",
                "mean_recovery",
            ]
        ].to_string(index=False)
    )


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Layer x token source-to-target causal tracing for residual, "
            "attention-output, and MLP-output activations."
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
        default=str(DEFAULT_DATASET_DIR),
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
    parser.add_argument("--source_distractors", type=int, default=40)
    parser.add_argument("--target_distractors", type=int, default=40)
    parser.add_argument(
        "--target_present_only",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--pair_filter",
        choices=[
            "all",
            "source_correct",
            "source_correct_target_incorrect",
            "both_correct",
        ],
        default="all",
    )
    parser.add_argument(
        "--components",
        type=str,
        default="attention,mlp",
        help="Comma-separated: residual,attention,mlp.",
    )
    parser.add_argument(
        "--token_offsets",
        type=str,
        default="-19:0",
        help="Offsets relative to final active input token, e.g. '-19:0' or '-5,-2,0'.",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="all",
        help="Window center layers: all, 0-27, 0,5,10-15.",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=3,
        help="Positive odd centered layer-window width.",
    )
    parser.add_argument(
        "--cache_device",
        choices=["cpu", "cuda"],
        default="cpu",
    )
    parser.add_argument(
        "--require_aligned_input_ids",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--max_pairs", type=int, default=None)
    parser.add_argument("--true_token_id", type=int, default=None)
    parser.add_argument("--false_token_id", type=int, default=None)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--out_path", type=str, required=True)

    main(parser.parse_args())