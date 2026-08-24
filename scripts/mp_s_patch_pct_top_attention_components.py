#!/usr/bin/env python3
"""
Fast cumulative attention head output (pre-o_proj) causal patching.

Patches full head outputs (A @ V) from source to target at a selectable
token offset without computing intermediate query-key decomposition matrices.


python mp_s_patch_pct_top_attention_components.py \

--dataset MP_S \

--token_offset -9 \

--head_ranking_csv ../Results/disj40_to_conj40_headpatch_qwen_-9.csv \

--ranking_metric delta_margin \

--rank_by_target_presence \

--patch_percentages 0.1,0.2,0.5,1,10,25,50,100 \

--out_path ../Results/mp_s_present_ranked_heads_offset_-9.csv

Unpaired counteractual control:

python mp_s_patch_pct_top_attention_components.py \
    --dataset MP_S \
    --token_offset -9 \
    --head_ranking_csv ../Results/disj40_to_conj40_headpatch_qwen_-9.csv \
    --ranking_metric delta_margin \
    --rank_by_target_presence \
    --patch_percentages 0.1,0.2,0.5,1,10,25,50,100 \
    --unpaired_counterfactuals \
    --out_path ../Results/mp_s_present_ranked_heads_unpaired_offset_-9.csv
"""

import argparse
import json
import os
import random
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


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_DATASET_DIR = os.path.abspath(
    os.path.join(
        SCRIPT_DIR,
        "..",
        "search-dataset",
        "visual_search_dataset_counterfactual",
    )
)

DEFAULT_MINIMAL_DATASET_DIR = os.path.abspath(
    os.path.join(
        SCRIPT_DIR,
        "..",
        "search-dataset",
        "minimal_pair_swap_dataset",
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
    r"(disjunctive|conjunctive)"
    r"_distractors_(\d+)"
    r"_trial_(\d+)"
    r"_target_(true|false)\.png"
)

MINIMAL_PAIR_FILENAME_PATTERN = re.compile(
    r"pair_(\d+)_(original|counterfactual_swap)\.png"
)


def parse_filename(filename: str) -> Dict[str, Any]:
    minimal_match = MINIMAL_PAIR_FILENAME_PATTERN.fullmatch(filename)
    if minimal_match is not None:
        return {
            "condition": minimal_match.group(2),
            "num_distractors": 0,
            "trial": int(minimal_match.group(1)),
            "target_present": False,
        }

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
    dataset: str,
    condition: str,
    num_distractors: int,
    trial: int,
    target_present: bool,
) -> str:
    if dataset == "MP_S":
        if condition not in {"original", "counterfactual_swap"}:
            raise ValueError(
                "MP_S condition must be 'original' or 'counterfactual_swap', "
                f"got {condition!r}."
            )
        return os.path.join(dataset_dir, f"pair_{trial:03d}_{condition}.png")

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
    dataset: str,
    conditions: Sequence[str],
    source_distractors: int,
    max_pairs: Optional[int],
) -> List[str]:
    source_paths: List[str] = []

    for condition in conditions:
        condition_dir = (
            dataset_dir if dataset == "MP_S" else os.path.join(dataset_dir, condition)
        )
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

            if (
                metadata["condition"] != condition
                or metadata["num_distractors"] != source_distractors
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


def build_shifted_source_map(source_paths: Sequence[str]) -> Dict[int, str]:
    """Map each MP_S trial to the next available source trial, cyclically.

    This provides a deterministic unpaired-source control without assuming that
    trial IDs are contiguous. For sorted trials [t0, t1, ..., tn], tn maps back
    to t0.
    """
    source_by_trial: Dict[int, str] = {}
    for path in source_paths:
        meta = parse_filename(os.path.basename(path))
        source_by_trial[int(meta["trial"])] = path

    sorted_trials = sorted(source_by_trial)
    if len(sorted_trials) < 2:
        raise ValueError(
            "--unpaired_counterfactuals requires at least two MP_S source trials."
        )

    return {
        trial: source_by_trial[sorted_trials[(i + 1) % len(sorted_trials)]]
        for i, trial in enumerate(sorted_trials)
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
    if token_offset > 0:
        raise ValueError("--token_offset must be <= 0.")

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


def score_next_token_logits(
    logits: torch.Tensor,
    last_position: int,
    tokenizer,
    binary_tokens: Dict[str, Any],
    target_present: bool,
) -> Dict[str, Any]:
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
        "first_token_logprob": float(next_token_logprobs[first_token_id].item()),
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


@dataclass
class DecoderStack:
    path: str
    layers: List[nn.Module]


def has_patchable_attention(layer: nn.Module) -> bool:
    self_attn = getattr(layer, "self_attn", None)
    return self_attn is not None and isinstance(
        getattr(self_attn, "o_proj", None), nn.Module
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
        raise RuntimeError("Could not locate decoder layers exposing self_attn.o_proj.")

    candidates.sort(key=lambda item: len(item[1]), reverse=True)
    path, layers = candidates[0]
    return DecoderStack(path=path, layers=layers)


def nested_config_value(model, names: Sequence[str]) -> Optional[int]:
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
        model, ["num_attention_heads", "n_head", "num_heads"]
    )
    if value is None:
        raise RuntimeError("Could not determine num_attention_heads.")
    return value


def resolve_head_dim(model, layer: nn.Module, num_heads: int) -> int:
    self_attn = layer.self_attn
    for attr_name in ["head_dim", "attention_head_size"]:
        value = getattr(self_attn, attr_name, None)
        if value is not None:
            return int(value)

    hidden_size = nested_config_value(model, ["hidden_size", "d_model", "n_embd"])
    if hidden_size is not None and hidden_size % num_heads == 0:
        return hidden_size // num_heads

    raise RuntimeError("Could not resolve head_dim.")


def pre_o_proj_to_heads(
    tensor: torch.Tensor, num_heads: int, expected_head_dim: int
) -> Tuple[torch.Tensor, str]:
    if tensor.ndim == 3:
        batch_size, sequence_length, hidden_size = tensor.shape
        head_dim = hidden_size // num_heads
        return (
            tensor.reshape(batch_size, sequence_length, num_heads, head_dim),
            "BSH",
        )
    if tensor.ndim == 4:
        if tensor.shape[1] == num_heads:
            return tensor.permute(0, 2, 1, 3), "BHSD"
        elif tensor.shape[2] == num_heads:
            return tensor, "BSHD"
    raise ValueError(f"Unexpected pre-o_proj tensor shape: {tensor.shape}")


def heads_to_pre_o_proj(heads: torch.Tensor, layout: str) -> torch.Tensor:
    if layout == "BSH":
        batch_size, sequence_length, num_heads, head_dim = heads.shape
        return heads.reshape(batch_size, sequence_length, num_heads * head_dim)
    if layout == "BHSD":
        return heads.permute(0, 2, 1, 3)
    if layout == "BSHD":
        return heads
    raise ValueError(f"Unknown layout: {layout}")


class FastOProjCache:
    """Caches pre-o_proj heads only at the specified token position in GPU VRAM."""

    def __init__(
        self,
        target_layers: Sequence[int],
        patch_position: int,
        num_heads: int,
        head_dim: int,
    ):
        self.target_layers = set(target_layers)
        self.patch_position = patch_position
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.values: Dict[int, torch.Tensor] = {}

    def make_hook(self, layer_index: int):
        def hook(module, args):
            if layer_index not in self.target_layers:
                return
            current = args[0]
            heads, _ = pre_o_proj_to_heads(current, self.num_heads, self.head_dim)
            # Keep in VRAM, only store the single token position [num_heads, head_dim]
            self.values[layer_index] = heads[0, self.patch_position].detach()

        return hook


class FastMultiHeadPatcher:
    """Patches specific heads directly in pre_o_proj inputs."""

    def __init__(
        self,
        source_heads: torch.Tensor,
        head_indices: Sequence[int],
        target_position: int,
        num_heads: int,
        head_dim: int,
    ):
        self.source_heads = source_heads
        self.head_indices = head_indices
        self.target_position = target_position
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.called = False

    def hook(self, module, args):
        current = args[0]
        heads, layout = pre_o_proj_to_heads(current, self.num_heads, self.head_dim)
        patched_heads = heads.clone()

        source = self.source_heads.to(device=current.device, dtype=current.dtype)
        for h in self.head_indices:
            patched_heads[0, self.target_position, h, :] = source[h]

        self.called = True
        return (heads_to_pre_o_proj(patched_heads, layout),) + tuple(args[1:])


def compute_recovery(
    source_margin: float,
    target_margin: float,
    patched_margin: float,
    min_denominator: float,
) -> Dict[str, Any]:
    margin_drop = source_margin - target_margin
    delta_margin = patched_margin - target_margin
    recovery_valid = np.isfinite(margin_drop) and margin_drop > min_denominator
    recovery = delta_margin / margin_drop if recovery_valid else np.nan

    return {
        "margin_drop": margin_drop,
        "delta_margin": delta_margin,
        "recovery": recovery,
        "recovery_valid": bool(recovery_valid),
    }


def load_head_ranking(
    ranking_csv: str,
    ranking_metric: str,
    num_layers: int,
    num_heads: int,
    target_present_filter: Optional[bool] = None,
) -> pd.DataFrame:
    ranking = pd.read_csv(ranking_csv)

    if target_present_filter is not None and "target_present" in ranking.columns:
        target_values = ranking["target_present"]
        if target_values.dtype == object:
            target_values = (
                target_values.astype(str)
                .str.strip()
                .str.lower()
                .map({"true": True, "false": False, "1": True, "0": False})
            )
        else:
            target_values = target_values.astype("boolean")
        ranking = ranking[target_values == target_present_filter].copy()

    ranking["layer"] = pd.to_numeric(ranking["layer"], errors="coerce").astype(int)
    ranking["head"] = pd.to_numeric(ranking["head"], errors="coerce").astype(int)
    ranking[ranking_metric] = pd.to_numeric(ranking[ranking_metric], errors="coerce")
    ranking = ranking.dropna(subset=["layer", "head", ranking_metric])

    ranking = (
        ranking.groupby(["layer", "head"], as_index=False)[ranking_metric]
        .mean()
        .sort_values([ranking_metric, "layer", "head"], ascending=[False, True, True])
        .reset_index(drop=True)
    )

    ranking["rank"] = np.arange(1, len(ranking) + 1)
    return ranking


def parse_percentages(text: str) -> List[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    return sorted(set(values))


def main(args) -> None:
    model, processor = load_vlm(args.model_name)
    tokenizer = get_tokenizer(processor)
    binary_tokens = resolve_binary_tokens(tokenizer)
    decoder_stack = discover_decoder_stack(model)

    num_layers = len(decoder_stack.layers)
    num_heads = resolve_num_heads(model, decoder_stack.layers[0])
    head_dim = resolve_head_dim(model, decoder_stack.layers[0], num_heads)
    total_heads = num_layers * num_heads
    percentages = parse_percentages(args.patch_percentages)

    # Ranking configuration
    ranking_presence_values = [True] if args.dataset == "MP_S" else [True, False]
    rankings = {
        tp: load_head_ranking(
            args.head_ranking_csv,
            args.ranking_metric,
            num_layers,
            num_heads,
            target_present_filter=tp if args.rank_by_target_presence else None,
        )
        for tp in ranking_presence_values
    }

    all_heads = [
        (layer, head) for layer in range(num_layers) for head in range(num_heads)
    ]
    rng = random.Random(args.random_seed)
    random_order = list(all_heads)
    rng.shuffle(random_order)

    # Pre-select heads
    selected_heads_dict: Dict[str, Dict[bool, Dict[float, List[Tuple[int, int]]]]] = {
        "ranked": {},
        "random": {},
    }
    for tp in ranking_presence_values:
        selected_heads_dict["ranked"][tp] = {}
        selected_heads_dict["random"][tp] = {}
        for pct in percentages:
            cnt = max(1, int(np.ceil(total_heads * pct / 100.0)))
            selected_df = rankings[tp].head(cnt)
            selected_heads_dict["ranked"][tp][pct] = [
                (int(r.layer), int(r.head)) for r in selected_df.itertuples()
            ]
            selected_heads_dict["random"][tp][pct] = random_order[:cnt]

    source_condition = args.source_condition or (
        "original" if args.dataset == "MP_S" else "disjunctive"
    )
    target_condition = args.target_condition or (
        "counterfactual_swap" if args.dataset == "MP_S" else "conjunctive"
    )

    source_paths = collect_source_paths(
        dataset_dir=args.dataset_dir,
        dataset=args.dataset,
        conditions=[source_condition],
        source_distractors=args.source_distractors,
        max_pairs=args.max_pairs,
    )

    if args.unpaired_counterfactuals and args.dataset != "MP_S":
        raise ValueError(
            "--unpaired_counterfactuals is currently supported only for --dataset MP_S."
        )

    shifted_source_map = (
        build_shifted_source_map(source_paths)
        if args.unpaired_counterfactuals
        else None
    )

    pairing_desc = "unpaired (+1 cyclic source)" if args.unpaired_counterfactuals else "matched"
    print(
        f"\nProcessing {len(source_paths)} pairs with fast head output patching "
        f"({pairing_desc})..."
    )

    # Determine which selection modes to run
    selection_types = ["ranked"] if args.skip_random_control else ["ranked", "random"]

    rows: List[Dict[str, Any]] = []
    model_device = next(model.parameters()).device

    for matched_source_image_path in tqdm(source_paths, desc="Pairs"):
        matched_source_meta = parse_filename(os.path.basename(matched_source_image_path))
        trial = matched_source_meta["trial"]
        target_present = matched_source_meta["target_present"]

        if shifted_source_map is not None:
            source_image_path = shifted_source_map[trial]
            source_pairing = "unpaired_shifted"
        else:
            source_image_path = matched_source_image_path
            source_pairing = "matched"

        source_meta = parse_filename(os.path.basename(source_image_path))
        source_trial = int(source_meta["trial"])

        target_image_path = corresponding_image_path(
            dataset_dir=args.dataset_dir,
            dataset=args.dataset,
            condition=target_condition,
            num_distractors=args.target_distractors,
            trial=trial,
            target_present=target_present,
        )
        if not os.path.exists(target_image_path):
            continue

        with Image.open(source_image_path) as img:
            source_inputs = build_inputs(
                processor, model_device, img.convert("RGB"), BINARY_PROMPT
            )
        with Image.open(target_image_path) as img:
            target_inputs = build_inputs(
                processor, model_device, img.convert("RGB"), BINARY_PROMPT
            )

        src_meta = selected_token_metadata(source_inputs, tokenizer, args.token_offset)
        tgt_meta = selected_token_metadata(target_inputs, tokenizer, args.token_offset)

        # 1. Identify which layers are actually needed for caching
        needed_layers = set()
        for st in selection_types:
            for tp in ranking_presence_values:
                for pct in percentages:
                    for l, _ in selected_heads_dict[st][tp][pct]:
                        needed_layers.add(l)

        # 2. Source Forward Pass (with selective caching)
        source_cache = FastOProjCache(
            target_layers=needed_layers,
            patch_position=src_meta["patch_position"],
            num_heads=num_heads,
            head_dim=head_dim,
        )
        handles = [
            decoder_stack.layers[l].self_attn.o_proj.register_forward_pre_hook(
                source_cache.make_hook(l)
            )
            for l in needed_layers
        ]
        with torch.inference_mode():
            src_out = model(**source_inputs, use_cache=False, return_dict=True)
        for h in handles:
            h.remove()

        src_result = score_next_token_logits(
            src_out.logits,
            src_meta["last_position"],
            tokenizer,
            binary_tokens,
            target_present,
        )
        source_margin = float(src_result["correct_logit_margin"])

        # 3. Target Unpatched Forward Pass
        with torch.inference_mode():
            tgt_out = model(**target_inputs, use_cache=False, return_dict=True)
        tgt_result = score_next_token_logits(
            tgt_out.logits,
            tgt_meta["last_position"],
            tokenizer,
            binary_tokens,
            target_present,
        )
        target_margin = float(tgt_result["correct_logit_margin"])

        # 4. Run Patched Forward Passes
        ranking_labels_to_eval = (
            ranking_presence_values
            if args.rank_by_target_presence
            else [bool(target_present)]
        )

        for pct in percentages:
            for r_label in ranking_labels_to_eval:
                for st in selection_types:
                    heads_to_patch = selected_heads_dict[st][r_label][pct]
                    heads_by_layer: Dict[int, List[int]] = {}
                    for l, h in heads_to_patch:
                        heads_by_layer.setdefault(l, []).append(h)

                    patchers = {
                        l: FastMultiHeadPatcher(
                            source_heads=source_cache.values[l],
                            head_indices=h_list,
                            target_position=tgt_meta["patch_position"],
                            num_heads=num_heads,
                            head_dim=head_dim,
                        )
                        for l, h_list in heads_by_layer.items()
                    }

                    p_handles = [
                        decoder_stack.layers[l].self_attn.o_proj.register_forward_pre_hook(
                            patcher.hook
                        )
                        for l, patcher in patchers.items()
                    ]

                    with torch.inference_mode():
                        patched_out = model(
                            **target_inputs, use_cache=False, return_dict=True
                        )
                    for h in p_handles:
                        h.remove()

                    p_result = score_next_token_logits(
                        patched_out.logits,
                        tgt_meta["last_position"],
                        tokenizer,
                        binary_tokens,
                        target_present,
                    )
                    patched_margin = float(p_result["correct_logit_margin"])
                    recovery = compute_recovery(
                        source_margin, target_margin, patched_margin, min_denominator=0.25
                    )

                    rows.append(
                        {
                            "trial": trial,
                            "target_trial": trial,
                            "source_trial": source_trial,
                            "source_pairing": source_pairing,
                            "target_present": target_present,
                            "ranking_target_present": r_label,
                            "patch_percentage": pct,
                            "selection_type": st,
                            "selected_head_count": len(heads_to_patch),
                            "source_margin": source_margin,
                            "target_margin": target_margin,
                            "patched_margin": patched_margin,
                            "source_correct": src_result["correct"],
                            "target_correct": tgt_result["correct"],
                            "patched_correct": p_result["correct"],
                            **recovery,
                        }
                    )

    os.makedirs(os.path.dirname(os.path.abspath(args.out_path)), exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.out_path, index=False)
    print(f"\nSaved results to {args.out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct"
    )
    parser.add_argument("--dataset", choices=["VS_CF", "MP_S"], default="MP_S")
    parser.add_argument("--dataset_dir", type=str, default=DEFAULT_MINIMAL_DATASET_DIR)
    parser.add_argument("--out_path", type=str, required=True)
    parser.add_argument("--head_ranking_csv", type=str, required=True)
    parser.add_argument("--ranking_metric", type=str, default="delta_margin")
    parser.add_argument("--rank_by_target_presence", action="store_true")
    parser.add_argument("--patch_percentages", type=str, default="0.1")
    parser.add_argument("--source_condition", type=str, default=None)
    parser.add_argument("--target_condition", type=str, default=None)
    parser.add_argument("--source_distractors", type=int, default=0)
    parser.add_argument("--target_distractors", type=int, default=0)
    parser.add_argument("--token_offset", type=int, default=-9)
    parser.add_argument("--max_pairs", type=int, default=None)
    parser.add_argument("--skip_random_control", action="store_true")
    parser.add_argument(
        "--unpaired_counterfactuals",
        action="store_true",
        help=(
            "For MP_S, source each target trial from the next available original "
            "trial (cyclic +1) instead of its matched original. This tests whether "
            "spatial/example correspondence is required for causal recovery."
        ),
    )
    parser.add_argument("--random_seed", type=int, default=42)
    main(parser.parse_args())