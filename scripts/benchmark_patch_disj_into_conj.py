#!/usr/bin/env python3
"""
Patch conjunctive-search target representations with matched disjunctive-search
target representations at every transformer layer.

The source representations are loaded from the output of the target hidden-state
extraction script. Only target-present conjunctive images can be patched because
the saved representation file contains target-present examples.

Example
-------
python benchmark_patch_disj_into_conj.py \
    --hidden_states_path /oscar/data/epavlick/aparasel/Binding-Problem-Brown/Results/target_hidden_states.pt \
    --out_path /oscar/data/epavlick/aparasel/Binding-Problem-Brown/Results/conj_disj_bbox_patch.csv \
    --distractors 5 10 20 30 40 50 \
    --model_name Qwen/Qwen2.5-VL-7B-Instruct \
    --patch_mode all_layers
"""

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
from tqdm import tqdm

from search_utils import (
    build_inputs,
    extract_answer,
    get_prompt,
    load_vlm,
)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.abspath(
    os.path.join(
        SCRIPT_DIR,
        "..",
        "search-dataset",
        "visual_search_dataset_counterfactual",
    )
)


def parse_filename(filename: str) -> Dict[str, Any]:
    pattern = (
        r"(disjunctive|conjunctive)"
        r"_distractors_(\d+)"
        r"_trial_(\d+)"
        r"_target_(true|false)\.png"
    )

    match = re.fullmatch(pattern, filename)
    if match is None:
        raise ValueError(f"Could not parse filename: {filename}")

    return {
        "condition": match.group(1),
        "num_distractors": int(match.group(2)),
        "trial": int(match.group(3)),
        "target_present": match.group(4) == "true",
    }


def load_target_metadata(image_path: str) -> Dict[str, Optional[float]]:
    json_path = image_path.replace(".png", ".json")

    if not os.path.exists(json_path):
        return {"target_x": None, "target_y": None}

    with open(json_path, "r") as f:
        meta = json.load(f)

    targets = [
        obj for obj in meta.get("objects", [])
        if obj.get("role") == "target"
    ]

    if not targets:
        return {"target_x": None, "target_y": None}

    target = targets[0]

    if "position_plot" in target:
        x, y = target["position_plot"]
    elif "position" in target:
        x, y = target["position"]
    else:
        return {"target_x": None, "target_y": None}

    return {"target_x": float(x), "target_y": float(y)}


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

    dist = np.sqrt(
        (x_norm - 0.5) ** 2
        + (y_norm - 0.5) ** 2
    )

    return {
        "target_x_norm": float(x_norm),
        "target_y_norm": float(y_norm),
        "target_dist_from_center": float(dist),
    }


def get_model_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def get_image_positions(
    inputs: Dict[str, torch.Tensor],
    model: torch.nn.Module,
) -> torch.Tensor:
    image_token_id = model.config.image_token_id
    return (
        inputs["input_ids"][0] == image_token_id
    ).nonzero(as_tuple=True)[0]


def find_transformer_layers(model: torch.nn.Module) -> torch.nn.ModuleList:
    """
    Find the decoder transformer blocks for common Hugging Face VLM layouts.

    Qwen2.5-VL normally resolves through one of the first two paths.
    """
    candidate_paths = [
        ("model", "language_model", "layers"),
        ("model", "layers"),
        ("language_model", "model", "layers"),
        ("language_model", "layers"),
    ]

    for path in candidate_paths:
        obj: Any = model
        valid = True

        for attr in path:
            if not hasattr(obj, attr):
                valid = False
                break
            obj = getattr(obj, attr)

        if valid and isinstance(obj, (torch.nn.ModuleList, list)):
            print("Transformer layers:", ".".join(path))
            return obj

    raise AttributeError(
        "Could not locate transformer layers. Checked: "
        + ", ".join(".".join(p) for p in candidate_paths)
    )


def as_python_int(value: Any) -> int:
    if torch.is_tensor(value):
        return int(value.item())
    return int(value)


def as_python_int_list(value: Any) -> List[int]:
    if torch.is_tensor(value):
        return [int(x) for x in value.detach().cpu().tolist()]
    return [int(x) for x in value]


@dataclass
class SavedExample:
    index: int
    condition: str
    num_distractors: int
    trial: int
    image_path: str
    bbox_token_indices: List[int]
    bbox_hidden: torch.Tensor
    token_h: int
    token_w: int
    target_token_idx: int


def build_saved_lookup(
    data: Dict[str, Any],
) -> Dict[Tuple[str, int, int], SavedExample]:
    required = [
        "conditions",
        "num_distractors",
        "trials",
        "image_paths",
        "bbox_token_indices",
        "bbox_hidden",
        "token_h",
        "token_w",
        "target_token_idx",
    ]

    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(
            "Hidden-state file is missing required keys: "
            + ", ".join(missing)
        )

    n = len(data["image_paths"])
    lookup: Dict[Tuple[str, int, int], SavedExample] = {}

    for i in range(n):
        condition = str(data["conditions"][i])
        num_distractors = as_python_int(data["num_distractors"][i])
        trial = as_python_int(data["trials"][i])
        key = (condition, num_distractors, trial)

        if key in lookup:
            raise ValueError(f"Duplicate saved representation key: {key}")

        lookup[key] = SavedExample(
            index=i,
            condition=condition,
            num_distractors=num_distractors,
            trial=trial,
            image_path=str(data["image_paths"][i]),
            bbox_token_indices=as_python_int_list(
                data["bbox_token_indices"][i]
            ),
            bbox_hidden=data["bbox_hidden"][i],
            token_h=as_python_int(data["token_h"][i]),
            token_w=as_python_int(data["token_w"][i]),
            target_token_idx=as_python_int(
                data["target_token_idx"][i]
            ),
        )

    return lookup


def validate_matched_examples(
    conjunctive: SavedExample,
    disjunctive: SavedExample,
) -> None:
    if conjunctive.bbox_token_indices != disjunctive.bbox_token_indices:
        raise ValueError(
            "Matched examples have different bbox token indices:\n"
            f"  conjunctive={conjunctive.bbox_token_indices}\n"
            f"  disjunctive={disjunctive.bbox_token_indices}"
        )

    if conjunctive.token_h != disjunctive.token_h:
        raise ValueError(
            f"token_h mismatch: {conjunctive.token_h} vs "
            f"{disjunctive.token_h}"
        )

    if conjunctive.token_w != disjunctive.token_w:
        raise ValueError(
            f"token_w mismatch: {conjunctive.token_w} vs "
            f"{disjunctive.token_w}"
        )

    if conjunctive.target_token_idx != disjunctive.target_token_idx:
        raise ValueError(
            "Center target token mismatch: "
            f"{conjunctive.target_token_idx} vs "
            f"{disjunctive.target_token_idx}"
        )

    if conjunctive.bbox_hidden.ndim != 3:
        raise ValueError(
            "Expected conjunctive bbox_hidden with shape "
            "[hidden_state_index, bbox_token, hidden_dim], got "
            f"{tuple(conjunctive.bbox_hidden.shape)}"
        )

    if disjunctive.bbox_hidden.ndim != 3:
        raise ValueError(
            "Expected disjunctive bbox_hidden with shape "
            "[hidden_state_index, bbox_token, hidden_dim], got "
            f"{tuple(disjunctive.bbox_hidden.shape)}"
        )

    expected_bbox_tokens = len(disjunctive.bbox_token_indices)
    if disjunctive.bbox_hidden.shape[1] != expected_bbox_tokens:
        raise ValueError(
            "Saved disjunctive bbox_hidden token count does not match "
            "bbox_token_indices: "
            f"{disjunctive.bbox_hidden.shape[1]} vs "
            f"{expected_bbox_tokens}"
        )


class BBoxPatchManager:
    """
    Register hooks that replace conjunctive bbox-token residuals with the
    corresponding matched disjunctive residuals.

    The extraction file stores:
        hidden_states[0]   = embedding output
        hidden_states[l+1] = output of transformer block l

    Therefore, a hook attached to block l uses bbox_hidden[l+1].
    """

    def __init__(
        self,
        layers: Sequence[torch.nn.Module],
        image_positions: torch.Tensor,
        bbox_token_indices: Sequence[int],
        source_bbox_hidden: torch.Tensor,
        patch_layers: Sequence[int],
    ) -> None:
        self.layers = layers
        self.image_positions_cpu = image_positions.detach().cpu().long()
        self.bbox_token_indices = [
            int(x) for x in bbox_token_indices
        ]
        self.source_bbox_hidden = source_bbox_hidden
        self.patch_layers = sorted(set(int(x) for x in patch_layers))
        self.handles: List[Any] = []
        self.patch_counts: Dict[int, int] = {
            layer_idx: 0 for layer_idx in self.patch_layers
        }

        if not self.bbox_token_indices:
            raise ValueError("Cannot patch an empty bbox token set.")

        if min(self.patch_layers, default=0) < 0:
            raise ValueError("Patch-layer indices must be nonnegative.")

        if self.patch_layers and max(self.patch_layers) >= len(self.layers):
            raise ValueError(
                f"Requested patch layer {max(self.patch_layers)}, but model "
                f"has only {len(self.layers)} layers."
            )

        required_hidden_states = len(self.layers) + 1
        if self.source_bbox_hidden.shape[0] < required_hidden_states:
            raise ValueError(
                "Saved bbox_hidden does not contain enough hidden states. "
                f"Need at least {required_hidden_states}, got "
                f"{self.source_bbox_hidden.shape[0]}."
            )

    def _make_hook(self, layer_idx: int):
        hidden_state_idx = layer_idx + 1

        def hook(
            module: torch.nn.Module,
            inputs: Tuple[Any, ...],
            output: Any,
        ) -> Any:
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output

            if not torch.is_tensor(hidden) or hidden.ndim != 3:
                return output

            # During cached autoregressive decoding, sequence length is usually
            # one. Patch only calls that still contain all prompt image tokens.
            image_positions = self.image_positions_cpu.to(hidden.device)

            if image_positions.numel() == 0:
                return output

            absolute_positions = image_positions[
                torch.tensor(
                    self.bbox_token_indices,
                    device=hidden.device,
                    dtype=torch.long,
                )
            ]

            if hidden.shape[1] <= int(absolute_positions.max().item()):
                return output

            source = self.source_bbox_hidden[
                hidden_state_idx
            ].to(
                device=hidden.device,
                dtype=hidden.dtype,
            )

            if source.shape[0] != len(self.bbox_token_indices):
                raise ValueError(
                    f"Layer {layer_idx}: source has {source.shape[0]} bbox "
                    f"tokens, expected {len(self.bbox_token_indices)}."
                )

            if source.shape[-1] != hidden.shape[-1]:
                raise ValueError(
                    f"Layer {layer_idx}: source hidden size "
                    f"{source.shape[-1]} != model hidden size "
                    f"{hidden.shape[-1]}."
                )

            # Clone to avoid unsafe in-place modification of a tensor that may
            # be reused elsewhere by the model implementation.
            patched_hidden = hidden.clone()
            patched_hidden[0, absolute_positions, :] = source
            self.patch_counts[layer_idx] += 1

            if isinstance(output, tuple):
                return (patched_hidden,) + output[1:]

            return patched_hidden

        return hook

    def __enter__(self):
        for layer_idx in self.patch_layers:
            handle = self.layers[layer_idx].register_forward_hook(
                self._make_hook(layer_idx)
            )
            self.handles.append(handle)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def assert_patched(self) -> None:
        missing = [
            layer_idx
            for layer_idx, count in self.patch_counts.items()
            if count == 0
        ]

        if missing:
            raise RuntimeError(
                "The following hooks never patched a prompt forward pass: "
                f"{missing}. This usually indicates an incorrect layer path "
                "or unexpected generation/model behavior."
            )


@torch.no_grad()
def patched_search_model_call(
    model: torch.nn.Module,
    processor: Any,
    image: Image.Image,
    condition: str,
    layers: Sequence[torch.nn.Module],
    image_bbox_token_indices: Sequence[int],
    source_bbox_hidden: torch.Tensor,
    patch_layers: Sequence[int],
    max_new_tokens: int,
) -> Tuple[str, Dict[int, int]]:
    prompt = get_prompt(condition)
    model_device = get_model_device(model)

    inputs = build_inputs(
        processor=processor,
        model_device=model_device,
        image=image,
        prompt=prompt,
    )

    image_positions = get_image_positions(inputs, model)

    if image_positions.numel() == 0:
        raise ValueError("No image-token positions found in model input.")

    max_bbox_idx = max(int(x) for x in image_bbox_token_indices)
    if max_bbox_idx >= image_positions.numel():
        raise IndexError(
            f"bbox token index {max_bbox_idx} exceeds available image-token "
            f"count {image_positions.numel()}."
        )

    with BBoxPatchManager(
        layers=layers,
        image_positions=image_positions,
        bbox_token_indices=image_bbox_token_indices,
        source_bbox_hidden=source_bbox_hidden,
        patch_layers=patch_layers,
    ) as patcher:
        gen_out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=False,
        )
        patcher.assert_patched()
        patch_counts = dict(patcher.patch_counts)

    generated_ids = gen_out.sequences
    input_len = inputs["input_ids"].shape[-1]
    new_token_ids = generated_ids[:, input_len:]

    response = processor.batch_decode(
        new_token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    return response, patch_counts


def collect_conjunctive_paths(
    distractors: Sequence[int],
    max_examples: Optional[int],
) -> List[str]:
    distractor_set = set(int(x) for x in distractors)
    folder = os.path.join(DATASET_DIR, "conjunctive")

    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Dataset folder not found: {folder}")

    paths: List[str] = []

    for filename in sorted(os.listdir(folder)):
        if not filename.endswith(".png"):
            continue

        meta = parse_filename(filename)

        # The hidden-state extraction file contains target-present examples.
        if not meta["target_present"]:
            continue

        if meta["num_distractors"] not in distractor_set:
            continue

        paths.append(os.path.join(folder, filename))

    if max_examples is not None:
        paths = paths[:max_examples]

    return paths


def resolve_patch_conditions(
    patch_mode: str,
    num_layers: int,
    requested_layers: Optional[Sequence[int]],
) -> List[Tuple[str, List[int]]]:
    if patch_mode == "all_layers":
        return [("all_layers", list(range(num_layers)))]

    if patch_mode == "layer_sweep":
        if requested_layers is None:
            layers = list(range(num_layers))
        else:
            layers = sorted(set(int(x) for x in requested_layers))

        invalid = [x for x in layers if x < 0 or x >= num_layers]
        if invalid:
            raise ValueError(
                f"Invalid patch layers {invalid}; model has {num_layers} layers."
            )

        return [(f"layer_{layer_idx}", [layer_idx]) for layer_idx in layers]

    if patch_mode == "selected_layers":
        if not requested_layers:
            raise ValueError(
                "--patch_layers is required for patch_mode=selected_layers"
            )

        layers = sorted(set(int(x) for x in requested_layers))
        invalid = [x for x in layers if x < 0 or x >= num_layers]

        if invalid:
            raise ValueError(
                f"Invalid patch layers {invalid}; model has {num_layers} layers."
            )

        label = "layers_" + "-".join(str(x) for x in layers)
        return [(label, layers)]

    raise ValueError(f"Unknown patch mode: {patch_mode}")


def save_results(
    results: List[Dict[str, Any]],
    out_path: str,
) -> None:
    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)

    df = pd.DataFrame(results)
    df.to_csv(out_path, index=False)

    if not df.empty:
        summary = (
            df.groupby(
                [
                    "patch_condition",
                    "num_distractors",
                ],
                dropna=False,
            )
            .agg(
                accuracy=("correct", "mean"),
                n=("correct", "size"),
                parsed_rate=("prediction", lambda x: x.notna().mean()),
            )
            .reset_index()
        )
    else:
        summary = pd.DataFrame(
            columns=[
                "patch_condition",
                "num_distractors",
                "accuracy",
                "n",
                "parsed_rate",
            ]
        )

    root, ext = os.path.splitext(out_path)
    summary_path = root + "_summary.csv"
    summary.to_csv(summary_path, index=False)


def main(args: argparse.Namespace) -> None:
    hidden_data = torch.load(
        args.hidden_states_path,
        map_location="cpu",
    )
    saved_lookup = build_saved_lookup(hidden_data)

    model, processor = load_vlm(args.model_name)
    model.eval()

    layers = find_transformer_layers(model)
    num_layers = len(layers)

    patch_conditions = resolve_patch_conditions(
        patch_mode=args.patch_mode,
        num_layers=num_layers,
        requested_layers=args.patch_layers,
    )

    image_paths = collect_conjunctive_paths(
        distractors=args.distractors,
        max_examples=args.max_examples,
    )

    print(f"Found {len(image_paths)} target-present conjunctive images.")
    print(f"Model transformer layers: {num_layers}")
    print(
        "Patch conditions:",
        [(name, patch_layers) for name, patch_layers in patch_conditions],
    )

    results: List[Dict[str, Any]] = []
    processed = 0

    for path in tqdm(image_paths):
        filename = os.path.basename(path)
        meta = parse_filename(filename)

        conj_key = (
            "conjunctive",
            meta["num_distractors"],
            meta["trial"],
        )
        disj_key = (
            "disjunctive",
            meta["num_distractors"],
            meta["trial"],
        )

        if conj_key not in saved_lookup:
            message = f"Missing conjunctive saved representation: {conj_key}"
            if args.strict_matching:
                raise KeyError(message)
            print("SKIP:", message)
            continue

        if disj_key not in saved_lookup:
            message = f"Missing disjunctive saved representation: {disj_key}"
            if args.strict_matching:
                raise KeyError(message)
            print("SKIP:", message)
            continue

        conj_saved = saved_lookup[conj_key]
        disj_saved = saved_lookup[disj_key]
        validate_matched_examples(conj_saved, disj_saved)

        target_meta = load_target_metadata(path)
        target_features = compute_target_position_features(
            target_meta["target_x"],
            target_meta["target_y"],
        )

        prompt_condition = (
            "conjunctive_alt"
            if args.prompt_type != "regular"
            else "conjunctive"
        )

        for patch_condition, patch_layers in patch_conditions:
            response: Optional[str]
            prediction: Optional[bool]
            correct: bool
            error: Optional[str]
            patch_counts: Dict[int, int]

            try:
                with Image.open(path) as img:
                    image = img.convert("RGB")

                response, patch_counts = patched_search_model_call(
                    model=model,
                    processor=processor,
                    image=image,
                    condition=prompt_condition,
                    layers=layers,
                    image_bbox_token_indices=conj_saved.bbox_token_indices,
                    source_bbox_hidden=disj_saved.bbox_hidden,
                    patch_layers=patch_layers,
                    max_new_tokens=args.max_new_tokens,
                )

                prediction = extract_answer(response)
                correct = prediction == meta["target_present"]
                error = None

            except Exception as exc:
                response = None
                prediction = None
                correct = False
                patch_counts = {}
                error = f"{type(exc).__name__}: {exc}"
                print(f"\nERROR on {filename}, {patch_condition}: {error}")

                if args.fail_fast:
                    raise

            results.append(
                {
                    "image_path": path,
                    "matched_disjunctive_path": disj_saved.image_path,
                    "condition": meta["condition"],
                    "num_distractors": meta["num_distractors"],
                    "trial": meta["trial"],
                    "target_present": meta["target_present"],
                    "patch_condition": patch_condition,
                    "patch_mode": args.patch_mode,
                    "patch_layers": json.dumps(patch_layers),
                    "patch_counts": json.dumps(patch_counts),
                    "num_bbox_tokens": len(
                        conj_saved.bbox_token_indices
                    ),
                    "bbox_token_indices": json.dumps(
                        conj_saved.bbox_token_indices
                    ),
                    "prediction": prediction,
                    "correct": correct,
                    "response": response,
                    "error": error,
                    "target_x": target_meta["target_x"],
                    "target_y": target_meta["target_y"],
                    "target_x_norm": target_features["target_x_norm"],
                    "target_y_norm": target_features["target_y_norm"],
                    "target_dist_from_center": target_features[
                        "target_dist_from_center"
                    ],
                }
            )

            processed += 1
            if processed % args.save_every == 0:
                save_results(results, args.out_path)

    save_results(results, args.out_path)

    print("\nSaved:")
    print(args.out_path)

    summary_path = os.path.splitext(args.out_path)[0] + "_summary.csv"
    print("\nSaved summary:")
    print(summary_path)

    if results:
        summary = pd.read_csv(summary_path)
        print("\nAccuracy summary:")
        print(summary.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct",
    )

    parser.add_argument(
        "--hidden_states_path",
        type=str,
        required=True,
        help="Path to target_hidden_states.pt.",
    )

    parser.add_argument(
        "--out_path",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--distractors",
        type=int,
        nargs="+",
        required=True,
        help="Distractor counts to evaluate, e.g. --distractors 5 10 20 30.",
    )

    parser.add_argument(
        "--patch_mode",
        type=str,
        choices=[
            "all_layers",
            "layer_sweep",
            "selected_layers",
        ],
        default="all_layers",
        help=(
            "all_layers: patch each transformer block during one generation; "
            "layer_sweep: run one generation per patched layer; "
            "selected_layers: patch only the layers supplied to --patch_layers."
        ),
    )

    parser.add_argument(
        "--patch_layers",
        type=int,
        nargs="*",
        default=None,
    )

    parser.add_argument(
        "--prompt_type",
        type=str,
        default="regular",
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--save_every",
        type=int,
        default=25,
    )

    parser.add_argument(
        "--strict_matching",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument(
        "--fail_fast",
        action="store_true",
    )

    args = parser.parse_args()
    main(args)
