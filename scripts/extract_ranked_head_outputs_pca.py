#!/usr/bin/env python3
"""Extract selected attention-head outputs and visualize them with PCA.

For each matched disjunctive/conjunctive example, this script captures the
input to the attention output projection (o_proj) at ``--token_offset``. Before
o_proj, the vector is a concatenation of per-head, attention-weighted value
vectors, so reshaping it recovers one ``head_dim``-dimensional output per head.

The script saves:
  * head_outputs.pt: raw [examples, selected_heads, head_dim] tensor + metadata
  * head_outputs_long.csv: one row per example/head (vector stored separately)
  * pca_coordinates.csv: PC coordinates and labels for every example/head
  * pca_explained_variance.csv: explained-variance ratios per head
  * pca_L{layer}H{head}.png: one PCA plot per selected head

Example:
python extract_ranked_head_outputs_pca.py \
    --ranking_csv ../Results/disj40_to_conj40_headpatch_qwen_-9.csv \
    --ranking_metric delta_margin \
    --ranking_target_present true \
    --top_k 8 \
    --distractors 40 \
    --token_offset -9 \
    --target_presence all \
    --pair_filter all \
    --out_dir ../Results/disj40_to_conj40_head_outputs_top8_offset_-9
"""

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from search_utils import build_inputs, load_vlm
from reconstruct_ranked_head_attention import (
    baseline_forward,
    find_decoder_layers,
    get_model_device,
    make_matched_pairs,
    resolve_boolean_token_id,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = (
    SCRIPT_DIR.parent / "search-dataset" /
    "visual_search_dataset_counterfactual"
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


def parse_bool_column(series):
    if series.dtype == object:
        return (
            series.astype(str).str.strip().str.lower()
            .map({"true": True, "false": False, "1": True, "0": False})
        )
    return series.astype("boolean")


def load_top_heads(path, metric, top_k, ranking_target_present):
    df = pd.read_csv(path).copy()
    required = {"layer", "head", metric}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Ranking CSV missing {sorted(missing)}; columns={list(df.columns)}"
        )

    if ranking_target_present != "all":
        if "target_present" not in df.columns:
            raise ValueError(
                "--ranking_target_present requires target_present in the CSV"
            )
        wanted = ranking_target_present == "true"
        df = df[parse_bool_column(df["target_present"]) == wanted].copy()

    for column in ["layer", "head", metric]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["layer", "head", metric])
    df[["layer", "head"]] = df[["layer", "head"]].astype(int)
    ranked = (
        df.groupby(["layer", "head"], as_index=False)[metric].mean()
        .sort_values([metric, "layer", "head"], ascending=[False, True, True])
        .head(top_k).reset_index(drop=True)
        .rename(columns={metric: "ranking_score"})
    )
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    return ranked


def get_last_active_position(inputs):
    mask = inputs.get("attention_mask")
    if mask is None:
        return int(inputs["input_ids"].shape[1] - 1)
    return int(torch.nonzero(mask[0], as_tuple=False).flatten()[-1].item())


def resolve_query_position(inputs, token_offset):
    if token_offset > 0:
        raise ValueError("--token_offset must be <= 0")
    last = get_last_active_position(inputs)
    query = last + token_offset
    if query < 0:
        raise ValueError(f"token_offset={token_offset} gives position {query}")
    return last, query


def safe_token_text(tokenizer, token_id):
    return tokenizer.decode(
        [int(token_id)], skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def get_o_proj(layer):
    attention = getattr(layer, "self_attn", None)
    if attention is None:
        raise AttributeError("Decoder layer has no self_attn module")
    for name in ("o_proj", "out_proj", "dense"):
        module = getattr(attention, name, None)
        if module is not None:
            return module
    raise AttributeError("Could not locate the attention output projection")


def infer_head_geometry(model, o_proj, ranked):
    config = getattr(model, "config", None)
    text_config = getattr(config, "text_config", config)
    n_heads = getattr(text_config, "num_attention_heads", None)
    width = getattr(o_proj, "in_features", None)
    if width is None:
        width = int(o_proj.weight.shape[1])
    if n_heads is None:
        n_heads = max(int(ranked["head"].max()) + 1, 1)
    if width % n_heads != 0:
        raise ValueError(f"o_proj input width {width} not divisible by {n_heads} heads")
    return int(n_heads), int(width // n_heads)


@torch.no_grad()
def extract_one(model, processor, layers, selected_by_layer, image_path,
                token_offset, true_id, false_id, n_heads, head_dim):
    with Image.open(image_path) as handle:
        image = handle.convert("RGB")
    inputs = build_inputs(
        processor=processor, model_device=get_model_device(model),
        image=image, prompt=BINARY_PROMPT,
    )
    last_position, query_position = resolve_query_position(inputs, token_offset)
    captured = {}
    handles = []

    def make_hook(layer_index):
        def hook(_module, hook_inputs):
            x = hook_inputs[0]
            # Qwen normally supplies [batch, sequence, hidden].
            if x.ndim != 3:
                raise RuntimeError(
                    f"Expected rank-3 o_proj input, got {tuple(x.shape)}"
                )
            if x.shape[0] == inputs["input_ids"].shape[0]:
                vector = x[0, query_position]
            elif x.shape[1] == inputs["input_ids"].shape[0]:
                vector = x[query_position, 0]
            else:
                raise RuntimeError(f"Unrecognized o_proj layout {tuple(x.shape)}")
            captured[layer_index] = vector.detach().float().cpu()
        return hook

    try:
        for layer_index in selected_by_layer:
            handles.append(
                get_o_proj(layers[layer_index]).register_forward_pre_hook(
                    make_hook(layer_index)
                )
            )
        baseline = baseline_forward(model, inputs, true_id, false_id)
    finally:
        for handle in handles:
            handle.remove()

    outputs = {}
    for layer_index, heads in selected_by_layer.items():
        if layer_index not in captured:
            raise RuntimeError(f"Hook for layer {layer_index} did not fire")
        vector = captured[layer_index]
        if vector.numel() != n_heads * head_dim:
            raise RuntimeError(
                f"Layer {layer_index}: captured {vector.numel()} values; "
                f"expected {n_heads} x {head_dim}"
            )
        split = vector.reshape(n_heads, head_dim)
        for head in heads:
            outputs[(layer_index, head)] = split[head].clone()

    query_id = int(inputs["input_ids"][0, query_position].item())
    return outputs, {
        "prediction": bool(baseline["prediction"]),
        "margin": float(baseline["margin"]),
        "last_position": last_position,
        "query_position": query_position,
        "query_token_id": query_id,
        "query_token_text": safe_token_text(processor.tokenizer, query_id),
    }


def fit_and_plot_pca(output_tensor, metadata, ranked, out_dir, standardize):
    coordinate_rows = []
    variance_rows = []
    colors = {"disjunctive": "#0072B2", "conjunctive": "#D55E00"}

    for head_index, row in enumerate(ranked.itertuples(index=False)):
        x = output_tensor[:, head_index, :].numpy()
        x_fit = StandardScaler().fit_transform(x) if standardize else x
        n_components = min(3, x_fit.shape[0], x_fit.shape[1])
        if n_components < 2:
            continue
        pca = PCA(n_components=n_components)
        z = pca.fit_transform(x_fit)
        variance_rows.append({
            "layer": int(row.layer), "head": int(row.head),
            "rank": int(row.rank),
            "pc1_explained_variance": float(pca.explained_variance_ratio_[0]),
            "pc2_explained_variance": float(pca.explained_variance_ratio_[1]),
            "pc3_explained_variance": (
                float(pca.explained_variance_ratio_[2]) if n_components >= 3
                else np.nan
            ),
        })
        for i, meta in enumerate(metadata):
            coordinate_rows.append({
                **meta, "layer": int(row.layer), "head": int(row.head),
                "rank": int(row.rank), "ranking_score": float(row.ranking_score),
                "pc1": float(z[i, 0]), "pc2": float(z[i, 1]),
                "pc3": float(z[i, 2]) if n_components >= 3 else np.nan,
            })

        fig, ax = plt.subplots(figsize=(6.2, 5.2))
        for condition in ("disjunctive", "conjunctive"):
            for present, marker in ((True, "o"), (False, "^")):
                mask = np.array([
                    m["condition"] == condition and
                    bool(m["target_present"]) == present for m in metadata
                ])
                if mask.any():
                    ax.scatter(
                        z[mask, 0], z[mask, 1], s=34, alpha=0.72,
                        color=colors[condition], marker=marker,
                        label=(f"{condition.title()} | target "
                               f"{'present' if present else 'absent'}"),
                    )
        ax.axhline(0, color="0.85", linewidth=0.7)
        ax.axvline(0, color="0.85", linewidth=0.7)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
        ax.set_title(f"Head output PCA — Rank {row.rank}: L{row.layer}H{row.head}")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"pca_L{row.layer}H{row.head}.png", dpi=250)
        plt.close(fig)

    return pd.DataFrame(coordinate_rows), pd.DataFrame(variance_rows)


def main(args):
    torch.set_grad_enabled(False)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ranked = load_top_heads(
        args.ranking_csv, args.ranking_metric, args.top_k,
        args.ranking_target_present,
    )
    ranked.to_csv(out_dir / "selected_heads.csv", index=False)
    print("Selected heads:\n", ranked.to_string(index=False))

    model, processor = load_vlm(args.model_name)
    model.eval()
    layers, layer_path = find_decoder_layers(model)
    print(f"Decoder: {layer_path} ({len(layers)} layers)")
    selected_by_layer = defaultdict(list)
    for row in ranked.itertuples(index=False):
        selected_by_layer[int(row.layer)].append(int(row.head))
    first_o_proj = get_o_proj(layers[int(ranked.iloc[0]["layer"])])
    n_heads, head_dim = infer_head_geometry(model, first_o_proj, ranked)
    print(f"Attention geometry: {n_heads} heads x {head_dim} dimensions")

    true_id = resolve_boolean_token_id(processor, "True")
    false_id = resolve_boolean_token_id(processor, "False")
    pairs = make_matched_pairs(
        dataset_dir=Path(args.dataset_dir),
        source_condition="disjunctive", target_condition="conjunctive",
        source_distractors=args.distractors,
        target_distractors=args.distractors,
        target_present_only=False,
    )
    if args.target_presence != "all":
        wanted = args.target_presence == "true"
        pairs = [p for p in pairs if bool(p["target_present"]) == wanted]
    if args.max_pairs is not None:
        pairs = pairs[:args.max_pairs]

    all_vectors, metadata = [], []
    for pair in tqdm(pairs, desc="Matched pairs"):
        pair_vectors, pair_metadata = [], []
        correctness = {}
        for condition, image_path in (
            ("disjunctive", pair["source_path"]),
            ("conjunctive", pair["target_path"]),
        ):
            outputs, info = extract_one(
                model, processor, layers, selected_by_layer, image_path,
                args.token_offset, true_id, false_id, n_heads, head_dim,
            )
            ordered = torch.stack([
                outputs[(int(r.layer), int(r.head))]
                for r in ranked.itertuples(index=False)
            ])
            correct = bool(info["prediction"] == bool(pair["target_present"]))
            correctness[condition] = correct
            pair_vectors.append(ordered)
            pair_metadata.append({
                "condition": condition, "trial": pair["trial"],
                "target_present": bool(pair["target_present"]),
                "image_path": str(image_path), "prediction": info["prediction"],
                "correct": correct, "margin": info["margin"],
                "token_offset": args.token_offset,
                "last_position": info["last_position"],
                "query_position": info["query_position"],
                "query_token_id": info["query_token_id"],
                "query_token_text": info["query_token_text"],
            })

        keep = (
            args.pair_filter == "all" or
            (args.pair_filter == "source_correct" and correctness["disjunctive"]) or
            (args.pair_filter == "source_correct_target_incorrect" and
             correctness["disjunctive"] and not correctness["conjunctive"]) or
            (args.pair_filter == "both_correct" and all(correctness.values()))
        )
        if keep:
            all_vectors.extend(pair_vectors)
            metadata.extend(pair_metadata)

    if not all_vectors:
        raise RuntimeError("No matched pairs passed the requested filters")
    output_tensor = torch.stack(all_vectors).float()
    payload = {
        "head_outputs": output_tensor,
        "metadata": metadata,
        "selected_heads": ranked.to_dict("records"),
        "head_dim": head_dim,
        "description": "Post-attention per-head vectors immediately before o_proj",
    }
    torch.save(payload, out_dir / "head_outputs.pt")

    long_rows = []
    for example_index, meta in enumerate(metadata):
        for head_index, row in enumerate(ranked.itertuples(index=False)):
            vector = output_tensor[example_index, head_index]
            long_rows.append({
                "example_index": example_index, "head_index": head_index,
                **meta, "layer": int(row.layer), "head": int(row.head),
                "rank": int(row.rank), "ranking_score": float(row.ranking_score),
                "output_norm": float(vector.norm()),
            })
    pd.DataFrame(long_rows).to_csv(out_dir / "head_outputs_long.csv", index=False)

    coordinates, variance = fit_and_plot_pca(
        output_tensor, metadata, ranked, out_dir, args.standardize,
    )
    coordinates.to_csv(out_dir / "pca_coordinates.csv", index=False)
    variance.to_csv(out_dir / "pca_explained_variance.csv", index=False)
    print(f"\nSaved {len(metadata)} examples to: {out_dir}")
    print(f"Raw tensor shape: {tuple(output_tensor.shape)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--dataset_dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--ranking_csv", required=True)
    parser.add_argument("--ranking_metric", default="delta_margin")
    parser.add_argument("--ranking_target_present", choices=["all", "true", "false"], default="true")
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--distractors", type=int, default=40)
    parser.add_argument("--token_offset", type=int, default=-9)
    parser.add_argument("--target_presence", choices=["all", "true", "false"], default="all")
    parser.add_argument(
        "--pair_filter",
        choices=["all", "source_correct", "source_correct_target_incorrect", "both_correct"],
        default="all",
    )
    parser.add_argument("--max_pairs", type=int, default=None)
    parser.add_argument(
        "--standardize", action="store_true",
        help="z-score head dimensions before PCA (default: PCA on raw outputs)",
    )
    parser.add_argument("--out_dir", required=True)
    main(parser.parse_args())
