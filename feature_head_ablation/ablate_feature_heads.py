#!/usr/bin/env python3
"""
Necessity test for color and shape attention heads in a vision-language model.

Each image in a manifest is asked a yes/no question that names a color and a
shape (default: "Is there a {color} letter {shape} in the image?").  The model
is run once unablated, then once per ablation spec:

    head set    color        top P% color heads, ablated at the color word
                shape        top P% shape heads, ablated at the shape word
                color+shape  both sets, each at its own word
    selection   ranked       heads from head_rankings/
                random       size-matched random heads (one draw per seed and
                             word; the color part of random color+shape is the
                             same draw as random color)
    type        zero         head output set to 0
                mean         head output set to its mean over ALL manifest
                             images (every condition, present + absent)

With 2 random seeds that is 3 x 3 x 2 = 18 ablated forwards per image.
Only the head output at the color/shape word token is replaced; every other
position is untouched.

Outputs in --out_dir (re-running the same command resumes):
    baseline.csv      unablated result per image (+ the tokens ablated)
    act_sums.pt       summed head outputs, used for the mean-ablation vectors
    head_sets.json    exact (layer, head) lists for every spec
    ablations.csv     one row per (image, spec)
    summary.csv       accuracy / margin per condition x spec, split into
                      pooled, present, absent; random seeds averaged as seed -2

Example:
    python ablate_feature_heads.py --model qwen2.5-vl-7b \\
        --manifest my_manifest.csv --out_dir results/qwen
"""

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from head_utils import (
    build_inputs, decode_token, discover_decoder_stack, find_word_position,
    from_heads, get_tokenizer, last_input_position, load_vlm,
    resolve_answer_tokens, resolve_head_dim, resolve_num_heads, score_answer,
    to_heads,
)

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = {
    "qwen2.5-vl-7b": "Qwen/Qwen2.5-VL-7B-Instruct",
    "internvl3.5-8b": "OpenGVLab/InternVL3_5-8B-HF",
    "gemma-3-12b": "google/gemma-3-12b-it",
}
DEFAULT_PROMPT = """Answer the following question with exactly one word:

True

or

False

Do not provide any explanation.

Question:
You are presented with an image containing letters.

Is there a {color} letter {shape} in the image?"""
FEATURES = ("color", "shape")
# Separate random streams per feature so random color and random shape heads
# differ; the values match the original runs so their random sets reproduce.
RANDOM_STREAM = {"color": 11, "shape": 9}


# --------------------------------------------------------------------------
# Hooks
# --------------------------------------------------------------------------
class Capture:
    """Store every head's output at fixed positions for one layer."""

    def __init__(self, positions: Dict[str, int], num_heads, head_dim):
        self.positions, self.num_heads, self.head_dim = positions, num_heads, head_dim
        self.values: Dict[str, torch.Tensor] = {}

    def hook(self, module, args):
        heads, _ = to_heads(args[0], self.num_heads, self.head_dim)
        for feature, pos in self.positions.items():
            self.values[feature] = heads[0, pos].detach().float().cpu()  # [H, D]


class Replace:
    """Overwrite selected heads at given positions with zeros or a fixed vector."""

    def __init__(self, assignments, num_heads, head_dim):
        self.assignments = assignments  # [(position, head, vector-or-None)]
        self.num_heads, self.head_dim = num_heads, head_dim
        self.called = False

    def hook(self, module, args):
        heads, layout = to_heads(args[0], self.num_heads, self.head_dim)
        heads = heads.clone()
        for pos, head, vector in self.assignments:
            heads[0, pos, head, :] = 0 if vector is None else vector.to(heads.device, heads.dtype)
        self.called = True
        return (from_heads(heads, layout),) + tuple(args[1:])


class Runner:
    def __init__(self, model, processor, color_offset=None, shape_offset=None):
        self.model, self.processor = model, processor
        self.tokenizer = get_tokenizer(processor)
        self.answers = resolve_answer_tokens(self.tokenizer)
        self.stack = discover_decoder_stack(model)
        self.num_layers = len(self.stack.layers)
        self.num_heads = resolve_num_heads(model, self.stack.layers[0])
        self.head_dim = resolve_head_dim(model, self.stack.layers[0], self.num_heads)
        self.device = next(model.parameters()).device
        self.offsets = {"color": color_offset, "shape": shape_offset}

    def inputs_and_positions(self, image, row):
        inputs = build_inputs(self.processor, self.device, image, row["prompt"])
        last = last_input_position(inputs)
        positions = {}
        for feature in FEATURES:
            offset = self.offsets[feature]
            positions[feature] = (last + offset if offset is not None else
                                  find_word_position(inputs["input_ids"], self.tokenizer,
                                                     str(row[f"{feature}_word"])))
        return inputs, last, positions

    def _forward(self, inputs, hooks):
        handles = [self.stack.layers[l].self_attn.o_proj.register_forward_pre_hook(h.hook)
                   for l, h in hooks.items()]
        try:
            with torch.inference_mode():
                return self.model(**inputs, use_cache=False, return_dict=True).logits
        finally:
            for h in handles:
                h.remove()

    def baseline(self, image, row):
        inputs, last, positions = self.inputs_and_positions(image, row)
        caps = {l: Capture(positions, self.num_heads, self.head_dim) for l in range(self.num_layers)}
        logits = self._forward(inputs, caps)
        acts = {f: torch.stack([caps[l].values[f] for l in range(self.num_layers)]) for f in FEATURES}
        tokens = {f: decode_token(self.tokenizer, inputs["input_ids"][0, p]) for f, p in positions.items()}
        offsets = {f: p - last for f, p in positions.items()}
        return score_answer(logits, last, self.answers, row["target_present"]), acts, tokens, offsets

    def ablate(self, image, row, heads_by_feature, means: Optional[Dict[str, torch.Tensor]]):
        inputs, last, positions = self.inputs_and_positions(image, row)
        by_layer: Dict[int, list] = {}
        for feature, heads in heads_by_feature.items():
            for layer, head in heads:
                vector = None if means is None else means[feature][layer, head]
                by_layer.setdefault(layer, []).append((positions[feature], head, vector))
        hooks = {l: Replace(a, self.num_heads, self.head_dim) for l, a in by_layer.items()}
        logits = self._forward(inputs, hooks)
        if not all(h.called for h in hooks.values()):
            raise RuntimeError("An ablation hook was not called.")
        return score_answer(logits, last, self.answers, row["target_present"])


# --------------------------------------------------------------------------
# Head sets
# --------------------------------------------------------------------------
def load_ranking(path, num_layers, num_heads) -> List[Tuple[int, int]]:
    ranking = pd.read_csv(path).sort_values("rank")
    heads = [(int(r.layer), int(r.head)) for r in ranking.itertuples()]
    if len(heads) != num_layers * num_heads or max(l for l, _ in heads) >= num_layers \
            or max(h for _, h in heads) >= num_heads:
        raise ValueError(f"{path} does not match a {num_layers} x {num_heads} model.")
    return heads


def build_specs(args, runner) -> List[Dict[str, Any]]:
    total = runner.num_layers * runner.num_heads
    count = max(1, int(np.ceil(total * args.percentage / 100.0)))
    rank_paths = {
        "color": args.color_ranking or os.path.join(HERE, "head_rankings", f"{args.model}_color_heads.csv"),
        "shape": args.shape_ranking or os.path.join(HERE, "head_rankings", f"{args.model}_shape_heads.csv"),
    }
    selections = {"ranked": {f: load_ranking(rank_paths[f], runner.num_layers, runner.num_heads)[:count]
                             for f in FEATURES}}
    for seed in args.random_seeds:
        selections[f"random_s{seed}"] = {}
        for feature in FEATURES:
            order = np.random.default_rng([seed, RANDOM_STREAM[feature]]).permutation(total)[:count]
            selections[f"random_s{seed}"][feature] = [
                (int(k // runner.num_heads), int(k % runner.num_heads)) for k in order]

    specs = []
    for selection, heads in selections.items():
        for head_set, features in (("color", ["color"]), ("shape", ["shape"]),
                                   ("color+shape", ["color", "shape"])):
            for ablation_type in args.ablation_types:
                specs.append({
                    "spec_id": f"{head_set}|{selection}|{ablation_type}",
                    "head_set": head_set,
                    "selection": "ranked" if selection == "ranked" else "random",
                    "random_seed": None if selection == "ranked" else int(selection.split("_s")[1]),
                    "ablation_type": ablation_type,
                    "heads": {f: heads[f] for f in features},
                })
    overlap = len(set(selections["ranked"]["color"]) & set(selections["ranked"]["shape"]))
    print(f"Top {args.percentage}% = {count} of {total} heads per set "
          f"(color/shape overlap: {overlap})\n  color: {rank_paths['color']}\n  shape: {rank_paths['shape']}")
    return specs


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_manifest(args) -> pd.DataFrame:
    df = pd.read_csv(args.manifest)
    missing = {"image_path", "condition", "target_present"} - set(df.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns {sorted(missing)}.")
    base = os.path.dirname(os.path.abspath(args.manifest))
    df["image_path"] = [p if os.path.isabs(p) else os.path.join(base, p) for p in df.image_path]
    df["target_present"] = df.target_present.map(
        lambda v: v if isinstance(v, (bool, np.bool_)) else str(v).strip().lower() in ("true", "1", "yes"))
    if "prompt" not in df.columns or df.prompt.isna().any():
        if not {"color_word", "shape_word"} <= set(df.columns):
            raise ValueError("Give a prompt column, or color_word and shape_word columns.")
        template = open(args.prompt_template).read().strip() if args.prompt_template else DEFAULT_PROMPT
        built = [template.format(color=c, shape=s) for c, s in zip(df.color_word, df.shape_word)]
        df["prompt"] = df["prompt"].fillna(pd.Series(built)) if "prompt" in df.columns else built
    for feature in FEATURES:
        if args.__dict__[f"{feature}_offset"] is None and f"{feature}_word" not in df.columns:
            raise ValueError(f"Give a {feature}_word column or --{feature}_offset.")
    if args.conditions:
        df = df[df.condition.isin(args.conditions)]
    if args.max_images_per_condition:
        df = df.groupby("condition", group_keys=False).head(args.max_images_per_condition)
    return df.reset_index(drop=True)


def open_image(path):
    with Image.open(path) as handle:
        return handle.convert("RGB")


# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------
def summarize(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df.error.isna()].copy()
    df["random_seed"] = df.random_seed.fillna(-1).astype(int)
    keys = ["condition", "head_set", "selection", "random_seed", "ablation_type"]
    agg = dict(n=("image_path", "size"),
               baseline_accuracy=("baseline_correct", "mean"),
               ablated_accuracy=("ablated_correct", "mean"),
               mean_baseline_margin=("baseline_margin", "mean"),
               mean_ablated_margin=("ablated_margin", "mean"),
               mean_delta_margin=("delta_margin", "mean"))
    seed_avg = df[df.selection == "random"].assign(random_seed=-2)
    frames = []
    for split, mask in (("pooled", None), ("present", True), ("absent", False)):
        for source in (df, seed_avg):
            sub = source if mask is None else source[source.target_present == mask]
            if not sub.empty:
                frames.append(sub.groupby(keys, as_index=False).agg(**agg).assign(split=split))
    out = pd.concat(frames, ignore_index=True)
    # balanced accuracy: mean of present and absent accuracy (insensitive to a yes/no bias)
    acc = out[out.split != "pooled"].pivot_table(index=keys, columns="split",
                                                 values=["baseline_accuracy", "ablated_accuracy"])
    if {"present", "absent"} <= set(acc.columns.get_level_values(1)):
        bal = pd.DataFrame({
            "balanced_baseline_accuracy": acc["baseline_accuracy"][["present", "absent"]].mean(axis=1),
            "balanced_ablated_accuracy": acc["ablated_accuracy"][["present", "absent"]].mean(axis=1),
        }).reset_index().assign(split="pooled")
        out = out.merge(bal, on=keys + ["split"], how="left")
    cols = ["split"] + keys + [c for c in out.columns if c not in keys + ["split"]]
    return out[cols].sort_values(["condition", "split", "head_set", "selection",
                                  "ablation_type", "random_seed"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    manifest = load_manifest(args)
    counts = manifest.groupby(["condition", "target_present"]).size().unstack(fill_value=0)
    print(f"Manifest: {len(manifest)} images\n{counts.to_string()}")

    model, processor = load_vlm(MODELS.get(args.model, args.model), args.cache_dir)
    runner = Runner(model, processor, args.color_offset, args.shape_offset)
    print(f"{runner.num_layers} layers x {runner.num_heads} heads (head_dim {runner.head_dim})")
    specs = build_specs(args, runner)
    with open(os.path.join(args.out_dir, "head_sets.json"), "w") as f:
        json.dump(specs, f, indent=1)

    # ---- pass 1: baselines and summed head outputs (for mean ablation) ----
    base_path = os.path.join(args.out_dir, "baseline.csv")
    sums_path = os.path.join(args.out_dir, "act_sums.pt")
    if os.path.exists(base_path) and os.path.exists(sums_path):
        baseline = pd.read_csv(base_path)
        sums = torch.load(sums_path)
        print(f"Loaded baselines from {base_path}")
    else:
        rows, sums = [], None
        for row in tqdm(manifest.to_dict("records"), desc="baseline"):
            result, acts, tokens, offsets = runner.baseline(open_image(row["image_path"]), row)
            if sums is None:
                sums = {f: torch.zeros_like(a, dtype=torch.float64) for f, a in acts.items()}
                print(f"Ablated tokens for the first image: color {tokens['color']!r} "
                      f"(offset {offsets['color']}), shape {tokens['shape']!r} (offset {offsets['shape']})")
            for f in FEATURES:
                sums[f] += acts[f].double()
            rows.append({**{k: row[k] for k in ("image_path", "condition", "target_present")},
                         "baseline_correct": result["correct"],
                         "baseline_margin": result["correct_logit_margin"],
                         "color_token": tokens["color"], "shape_token": tokens["shape"],
                         "color_offset": offsets["color"], "shape_offset": offsets["shape"]})
        baseline = pd.DataFrame(rows)
        baseline.to_csv(base_path, index=False)
        torch.save({"sums": sums, "n_images": len(baseline)}, sums_path)
        sums = {"sums": sums, "n_images": len(baseline)}
        print("Baseline accuracy by condition:\n"
              + baseline.groupby(["condition", "target_present"]).baseline_correct.mean().to_string())
        print(f"Ablated color tokens: {baseline.color_token.value_counts().to_dict()}\n"
              f"Ablated shape tokens: {baseline.shape_token.value_counts().to_dict()}")
    means = {f: (s / sums["n_images"]).float() for f, s in sums["sums"].items()}

    # ---- pass 2: ablations -------------------------------------------------
    rows_path = os.path.join(args.out_dir, "ablations.csv")
    rows: List[Dict[str, Any]] = []
    done = set()
    if os.path.exists(rows_path):
        previous = pd.read_csv(rows_path)
        previous = previous[previous.error.isna()]
        rows, done = previous.to_dict("records"), set(zip(previous.image_path, previous.spec_id))
        print(f"Resuming: {len(done)} rows already done")
    prompts = dict(zip(manifest.image_path, manifest.to_dict("records")))
    for i, b in enumerate(tqdm(baseline.to_dict("records"), desc="ablations")):
        todo = [s for s in specs if (b["image_path"], s["spec_id"]) not in done]
        if not todo:
            continue
        row, image = prompts[b["image_path"]], open_image(b["image_path"])
        for spec in todo:
            out = {**b, **{k: spec[k] for k in ("spec_id", "head_set", "selection",
                                                "random_seed", "ablation_type")},
                   "percentage": args.percentage,
                   "n_heads": sum(len(v) for v in spec["heads"].values())}
            try:
                result = runner.ablate(image, row, spec["heads"],
                                       means if spec["ablation_type"] == "mean" else None)
                out.update(ablated_correct=result["correct"],
                           ablated_margin=result["correct_logit_margin"], error=None)
                out["delta_margin"] = out["ablated_margin"] - out["baseline_margin"]
            except Exception as error:  # errored rows are retried on resume
                out["error"] = repr(error)
            rows.append(out)
        if (i + 1) % args.save_every == 0:
            pd.DataFrame(rows).to_csv(rows_path, index=False)
    results = pd.DataFrame(rows)
    results.to_csv(rows_path, index=False)
    print(f"Saved {rows_path} ({len(results)} rows, {int(results.error.notna().sum())} errors)")

    summary = summarize(results)
    summary.to_csv(os.path.join(args.out_dir, "summary.csv"), index=False)
    show = summary[(summary.split == "pooled")
                   & ((summary.selection == "ranked") | (summary.random_seed == -2))]
    print(show[["condition", "head_set", "selection", "ablation_type", "n", "baseline_accuracy",
                "ablated_accuracy", "mean_delta_margin"]].to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True,
                   help=f"One of {list(MODELS)} (uses head_rankings/<model>_*.csv), "
                        "or any Hugging Face model id together with --color_ranking/--shape_ranking.")
    p.add_argument("--manifest", required=True, help="CSV of images; see README.")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--conditions", nargs="+", default=None, help="Subset of manifest conditions.")
    p.add_argument("--percentage", type=float, default=5, help="Top P%% of heads per set.")
    p.add_argument("--ablation_types", nargs="+", default=["zero", "mean"], choices=["zero", "mean"])
    p.add_argument("--random_seeds", nargs="+", type=int, default=[0, 1])
    p.add_argument("--color_ranking", default=None)
    p.add_argument("--shape_ranking", default=None)
    p.add_argument("--prompt_template", default=None,
                   help="Text file with {color} and {shape} placeholders; overrides the default prompt.")
    p.add_argument("--color_offset", type=int, default=None,
                   help="Ablate at a fixed offset from the last input token instead of finding color_word.")
    p.add_argument("--shape_offset", type=int, default=None,
                   help="Ablate at a fixed offset from the last input token instead of finding shape_word.")
    p.add_argument("--max_images_per_condition", type=int, default=None, help="For quick tests.")
    p.add_argument("--cache_dir", default=None, help="Hugging Face cache dir (default: $HF_HOME).")
    p.add_argument("--save_every", type=int, default=10, help="Save ablation rows every N images.")
    main(p.parse_args())
