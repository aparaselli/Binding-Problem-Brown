#!/usr/bin/env python3
"""
Worked example: build a manifest for ablate_feature_heads.py from our
color x shape visual-search dataset.  Adapt this to your own dataset; the
ablation script only needs the manifest columns described in the README.

Our layout (one PNG + JSON per image):
    <root>/<color_variant>/<shape_variant>/<condition>/
        <prefix>_distractors_<N>_trial_<TTT>_target_<true|false>.png
        <prefix>_distractors_<N>_trial_<TTT>_target_<true|false>.json
    where the JSON holds "target_color" and "target_shape".

Usage:
    python make_manifest_example.py --root /path/to/dataset \\
        --conditions conjunctive disjunctive --distractors 5 --out manifest.csv
"""

import argparse
import json
import os
import re

import pandas as pd

PATTERN = re.compile(r".*_distractors_(\d+)_trial_(\d+)_target_(true|false)\.png")


def main(args):
    rows = []
    for color_variant in sorted(os.listdir(args.root)):
        if not re.fullmatch(r"[a-z]+_target_[a-z]+_distractor", color_variant):
            continue
        for shape_variant in sorted(os.listdir(os.path.join(args.root, color_variant))):
            for condition in args.conditions:
                folder = os.path.join(args.root, color_variant, shape_variant, condition)
                if not os.path.isdir(folder):
                    continue
                for name in os.listdir(folder):
                    match = PATTERN.fullmatch(name)
                    if not match or int(match.group(1)) != args.distractors:
                        continue
                    path = os.path.abspath(os.path.join(folder, name))
                    with open(path[:-4] + ".json") as handle:
                        meta = json.load(handle)
                    rows.append({
                        "image_path": path,
                        "condition": condition,
                        "target_present": match.group(3) == "true",
                        "color_word": meta["target_color"],
                        "shape_word": meta["target_shape"],
                        "combo": f"{color_variant}/{shape_variant}",
                        "trial": int(match.group(2)),
                    })
    df = pd.DataFrame(rows).sort_values(["condition", "combo", "trial", "target_present"])
    if args.n_per_cell:  # first N trials per (condition, combo, present/absent)
        df = df.groupby(["condition", "combo", "target_present"]).head(args.n_per_cell)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} images to {args.out}")
    print(df.groupby(["condition", "target_present"]).size().unstack().to_string())


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True)
    p.add_argument("--conditions", nargs="+", default=["conjunctive", "disjunctive"])
    p.add_argument("--distractors", type=int, default=5)
    p.add_argument("--n_per_cell", type=int, default=None)
    p.add_argument("--out", default="manifest.csv")
    main(p.parse_args())
