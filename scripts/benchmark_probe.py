# benchmark_probe.py

import os
import re
import argparse
import random

import numpy as np
import pandas as pd
import torch

from PIL import Image
from tqdm import tqdm

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

from search_utils import (
    load_vlm,
    get_prompt,
    build_inputs,
)

# ==================================================
# CONFIG
# ==================================================

DATASET_DIR = (
    "/oscar/data/epavlick/aparasel/"
    "Binding-Problem-Brown/search-dataset/"
    "visual_search_dataset"
)

SEED = 42

random.seed(SEED)
np.random.seed(SEED)

# ==================================================
# PARSE
# ==================================================

def parse_filename(filename):

    pattern = (
        r"(disjunctive|conjunctive)"
        r"_distractors_(\d+)"
        r"_trial_(\d+)"
        r"_target_(true|false)\.png"
    )

    m = re.match(pattern, filename)

    if m is None:
        raise ValueError(filename)

    return {
        "condition": m.group(1),
        "num_distractors": int(m.group(2)),
        "trial": int(m.group(3)),
        "target_present": (
            m.group(4) == "true"
        ),
    }

# ==================================================
# SPLIT
# ==================================================

def make_split(df):

    rng = np.random.RandomState(SEED)

    train_idx = []
    test_idx = []

    for nd in sorted(df.num_distractors.unique()):

        for target in [False, True]:

            subset = df[
                (df.num_distractors == nd)
                &
                (df.target_present == target)
            ]

            idx = subset.index.values.copy()
            if len(subset) != 50:
                raise ValueError(
                    f"Expected 50 examples for "
                    f"num_distractors={nd}, target_present={target}, "
                    f"but got {len(subset)}"
                )

            rng.shuffle(idx)

            train_idx.extend(idx[:25])
            test_idx.extend(idx[25:])

    return train_idx, test_idx

# ==================================================
# MAIN
# ==================================================

def main(args):

    model, processor = load_vlm(args.model_name)

    model.eval()
    model_device = next(model.parameters()).device
    # ------------------------------------------------

    image_paths = []

    for condition in [
        "disjunctive",
        "conjunctive",
    ]:

        folder = os.path.join(
            DATASET_DIR,
            condition,
        )

        for fname in sorted(os.listdir(folder)):

            if fname.endswith(".png"):

                image_paths.append(
                    os.path.join(folder, fname)
                )

    print(
        f"Found {len(image_paths)} images"
    )

    # ------------------------------------------------

    layer_features = None

    metadata = []

    with torch.no_grad():

        for path in tqdm(image_paths):

            meta = parse_filename(
                os.path.basename(path)
            )

            image = Image.open(path).convert("RGB")

            prompt = get_prompt(meta["condition"])

            inputs = build_inputs(
                processor=processor,
                model_device=model_device,
                image=image,
                prompt=prompt,
            )

            outputs = model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )

            if layer_features is None:
                print(f"Num hidden states: {len(outputs.hidden_states)}")
                print(f"Last layer shape: {outputs.hidden_states[-1].shape}")
            if layer_features is None:

                num_layers = len(
                    outputs.hidden_states
                )

                layer_features = [
                    []
                    for _ in range(num_layers)
                ]

            last_token_idx = (
                inputs["attention_mask"]
                .sum(dim=1)
                .item()
                - 1
            )

            if layer_features is None:
                print(f"Last token index: {last_token_idx}")
                print(f"Sequence length: {outputs.hidden_states[-1].shape[1]}")
            for layer, h in enumerate(
                outputs.hidden_states
            ):


                layer_features[layer].append(
                    h[0, last_token_idx]
                    .float()
                    .cpu()
                    .numpy()
                )

            metadata.append(meta)

    metadata = pd.DataFrame(metadata)

    # ==================================================
    # PROBES
    # ==================================================

    results = []

    for condition in [
        "disjunctive",
        "conjunctive",
    ]:

        print()
        print("=" * 60)
        print(condition.upper())
        print("=" * 60)

        for nd in sorted(metadata.num_distractors.unique()):

            subset = metadata[
                (metadata.condition == condition)
                &
                (metadata.num_distractors == nd)
            ].copy()

            if len(subset) != 100:
                raise ValueError(
                    f"Expected 100 examples for "
                    f"{condition}, {nd} distractors "
                    f"but got {len(subset)}"
                )

            rng = np.random.RandomState(SEED)

            present = subset[
                subset.target_present
            ]

            absent = subset[
                ~subset.target_present
            ]

            if len(present) != 50 or len(absent) != 50:
                raise ValueError(
                    f"Expected 50 present and 50 absent "
                    f"for {condition}, {nd}"
                )

            present_idx = present.index.values.copy()
            absent_idx = absent.index.values.copy()

            rng.shuffle(present_idx)
            rng.shuffle(absent_idx)

            train_global = np.concatenate([
                present_idx[:25],
                absent_idx[:25],
            ])

            test_global = np.concatenate([
                present_idx[25:],
                absent_idx[25:],
            ])

            y_train = (
                metadata.loc[
                    train_global,
                    "target_present",
                ]
                .astype(int)
                .values
            )

            y_test = (
                metadata.loc[
                    test_global,
                    "target_present",
                ]
                .astype(int)
                .values
            )

            print(f"\n{nd} distractors")

            for layer in range(num_layers):

                X = np.stack(
                    layer_features[layer]
                )

                X_train = X[train_global]
                X_test = X[test_global]

                scaler = StandardScaler()

                X_train = scaler.fit_transform(
                    X_train
                )

                X_test = scaler.transform(
                    X_test
                )

                probe = LogisticRegression(
                    max_iter=5000,
                    random_state=SEED,
                )

                probe.fit(
                    X_train,
                    y_train,
                )

                train_pred = probe.predict(X_train)
                test_pred = probe.predict(X_test)

                train_acc = accuracy_score(
                    y_train,
                    train_pred,
                )

                test_acc = accuracy_score(
                    y_test,
                    test_pred,
                )

                train_bal_acc = balanced_accuracy_score(
                    y_train,
                    train_pred,
                )

                test_bal_acc = balanced_accuracy_score(
                    y_test,
                    test_pred,
                )

                print(
                    f"Layer {layer:2d} | "
                    f"Train {train_bal_acc:.3f} | "
                    f"Test {test_bal_acc:.3f}"
                )

                results.append({

                    "condition": condition,
                    "num_distractors": nd,
                    "layer": layer,

                    "train_acc": train_acc,
                    "test_acc": test_acc,

                    "train_bal_acc": train_bal_acc,
                    "test_bal_acc": test_bal_acc,

                })

    results = pd.DataFrame(results)

    results.to_csv(
        args.out_path,
        index=False,
    )

    print()
    print(results)

# ==================================================
# CLI
# ==================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct",
    )

    parser.add_argument(
        "--out_path",
        type=str,
        default="probe_results.csv",
    )

    args = parser.parse_args()

    main(args)