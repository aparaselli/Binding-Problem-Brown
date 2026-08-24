# benchmark.py

import os
import re
import argparse
import json
import numpy as np

import pandas as pd

from PIL import Image
from tqdm import tqdm

from search_utils import (
    load_vlm,
    search_model_call,
    extract_answer,
)



SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATASET_DIR = os.path.join(
    SCRIPT_DIR,
    "..",
    "search-dataset",
    "visual_search_dataset_counterfactual",
)

DATASET_DIR = os.path.abspath(DATASET_DIR)


DATASET_DIR_MINIMAL = os.path.join(
    SCRIPT_DIR,
    "..",
    "search-dataset",
    "minimal_pair_swap_dataset",
)

DATASET_DIR_MINIMAL = os.path.abspath(DATASET_DIR_MINIMAL)



def parse_filename(filename):
    # Match Minimal Pair Swap: pair_000_original.png or pair_000_counterfactual_swap.png
    mp_match = re.match(r"pair_(\d+)_(original|counterfactual_swap)\.png", filename)
    if mp_match:
        return {
            "condition": mp_match.group(2),
            "num_distractors": 0,
            "trial": int(mp_match.group(1)),
            "target_present": False,  # never present in MP_S
        }

    # Match Visual Search: disjunctive_distractors_0_trial_000_target_true.png
    vs_match = re.match(
        r"(disjunctive|conjunctive|disjunctive_T)_distractors_(\d+)_trial_(\d+)_target_(true|false)\.png",
        filename,
    )
    if vs_match:
        return {
            "condition": vs_match.group(1),
            "num_distractors": int(vs_match.group(2)),
            "trial": int(vs_match.group(3)),
            "target_present": vs_match.group(4) == "true",
        }

    raise ValueError(f"Could not parse filename: {filename}")


def load_target_metadata(image_path):
    json_path = image_path.replace(".png", ".json")

    if not os.path.exists(json_path):
        return {
            "target_x": None,
            "target_y": None,
        }

    with open(json_path, "r") as f:
        meta = json.load(f)

    objects = meta.get("objects", [])

    # Check for target role if present (VS_CF dataset)
    target_objects = [
        obj for obj in objects
        if obj.get("role") == "target"
    ]

    # If no target role exists (e.g. MP_S dataset), fallback to the first object
    if len(target_objects) == 0:
        if len(objects) > 0:
            target = objects[0]
        else:
            return {
                "target_x": None,
                "target_y": None,
            }
    else:
        target = target_objects[0]

    x, y = target["position_plot"]

    return {
        "target_x": x,
        "target_y": y,
    }


def compute_target_position_features(target_x, target_y):

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
        +
        (y_norm - 0.5) ** 2
    )

    return {
        "target_x_norm": x_norm,
        "target_y_norm": y_norm,
        "target_dist_from_center": dist,
    }

def main(args):

    model, processor = load_vlm(args.model_name)

    results = []

    image_paths = []

    if args.dataset == "VS_CF":
        final_DS = DATASET_DIR
        condition_list = args.conditions
    elif args.dataset == "MP_S":
        final_DS = DATASET_DIR_MINIMAL
        condition_list = [""]
    else:
        raise ValueError(f"{args.dataset} is an invalid dataset. Please choose VS_CF (visual search coutnerfacts) or MP_S (minimal pairs!")


    for condition in condition_list:

        condition_dir = os.path.join(
            final_DS,
            condition,
        )

        for fname in sorted(os.listdir(condition_dir)):

            if not fname.endswith(".png"):
                continue

            image_paths.append(
                os.path.join(condition_dir, fname)
            )

    print(f"Found {len(image_paths)} images")

    for idx, path in enumerate(
        tqdm(image_paths)
    ):

        fname = os.path.basename(path)

        meta = parse_filename(fname)

        target_meta = load_target_metadata(path)

        target_features = compute_target_position_features(
            target_meta["target_x"],
            target_meta["target_y"],
        )

        try:

            with Image.open(path) as img:
                image = img.convert("RGB")

            result = search_model_call(
                model=model,
                processor=processor,
                image=image,
                condition=meta["condition"],
                prompt_type=args.prompt_type,
            )

            response = result["response"]


            prediction = extract_answer(
                response,
                prompt_type=args.prompt_type,
            )

            correct = (
                prediction
                ==
                meta["target_present"]
            )

        except Exception as e:

            response = None
            prediction = None
            correct = False

            print(
                f"ERROR on {fname}: {e}"
            )

        results.append({
            "image_path": path,

            "condition":
                meta["condition"],

            "num_distractors":
                meta["num_distractors"],

            "trial":
                meta["trial"],

            "target_present":
                meta["target_present"],

            "prediction":
                prediction,

            "correct":
                correct,

            "response":
                response,

            "target_x": target_meta["target_x"],
            "target_y": target_meta["target_y"],

            "target_x_norm": target_features["target_x_norm"],
            "target_y_norm": target_features["target_y_norm"],
            "target_dist_from_center": target_features["target_dist_from_center"],
            "first_token": result["first_token"],
            "first_token_id": result["first_token_id"],
            "first_token_logprob": result["first_token_logprob"],
        })

        # periodic save

        if (idx + 1) % 50 == 0:

            pd.DataFrame(results).to_csv(
                args.out_path,
                index=False,
            )

    df = pd.DataFrame(results)

    df.to_csv(
        args.out_path,
        index=False,
    )

    print("\nSaved:")
    print(args.out_path)

    # Dynamic grouping depending on dataset
    groupby_cols = (
        ["condition", "num_distractors"]
        if args.dataset == "VS_CF"
        else ["condition"]
    )

    summary = (
        df.groupby(groupby_cols)["correct"]
        .mean()
        .reset_index()
    )

    print("\nAccuracy Summary\n")
    print(summary)

    summary_path = (
        args.out_path
        .replace(".csv", "_summary.csv")
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print("\nSaved summary:")
    print(summary_path)


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
        required=True,
    )

    parser.add_argument(
        "--prompt_type",
        type=str,
        default="binary",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="VS_CF", #"VS_CF" == conjunctive/dijunctive coutnerfacutals, "MP_S" == minimal_pair_swap
    )

    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["disjunctive", "conjunctive", "disjunctive_T"],
        help="Conditions to benchmark for VS_CF (e.g. --conditions disjunctive_T or --conditions disjunctive conjunctive disjunctive_T)",
    )
    args = parser.parse_args()

    main(args)