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


def parse_filename(filename):

    pattern = (
        r"(disjunctive|conjunctive)"
        r"_distractors_(\d+)"
        r"_trial_(\d+)"
        r"_target_(true|false)\.png"
    )

    match = re.match(pattern, filename)

    if match is None:
        raise ValueError(f"Could not parse: {filename}")

    condition = match.group(1)
    num_distractors = int(match.group(2))
    trial = int(match.group(3))
    target_present = match.group(4) == "true"

    return {
        "condition": condition,
        "num_distractors": num_distractors,
        "trial": trial,
        "target_present": target_present,
    }


def load_target_metadata(image_path):

    json_path = image_path.replace(".png", ".json")

    if not os.path.exists(json_path):
        return {
            "target_x": None,
            "target_y": None,
        }

    with open(json_path, "r") as f:
        meta = json.load(f)

    target_objects = [
        obj for obj in meta["objects"]
        if obj["role"] == "target"
    ]

    if len(target_objects) == 0:
        return {
            "target_x": None,
            "target_y": None,
        }

    target = target_objects[0]
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

    for condition in ["disjunctive", "conjunctive"]:

        condition_dir = os.path.join(
            DATASET_DIR,
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


    summary = (
        df
        .groupby(
            [
                "condition",
                "num_distractors",
            ]
        )["correct"]
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
        default="regular",
    )


    args = parser.parse_args()

    main(args)