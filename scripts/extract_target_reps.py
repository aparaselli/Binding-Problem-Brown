# extract_target_reps.py

import os
import re
import json
import argparse

import torch
from PIL import Image
from tqdm import tqdm

from search_utils import (
    load_vlm,
    get_prompt,
    build_inputs,
)

DATASET_DIR = (
    "/oscar/data/epavlick/aparasel/"
    "Binding-Problem-Brown/search-dataset/"
    "visual_search_dataset"
)

# ==================================================
# FILENAME PARSING
# ==================================================

def parse_filename(filename):

    pattern = (
        r"(disjunctive|conjunctive)"
        r"_distractors_(\d+)"
        r"_trial_(\d+)"
        r"_target_(true|false)\.png"
    )

    m = re.match(pattern, filename)

    return {
        "condition": m.group(1),
        "num_distractors": int(m.group(2)),
        "trial": int(m.group(3)),
        "target_present": (
            m.group(4) == "true"
        ),
    }

# ==================================================
# TARGET POSITION
# ==================================================

def load_target_position(image_path):

    json_path = image_path.replace(
        ".png",
        ".json",
    )

    with open(json_path, "r") as f:
        meta = json.load(f)

    target = [
        x
        for x in meta["objects"]
        if x["role"] == "target"
    ]

    if len(target) == 0:
        return None

    return target[0]["position"]

# ==================================================
# TARGET PATCHES
# ==================================================

def get_target_patch_indices(
    target_pos,
    grid_h,
    grid_w,
):

    if target_pos is None:
        return None

    x, y = target_pos

    # map to [0,1]
    x = (x - 0.5) / 9.0
    y = (y - 0.5) / 9.0

    row = int(y * grid_h)
    col = int(x * grid_w)

    row = max(
        0,
        min(row, grid_h - 1)
    )

    col = max(
        0,
        min(col, grid_w - 1)
    )

    patch_indices = []

    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:

            r = row + dr
            c = col + dc

            if (
                r < 0
                or r >= grid_h
                or c < 0
                or c >= grid_w
            ):
                continue

            idx = r * grid_w + c

            patch_indices.append(idx)

    return sorted(
        list(set(patch_indices))
    )

# ==================================================
# MAIN
# ==================================================

def main(args):

    model, processor = load_vlm(
        args.model_name
    )

    model.eval()

    model_device = next(
        model.parameters()
    ).device

    image_paths = []

    for condition in [
        "disjunctive",
        "conjunctive",
    ]:

        folder = os.path.join(
            DATASET_DIR,
            condition,
        )

        for fname in sorted(
            os.listdir(folder)
        ):

            if fname.endswith(".png"):

                image_paths.append(
                    os.path.join(
                        folder,
                        fname,
                    )
                )

    records = []

    with torch.no_grad():

        for path in tqdm(image_paths):

            meta = parse_filename(
                os.path.basename(path)
            )

            if not meta[
                "target_present"
            ]:
                continue

            image = Image.open(
                path
            ).convert("RGB")

            prompt = get_prompt(
                meta["condition"]
            )

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
            # ==================================================
            # IMAGE GRID
            # ==================================================

            image_grid = inputs[
                "image_grid_thw"
            ][0]

            # original grid (34 x 34)
            grid_h = int(image_grid[1])
            grid_w = int(image_grid[2])

            # Qwen merges patches spatially
            merge_size = (
                processor.image_processor
                .merge_size
            )

            grid_h //= merge_size
            grid_w //= merge_size

            num_vision_tokens = (
                grid_h * grid_w
            )

            patch_indices = (
                get_target_patch_indices(
                    load_target_position(path),
                    grid_h,
                    grid_w,
                )
            )

            if patch_indices is None:
                continue
            assert (
                max(patch_indices)
                < num_vision_tokens
            )

            assert (
                min(patch_indices)
                >= 0
            )

            # Qwen places vision tokens first
            # Qwen puts vision tokens first
            vision_start = 0

            for layer, h in enumerate(
                outputs.hidden_states
            ):

                h = h[0]

                token_idx = [
                    vision_start + p
                    for p in patch_indices
                ]

                assert (
                    max(token_idx)
                    < h.shape[0]
                )

                rep = (
                    h[token_idx]
                    .mean(dim=0)
                    .cpu()
                )
                if len(records) == 0:
                    print(
                        "grid:",
                        grid_h,
                        grid_w,
                    )
                    print(
                        "vision tokens:",
                        num_vision_tokens,
                    )
                    print(
                        "sequence length:",
                        outputs.hidden_states[-1].shape[1]
                    )
                    print(
                        "patches:",
                        patch_indices
                    )

                records.append({

                    "condition":
                        meta["condition"],

                    "num_distractors":
                        meta["num_distractors"],

                    "trial":
                        meta["trial"],

                    "target_present":
                        True,

                    "layer":
                        layer,

                    "target_rep":
                        rep,

                    "target_patch_indices":
                        patch_indices,

                })

    torch.save(
        records,
        args.out_path,
    )

    print(
        f"Saved {len(records)} "
        f"representations."
    )

# ==================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        default="Qwen/Qwen2.5-VL-7B-Instruct",
    )

    parser.add_argument(
        "--out_path",
        required=True,
    )

    args = parser.parse_args()

    main(args)