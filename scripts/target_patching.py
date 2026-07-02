# patch_target_with_5dist.py

import os
import re
import json
import argparse

import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm

from search_utils import (
    load_vlm,
    get_prompt,
    build_inputs,
    extract_answer,
)

from PIL import Image, ImageDraw

import numpy as np
import matplotlib.pyplot as plt

import io
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def render_image_from_metadata(
    metadata,
    keep_only_target=False,
):
    fig, ax = plt.subplots(
        figsize=(6, 6),
        dpi=100,
    )

    ax.set_xlim(0.5, 9.5)
    ax.set_ylim(0.5, 9.5)
    ax.axis("off")

    for obj in metadata["objects"]:

        if (
            keep_only_target
            and obj["role"] != "target"
        ):
            continue

        x, y = obj["position"]

        shape = obj["shape"]
        color = obj["color"]


        if shape == "circle":

            ax.plot(
                x,
                y,
                marker="o",
                color=color,
                markersize=14,
                linestyle="None",
            )


        else:

            ax.text(
                x,
                y,
                shape,
                color=color,
                fontsize=22,
                ha="center",
                va="center",
                fontweight="bold",
            )

    buf = io.BytesIO()

    plt.savefig(
        buf,
        format="png",
        bbox_inches="tight",
        pad_inches=0.1,
    )

    buf.seek(0)

    image = Image.open(buf).convert(
        "RGB"
    )

    plt.close(fig)
    buf.close()

    return image

def create_clean_counterfactual_image(
    target_pos,
    condition,
):

    fig, ax = plt.subplots(
        figsize=(6, 6),
        dpi=100,
    )

    ax.set_xlim(0.5, 9.5)
    ax.set_ylim(0.5, 9.5)
    ax.axis("off")

    x, y = target_pos

    if condition == "conjunctive":

        ax.text(
            x,
            y,
            "L",
            color="green",
            fontsize=22,
            ha="center",
            va="center",
            fontweight="bold",
        )

    else:

        ax.plot(
            x,
            y,
            marker="o",
            color="green",
            markersize=14,
            linestyle="None",
        )

    import io

    buf = io.BytesIO()

    plt.savefig(
        buf,
        format="png",
        bbox_inches="tight",
        pad_inches=0.1,
    )

    buf.seek(0)

    image = Image.open(buf).convert("RGB")

    plt.close(fig)

    return image

DATASET_DIR = (
    "/oscar/data/epavlick/aparasel/"
    "Binding-Problem-Brown/search-dataset/"
    "visual_search_dataset"
)


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
        "target_present": m.group(4) == "true",
    }


def load_target_position(image_path):

    json_path = image_path.replace(".png", ".json")

    with open(json_path, "r") as f:
        meta = json.load(f)

    targets = [
        obj for obj in meta["objects"]
        if obj["role"] == "target"
    ]

    if len(targets) == 0:
        return None

    return targets[0]["position"]


def get_target_patch_indices(target_pos, grid_h, grid_w):

    if target_pos is None:
        return None

    x, y = target_pos

    x = (x - 0.5) / 9.0
    y = (y - 0.5) / 9.0

    row = int(y * grid_h)
    col = int(x * grid_w)

    row = max(0, min(row, grid_h - 1))
    col = max(0, min(col, grid_w - 1))

    patch_indices = []

    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:

            r = row + dr
            c = col + dc

            if r < 0 or r >= grid_h:
                continue

            if c < 0 or c >= grid_w:
                continue

            patch_indices.append(
                r * grid_w + c
            )

    return sorted(list(set(patch_indices)))



def get_decoder_layers(model):

    candidates = [
        ["model", "layers"],
        ["model", "model", "layers"],
        ["language_model", "model", "layers"],
        ["model", "language_model", "layers"],
    ]

    for path in candidates:

        obj = model

        try:
            for attr in path:
                obj = getattr(obj, attr)

            return obj

        except AttributeError:
            pass

    raise RuntimeError(
        "Could not find transformer layers."
    )



def load_five_dist_means(rep_path):

    records = torch.load(rep_path)
    df = pd.DataFrame(records)

    df = df[
        (df["condition"] == "conjunctive")
        &
        (df["num_distractors"] == 5)
    ]

    means = {}

    for layer in sorted(df["layer"].unique()):

        reps = torch.stack(
            df[df["layer"] == layer]["target_rep"].tolist()
        ).float()

        means[int(layer)] = reps.mean(dim=0)

    return means

@torch.no_grad()
def extract_clean_target_reps(
    model,
    processor,
    image,
    condition,
    patch_indices,
):

    prompt = get_prompt(condition)

    model_device = next(
        model.parameters()
    ).device

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

    reps = {}

    idx = torch.tensor(
        patch_indices,
        device=model_device,
        dtype=torch.long,
    )

    for layer, h in enumerate(
        outputs.hidden_states
    ):

        h = h[0]

        reps[layer] = (
            h[idx]
            .mean(dim=0)
            .detach()
            .cpu()
        )

    return reps

@torch.no_grad()
def patched_search_model_call(
    model,
    processor,
    image,
    condition,
    patch_indices,
    target_means,
    max_new_tokens=256,
):

    prompt = get_prompt(condition)

    model_device = next(model.parameters()).device

    inputs = build_inputs(
        processor=processor,
        model_device=model_device,
        image=image,
        prompt=prompt,
    )

    layers = get_decoder_layers(model)

    idx = torch.tensor(
        patch_indices,
        device=model_device,
        dtype=torch.long,
    )

    handles = []

    def make_hook(block_idx):

        mean_layer = block_idx + 1

        patch_vec = (
            target_means[mean_layer]
            .to(model_device)
        )

        def hook(module, module_input, output):

            if isinstance(output, tuple):
                h = output[0]
                rest = output[1:]
            else:
                h = output
                rest = None

            if h.dim() == 3 and h.shape[1] > idx.max().item():

                h = h.clone()

                h[:, idx, :] = patch_vec.to(
                    dtype=h.dtype
                ).view(1, 1, -1)

            if rest is None:
                return h

            return (h,) + rest

        return hook

    for block_idx, layer in enumerate(layers):

        handle = layer.register_forward_hook(
            make_hook(block_idx)
        )

        handles.append(handle)

    try:

        gen_out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )

    finally:

        for handle in handles:
            handle.remove()

    generated_ids = gen_out.sequences

    input_len = inputs["input_ids"].shape[-1]

    new_token_ids = generated_ids[:, input_len:]

    response = processor.batch_decode(
        new_token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    return response



def main(args):

    model, processor = load_vlm(args.model_name)

    model.eval()


    image_paths = []

    condition_dir = os.path.join(
        DATASET_DIR,
        "conjunctive",
    )

    for fname in sorted(os.listdir(condition_dir)):

        if not fname.endswith(".png"):
            continue

        meta = parse_filename(fname)

        if not meta["target_present"]:
            continue

        image_paths.append(
            os.path.join(condition_dir, fname)
        )

    print(f"Found {len(image_paths)} conjunctive target-present images")

    results = []

    for idx_img, path in enumerate(tqdm(image_paths)):

        fname = os.path.basename(path)
        meta = parse_filename(fname)

        try:

            with Image.open(path) as img:
                image = img.convert("RGB")

            prompt = get_prompt(meta["condition"])

            model_device = next(model.parameters()).device

            tmp_inputs = build_inputs(
                processor=processor,
                model_device=model_device,
                image=image,
                prompt=prompt,
            )

            image_grid = tmp_inputs["image_grid_thw"][0]

            grid_h = int(image_grid[1])
            grid_w = int(image_grid[2])

            merge_size = processor.image_processor.merge_size

            grid_h //= merge_size
            grid_w //= merge_size

            target_pos = load_target_position(path)

            patch_indices = get_target_patch_indices(
                target_pos,
                grid_h,
                grid_w,
            )

            assert patch_indices is not None
            assert max(patch_indices) < grid_h * grid_w
            with open(
                path.replace(".png", ".json"),
                "r",
            ) as f:
                metadata = json.load(f)

            clean_image = render_image_from_metadata(
                metadata,
                keep_only_target=True,
            )

            clean_target_reps = (
                extract_clean_target_reps(
                    model=model,
                    processor=processor,
                    image=clean_image,
                    condition=meta["condition"],
                    patch_indices=patch_indices,
                )
            )

            response = patched_search_model_call(
                model=model,
                processor=processor,
                image=image,
                condition=meta["condition"],
                patch_indices=patch_indices,
                target_means=clean_target_reps,
                max_new_tokens=args.max_new_tokens,
            )

            prediction = extract_answer(response)

            correct = (
                prediction
                ==
                meta["target_present"]
            )

        except Exception as e:

            print(f"ERROR on {fname}: {e}")

            response = None
            prediction = None
            correct = False
            patch_indices = None
            target_pos = None

        results.append({

            "image_path": path,

            "condition": meta["condition"],

            "num_distractors": meta["num_distractors"],

            "trial": meta["trial"],

            "target_present": meta["target_present"],

            "prediction": prediction,

            "correct": correct,

            "response": response,

            "patched_with":
                "counterfactual_clean_target",

            "target_patch_indices": patch_indices,

            "target_x": None if target_pos is None else target_pos[0],

            "target_y": None if target_pos is None else target_pos[1],

        })

        if idx_img == 0:
            clean_image.save(
                "debug_clean_image.png"
            )

        if (idx_img + 1) % 25 == 0:

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
        df.groupby("num_distractors")["correct"]
        .mean()
        .reset_index()
    )

    print("\nAccuracy Summary")
    print(summary)

    summary_path = args.out_path.replace(
        ".csv",
        "_summary.csv",
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
        "--max_new_tokens",
        type=int,
        default=256,
    )

    args = parser.parse_args()

    main(args)
