import os
import re
import json
import argparse

import torch
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from search_utils import load_vlm



SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.abspath(
    os.path.join(
        SCRIPT_DIR,
        "..",
        "search-dataset",
        "visual_search_dataset_counterfactual",
    )
)

PATCH_SIZE = 14


def get_model_device(model):
    return next(model.parameters()).device


def parse_filename(filename):
    pattern = (
        r"(disjunctive|conjunctive)"
        r"_distractors_(\d+)"
        r"_trial_(\d+)"
        r"_target_(true|false)\.png"
    )

    match = re.match(pattern, filename)
    if match is None:
        raise ValueError(f"Could not parse filename: {filename}")

    return {
        "condition": match.group(1),
        "num_distractors": int(match.group(2)),
        "trial": int(match.group(3)),
        "target_present": match.group(4) == "true",
    }


def get_merge_size(model):
    try:
        return int(model.config.vision_config.spatial_merge_size)
    except Exception:
        return 2


def load_target_metadata(image_path):

    json_path = image_path.replace(".png", ".json")

    with open(json_path, "r") as f:
        meta = json.load(f)

    target_objects = [
        obj for obj in meta.get("objects", [])
        if obj.get("role") == "target"
    ]

    if len(target_objects) == 0:
        raise ValueError(f"No target object found in {json_path}")

    target = target_objects[0]

    if "position_plot" in target:
        target_x_plot, target_y_plot = target["position_plot"]
    elif "position" in target:
        target_x_plot, target_y_plot = target["position"]
    else:
        raise KeyError(
            f"Target in {json_path} has neither 'position_plot' nor 'position'."
        )

    with Image.open(image_path) as img:
        image_width, image_height = img.size

    image_size_meta = meta.get("image_size", None)

    result = {
        "json_path": json_path,
        "image_width": int(image_width),
        "image_height": int(image_height),
        "image_size_meta": image_size_meta,

        "target_x_plot": float(target_x_plot),
        "target_y_plot": float(target_y_plot),

        # Backward-compatible names used by your previous scripts.
        "target_x": float(target_x_plot),
        "target_y": float(target_y_plot),

        "target_x_px": None,
        "target_y_px": None,
        "bbox_left": None,
        "bbox_top": None,
        "bbox_right": None,
        "bbox_bottom": None,
    }

    if "position_pixel" in target:
        x_px_display, y_px_display = target["position_pixel"]
        result["target_x_px"] = float(x_px_display)
        result["target_y_px"] = float(image_height - y_px_display)

    if "bbox_pixel" in target:
        bbox = target["bbox_pixel"]

        x0 = float(bbox["x0"])
        x1 = float(bbox["x1"])
        y0_display = float(bbox["y0"])
        y1_display = float(bbox["y1"])

        left = x0
        right = x1
        top = image_height - y1_display
        bottom = image_height - y0_display

        result.update({
            "bbox_left": left,
            "bbox_top": top,
            "bbox_right": right,
            "bbox_bottom": bottom,
        })

        if result["target_x_px"] is None:
            result["target_x_px"] = (left + right) / 2.0
        if result["target_y_px"] is None:
            result["target_y_px"] = (top + bottom) / 2.0

    if result["target_x_px"] is None or result["target_y_px"] is None:
        x_px = (float(target_x_plot) - 0.5) / 9.0 * image_width
        y_px = image_height - ((float(target_y_plot) - 0.5) / 9.0 * image_height)

        result["target_x_px"] = x_px
        result["target_y_px"] = y_px

    return result


def build_inputs(path, processor):
    image = Image.open(path).convert("RGB")

    prompt = (
        "Is there a green L present in this image? "
        "Answer only True or False."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
    )

    return image, inputs


def move_inputs_to_device(inputs, device):
    moved = {}
    for k, v in inputs.items():
        if torch.is_tensor(v):
            moved[k] = v.to(device)
        else:
            moved[k] = v
    return moved


def get_image_positions(inputs, model):
    image_token_id = model.config.image_token_id

    image_positions = (
        inputs["input_ids"][0]
        == image_token_id
    ).nonzero(as_tuple=True)[0]

    return image_positions


def get_visual_grid_info(inputs, model):

    merge_size = get_merge_size(model)

    if "image_grid_thw" not in inputs:
        raise KeyError("inputs does not contain image_grid_thw")

    grid_thw = inputs["image_grid_thw"][0].detach().cpu().tolist()
    raw_t, raw_h, raw_w = [int(x) for x in grid_thw]

    if raw_h % merge_size != 0 or raw_w % merge_size != 0:
        raise ValueError(
            f"Raw grid ({raw_h}, {raw_w}) is not divisible by merge_size={merge_size}."
        )

    token_h = raw_h // merge_size
    token_w = raw_w // merge_size

    processed_h = raw_h * PATCH_SIZE
    processed_w = raw_w * PATCH_SIZE

    expected_tokens = raw_t * token_h * token_w

    return {
        "merge_size": merge_size,
        "raw_t": raw_t,
        "raw_h": raw_h,
        "raw_w": raw_w,
        "token_h": token_h,
        "token_w": token_w,
        "processed_h": processed_h,
        "processed_w": processed_w,
        "expected_tokens": expected_tokens,
    }




def clip_int(x, lo, hi):
    return max(lo, min(int(x), hi))


def target_to_visual_tokens(target_meta, grid_info):

    image_w = target_meta["image_width"]
    image_h = target_meta["image_height"]

    raw_h = grid_info["raw_h"]
    raw_w = grid_info["raw_w"]
    token_h = grid_info["token_h"]
    token_w = grid_info["token_w"]
    merge_size = grid_info["merge_size"]
    processed_h = grid_info["processed_h"]
    processed_w = grid_info["processed_w"]

    scale_x = processed_w / image_w
    scale_y = processed_h / image_h

    cx = target_meta["target_x_px"] * scale_x
    cy = target_meta["target_y_px"] * scale_y

    raw_patch_x = clip_int(cx // PATCH_SIZE, 0, raw_w - 1)
    raw_patch_y = clip_int(cy // PATCH_SIZE, 0, raw_h - 1)

    token_x = clip_int(raw_patch_x // merge_size, 0, token_w - 1)
    token_y = clip_int(raw_patch_y // merge_size, 0, token_h - 1)

    center_token_idx = token_y * token_w + token_x
    bbox_available = (
        target_meta["bbox_left"] is not None
        and target_meta["bbox_top"] is not None
        and target_meta["bbox_right"] is not None
        and target_meta["bbox_bottom"] is not None
    )

    if bbox_available:
        left = target_meta["bbox_left"] * scale_x
        right = target_meta["bbox_right"] * scale_x
        top = target_meta["bbox_top"] * scale_y
        bottom = target_meta["bbox_bottom"] * scale_y

        raw_x0 = clip_int(left // PATCH_SIZE, 0, raw_w - 1)
        raw_x1 = clip_int((right - 1e-6) // PATCH_SIZE, 0, raw_w - 1)
        raw_y0 = clip_int(top // PATCH_SIZE, 0, raw_h - 1)
        raw_y1 = clip_int((bottom - 1e-6) // PATCH_SIZE, 0, raw_h - 1)

        token_x0 = clip_int(raw_x0 // merge_size, 0, token_w - 1)
        token_x1 = clip_int(raw_x1 // merge_size, 0, token_w - 1)
        token_y0 = clip_int(raw_y0 // merge_size, 0, token_h - 1)
        token_y1 = clip_int(raw_y1 // merge_size, 0, token_h - 1)

        bbox_token_indices = []
        for y in range(token_y0, token_y1 + 1):
            for x in range(token_x0, token_x1 + 1):
                bbox_token_indices.append(y * token_w + x)
    else:
        bbox_token_indices = [center_token_idx]
        token_x0 = token_x1 = token_x
        token_y0 = token_y1 = token_y
        raw_x0 = raw_x1 = raw_patch_x
        raw_y0 = raw_y1 = raw_patch_y

    return {
        "target_x_processed": float(cx),
        "target_y_processed": float(cy),

        "raw_patch_x": int(raw_patch_x),
        "raw_patch_y": int(raw_patch_y),

        "target_token_x": int(token_x),
        "target_token_y": int(token_y),
        "target_token_idx": int(center_token_idx),

        "bbox_raw_x0": int(raw_x0),
        "bbox_raw_x1": int(raw_x1),
        "bbox_raw_y0": int(raw_y0),
        "bbox_raw_y1": int(raw_y1),

        "bbox_token_x0": int(token_x0),
        "bbox_token_x1": int(token_x1),
        "bbox_token_y0": int(token_y0),
        "bbox_token_y1": int(token_y1),
        "bbox_token_indices": bbox_token_indices,
    }


def get_token_window_indices(token_x, token_y, token_h, token_w):
    indices = []

    for y in range(max(0, token_y - 1), min(token_h, token_y + 2)):
        for x in range(max(0, token_x - 1), min(token_w, token_x + 2)):
            indices.append(y * token_w + x)

    return indices


# visual confimraiton

def visualize_target_tokens(
    image,
    target_meta,
    token_info,
    window_indices,
    grid_info,
):
    token_h = grid_info["token_h"]
    token_w = grid_info["token_w"]

    processed_h = grid_info["processed_h"]
    processed_w = grid_info["processed_w"]

    token_cell_w = processed_w / token_w
    token_cell_h = processed_h / token_h

    # Visualize in the same coordinate system as the actual input image.
    image_for_plot = image.resize((processed_w, processed_h))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image_for_plot)

    for x in range(token_w + 1):
        ax.axvline(x * token_cell_w, color="white", alpha=0.25)

    for y in range(token_h + 1):
        ax.axhline(y * token_cell_h, color="white", alpha=0.25)

    for idx in window_indices:
        y = idx // token_w
        x = idx % token_w

        rect = patches.Rectangle(
            (x * token_cell_w, y * token_cell_h),
            token_cell_w,
            token_cell_h,
            linewidth=2,
            edgecolor="yellow",
            facecolor="none",
        )
        ax.add_patch(rect)

        ax.text(
            x * token_cell_w + token_cell_w / 2,
            y * token_cell_h + token_cell_h / 2,
            str(idx),
            color="white",
            fontsize=8,
            ha="center",
            va="center",
        )

    tx = token_info["target_token_x"]
    ty = token_info["target_token_y"]
    rect = patches.Rectangle(
        (tx * token_cell_w, ty * token_cell_h),
        token_cell_w,
        token_cell_h,
        linewidth=3,
        edgecolor="red",
        facecolor="none",
    )
    ax.add_patch(rect)

    for idx in token_info["bbox_token_indices"]:
        y = idx // token_w
        x = idx % token_w
        rect = patches.Rectangle(
            (x * token_cell_w, y * token_cell_h),
            token_cell_w,
            token_cell_h,
            linewidth=2,
            edgecolor="cyan",
            facecolor="none",
        )
        ax.add_patch(rect)

    ax.scatter(
        token_info["target_x_processed"],
        token_info["target_y_processed"],
        c="magenta",
        s=80,
    )

    ax.set_title(
        f"center token=({tx},{ty}), idx={token_info['target_token_idx']}"
    )
    ax.axis("off")
    plt.show()

#used for debugging

def debug_first_examples(image_paths, model, processor):
    model_device = get_model_device(model)
    merge_size = get_merge_size(model)

    print("\n" + "=" * 80)
    print("RUNNING DEBUG ON FIRST 3 EXAMPLES")
    print("=" * 80)
    print("merge_size:", merge_size)

    for path in image_paths[:3]:
        print("\n" + "=" * 80)
        print(path)
        print("=" * 80)

        target_meta = load_target_metadata(path)
        image, inputs = build_inputs(path, processor)

        grid_info = get_visual_grid_info(inputs, model)
        token_info = target_to_visual_tokens(target_meta, grid_info)

        window_indices = get_token_window_indices(
            token_info["target_token_x"],
            token_info["target_token_y"],
            grid_info["token_h"],
            grid_info["token_w"],
        )

        print("image size:", image.size)
        print("image_grid_thw:", inputs.get("image_grid_thw", None))
        print("grid_info:", grid_info)
        print("target plot position:", target_meta["target_x"], target_meta["target_y"])
        print("target pixel center:", target_meta["target_x_px"], target_meta["target_y_px"])
        print(
            "target bbox:",
            target_meta["bbox_left"],
            target_meta["bbox_top"],
            target_meta["bbox_right"],
            target_meta["bbox_bottom"],
        )
        print("raw patch:", token_info["raw_patch_x"], token_info["raw_patch_y"])
        print("center token:", token_info["target_token_x"], token_info["target_token_y"])
        print("center token idx:", token_info["target_token_idx"])
        print("bbox token indices:", token_info["bbox_token_indices"])
        print("window indices:", window_indices)

        inputs = move_inputs_to_device(inputs, model_device)
        image_positions = get_image_positions(inputs, model)

        print("image_positions shape:", image_positions.shape)
        print("expected:", grid_info["expected_tokens"])

        if len(image_positions) != grid_info["expected_tokens"]:
            raise ValueError(
                f"Expected {grid_info['expected_tokens']} image tokens, "
                f"got {len(image_positions)} for {path}"
            )

        with torch.no_grad():
            outputs = model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )

        hidden_states = outputs.hidden_states

        print("hidden_states len:", len(hidden_states))
        print("hidden_states[0]:", hidden_states[0].shape)
        print("hidden_states[-1]:", hidden_states[-1].shape)

        h = hidden_states[-1][0]
        image_h = h[image_positions]

        center = image_h[token_info["target_token_idx"]]
        window = image_h[
            torch.tensor(window_indices, device=image_h.device, dtype=torch.long)
        ]
        bbox_tokens = image_h[
            torch.tensor(
                token_info["bbox_token_indices"],
                device=image_h.device,
                dtype=torch.long,
            )
        ]

        print("image_h:", image_h.shape)
        print("center:", center.shape)
        print("window:", window.shape)
        print("bbox_tokens:", bbox_tokens.shape)

        visualize_target_tokens(
            image=image,
            target_meta=target_meta,
            token_info=token_info,
            window_indices=window_indices,
            grid_info=grid_info,
        )

        input("\nPress enter for next example...")



def main(args):
    model, processor = load_vlm(args.model_name)
    model.eval()

    model_device = get_model_device(model)

    image_paths = []

    for condition in ["disjunctive", "conjunctive"]:
        folder = os.path.join(DATASET_DIR, condition)

        for fname in sorted(os.listdir(folder)):
            if fname.endswith(".png") and "target_true" in fname:
                image_paths.append(os.path.join(folder, fname))

    print(f"Found {len(image_paths)} target-present images.")
    print("merge_size:", get_merge_size(model))

    if args.debug:
        debug_first_examples(image_paths, model, processor)
        return

    if args.max_examples is not None:
        image_paths = image_paths[:args.max_examples]

    results = {
        "image_paths": [],
        "conditions": [],
        "num_distractors": [],
        "trials": [],

        "image_width": [],
        "image_height": [],

        "target_x": [],
        "target_y": [],
        "target_x_px": [],
        "target_y_px": [],

        "bbox_left": [],
        "bbox_top": [],
        "bbox_right": [],
        "bbox_bottom": [],

        "raw_h": [],
        "raw_w": [],
        "token_h": [],
        "token_w": [],
        "merge_size": [],

        "raw_patch_x": [],
        "raw_patch_y": [],

        "target_token_x": [],
        "target_token_y": [],
        "target_token_idx": [],

        "bbox_token_indices": [],
        "window3x3_indices": [],

        "center_token_hidden": [],
        "bbox_mean_hidden": [],
        "window3x3_mean_hidden": [],
        "bbox_hidden": [],
        "window3x3_hidden": [],
    }

    with torch.no_grad():
        for path in tqdm(image_paths):
            meta = parse_filename(os.path.basename(path))

            target_meta = load_target_metadata(path)
            image, inputs = build_inputs(path, processor)
            grid_info = get_visual_grid_info(inputs, model)
            token_info = target_to_visual_tokens(target_meta, grid_info)

            window_indices = get_token_window_indices(
                token_info["target_token_x"],
                token_info["target_token_y"],
                grid_info["token_h"],
                grid_info["token_w"],
            )

            inputs = move_inputs_to_device(inputs, model_device)
            image_positions = get_image_positions(inputs, model)

            if len(image_positions) != grid_info["expected_tokens"]:
                raise ValueError(
                    f"Expected {grid_info['expected_tokens']} image tokens, "
                    f"got {len(image_positions)} for {path}"
                )

            outputs = model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )

            hidden_states = outputs.hidden_states

            center_hidden = []
            bbox_mean_hidden = []
            window_mean_hidden = []
            bbox_hidden = []
            window_hidden = []

            center_idx = token_info["target_token_idx"]

            bbox_tensor = torch.tensor(
                token_info["bbox_token_indices"],
                device=model_device,
                dtype=torch.long,
            )

            window_tensor = torch.tensor(
                window_indices,
                device=model_device,
                dtype=torch.long,
            )

            for layer in hidden_states:
                h = layer[0]
                image_h = h[image_positions]

                center = image_h[center_idx]
                bbox = image_h[bbox_tensor]
                window = image_h[window_tensor]

                center_hidden.append(center.detach().cpu())
                bbox_mean_hidden.append(bbox.mean(0).detach().cpu())
                window_mean_hidden.append(window.mean(0).detach().cpu())
                bbox_hidden.append(bbox.detach().cpu())
                window_hidden.append(window.detach().cpu())

            center_hidden = torch.stack(center_hidden).half()
            bbox_mean_hidden = torch.stack(bbox_mean_hidden).half()
            window_mean_hidden = torch.stack(window_mean_hidden).half()

            # These may have variable token counts, so keep them as per-example tensors.
            bbox_hidden = torch.stack(bbox_hidden).half()
            window_hidden = torch.stack(window_hidden).half()

            results["image_paths"].append(path)
            results["conditions"].append(meta["condition"])
            results["num_distractors"].append(meta["num_distractors"])
            results["trials"].append(meta["trial"])

            results["image_width"].append(target_meta["image_width"])
            results["image_height"].append(target_meta["image_height"])

            results["target_x"].append(target_meta["target_x"])
            results["target_y"].append(target_meta["target_y"])
            results["target_x_px"].append(target_meta["target_x_px"])
            results["target_y_px"].append(target_meta["target_y_px"])

            results["bbox_left"].append(target_meta["bbox_left"])
            results["bbox_top"].append(target_meta["bbox_top"])
            results["bbox_right"].append(target_meta["bbox_right"])
            results["bbox_bottom"].append(target_meta["bbox_bottom"])

            results["raw_h"].append(grid_info["raw_h"])
            results["raw_w"].append(grid_info["raw_w"])
            results["token_h"].append(grid_info["token_h"])
            results["token_w"].append(grid_info["token_w"])
            results["merge_size"].append(grid_info["merge_size"])

            results["raw_patch_x"].append(token_info["raw_patch_x"])
            results["raw_patch_y"].append(token_info["raw_patch_y"])

            results["target_token_x"].append(token_info["target_token_x"])
            results["target_token_y"].append(token_info["target_token_y"])
            results["target_token_idx"].append(token_info["target_token_idx"])

            results["bbox_token_indices"].append(token_info["bbox_token_indices"])
            results["window3x3_indices"].append(window_indices)

            results["center_token_hidden"].append(center_hidden)
            results["bbox_mean_hidden"].append(bbox_mean_hidden)
            results["window3x3_mean_hidden"].append(window_mean_hidden)
            results["bbox_hidden"].append(bbox_hidden)
            results["window3x3_hidden"].append(window_hidden)

    tensor_keys = [
        "num_distractors",
        "trials",
        "image_width",
        "image_height",
        "target_x",
        "target_y",
        "target_x_px",
        "target_y_px",
        "bbox_left",
        "bbox_top",
        "bbox_right",
        "bbox_bottom",
        "raw_h",
        "raw_w",
        "token_h",
        "token_w",
        "merge_size",
        "raw_patch_x",
        "raw_patch_y",
        "target_token_x",
        "target_token_y",
        "target_token_idx",
    ]

    for key in tensor_keys:
        results[key] = torch.tensor(results[key])

    results["center_token_hidden"] = torch.stack(results["center_token_hidden"])
    results["bbox_mean_hidden"] = torch.stack(results["bbox_mean_hidden"])
    results["window3x3_mean_hidden"] = torch.stack(results["window3x3_mean_hidden"])

    torch.save(results, args.out_path)

    print("\nSaved:")
    print(args.out_path)

    print("\nShapes:")
    print("center_token_hidden:", results["center_token_hidden"].shape)
    print("bbox_mean_hidden:", results["bbox_mean_hidden"].shape)
    print("window3x3_mean_hidden:", results["window3x3_mean_hidden"].shape)
    print("bbox_hidden: list length", len(results["bbox_hidden"]))
    print("window3x3_hidden: list length", len(results["window3x3_hidden"]))


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
        "--debug",
        action="store_true",
    )

    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
    )

    args = parser.parse_args()
    main(args)
