# backpatch.py

import os
import re
import argparse

import torch
import pandas as pd

from PIL import Image
from tqdm import tqdm

from search_utils import (
    load_vlm,
    search_model_call,
    extract_answer,
    build_inputs,
    get_prompt,
)


DATASET_DIR = (
    "/oscar/data/epavlick/aparasel/"
    "Binding-Problem-Brown/search-dataset/"
    "visual_search_dataset"
)

torch.set_grad_enabled(False)


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

    return {
        "condition": match.group(1),
        "num_distractors": int(match.group(2)),
        "trial": int(match.group(3)),
        "target_present": match.group(4) == "true",
    }


def get_language_model(model):

    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        return model.model.language_model

    if hasattr(model, "language_model"):
        return model.language_model

    if hasattr(model, "model") and hasattr(model.model, "transformer"):
        return model.model.transformer

    if hasattr(model, "transformer"):
        return model.transformer

    for _, module in model.named_modules():
        if hasattr(module, "layers"):
            return module

    raise AttributeError("Could not find language model.")


def get_layers(model):

    lm = get_language_model(model)

    if hasattr(lm, "layers"):
        return lm.layers

    raise AttributeError("Language model has no .layers")


def get_first_tensor(output):

    if torch.is_tensor(output):
        return output

    if isinstance(output, (tuple, list)):
        for item in output:
            if torch.is_tensor(item):
                return item

    raise TypeError(f"Could not find tensor in output type {type(output)}")


def replace_first_tensor(output, new_tensor):

    if torch.is_tensor(output):
        return new_tensor

    if isinstance(output, tuple):
        out = list(output)
        for i, item in enumerate(out):
            if torch.is_tensor(item):
                out[i] = new_tensor
                return tuple(out)

    if isinstance(output, list):
        for i, item in enumerate(output):
            if torch.is_tensor(item):
                output[i] = new_tensor
                return output

    return output


def get_visual_positions(processor, input_ids_1d):

    tok = processor.tokenizer

    # Qwen-style markers
    vs_id = tok.convert_tokens_to_ids("<|vision_start|>")
    ve_id = tok.convert_tokens_to_ids("<|vision_end|>")

    ids = input_ids_1d.tolist()

    if vs_id is not None and ve_id is not None and vs_id >= 0 and ve_id >= 0:
        try:
            s = ids.index(vs_id)
            e = ids.index(ve_id)

            if e > s + 1:
                return torch.arange(
                    s + 1,
                    e,
                    device=input_ids_1d.device,
                    dtype=torch.long,
                )
        except ValueError:
            pass

    # Gemma / generic image tokens
    candidate_tokens = [
        "<image>",
        "<IMAGE>",
        "<img>",
        "<|image|>",
        "<|image_pad|>",
        "<|vision_pad|>",
        "<|visual_pad|>",
        "<image_soft_token>",
        "<start_of_image>",
        "<end_of_image>",
    ]

    candidate_ids = set()

    for token in candidate_tokens:
        try:
            tid = tok.convert_tokens_to_ids(token)
            if tid is not None and tid >= 0:
                candidate_ids.add(int(tid))
        except Exception:
            pass

    for attr in ["image_token_id", "img_token_id", "vision_token_id"]:
        if hasattr(tok, attr):
            val = getattr(tok, attr)
            if val is not None:
                candidate_ids.add(int(val))

    mask = torch.zeros_like(input_ids_1d, dtype=torch.bool)

    for tid in candidate_ids:
        mask |= input_ids_1d == tid

    return mask.nonzero(as_tuple=False).view(-1).long()


def strip_special(inputs):

    return {
        k: v
        for k, v in inputs.items()
        if not str(k).startswith("_")
    }


def register_capture_hook(layers, source_layer, positions, cache):

    def hook(module, inputs, output):

        t = get_first_tensor(output)

        if t.dim() != 3:
            return output

        pos = positions.to(t.device)
        pos = pos[pos < t.shape[1]]

        if pos.numel() == 0:
            return output

        cache["hidden"] = (
            t[:, pos, :]
            .detach()
            .clone()
        )

        return output

    return layers[source_layer].register_forward_hook(hook)


def register_patch_hook(
    layers,
    destination_layer,
    positions,
    cache,
    alpha=1.0,
):

    def hook(module, inputs, output):

        if "hidden" not in cache:
            return output

        t = get_first_tensor(output)

        if t.dim() != 3:
            return output

        out = t.clone()

        pos = positions.to(out.device)
        pos = pos[pos < out.shape[1]]

        if pos.numel() == 0:
            return output

        src = cache["hidden"].to(
            device=out.device,
            dtype=out.dtype,
        )

        if src.shape[1] != pos.numel():
            m = min(src.shape[1], pos.numel())
            src = src[:, :m, :]
            pos = pos[:m]

        out[:, pos, :] = (
            (1.0 - alpha) * out[:, pos, :]
            +
            alpha * src
        )

        return replace_first_tensor(output, out)

    return layers[destination_layer].register_forward_hook(hook)


@torch.no_grad()
def generate_from_inputs(model, processor, inputs, max_new_tokens=256):

    gen_inputs = strip_special(inputs)

    gen_out = model.generate(
        **gen_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        return_dict_in_generate=True,
        output_scores=False,
    )

    input_len = gen_inputs["input_ids"].shape[-1]
    new_token_ids = gen_out.sequences[:, input_len:]

    response = processor.batch_decode(
        new_token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    return response


@torch.no_grad()
def run_same_image_backpatch(
    model,
    processor,
    image,
    condition,
    source_layer,
    destination_layer,
    alpha=1.0,
    max_new_tokens=256,
):

    layers = get_layers(model)

    model_device = next(model.parameters()).device
    prompt = get_prompt(condition)

    inputs = build_inputs(
        processor=processor,
        model_device=model_device,
        image=image,
        prompt=prompt,
    )

    model_inputs = strip_special(inputs)

    positions = get_visual_positions(
        processor,
        inputs["input_ids"][0],
    )

    if positions.numel() == 0:
        raise ValueError("No visual positions found.")


    original_response = generate_from_inputs(
        model=model,
        processor=processor,
        inputs=inputs,
        max_new_tokens=max_new_tokens,
    )


    cache = {}

    capture_hook = register_capture_hook(
        layers=layers,
        source_layer=source_layer,
        positions=positions,
        cache=cache,
    )

    try:
        _ = model(
            **model_inputs,
            use_cache=False,
        )
    finally:
        capture_hook.remove()

    if "hidden" not in cache:
        raise RuntimeError("Source hook did not capture hidden state.")


    patch_hook = register_patch_hook(
        layers=layers,
        destination_layer=destination_layer,
        positions=positions,
        cache=cache,
        alpha=alpha,
    )

    try:
        patched_response = generate_from_inputs(
            model=model,
            processor=processor,
            inputs=inputs,
            max_new_tokens=max_new_tokens,
        )
    finally:
        patch_hook.remove()

    return {
        "original_response": original_response,
        "patched_response": patched_response,
        "pos_count": int(positions.numel()),
    }



def main(args):

    model, processor = load_vlm(args.model_name)

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)

    layers = get_layers(model)
    n_layers = len(layers)

    if args.source_layer < 0 or args.source_layer >= n_layers:
        raise ValueError(f"source_layer out of range. n_layers={n_layers}")

    if args.destination_layer < 0 or args.destination_layer >= n_layers:
        raise ValueError(f"destination_layer out of range. n_layers={n_layers}")

    image_paths = []

    condition_dir = os.path.join(
        DATASET_DIR,
        args.condition,
    )

    for fname in sorted(os.listdir(condition_dir)):

        if not fname.endswith(".png"):
            continue

        meta = parse_filename(fname)

        if meta["num_distractors"] not in args.num_distractors:
            continue

        image_paths.append(
            os.path.join(condition_dir, fname)
        )

    if args.limit is not None:
        image_paths = image_paths[:args.limit]

    print(f"Found {len(image_paths)} images")
    print(f"source_layer={args.source_layer}")
    print(f"destination_layer={args.destination_layer}")
    print(f"alpha={args.alpha}")

    results = []

    for idx, path in enumerate(tqdm(image_paths)):

        fname = os.path.basename(path)
        meta = parse_filename(fname)

        try:
            image = Image.open(path).convert("RGB")

            out = run_same_image_backpatch(
                model=model,
                processor=processor,
                image=image,
                condition=meta["condition"],
                source_layer=args.source_layer,
                destination_layer=args.destination_layer,
                alpha=args.alpha,
                max_new_tokens=args.max_new_tokens,
            )

            original_response = out["original_response"]
            patched_response = out["patched_response"]
            pos_count = out["pos_count"]

            original_prediction = extract_answer(original_response)
            patched_prediction = extract_answer(patched_response)

            original_correct = (
                original_prediction
                ==
                meta["target_present"]
            )

            patched_correct = (
                patched_prediction
                ==
                meta["target_present"]
            )

            error = None

        except Exception as e:

            original_response = None
            patched_response = None
            original_prediction = None
            patched_prediction = None
            original_correct = False
            patched_correct = False
            pos_count = -1
            error = repr(e)

            print(f"ERROR on {fname}: {e}")

        result = {
            "image_path": path,
            "condition": meta["condition"],
            "num_distractors": meta["num_distractors"],
            "trial": meta["trial"],
            "target_present": meta["target_present"],

            "source_layer": args.source_layer,
            "destination_layer": args.destination_layer,
            "alpha": args.alpha,
            "pos_count": pos_count,

            "original_prediction": original_prediction,
            "patched_prediction": patched_prediction,

            "original_correct": original_correct,
            "patched_correct": patched_correct,

            "changed_prediction": (
                original_prediction != patched_prediction
                if original_prediction is not None
                and patched_prediction is not None
                else False
            ),

            "correctness_delta": (
                int(patched_correct)
                -
                int(original_correct)
            ),

            "original_response": original_response,
            "patched_response": patched_response,

            "error": error,
        }

        results.append(result)

        if (idx + 1) % args.flush_every == 0:
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
                "source_layer",
                "destination_layer",
            ]
        )
        .agg(
            n=("patched_correct", "size"),
            original_accuracy=("original_correct", "mean"),
            patched_accuracy=("patched_correct", "mean"),
            changed_rate=("changed_prediction", "mean"),
            mean_correctness_delta=("correctness_delta", "mean"),
            mean_pos_count=("pos_count", "mean"),
        )
        .reset_index()
    )

    summary_path = args.out_path.replace(
        ".csv",
        "_summary.csv",
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print("\nSummary:")
    print(summary)

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
        "--source_layer",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--destination_layer",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--condition",
        type=str,
        choices=[
            "disjunctive",
            "conjunctive",
        ],
        required=True,
    )

    parser.add_argument(
        "--num_distractors",
        type=int,
        nargs="+",
        required=True,
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--out_path",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--flush_every",
        type=int,
        default=50,
    )

    args = parser.parse_args()

    main(args)