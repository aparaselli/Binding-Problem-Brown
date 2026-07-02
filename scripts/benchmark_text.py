
# benchmark_text.py

import argparse
import json
import random
import re

import pandas as pd
from tqdm import tqdm

from search_utils_text import (
    load_lm,
    search_model_call,
    extract_answer,
    build_sequence,
    get_prompt,
)

DISTRACTOR_COUNTS = [5,10,15,20,25,30,35,40,45,50]

def build_objects(condition, num_distractors, target_present):
    objs = []

    if condition == "disjunctive":
        if target_present:
            objs.append("green circle")
        objs.extend(["red circle"] * num_distractors)

    elif condition == "conjunctive":
        if target_present:
            objs.append("green L")

        for _ in range(num_distractors):
            objs.append(random.choice(["red L", "green T"]))

    else:
        raise ValueError(condition)

    random.shuffle(objs)
    return objs


def main(args):

    random.seed(args.seed)

    model, tokenizer = load_lm(args.model_name)

    results = []

    trials = []

    for condition in ["disjunctive", "conjunctive"]:
        for nd in DISTRACTOR_COUNTS:

            labels = [True]*50 + [False]*50
            random.shuffle(labels)

            for trial, target_present in enumerate(labels):
                trials.append(
                    (condition, nd, trial, target_present)
                )

    print(f"Running {len(trials)} trials")

    for idx, (condition, nd, trial, target_present) in enumerate(tqdm(trials)):

        objects = build_objects(
            condition,
            nd,
            target_present,
        )

        sequence = build_sequence(
            tokenizer=tokenizer,
            object_strings=objects,
            filler_token=args.filler_token,
            target_tokens=args.target_tokens,
        )

        prompt = get_prompt(condition)

        full_prompt = (
            f"Sequence:\n\n{sequence}\n\n{prompt}"
        )

        response = search_model_call(
            model=model,
            tokenizer=tokenizer,
            sequence=sequence,
            condition=condition,
        )


        prediction = extract_answer(response)

        correct = (
            prediction == target_present
            if prediction is not None
            else False
        )

        token_length = len(
            tokenizer(
                sequence,
                add_special_tokens=False,
            )["input_ids"]
        )

        results.append({
            "condition": condition,
            "num_distractors": nd,
            "trial": trial,
            "target_present": target_present,
            "sequence": sequence,
            "object_list": json.dumps(objects),
            "token_length": token_length,
            "prediction": prediction,
            "correct": correct,
            "response": response,
            "prompt": full_prompt,
        })

        if (idx + 1) % 50 == 0:
            pd.DataFrame(results).to_csv(
                args.out_path,
                index=False,
            )

    df = pd.DataFrame(results)
    df.to_csv(args.out_path, index=False)

    summary = (
        df.groupby(
            ["condition", "num_distractors"]
        )["correct"]
        .mean()
        .reset_index()
    )

    summary_path = args.out_path.replace(
        ".csv",
        "_summary.csv",
    )

    summary.to_csv(summary_path, index=False)

    print("\nSaved:")
    print(args.out_path)

    print("\nSaved summary:")
    print(summary_path)

    print("\nAccuracy Summary\n")
    print(summary)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--out_path",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--filler_token",
        type=str,
        default="blank",
    )

    parser.add_argument(
        "--target_tokens",
        type=int,
        default=289,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    args = parser.parse_args()

    main(args)
