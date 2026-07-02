# search_utils_text.py

import re
import torch

from transformers import AutoTokenizer

try:
    from transformers import AutoModelForVision2Seq
except Exception:
    AutoModelForVision2Seq = None

try:
    from transformers import AutoModelForImageTextToText
except Exception:
    AutoModelForImageTextToText = None

try:
    from transformers import AutoModelForCausalLM
except Exception:
    AutoModelForCausalLM = None


# ==================================================
# PROMPTS
# ==================================================

DISJUNCTIVE_PROMPT = """
You are presented with a sequence containing descriptions of several shapes.

Your task is to determine whether there are any green shapes in the sequence.

Follow these steps carefully:

1. Briefly describe each shape in the sequence, noting its color.

2. Conclude your response by stating [True] if there are any green shapes,
or [False] if there are none.

Enclose your final answer in square brackets.
""".strip()


CONJUNCTIVE_PROMPT = """
You are presented with a sequence containing descriptions of letters.
The letters are either 'L' or 'T', and each letter is either red or green.

Your task is to determine whether there are any green 'L's in the sequence.

Follow these steps carefully:

1. Briefly describe each letter in the sequence, noting its color.

2. Conclude your response by stating [True] if there are any green 'L's,
or [False] if there are none.

Enclose your final answer in square brackets.
""".strip()


def get_prompt(condition):

    if condition.lower() == "disjunctive":
        return DISJUNCTIVE_PROMPT

    if condition.lower() == "conjunctive":
        return CONJUNCTIVE_PROMPT

    raise ValueError(condition)


# ==================================================
# MODEL
# ==================================================

def load_lm(model_name):

    bf16_ok = (
        torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
    )

    dtype = (
        torch.bfloat16
        if bf16_ok
        else torch.float16
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    name = model_name.lower()

    # ----------------------------------
    # GEMMA
    # ----------------------------------

    if "gemma" in name:

        model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
        )

    # ----------------------------------
    # QWEN-VL
    # ----------------------------------

    elif "qwen" in name and "vl" in name:

        model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
        )

    # ----------------------------------
    # Normal LLMs
    # ----------------------------------

    else:

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
        )
    model.eval()

    return model, tokenizer


# ==================================================
# BUILD SEQUENCE
# ==================================================
import random

def build_sequence(
    tokenizer,
    object_strings,
    filler_token="blank",
    target_tokens=289,
    max_adjust_iters=200,
):
    """
    Creates a fixed-length sequence of slots.

    If target_tokens <= 0, no padding is used and the
    objects are simply concatenated.
    """

    # ----------------------------------
    # NO PADDING MODE
    # ----------------------------------
    if target_tokens <= 0:
        return ", ".join(object_strings)

    # ----------------------------------
    # PADDED MODE
    # ----------------------------------
    num_slots = target_tokens

    for _ in range(max_adjust_iters):

        slots = [filler_token] * num_slots

        positions = random.sample(
            range(num_slots),
            len(object_strings),
        )

        for pos, obj in zip(positions, object_strings):
            slots[pos] = obj

        sequence = " ".join(slots)

        token_len = len(
            tokenizer(
                sequence,
                add_special_tokens=False,
            )["input_ids"]
        )

        if token_len == target_tokens:
            return sequence

        if token_len < target_tokens:
            num_slots += 1
        else:
            num_slots -= 1

    return sequence

# ==================================================
# INPUTS
# ==================================================

def build_inputs(
    tokenizer,
    model_device,
    sequence,
    prompt,
):

    messages = [
        {
            "role": "user",
            "content":
                f"Sequence:\n\n{sequence}\n\n"
                + prompt
        }
    ]

    if hasattr(tokenizer, "apply_chat_template"):

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    else:

        text = (
            f"Sequence:\n\n{sequence}\n\n"
            + prompt
        )

    inputs = tokenizer(
        text,
        return_tensors="pt",
    )

    for k, v in inputs.items():
        inputs[k] = v.to(model_device)

    return inputs


# ==================================================
# GENERATION
# ==================================================

@torch.no_grad()
def search_model_call(
    model,
    tokenizer,
    sequence,
    condition,
    max_new_tokens=1048,
):

    prompt = get_prompt(condition)

    # DEBUGGING
    # print("=" * 80)
    # print(prompt)
    # print("=" * 80)

    device = next(model.parameters()).device

    inputs = build_inputs(
        tokenizer,
        device,
        sequence,
        prompt,
    )

    outputs = model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=max_new_tokens,
    )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()

    return response


# ==================================================
# ANSWER PARSING
# ==================================================

def extract_answer(response):

    match = re.search(
        r"\[\s*(true|false)\s*\]",
        response,
        flags=re.IGNORECASE,
    )

    if match:
        return match.group(1).lower() == "true"

    r = response.lower()

    if "[true]" in r:
        return True

    if "[false]" in r:
        return False

    return None


# ==================================================
# EVALUATION
# ==================================================

def evaluate_prediction(
    response,
    target_present,
):

    pred = extract_answer(response)

    return {
        "prediction": pred,
        "correct": pred == target_present if pred is not None else False,
    }