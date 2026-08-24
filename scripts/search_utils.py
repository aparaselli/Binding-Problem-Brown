# search_utils.py

import re
import torch

from transformers import AutoProcessor

try:
    from transformers import AutoModelForVision2Seq
except Exception:
    AutoModelForVision2Seq = None

try:
    from transformers import AutoModelForImageTextToText
except Exception:
    AutoModelForImageTextToText = None



DISJUNCTIVE_PROMPT = """
You are presented with an image containing several letters.
Your task is to determine if there are any green letters in the image.

Follow these steps carefully:

1. Describe each shape in the image, noting their color.

2. Conclude your response by stating [True] if there are any green shapes,
or [False] if there are none.

Enclose your final answer in square brackets, as shown.
""".strip()


CONJUNCTIVE_PROMPT = """
You are presented with an image containing a set of letters, specifically
the letters 'L' and 'T'. These letters will appear in either red or green.

Your task is to determine if there are any green 'L's in the image.

Follow these steps carefully:

1. Describe each shape in the image, noting their color.

2. Conclude your response by stating [True] if the letter 'L' appears in green,
or [False] if there are no green 'L's.

Enclose your final answer in square brackets, as shown.
""".strip()

CONJUNCTIVE_PROMPT_ALT = """
You are presented with an image containing a set of letters, specifically
the letters 'L' and 'T'. These letters will appear in either red or green.

Your task is to determine if there are any green 'L's in the image.
Conclude your response by stating [True] if the letter 'L' appears in green,
or [False] if there are no green 'L's.

Enclose your final answer in square brackets, as shown.
""".strip()


PROMPT_BINARY = """
Answer the following question with exactly one word:

True

or

False

Do not provide any explanation.

Question:
You are presented with an image containing letters.

Is there a green letter L in the image?
""".strip()



def get_prompt(condition, prompt_type="regular"):

    condition = condition.lower()

    if prompt_type == "binary":

        return PROMPT_BINARY

    if condition == "disjunctive":
        return DISJUNCTIVE_PROMPT

    if condition == "conjunctive":
        return CONJUNCTIVE_PROMPT

    if condition == "conjunctive_alt":
        return CONJUNCTIVE_PROMPT_ALT

    raise ValueError(f"Unknown condition: {condition}")



def load_vlm(model_name):

    name_lc = model_name.lower()

    bf16_ok = (
        torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
    )

    dtype = (
        torch.bfloat16
        if bf16_ok
        else torch.float16
    )

    print(f"Loading {model_name}")

    processor = AutoProcessor.from_pretrained(
        model_name,
        cache_dir="/oscar/scratch/aparasel/hf_cache",
    )


    if "gemma" in name_lc:

        if AutoModelForImageTextToText is None:
            raise RuntimeError(
                "Gemma requires AutoModelForImageTextToText"
            )

        model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
            low_cpu_mem_usage=True,
            cache_dir="/oscar/scratch/aparasel/hf_cache",
        )


    else:

        if AutoModelForVision2Seq is not None:

            model = AutoModelForVision2Seq.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map="auto",
                low_cpu_mem_usage=True,
                cache_dir="/oscar/scratch/aparasel/hf_cache",
            )

        elif AutoModelForImageTextToText is not None:

            model = AutoModelForImageTextToText.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map="auto",
                low_cpu_mem_usage=True,
                cache_dir="/oscar/scratch/aparasel/hf_cache",
            )

        else:
            raise RuntimeError(
                "No compatible model loader found."
            )

    model.eval()

    print("Model loaded")

    return model, processor



def build_inputs(
    processor,
    model_device,
    image,
    prompt="",
    assistant_prefill=None,
):

    if assistant_prefill is None:

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ]

        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    else:

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    }
                ],
            },
            {
                "role": "assistant",
                "content": assistant_prefill,
            },
        ]

        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            continue_final_message=True
        )

    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
        padding=True,
    )

    for k, v in inputs.items():
        if torch.is_tensor(v):
            inputs[k] = v.to(model_device)

    return inputs



@torch.no_grad()
def search_model_call(
    model,
    processor,
    image,
    condition,
    prompt_type="regular",
    max_new_tokens=1024,
):

    prompt = get_prompt(
        condition,
        prompt_type=prompt_type,
    )

    model_device = next(model.parameters()).device

    inputs = build_inputs(
        processor=processor,
        model_device=model_device,
        image=image,
        prompt=prompt,
    )

    gen_out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        return_dict_in_generate=True,
        output_scores=True,
    )

    generated_ids = gen_out.sequences

    input_len = inputs["input_ids"].shape[-1]

    new_token_ids = generated_ids[:, input_len:]

    response = processor.batch_decode(
        new_token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


    scores = gen_out.scores[0]

    log_probs = scores.log_softmax(dim=-1)

    first_token_id = new_token_ids[0, 0].item()

    first_token = processor.tokenizer.decode(
        [first_token_id],
        clean_up_tokenization_spaces=False,
    )

    first_token_logprob = (
        log_probs[0, first_token_id]
        .float()
        .cpu()
        .item()
    )

    return {
        "response": response,
        "first_token": first_token,
        "first_token_id": first_token_id,
        "first_token_logprob": first_token_logprob,
    }



def extract_answer(response, prompt_type="regular"):

    if response is None:
        return None

    if prompt_type == "binary":

        text = response.strip().lower()

        if text == "true":
            return True

        if text == "false":
            return False

        return None

    # regular prompt type parser
    matches = re.findall(
        r"\[\s*(true|false)\s*\]",
        response,
        flags=re.IGNORECASE,
    )

    if matches:
        answer = matches[-1].lower()
        return answer == "true"

    response_lc = response.lower()
    last_true = response_lc.rfind("[true]")
    last_false = response_lc.rfind("[false]")

    if last_true == -1 and last_false == -1:
        return None

    return last_true > last_false



def evaluate_prediction(
    response,
    target_present,
):

    pred = extract_answer(response)

    if pred is None:

        return {
            "prediction": None,
            "correct": False,
        }

    return {
        "prediction": pred,
        "correct": pred == target_present,
    }