import torch
import pandas as pd
from transformers import AutoTokenizer

# ==================================================
# CONFIG
# ==================================================

PATH = ("/oscar/data/epavlick/aparasel/Binding-Problem-Brown/Results/target_hidden_states.pt")

REPRESENTATION = "bbox_mean_hidden"
N_EXAMPLES = 5
TOPK = 10

# ==================================================
# LOAD
# ==================================================

data = torch.load(
    PATH,
    map_location="cpu",
)

tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct"
)

from search_utils import load_vlm

model, processor = load_vlm(
    "Qwen/Qwen2.5-VL-7B-Instruct"
)

model.eval()

# ==================================================
# INSPECT TOKENIZATION
# ==================================================

green_tokens = tokenizer.encode(
    " green",
    add_special_tokens=False,
)

L_tokens = tokenizer.encode(
    " L",
    add_special_tokens=False,
)

print("green token ids:", green_tokens)
print("L token ids:", L_tokens)

green_id = green_tokens[0]
L_id = L_tokens[0]

# ==================================================
# LOGIT LENS
# ==================================================

rows = []

N = min(
    N_EXAMPLES,
    data[REPRESENTATION].shape[0],
)

for ex in range(N):

    hidden = data[
        REPRESENTATION
    ][ex]

    n_layers = hidden.shape[0]

    for layer in range(n_layers):

        h = hidden[layer].to(
            model.device
        )

        with torch.no_grad():

            h = model.model.norm(h)

            logits = model.lm_head(h)

            probs = logits.softmax(-1)

            topk = torch.topk(
                probs,
                k=TOPK,
            )

        ids = topk.indices.cpu()

        toks = tokenizer.batch_decode(
            ids.unsqueeze(-1),
            skip_special_tokens=False,
        )

        rows.append(
            {
                "representation":
                    REPRESENTATION,

                "example":
                    ex,

                "layer":
                    layer,

                "top1":
                    toks[0],

                "top5":
                    toks[:5],

                "top10":
                    toks,

                "green_logit":
                    logits[
                        green_id
                    ].item(),

                "L_logit":
                    logits[
                        L_id
                    ].item(),

                "green_prob":
                    probs[
                        green_id
                    ].item(),

                "L_prob":
                    probs[
                        L_id
                    ].item(),
            }
        )

df = pd.DataFrame(rows)

out_path = (
    f"logit_lens_{REPRESENTATION}.csv"
)

df.to_csv(
    out_path,
    index=False,
)

print("\nSaved:")
print(out_path)

print(
    df[
        [
            "example",
            "layer",
            "top1",
            "green_prob",
            "L_prob",
        ]
    ]
    .head(20)
)
