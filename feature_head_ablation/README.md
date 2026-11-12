# Color- and shape-head ablation

This folder tests whether a small set of attention heads is **necessary** for
visual search in vision-language models. Each image is shown with a yes/no
question that names a color and a shape, for example *"Is there a blue letter L
in the image?"*. We remove the output of the top 5% **color heads** at the
color word, the top 5% **shape heads** at the shape word, or both, and measure
how the model's answer changes.

Supported out of the box (rankings included):

| `--model` | Hugging Face model | Heads | GPU memory |
|---|---|---|---|
| `qwen2.5-vl-7b` | `Qwen/Qwen2.5-VL-7B-Instruct` | 28 layers x 28 | ~17 GB (1 x 24 GB) |
| `internvl3.5-8b` | `OpenGVLab/InternVL3_5-8B-HF` | 36 x 32 | ~18 GB (1 x 24 GB) |
| `gemma-3-12b` | `google/gemma-3-12b-it` | 48 x 16 | ~26 GB (2 x 24 GB) |

## Files

```
ablate_feature_heads.py    main script
head_utils.py              model loading, attention-head hooks, yes/no scoring
make_manifest_example.py   builds a manifest from our dataset (template for yours)
head_rankings/             <model>_color_heads.csv, <model>_shape_heads.csv
```

## How the heads were ranked

Heads were ranked by single-head activation patching on counterfactual image
pairs, one head at a time:

- **Color heads.** The two images in a pair differ only in object colors
  (letters and positions are the same). We copy one head's output at the
  **color word** token from the run on one image into the run on the other, and
  measure how far this moves the yes/no logit margin toward the answer that
  would be correct for the color-swapped scene.
- **Shape heads.** Same procedure, with pairs that differ only in letter
  identity, patched at the **shape word** token.

Each head's `score` is the mean margin change over 864 pairs (two pair types x
432), and `rank` sorts heads by that score (1 = strongest). The script selects
the top `--percentage` of all heads from each file.

## Your data: the manifest

The script reads a CSV with one row per image:

| column | required | meaning |
|---|---|---|
| `image_path` | yes | absolute, or relative to the manifest's folder |
| `condition` | yes | any label, e.g. `conjunctive` / `disjunctive`; results are reported per condition |
| `target_present` | yes | `True` / `False`: the correct answer |
| `color_word` | yes* | the color named in the question, e.g. `blue` |
| `shape_word` | yes* | the shape named in the question, e.g. `L` |
| `prompt` | no | the full prompt; if absent, the default prompt below is filled with the two words |

\* Used to find the token to ablate. You can instead pass fixed positions with
`--color_offset` / `--shape_offset` (offset from the last input token).

Default prompt (change it per row with `prompt`, or for all rows with
`--prompt_template file.txt` containing `{color}` and `{shape}`):

```
Answer the following question with exactly one word:

True

or

False

Do not provide any explanation.

Question:
You are presented with an image containing letters.

Is there a {color} letter {shape} in the image?
```

The answer is scored from the next-token logits of `True` vs `False`, so
prompts should keep that answer format. The ablated token for each feature is
the **last token of the last occurrence** of `color_word` / `shape_word` in
the tokenized prompt; with the default prompt that is 11 and 9 tokens from the
end for all three models. The script prints the tokens it ablates and stores
them in `baseline.csv`, so check them if you change the prompt.

`make_manifest_example.py` shows how we build a manifest from our own dataset
(9 color x shape combinations, 5 distractors, conjunctive and disjunctive
search, 125 target-present + 125 target-absent images per combination).

## What one run does

For every image the model is run once without intervention (baseline), then
once per ablation spec:

| | options |
|---|---|
| head set | `color` (color heads at the color word), `shape` (shape heads at the shape word), `color+shape` (both, each at its own word) |
| selection | `ranked` (top 5% from `head_rankings/`) or `random` (same number of random heads; seeds 0 and 1) |
| ablation | `zero` (head output set to 0) or `mean` (head output set to its average over **all** manifest images, every condition, present and absent) |

That is 18 ablated forward passes per image. Only the selected heads at the
color/shape word token are changed; all other positions are untouched.

## Running it

```bash
pip install torch transformers pandas numpy pillow tqdm accelerate
# we used torch 2.9.0; transformers 4.57.1 (Qwen) and 5.17.0 (InternVL, Gemma-3)

python make_manifest_example.py --root /path/to/dataset --out manifest.csv   # or write your own

python ablate_feature_heads.py --model qwen2.5-vl-7b \
    --manifest manifest.csv --out_dir results/qwen
```

Quick test first (first 10 images per condition):

```bash
python ablate_feature_heads.py --model qwen2.5-vl-7b --manifest manifest.csv \
    --out_dir results/qwen_test --max_images_per_condition 10
```

Useful options: `--percentage` (default 5), `--ablation_types zero mean`,
`--random_seeds 0 1`, `--conditions` (subset of manifest conditions),
`--color_ranking` / `--shape_ranking` (your own ranking CSVs with
`rank,layer,head` columns, e.g. for another model), `--cache_dir` (Hugging Face
cache; default `$HF_HOME`).

**Runtime.** About 0.17-0.2 s per forward pass for Qwen and InternVL on one
RTX A5000, so 4,500 images x 19 passes takes about 4-5 hours per model. If a
run stops, rerun the same command: finished baselines and ablation rows are
reused.

## Outputs (`--out_dir`)

| file | contents |
|---|---|
| `baseline.csv` | unablated correctness and margin per image, plus the tokens ablated |
| `act_sums.pt` | summed head outputs over all images (the mean-ablation vectors are these / `n_images`) |
| `head_sets.json` | the exact (layer, head) list for every spec |
| `ablations.csv` | one row per (image, spec) |
| `summary.csv` | per condition x spec: accuracy, mean margins, `mean_delta_margin` |

In `summary.csv`, `split` is `pooled`, `present` or `absent`; `random_seed` is
-1 for ranked heads, the seed for single random draws, and -2 for the average
over seeds. Pooled rows also have `balanced_*_accuracy` (mean of present and
absent accuracy).

**Margin** = logit(correct answer) - logit(wrong answer), so positive means the
model answers correctly. `delta_margin` = ablated - baseline margin.

## Reading the results

- **Look at present and absent separately.** In our runs, ablating the top
  heads often pushes the model toward one answer for every image (e.g. color
  heads toward "no"). Pooled accuracy then sits near 50% even though accuracy
  on present and on absent images moves in opposite directions. Compare
  margins and the present/absent split, or use balanced accuracy.
- **Random heads are the control.** An effect counts only if the ranked heads
  move the margin much more than the size-matched random heads.
- **Zero vs mean.** Zero ablation also removes the part of a head's output
  that is the same for every image; mean ablation keeps that part and removes
  only what differs between images. If the two disagree, the heads' average
  output itself matters (e.g. it carries a general bias toward "yes").
