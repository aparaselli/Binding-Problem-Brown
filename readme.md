# Binding problem

***NOTE:*** 
```
Binding-Problem-Brown/Analysis/write_up_replication.ipynb 
```
is a notebook explaining step by step how to recreate figures in write up

## Data Generation
In /search-dataset there exists two ipynb notebooks for dataset generation

search_counterfact_dataset_gen.ipynb - Generates updated experiment, where disjunctive search is red and green Ls instead of red and green circles. Furthermore, between trial numbers (so for example: conjunctive_distractors_0_**trial_**000**_target_true.png and conjunctive_distractors_10_**trial_000**_target_true.png) the target will always be in the same positions. This ensures that comparison across distractors is not influenced by positional embeddings.
- The first cell creates the dataset
- The second cell creates a disjunctive condiiton where the distractors are green Ts and the target is still a green L.
- The third cell creates two more dijunctive conditions, one where the target is green T and the distractors are a green L (This can be used to create a ''green" spatial mask while leaving out the green L). The second is a red L target among distractor red Ts (This can be used to create a ''L" spatial mask while leaving out the green L again)

minimal_pair_swap.ipynb - Generates a dataset with one green T and one red L, and a corresponding counterfactual where the positions are swapped. This can be intervened on to make the model hallucinate a green L.

## Benchmarking

`benchmark.py` evaluates a vision-language model on one of two visual-search datasets:

* `VS_CF`: the conjunctive/disjunctive visual-search counterfactual dataset.
* `MP_S`: the minimal-pair swap dataset.

For each image, the script loads the image, prompts the model with the selected task prompt, extracts a binary prediction, compares it to the ground-truth target label, and saves trial-level results to a CSV file.

### Supported datasets

#### `VS_CF`: Visual-search counterfactual dataset

This dataset contains images organized into condition folders, such as:

```text
visual_search_dataset_counterfactual/
├── disjunctive/
├── conjunctive/
└── disjunctive_T/
```

Expected filenames follow this format:

```text
conjunctive_distractors_40_trial_012_target_true.png
```

The filename encodes:

* `condition`: e.g., `disjunctive`, `conjunctive`, or `disjunctive_T`
* `num_distractors`: the number of distractors in the image
* `trial`: trial index
* `target_present`: whether the green `L` target is present

When a matching `.json` metadata file exists, the script also records the target’s plot coordinates, normalized position, and distance from the image center.

#### `MP_S`: Minimal-pair swap dataset

This dataset contains files such as:

```text
pair_000_original.png
pair_000_counterfactual_swap.png
```

The script treats these as the `original` and `counterfactual_swap` conditions. Since the minimal-pair dataset does not use the standard visual-search filename structure, it assigns:

* `num_distractors = 0`
* `target_present = False`

### Outputs

The main output CSV contains one row per evaluated image, including:

* image path and condition
* distractor count and trial number
* ground-truth target presence
* model prediction and correctness
* full model response
* target-position metadata, when available
* the first generated token, token ID, and log probability

The script also creates a summary CSV by averaging accuracy:

* grouped by `condition` and `num_distractors` for `VS_CF`
* grouped by `condition` for `MP_S`

For example, if the output path is:

```text
../Results/qwen_vs_cf.csv
```

the summary is saved as:

```text
../Results/qwen_vs_cf_summary.csv
```

### Running the visual-search counterfactual benchmark

To evaluate Qwen on the disjunctive and conjunctive conditions:

```bash
python benchmark.py \
  --model_name Qwen/Qwen2.5-VL-7B-Instruct \
  --dataset VS_CF \
  --conditions disjunctive conjunctive \
  --prompt_type binary \
  --out_path ../Results/qwen_vs_cf_benchmark.csv
```

To evaluate only the matched 40-distractor images, you would currently run the command above and filter the resulting CSV by `num_distractors == 40`.

To include the `disjunctive_T` condition as well:

```bash
python benchmark.py \
  --model_name Qwen/Qwen2.5-VL-7B-Instruct \
  --dataset VS_CF \
  --conditions disjunctive conjunctive disjunctive_T \
  --prompt_type binary \
  --out_path ../Results/qwen_all_conditions_benchmark.csv
```

### Running the minimal-pair swap benchmark

```bash
python benchmark.py \
  --model_name Qwen/Qwen2.5-VL-7B-Instruct \
  --dataset MP_S \
  --prompt_type binary \
  --out_path ../Results/qwen_minimal_pair_swap_benchmark.csv
```

## Causal Tracing (Layer x Token)

`layer_token_component_causal_tracing.py` identifies where information supporting successful visual search is represented in a VLM.

For each matched source–target image pair—typically a successful **disjunctive** image and its harder **conjunctive** counterpart—the script copies one source activation into the target forward pass and measures whether the target’s answer improves.

### What it patches

The script can patch one token position at a time across decoder layers, separately for:

* `attention`: the output of the self-attention sublayer
* `mlp`: the output of the MLP sublayer
* `residual`: the decoder-layer input residual stream

A patch may cover a centered window of layers. For example, `--window_size 3 --layers 10` patches layers 9, 10, and 11.

### Procedure

For every matched pair, the script:

1. Runs the source image and caches activations at every decoder layer.
2. Runs the target image normally to establish its baseline performance.
3. Re-runs the target image while replacing the selected target activation with the corresponding clean source activation.
4. Measures whether that intervention moves the target toward the source behavior.

The source and target are matched by trial number and target-presence label. Thus, with the spatially matched dataset generator, each pair has the same target and distractor locations but differs in its visual-search condition.

### Main metrics

* `cmm`: Correct Margin Metric—the True-minus-False logit margin for target-present trials, with the sign reversed for target-absent trials. Higher is better in both cases.
* `delta_cmm`: improvement over the clean target baseline.
* `recovery`: the fraction of the source–target CMM gap recovered by the patch.
* `patched_accuracy`: accuracy after the intervention.

### Outputs

Given an output path such as:

```text
../Results/disj40_to_conj40_layer_token_components_w3.csv
```

the script saves:

* `disj40_to_conj40_layer_token_components_w3.csv`: one row per trial, component, token offset, and layer window.
* `disj40_to_conj40_layer_token_components_w3_summary.csv`: results averaged over trials for each component/layer/token cell.
* `disj40_to_conj40_layer_token_components_w3_config.json`: the exact command-line configuration.
* Partial CSVs during the run, allowing recovery from interrupted jobs.

### Example

This traces attention and MLP outputs from matched 40-distractor disjunctive images into their conjunctive counterparts. It tests every token from offsets `-19` through `0`, uses all decoder layers as window centers, and patches three adjacent layers at once.

(This includes the most interest text tokens: "Is there a green L in the image?")

```bash
python layer_token_component_causal_tracing.py \
  --source_condition disjunctive \
  --target_condition conjunctive \
  --source_distractors 40 \
  --target_distractors 40 \
  --components attention,mlp \
  --token_offsets=-19:0 \
  --layers all \
  --window_size 5 \
  --pair_filter all \
  --out_path ../Results/disj40_to_conj40_layer_token_components_w3.csv
```


## Attention head search/analysis

`benchmark_headpatch.py` performs a fine-grained follow-up to layer × token causal tracing. Rather than patching an entire attention sublayer, it tests **each attention head individually**.

For every matched source–target pair, it copies one source attention head into the corresponding target layer and measures whether that single intervention restores correct visual-search behavior.

### What is patched

The script caches each head’s contribution immediately before the attention output projection (`self_attn.o_proj`) at one selected prompt-token position.

It then runs one target forward pass for every `(layer, head)` combination, replacing only that head’s vector at the selected token position. All other heads, tokens, layers, and computations remain unchanged.

`--token_offset` is relative to the final input token:

The final text is:
"Is there a green letter L in the image?"
Therefore the following offsets correspond to the following tokens:

* `0`: final input token
* `-9`: the `L` token.
* `-11`: the `green` token.

### Key outputs

The detailed output CSV contains one row per matched pair and attention head, including:

* source, target, and patched correct-logit margins
* `delta_margin`: improvement over the unpatched target
* `recovery`: fraction of the source-to-target margin drop restored by the patch
* whether the patch flips an incorrect target prediction to correct
* target-location metadata and patching diagnostics

The script also creates:

* `*_head_summary.csv`: average performance for every layer/head, ranked by mean margin improvement.
* `*_pair_summary.csv`: best and mean head-patching results for each image pair.

### Example

This evaluates every Qwen attention head by patching its disjunctive activation into the matched 40-distractor conjunctive image at token offset `-9` (The 'L' token):

```bash
python benchmark_headpatch.py \
  --source_condition disjunctive \
  --target_condition conjunctive \
  --source_distractors 40 \
  --target_distractors 40 \
  --token_offset -9 \
  --out_path ../Results/disj40_to_conj40_headpatch_qwen_-9.csv
```


***NOTE***: The next set of files largely run analysis/experiments dependent on the output of the above script (aka the heads ranked by delta CMM)

## Patch Percent Top Heads ("sparsity" experiment)

`patch_pct_top_heads.py` tests whether the visual-search behavior can be recovered by patching **small, cumulative sets of previously ranked attention heads** from a source image into its matched target image.

Unlike the single-head benchmark, this script ***does not discover important heads.*** It takes a prior head-patching CSV, ranks heads using a selected metric, then evaluates the top (N%) of those heads together.

### Procedure

For each matched target image, the script:

1. Loads the corresponding source image and caches each attention head’s full output immediately before `o_proj`.
2. Runs the target image without intervention.
3. Patches the top-ranked (N%) of heads into the target at the requested token position(s).
4. Measures patched accuracy, change in correct-answer margin, and normalized recovery.

Each percentage is cumulative: a 5% patch includes the heads used in the 1% and 3% patches.

### Ranking heads

The ranking CSV should come from the per-head benchmark script.

* Use `--ranking_metric delta_margin` with a **raw per-trial head-patching CSV**.
* Use `--ranking_metric mean_delta_margin` with an aggregated head-summary CSV.

With `--rank_by_target_presence`, the script builds separate rankings for target-present and target-absent trials. `--always_use_target_present_ranking` instead uses the target-present ranking for every target image.

### Multiple token positions

The optional `--patch_config` argument enables simultaneous patching at multiple prompt positions. Each entry has the form:

```text
OFFSET|SOURCE_DATASET_DIR|HEAD_RANKING_CSV
```

Each position receives its own source image, head ranking, and independently selected top-(N%) set. This supports experiments such as patching:

* the `green` representation at offset `-11` from `src_green_mask`
* the `L` representation at offset `-9` from `src_L_mask`

in the same conjunctive target forward pass.

### Outputs

The main CSV contains one row per trial and patch percentage, including:

* selected head sets and their actual count
* source, target, and patched margins
* `delta_margin` and normalized `recovery`
* accuracy and whether patching flips an incorrect target answer to correct
* source dataset and ranking file used at each token offset

The script additionally saves:

* `*_percentage_summary.csv`: results averaged by patch percentage.
* `*_ranking_used.csv` or `*_target_present_ranking_used.csv`: the exact ranking(s) used for the run.

### Example: patch top heads at the `L` token

```bash
python patch_pct_top_heads.py \
  --source_condition disjunctive \
  --target_condition conjunctive \
  --source_distractors 40 \
  --target_distractors 40 \
  --token_offset -9 \
  --head_ranking_csv ../Results/disj40_to_conj40_headpatch_qwen_-9.csv \
  --ranking_metric delta_margin \
  --rank_by_target_presence \
  --patch_percentages 0.1,0.5,1,3,5,10,20,50,100 \
  --out_path ../Results/disj40_to_conj40_cumulative_offset_-9.csv
```

### Example: jointly patch `green` and `L`

```bash
python patch_pct_top_heads.py \
  --source_condition disjunctive \
  --target_condition conjunctive \
  --source_distractors 40 \
  --target_distractors 40 \
  --patch_config='-11|../search-dataset/visual_search_dataset_counterfactual/src_green_mask|../Results/disj35_T_to_conj35_headpatch_qwen_-11.csv' \
  --patch_config='-9|../search-dataset/visual_search_dataset_counterfactual/src_L_mask|../Results/disj40_to_conj40_headpatch_qwen_-9.csv' \
  --ranking_metric delta_margin \
  --rank_by_target_presence \
  --always_use_target_present_ranking \
  --patch_percentages 0.1,0.5,1,3,5,10,20,50,100 \
  --out_path ../Results/green_L_mask_to_conj40.csv
```

Use `--random_heads` with the same percentages as a random-head control.

## Minimal Pair Swap Patch Percent Top Heads (Evidence for atomic feature spatial maps)

This script is a streamlined version of cumulative head patching for testing whether heads identified on the visual-search task transfer to the **minimal-pair swap (`MP_S`) dataset**. This dataset contain a single green T and a red L with the goal of making the model hallucinate a green L.

It patches the full attention-head output immediately before `o_proj` at one token offset, using head rankings from a prior visual-search head-patching run.

### What it adds

Relative to the preceding cumulative-patching script, this version is designed for two controls:

* **Transfer:** Do heads ranked as causal for disjunctive → conjunctive recovery also improve the matched `original` → `counterfactual_swap` minimal-pair examples?
* **Unpaired-source control:** Does recovery require copying activations from the *matching* original image, rather than an arbitrary original image?

By default, it evaluates both:

* `selection_type = ranked`: the top heads from the supplied ranking CSV.
* `selection_type = random`: an equally sized, seed-controlled random head set.

Use `--skip_random_control` to run only the ranked condition.

### Unpaired counterfactual control

With `--unpaired_counterfactuals`, each minimal-pair target is patched from the next original trial in a cyclic ordering rather than its matched original image.

If matched patching succeeds but unpaired patching does not, that supports the claim that recovery depends on spatial positions of objects.

### Example: matched transfer experiment (patch heads over L position)

```bash
python mp_s_patch_pct_top_attention_components.py \
  --dataset MP_S \
  --token_offset -9 \
  --head_ranking_csv ../Results/disj40_to_conj40_headpatch_qwen_-9.csv \
  --ranking_metric delta_margin \
  --rank_by_target_presence \
  --patch_percentages 0.1,0.2,0.5,1,10,25,50,100 \
  --out_path ../Results/mp_s_present_ranked_heads_offset_-9.csv
```

### Example: unpaired-source control

```bash
python mp_s_patch_pct_top_attention_components.py \
  --dataset MP_S \
  --token_offset -9 \
  --head_ranking_csv ../Results/disj40_to_conj40_headpatch_qwen_-9.csv \
  --ranking_metric delta_margin \
  --rank_by_target_presence \
  --patch_percentages 0.1,0.2,0.5,1,10,25,50,100 \
  --unpaired_counterfactuals \
  --out_path ../Results/mp_s_present_ranked_heads_unpaired_offset_-9.csv
```

The output CSV records the patch percentage, ranked or random selection type, matched or unpaired source pairing, margins, accuracy, and recovery for every trial.

## PCA on top attention heads

`extract_ranked_head_outputs_pca.py` is an **observational analysis** script: it does not patch or alter the model. Instead, it extracts the output vector of selected attention heads and visualizes how those vectors differ across matched disjunctive and conjunctive images.

The extracted vector is each head’s attention output immediately before `o_proj`—the head-specific, attention-weighted value representation.

This allows us to see how these heads representations differ under conjunctive vs dijsunctive AND target present vs target absent.

### Procedure

The script:

1. Selects the top `--top_k` heads from a prior head-ranking CSV.
2. Extracts each selected head’s output at `--token_offset` for both images in every matched pair.
3. Fits PCA **separately for each head** across all retained examples.
4. Colors points by condition and uses marker shape to distinguish target-present from target-absent trials.

This lets you assess whether a causally important head encodes distinguishable representations in the easy disjunctive versus difficult conjunctive condition.

### Outputs

The output directory contains:

* `selected_heads.csv`: heads selected from the ranking file.
* `head_outputs.pt`: raw tensor of shape `[examples, selected_heads, head_dim]`, plus metadata.
* `head_outputs_long.csv`: trial-level metadata and output-vector norms.
* `pca_coordinates.csv`: PC1/PC2/PC3 coordinates for every example and head.
* `pca_explained_variance.csv`: explained variance of each head’s principal components.
* `pca_L{layer}H{head}.png`: one PCA scatter plot per selected head.

### Example
runs PCA on attention heads ranked by target presence case

```bash
python extract_ranked_head_outputs_pca.py \
  --ranking_csv ../Results/disj40_to_conj40_headpatch_qwen_-9.csv \
  --ranking_metric delta_margin \
  --ranking_target_present true \
  --top_k 8 \
  --distractors 40 \
  --token_offset -9 \
  --target_presence all \
  --pair_filter all \
  --out_dir ../Results/disj40_to_conj40_head_outputs_top8_offset_-9
```

