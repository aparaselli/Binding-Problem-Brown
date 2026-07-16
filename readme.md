# Binding problem


## Data Generation
In /search-dataset there exists two ipynb notebooks for dataset generation

search_counterfact_dataset_gen.ipynb - Generates updated experiment, where disjunctive search is red and green Ls instead of red and green circles. Furthermore, between trial numbers (so for example: conjunctive_distractors_0_**trial_**000**_target_true.png and conjunctive_distractors_10_**trial_000**_target_true.png) the target will always be in the same positions. This ensures that comparison across distractors is not influenced by positional embeddings.
- The first cell creates the dataset
- The second cell verifies all images are of the same size

search_dataset_gen_old.ipynb - Generates previous version where disjunctive search was red and green circles (which was potentially a confound). aditionally across number of distractors, target position was randomized 

## Hidden state extraction 

scripts/extract_target_hidden_states.py - Loads the counterfactual visual-search dataset (if you want to run the original dataset you can change hthe file). For each example, the script reads the paired JSON metadata, maps the target object's plot/pixel location and bounding box onto the model's visual-token grid, then extracts per-layer hidden representations for the target center token, the target bounding-box tokens, and the surrounding 3x3 token window. It saves a `.pt` file containing image metadata, distractor counts, trial IDs, token-grid coordinates, token indices, and the stacked hidden-state tensors (`center_token_hidden`, `bbox_mean_hidden`, `window3x3_mean_hidden`, plus the raw bbox/window token tensors). The `--debug` flag visualizes the target-token alignment for the first examples, and `--max_examples` can limit extraction for quick tests.

Example call:
python extract_target_hidden_states.py --out_path results/target_hidden_states.pt

Binding-Problem-Brown/Analysis/activations_analysis_example.ipynb - This noteboook contains an example of loading the hidden states!

## Benchmarking

scripts/benchmark.py - Runs the selected VLM on every image in the counterfactual visual-search dataset, records the model response/prediction/correctness plus target-position metadata, and writes both a full CSV and an accuracy summary grouped by condition and distractor count.

Example call:
```bash
python scripts/benchmark.py --out_path results/benchmark.csv
```

## Attention head search/analysis

This script performs a causal attention head patching experiment on the visual search benchmark.

For each matched source-target image pair, it:

1. Runs the behavioral evaluation on both the source (e.g., 10 distractors) and target (e.g., 40 distractors) images. (This is usually very fast/cheap)
2. Caches the pre-`o_proj` activations for every attention head at the final input token from the source example.
3. Individually patches each attention head into the corresponding target example during generation.
4. Measures the change in the model's True/False decision margin after patching.
5. Computes a normalized recovery score indicating how much each head restores the source behavior.

The script automatically discovers the language decoder architecture, supports multiple vision-language models, and saves:

- **Per-head results** for every trial, layer, and head.
- **Head summary** containing aggregate recovery statistics.
- **Pair summary** summarizing the best-performing heads for each source-target image pair.

This experiment is designed to identify attention heads that causally contribute to visual search performance under increased distractor difficulty.

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--model_name` | Vision-language model to evaluate. | `Qwen/Qwen2.5-VL-7B-Instruct` |
| `--dataset_dir` | Path to the visual search dataset. | `search-dataset/visual_search_dataset_counterfactual` |
| `--out_path` | Output CSV containing per-head patching results. | **Required** |
| `--source_condition` | Source search condition used to cache attention heads (`conjunctive` or `disjunctive`). | `disjunctive` |
| `--target_condition` | Target search condition that receives the patched activations. | `conjunctive` |
| `--source_distractors` | Number of distractors in the source images. | `10` |
| `--target_distractors` | Number of distractors in the target images. | `40` |
### Example Usage

```bash
python benchmark_headpatch.py \
    --source_condition conjunctive \
    --target_condition conjunctive \
    --source_distractors 10 \
    --target_distractors 40 \
    --out_path results/headpatch.csv