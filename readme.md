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
