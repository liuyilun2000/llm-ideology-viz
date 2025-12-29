# Experiment Scripts

This directory contains shell scripts for running corpus-based framework experiments as described in the paper.

## Available Scripts

### 1. `run_uk_parlasent_experiment.sh`
Runs the UK ParlaSent experiment with LLaMA3-8B, replicating the main experiment from the paper.

**Usage:**
```bash
./scripts/run_uk_parlasent_experiment.sh
```

**Parameters:**
- Dataset: ParlaSent EN (UK Parliament)
- Model: LLaMA3-8B
- Parties: Conservative, Labour, Liberal Democrat, Scottish National Party
- Layers: 8, 16, 24
- Output: `output/experiments/uk_parlasent/`

### 2. `run_cross_model_experiment.sh`
Runs the same experiment across multiple models for cross-model comparison.

**Usage:**
```bash
./scripts/run_cross_model_experiment.sh
```

**Models tested:**
- LLaMA2-7B
- LLaMA3-8B
- Mistral-7B
- Qwen2.5-7B

**Parameters:**
- Layers: 2 (early) and 28 (late) for comparison
- Output: `output/experiments/cross_model/`

### 3. `run_german_bundestag_experiment.sh`
Runs experiment on German Bundestag data (requires German dataset).

**Usage:**
```bash
./scripts/run_german_bundestag_experiment.sh
```

**Note:** This script assumes you have German Bundestag data in ParlaSent format or a compatible structure.

### 4. `run_layer_analysis_experiment.sh`
Analyzes all layers to study the evolution of ideological representations through model depth.

**Usage:**
```bash
./scripts/run_layer_analysis_experiment.sh
```

## Main Script

### `run_corpus_based_experiment.py`

The main Python script that all shell scripts call. Can be used directly with custom parameters.

**Usage:**
```bash
python scripts/run_corpus_based_experiment.py \
    --dataset_name ParlaSent \
    --dataset_subset EN \
    --model_name meta-llama/Meta-Llama-3-8B \
    --activation_type mlp \
    --pooling_type mean \
    --country UK \
    --parties Conservative Labour "Liberal Democrat" "Scottish National Party" \
    --n_components 2 \
    --layers 8 16 24 \
    --output_dir output/experiments \
    --save_activations \
    --save_projections \
    --visualize_2d
```

**Key Arguments:**

- **Dataset:**
  - `--dataset_name`: Dataset name (currently: ParlaSent)
  - `--dataset_subset`: Subset code (EN, BCS, CZ, SK, SL)
  - `--country`: Filter by country
  - `--parties`: List of parties to include
  - `--min_speeches_per_party`: Minimum speeches per party

- **Model:**
  - `--model_name`: HuggingFace model identifier
  - `--device`: cuda or cpu
  - `--cache_dir`: Model cache directory

- **Activation:**
  - `--activation_type`: mlp or attention
  - `--pooling_type`: mean, max, or last
  - `--use_cached_activations`: Use cached activations if available
  - `--save_activations`: Save extracted activations

- **LDA:**
  - `--n_components`: Number of LDA components (default: 2)
  - `--layers`: Specific layers to analyze (default: all)
  - `--min_samples_per_class`: Minimum samples per class

- **Output:**
  - `--output_dir`: Output directory
  - `--save_projections`: Save LDA projections
  - `--visualize_2d`: Generate 2D plots
  - `--visualize_3d`: Generate 3D plots (requires n_components >= 3)
  - `--party_colors`: Party colors as "Party:Color" pairs

## ParlaSent Dataset Subsets

The ParlaSent dataset supports the following subsets:

- **EN**: English (UK Parliament, 2015-2021)
- **EN_additional_test**: English additional test set
- **BCS**: Bosnian/Croatian/Serbian
- **BCS_additional_test**: BCS additional test set
- **CZ**: Czech
- **SK**: Slovak
- **SL**: Slovenian

## Output Structure

Experiments generate the following output structure:

```
output/experiments/
└── {dataset_subset}_{model_name}/
    ├── layer_{layer_idx}/
    │   ├── lda_2d.html
    │   ├── lda_2d.png
    │   └── lda_3d.html (if 3D visualization enabled)
    └── projections/
        ├── layer_{layer_idx}_projections.npy
        └── layer_{layer_idx}_metadata.csv
```

## Examples

### Custom Experiment

```bash
python scripts/run_corpus_based_experiment.py \
    --dataset_subset EN \
    --model_name mistralai/Mistral-7B-v0.3 \
    --country UK \
    --parties Conservative Labour \
    --layers 10 20 30 \
    --n_components 3 \
    --visualize_3d \
    --output_dir output/custom_experiment
```

### Using Cached Activations

```bash
python scripts/run_corpus_based_experiment.py \
    --dataset_subset EN \
    --model_name meta-llama/Meta-Llama-3-8B \
    --use_cached_activations \
    --layers 16 \
    --output_dir output/quick_analysis
```

## Requirements

- Python 3.8+
- All dependencies from `requirements.txt`
- GPU recommended for activation extraction (CPU supported but slow)
- Sufficient disk space for cached activations (~GB per model/dataset combination)

## Notes

- Activation extraction can take significant time depending on dataset size and model
- Use `--save_activations` to cache activations for faster subsequent runs
- Use `--use_cached_activations` to skip extraction if activations are already cached
- Party names with spaces should be quoted in shell scripts
- Colors should be specified in hex format (e.g., `#0194E1`)

