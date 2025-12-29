# Experiment Setup Guide

Guide for running corpus-based framework experiments using the ParlaSent dataset.

## Quick Start

### Shell Scripts

**2D Visualization:**
```bash
# UK ParlaSent experiment (paper replication)
bash scripts/run_uk_parlasent_2d_experiment.sh

# Czech ParlaSent
bash scripts/run_cz_parlasent_2d_experiment.sh

# Cross-model comparison
bash scripts/run_cross_model_2d_experiment.sh

# Layer analysis
bash scripts/run_layer_analysis_2d_experiment.sh
```

**1D Visualization:**
```bash
# UK ParlaSent 1D distribution
bash scripts/run_uk_parlasent_1d_experiment.sh

# Czech ParlaSent 1D distribution
bash scripts/run_cz_parlasent_1d_experiment.sh

# Generic 1D visualization
bash scripts/run_1d_visualization_experiment.sh
```

### Python Script

```bash
python scripts/run_corpus_based_experiment.py \
    --dataset_subset EN \
    --model_name meta-llama/Meta-Llama-3-8B \
    --country UK \
    --parties Conservative Labour "Liberal Democrat" "Scottish National Party" \
    --layers 8 16 24 \
    --output_dir output/experiments
```

## Parameters

### Dataset
- `--dataset_subset`: `EN`, `BCS`, `CZ`, `SK`, `SL` (required)
- `--country`: Filter by country code (e.g., `UK`)
- `--parties`: List of parties to include
- `--min_speeches_per_party`: Minimum speeches per party (default: 1)

### Model
- `--model_name`: HuggingFace model identifier (required)
- `--device`: `cuda` or `cpu` (default: `cuda`)

### Activation
- `--activation_type`: `mlp`, `attention`, `residual`, or `hidden` (default: `mlp`)
- `--pooling_type`: `mean`, `max`, or `last` (default: `mean`)
- `--use_cached_activations`: Use cached activations if available
- `--save_activations`: Save extracted activations to cache

### LDA
- `--n_components`: Number of LDA components (default: 2)
- `--layers`: Specific layers to analyze (default: all layers)

### Output
- `--output_dir`: Output directory (default: `output/experiments`)
- `--visualize_2d`: Generate 2D scatter plots (default: True, disabled if `--visualize_1d` is set)
- `--visualize_1d`: Generate 1D distribution plots
- `--lda_dimension`: Which LDA dimension for 1D plots (0-indexed, default: 0)
- `--visualization_style`: `centroids` or `heatmap` (default: `heatmap`, for 2D only)
- `--show_individual_speeches`: Show individual speech points (default: False)
- `--save_projections`: Save LDA projections to `.npy` files

## ParlaSent Dataset

**Available Subsets:**
- `EN`: English (UK Parliament, 2015-2021) - ~5,044 speech segments
- `BCS`: Bosnian/Croatian/Serbian
- `CZ`: Czech
- `SK`: Slovak
- `SL`: Slovenian

**Columns:**
- `sentence`: Text/speech content
- `party`: Party affiliation
- `country`: Country code
- `name`: Speaker name
- `date`, `gender`, `birth_year`, `ruling`

## Example Experiments

### Paper Replication (UK ParlaSent)

```bash
python scripts/run_corpus_based_experiment.py \
    --dataset_subset EN \
    --model_name meta-llama/Meta-Llama-3-8B \
    --activation_type mlp \
    --pooling_type mean \
    --country UK \
    --parties Conservative Labour "Liberal Democrat" "Scottish National Party" \
    --n_components 2 \
    --layers 8 16 24 \
    --output_dir output/experiments/uk_llama3 \
    --save_activations \
    --save_projections \
    --visualize_2d
```

### Cross-Model Comparison

```bash
for MODEL in \
    "meta-llama/Llama-2-7b-hf" \
    "meta-llama/Meta-Llama-3-8B" \
    "mistralai/Mistral-7B-v0.3" \
    "Qwen/Qwen2.5-7B"
do
    python scripts/run_corpus_based_experiment.py \
        --dataset_subset EN \
        --model_name "$MODEL" \
        --country UK \
        --parties Conservative Labour "Liberal Democrat" "Scottish National Party" \
        --layers 2 28 \
        --output_dir "output/experiments/cross_model" \
        --use_cached_activations
done
```

### 3D Visualization

```bash
python scripts/run_corpus_based_experiment.py \
    --dataset_subset EN \
    --model_name meta-llama/Meta-Llama-3-8B \
    --country UK \
    --parties Conservative Labour "Liberal Democrat" "Scottish National Party" \
    --n_components 3 \
    --layers 16 \
    --visualize_3d \
    --output_dir output/experiments/3d_visualization
```

### 1D Distribution Visualization

```bash
python scripts/run_corpus_based_experiment.py \
    --dataset_subset EN \
    --model_name meta-llama/Meta-Llama-3-8B \
    --country UK \
    --parties Conservative Labour "Liberal Democrat" "Scottish National Party" \
    --n_components 2 \
    --layers 8 16 24 \
    --visualize_1d \
    --lda_dimension 0 \
    --show_individual_speeches \
    --output_dir output/experiments/1d_visualization
```

**1D Visualization Features:**
- Area histograms proportional to number of speeches per party
- Speaker centroids as square markers (positioned below axis)
- Individual speech points (optional, positioned below axis)
- Styling matches 2D (no axis labels, light gray grid)

## Output Structure

```
output/experiments/
└── {dataset_subset}_{model_name}/
    ├── layer_{layer_idx}/
    │   ├── lda_2d.html          # Interactive 2D plot
    │   ├── lda_2d.png           # Static 2D plot
    │   └── lda_3d.html          # Interactive 3D plot (if enabled)
    └── projections/
        ├── layer_{layer_idx}_projections.npy
        └── layer_{layer_idx}_metadata.csv
```

## Activation Caching

### ID-Based Caching (New)

The system now uses ID-based caching where each text is cached individually by its unique ID. This enables:

- **Incremental Updates**: When you add/remove parties, only new texts are extracted
- **Selective Loading**: Only loads activations for texts that are cached
- **Cross-Experiment Reuse**: Same text cached once, reused across different party filters

**First run** (extract and cache):
```bash
python scripts/run_corpus_based_experiment.py \
    --dataset_subset EN \
    --model_name meta-llama/Meta-Llama-3-8B \
    --save_activations
```

**Subsequent runs** (automatically uses cache):
```bash
# Change party filter - only new texts are extracted
python scripts/run_corpus_based_experiment.py \
    --dataset_subset EN \
    --model_name meta-llama/Meta-Llama-3-8B \
    --parties Conservative Labour  # Only extracts if new parties added
```

**Cache structure**: 
- `activation_index.json`: Maps text IDs to batch files
- `0.pt`, `1.pt`, ...: Batch files with activations

**Cache location**: `data/cache/{DatasetName}/{ModelName}/{DatasetSubset}/`

## Paper Parameters

- **Activation Type**: MLP (post-activation outputs from FFN)
- **Pooling**: Mean pooling across tokens
- **LDA Components**: 2 (for 2D visualization)
- **Layers**: 8, 16, 24 (cross-layer) or 2, 28 (cross-model)
- **Random Seed**: 42

## Troubleshooting

**Out of Memory:**
- Use `--device cpu`
- Reduce `--batch_size 50`
- Process fewer layers at once

**Missing Dependencies:**
```bash
pip install -r requirements.txt
pip install transformer_lens
```

**HuggingFace Authentication:**
```bash
huggingface-cli login
# Or set HF_TOKEN in .env file
```

**Dataset Download Issues:**
- Check internet connection
- Verify HuggingFace access
- Manual test: `python -c "from datasets import load_dataset; load_dataset('classla/ParlaSent', 'EN')"`

## See Also

- `ARCHITECTURE.md` - Code structure and design decisions
- `STATUS.md` - Current project status
- `README.md` - General project information
