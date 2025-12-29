# Ideological Neural Manifolds of Large Language Models

[![IC2S2](https://img.shields.io/badge/IC2S2-2025-8c1b13.svg)](https://arxiv.org/abs/2311.15983)

Official repository for **Ideological Neural Manifolds of Large Language Models** by Yilun Liu, Daniel Matter and Jürgen Pfeffer. Tools for analyzing and visualizing political speech embeddings using LLMs.

## Overview

Two complementary frameworks for analyzing ideological representations in LLMs:

1. **Corpus-Based Framework** ✅: Discovers optimal ideological manifolds from labeled political texts using Linear Discriminant Analysis (LDA)
2. **Reference-Based Framework** ⏳: Constructs interpretable ideological spaces using curated reference texts (coming soon)

## Quick Start

### Installation

```bash
pip install -r requirements.txt
pip install transformer_lens
```

### Running Experiments

```bash
# UK ParlaSent experiment (paper replication)
bash scripts/run_uk_parlasent_experiment.sh

# Or use Python script directly
python scripts/run_corpus_based_experiment.py \
    --dataset_subset EN \
    --model_name meta-llama/Meta-Llama-3-8B \
    --country UK \
    --parties Conservative Labour "Liberal Democrat" "Scottish National Party" \
    --layers 8 16 24 \
    --output_dir output/experiments
```

### Python API

```python
from corpus_based import CorpusBasedAnalyzer
from data import load_parlasent, filter_dataset

# Load and prepare data
df = load_parlasent(datasets=['EN'])
df_filtered = filter_dataset(df, country='UK', parties=['Conservative', 'Labour'])

# Initialize analyzer
analyzer = CorpusBasedAnalyzer(model_name="meta-llama/Meta-Llama-3-8B", device="cuda")

# Extract activations and fit LDA
texts = df_filtered['sentence'].tolist()
pooled_activations = analyzer.extract_and_pool(texts, activation_type="mlp")
lda_results = analyzer.fit_lda(
    labels=df_filtered['party'].values,
    n_components=2,
    layer_indices=[8, 16, 24]
)

# Visualize
for layer_idx in [8, 16, 24]:
    analyzer.visualize_2d(
        layer_idx=layer_idx,
        metadata=df_filtered,
        party_column='party',
        save_path=f"output/layer_{layer_idx}_lda_2d"
    )
```

## Documentation

- **[STATUS.md](STATUS.md)**: Current project status and completed features
- **[ARCHITECTURE.md](ARCHITECTURE.md)**: Code structure, design decisions, and architecture details
- **[EXPERIMENT_SETUP.md](EXPERIMENT_SETUP.md)**: Experiment setup guide and parameter reference
- **[CHANGELOG.md](CHANGELOG.md)**: Version history and changes

## Methodology

1. **Extract** neural activations from FFN layers
2. **Pool** to obtain fixed-dimension representations (mean/max/last token)
3. **Analyze** using Linear Discriminant Analysis to discover separability dimensions
4. **Project** activations onto discovered ideological manifolds
5. **Visualize** with 2D/3D interactive scatter plots

## Features

- **Activation Types**: MLP, attention, residual/hidden states
- **Pooling Strategies**: Mean (default), max, last token
- **Datasets**: ParlaSent (EN, BCS, CZ, SK, SL)
- **Models**: Any HuggingFace model compatible with `transformer_lens`
- **Visualization**: Interactive plots with centroids or heatmap styles

## Requirements

- Python 3.8+
- PyTorch, transformer_lens, scikit-learn, pandas, numpy, plotly, datasets

See `requirements.txt` for complete list.

## Citation

```bibtex
@article{liu2024ideological,
  title={Ideological Neural Manifolds of Large Language Models},
  author={Liu, Yilun and Matter, Daniel and Pfeffer, Jürgen},
  journal={IC2S2},
  year={2025}
}
```
