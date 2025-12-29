# Architecture Documentation

## Overview

The corpus-based framework is organized into a modular, professional structure following the methodology described in the thesis. Each component has clear responsibilities and can be used independently.

## Package Structure

```
corpus_based/
├── __init__.py          # Public API exports
├── extraction.py        # Neural activation extraction
├── pooling.py           # Pooling operations
├── lda.py               # Linear Discriminant Analysis
├── projection.py        # Manifold projection
├── visualization.py     # Plotting utilities
└── analyzer.py          # Main orchestrator

data/
├── __init__.py
├── datasets.py          # Dataset loading (ParlaSent)
└── activation_cache.py  # Activation caching

utils/
└── env_loader.py        # Environment variable loading
```

## Core Components

### 1. Activation Extraction (`extraction.py`)

**Purpose**: Extract neural activations from LLM feed-forward networks.

**Key Design Decisions**:
- **Immediate Pooling**: Activations are pooled immediately in the hook function to handle variable-length sequences naturally
- **Layer Filtering**: Layers are filtered during extraction, not after, for efficiency
- **Multiple Activation Types**: Supports MLP, attention, and residual/hidden states

**Main Class**: `ActivationExtractor`

**Flow**:
1. Hook function extracts layer index from hook name
2. Filters by target layers (if specified)
3. Gets activation tensor `[seq_len, hidden_dim]`
4. **Pools immediately** using specified strategy → `[hidden_dim]`
5. Stores in dictionary by layer index
6. Returns `[N, L, D]` where N=texts, L=layers, D=activation dimension

### 2. Pooling (`pooling.py`)

**Purpose**: Convert variable-length token-level activations into fixed-dimension representations.

**Strategy Pattern**: Abstract base class with concrete implementations:
- `MeanPooling` (default, as per methodology)
- `MaxPooling`
- `LastTokenPooling`

**Key Design Decision**: Pooling happens in the extraction hook, ensuring:
- Variable-length sequences handled naturally
- Memory efficient (no full sequences stored)
- Uses correct pooling type from configuration

### 3. LDA Analysis (`lda.py`)

**Purpose**: Perform Linear Discriminant Analysis to discover optimal ideological manifolds.

**Key Features**:
- Multi-class LDA for discovering separability dimensions
- Layer-wise analysis support
- Automatic component selection
- Class filtering based on minimum samples

**Main Class**: `LDAAnalyzer`

### 4. Projection (`projection.py`)

**Purpose**: Project activations onto discovered ideological manifolds.

**Main Class**: `ManifoldProjector`

### 5. Visualization (`visualization.py`)

**Purpose**: Plotting and visualization utilities.

**2D Styles**:
- `centroids`: Large circles for party/speaker centroids
- `heatmap`: Density contours/heatmaps showing distribution shapes

**1D Visualization**:
- Distribution plots showing single LDA dimension
- Party area histograms (proportional to number of speeches)
- Speaker centroids as square markers
- Individual speech points (optional)
- Area plots trimmed to show only regions with values > threshold

**Features**:
- 2D/3D/1D scatter plots
- Interactive hover data
- Square root scaling for speaker sizes
- Customizable colors
- Consistent styling across visualization types

**Main Class**: `CorpusVisualizer`

### 6. Analyzer (`analyzer.py`)

**Purpose**: Main orchestrator providing high-level interface.

**Main Class**: `CorpusBasedAnalyzer`

Orchestrates the complete pipeline:
1. Extract neural activations
2. Apply pooling
3. Perform LDA
4. Project onto manifolds
5. Visualize results

## Data Module

### Dataset Loading (`datasets.py`)

- `load_parlasent()`: Load ParlaSent datasets from HuggingFace
- `filter_dataset()`: Filter by country, parties, etc.

### Activation Caching (`activation_cache.py`)

**ID-Based Caching (New)**:
- `cache_activations_by_id()`: Cache activations with per-text ID tracking
  - Only extracts texts not already cached
  - Supports incremental updates when party filters change
  - Maintains `activation_index.json` mapping text IDs to batch files
- `load_activations_by_id()`: Load activations by text IDs
  - Selective loading (only requested texts)
  - Preserves DataFrame order
- `compute_text_id()`: Generate unique hash IDs for texts

**Legacy Functions** (backward compatible):
- `cache_activations()`: Sequential caching (old format)
- `load_activations()`: Load sequential cache
- `load_activations_to_dataframe()`: Load and add to DataFrame (auto-detects ID-based cache)

**Cache Structure**:
```
data/cache/{DatasetName}/{ModelName}/{DatasetSubset}/
├── activation_index.json  # Maps text_id -> batch_file
├── 0.pt                   # Batch files
├── 1.pt
└── ...
```

## Key Design Decisions

### 1. Immediate Pooling in Hook Function

**Decision**: Pool activations immediately in the extraction hook function, not after stacking.

**Rationale**:
- Handles variable-length sequences naturally
- Memory efficient (no need to store full sequences)
- Uses specified pooling strategy from configuration
- Produces consistent `[N, L, D]` shape

**Alternative Considered**: Pool after stacking, but requires padding or complex variable-length handling.

### 2. Layer Filtering During Extraction

**Decision**: Filter layers during extraction in the hook function, not after.

**Rationale**:
- More efficient (don't extract unnecessary layers)
- Cleaner code (layer selection logic in one place)
- Preserves layer order from `layer_indices` parameter
- Reduces memory usage

### 3. Modular Package Structure

**Decision**: Separate extraction, pooling, LDA, projection, and visualization into different modules.

**Rationale**:
- Single responsibility principle
- Components can be used independently
- Easy to test individual components
- Easy to extend (e.g., add new pooling strategies)
- Clear separation of concerns

### 4. Pooling Strategy Pattern

**Decision**: Use strategy pattern with abstract base class for pooling operations.

**Rationale**:
- Easy to add new pooling strategies
- Consistent interface
- Easy to test each strategy independently
- Can specify pooling type in configuration

### 5. Environment Variable for HuggingFace Token

**Decision**: Load HuggingFace token from `.env` file and set as environment variable.

**Rationale**:
- Security: Token not in code or version control
- Convenience: Set once in `.env` file
- Compatibility: `transformer_lens` reads from environment automatically
- Standard practice for sensitive credentials

### 6. Separate Dataset Loading from Activation Caching

**Decision**: Split dataset loading (`datasets.py`) from activation caching (`activation_cache.py`).

**Rationale**:
- Separation of concerns
- Can load datasets without caching, or cache without loading
- Clear what each module does
- Easier to maintain

## Usage Pattern

```python
from corpus_based import CorpusBasedAnalyzer
from data import load_parlasent, filter_dataset

# 1. Load and prepare data
df = load_parlasent(datasets=['EN'])
df_filtered = filter_dataset(df, country='UK', parties=['Conservative', 'Labour'])

# 2. Initialize analyzer
analyzer = CorpusBasedAnalyzer(
    model_name="meta-llama/Meta-Llama-3-8B",
    device="cuda"
)

# 3. Extract and pool activations
texts = df_filtered['sentence'].tolist()
pooled_activations = analyzer.extract_and_pool(texts, activation_type="mlp")

# 4. Fit LDA models
labels = df_filtered['party'].values
lda_results = analyzer.fit_lda(
    labels=labels,
    n_components=2,
    layer_indices=[8, 16, 24]
)

# 5. Visualize results
for layer_idx in [8, 16, 24]:
    fig = analyzer.visualize_2d(
        layer_idx=layer_idx,
        metadata=df_filtered,
        party_column='party',
        save_path=f"output/layer_{layer_idx}_lda_2d"
    )
```

## Activation Types

1. **`mlp`**: MLP post-activation outputs (default)
   - Hook pattern: `("mlp" in name) and ("hook_post" in name)`
   - Dimension: `d_mlp`

2. **`attention`**: Attention post-activation outputs
   - Hook pattern: `("attn" in name) and ("hook_post" in name)`
   - Dimension: `d_model`

3. **`residual`/`hidden`**: Intermediate hidden states in residual stream
   - Hook pattern: `("blocks" in name) and ("hook_resid_post" in name)`
   - Dimension: `d_model`

## Migration from Old Code

**Old**: `PoliticalSpeechAnalyzer` (monolithic class)

**New**: Modular components with `CorpusBasedAnalyzer` as main interface

**Key Changes**:
- Activation extraction is now separate and reusable
- Pooling strategies are pluggable
- LDA analysis is independent of visualization
- Visualization supports multiple plot types

## Future Considerations

### Reference-Based Framework
When organizing the reference-based framework, consider:
- Similar modular structure
- Reference corpus generation
- Reference direction optimization
- Integration with corpus-based framework

### Performance Improvements
- Batch processing for activation extraction
- Better caching strategy
- Incremental loading
- Compression for cached activations

