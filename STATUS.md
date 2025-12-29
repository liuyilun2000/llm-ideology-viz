# Project Status

## Current State (2024)

### ✅ Completed

**Corpus-Based Framework**
- ✅ Modular package structure with clear separation of concerns
- ✅ Neural activation extraction from LLM feed-forward networks
- ✅ Multiple pooling strategies (mean, max, last token)
- ✅ Linear Discriminant Analysis (LDA) for manifold discovery
- ✅ 2D/3D/1D visualization with interactive plots
- ✅ ParlaSent dataset integration
- ✅ ID-based activation caching with incremental updates
- ✅ Comprehensive experiment scripts with CLI
- ✅ Environment variable management for HuggingFace tokens

**Key Features**
- Supports MLP, attention, and residual/hidden state activations
- Layer-wise analysis (single, multiple, or all layers)
- Variable-length sequence handling via immediate pooling
- Multiple visualization styles:
  - 2D: centroids, heatmap/contours
  - 1D: distribution plots with area histograms
- Cross-model comparison support
- ID-based caching: incremental updates when party filters change

**Documentation**
- ✅ Architecture documentation
- ✅ Experiment setup guide
- ✅ Design decisions documented
- ✅ Code reorganization completed

### ⏳ Pending

**Reference-Based Framework**
- ⏳ Framework organization (not started)
- ⏳ Reference corpus generation utilities
- ⏳ Reference direction optimization
- ⏳ Integration with corpus-based framework

**Improvements**
- ⏳ Batch processing for activation extraction
- ⏳ Enhanced error handling
- ⏳ Structured logging
- ⏳ Unit tests
- ⏳ Performance optimization for large datasets

## Codebase Structure

```
llm-ideology-viz/
├── corpus_based/          # ✅ Organized - Corpus-based framework
│   ├── extraction.py      # Activation extraction
│   ├── pooling.py         # Pooling strategies
│   ├── lda.py             # LDA analysis
│   ├── projection.py      # Manifold projection
│   ├── visualization.py   # Plotting utilities
│   └── analyzer.py        # Main orchestrator
├── data/                  # ✅ Organized - Data management
│   ├── datasets.py        # Dataset loading
│   └── activation_cache.py # Activation caching
├── utils/                 # ✅ Organized - Utilities
│   └── env_loader.py      # Environment variable loading
├── scripts/               # ✅ Organized - Experiment scripts
│   └── run_corpus_based_experiment.py
└── examples/              # ✅ Examples available
    └── corpus_based_example.py
```

## Supported Features

### Activation Types
- `mlp`: MLP post-activation outputs (default)
- `attention`: Attention post-activation outputs
- `residual`/`hidden`: Intermediate hidden states

### Pooling Strategies
- `mean`: Mean pooling (default, as per methodology)
- `max`: Max pooling
- `last`: Last token pooling

### Datasets
- ParlaSent (EN, BCS, CZ, SK, SL subsets)
- Custom dataset support via DataFrame interface

### Models
- Any HuggingFace model compatible with `transformer_lens`
- Tested with: LLaMA2-7B, LLaMA3-8B, Mistral-7B, Qwen2.5-7B

## Known Issues

1. **Variable-Length Sequences**: ✅ Fixed - Handled via immediate pooling in hook function
2. **HuggingFace Authentication**: ✅ Fixed - Automatic token loading from `.env`
3. **Layer Filtering Efficiency**: ✅ Fixed - Filtering during extraction
4. **Double Activation Extraction**: ✅ Fixed - Efficient caching system
5. **Area Plot Rendering**: ✅ Fixed - Corrected indentation in visualization code
6. **Area Proportionality**: ✅ Fixed - Area now proportional to number of speeches

## Migration Notes

The codebase was reorganized from a monolithic structure to a modular architecture. Old files (`corpus_based.py`, `load_dataset.py`) are kept for reference but are deprecated. See `ARCHITECTURE.md` for details.

## Next Steps

1. Organize reference-based framework (similar modular structure)
2. Add comprehensive unit tests
3. Implement batch processing for efficiency
4. Add structured logging
5. Performance optimization for large-scale experiments

