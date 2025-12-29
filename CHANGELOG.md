# Changelog

All notable changes to this project are documented in this file.

## [2024] - Major Reorganization

### Added
- Modular `corpus_based/` package structure
- `data/` module for dataset loading and activation caching
- `utils/env_loader.py` for environment variable management
- Comprehensive experiment scripts with CLI
- Full documentation suite

### Changed
- **Immediate Pooling**: Pool activations in hook function to handle variable-length sequences
- **Layer Filtering**: Filter during extraction for efficiency
- **HuggingFace Token**: Automatic loading from `.env` file
- **Pooling Strategy**: Strategy pattern with abstract base class
- **Dataset Loading**: Separated from activation caching

### Fixed
- Variable-length sequence handling
- HuggingFace authentication
- Import errors
- Script path issues
- Token parameter conflicts

### Deprecated
- `corpus_based.py` - Old monolithic file (kept for reference)
- `load_dataset.py` - Functionality moved to `data/`

**Note**: See `ARCHITECTURE.md` for detailed design decisions and migration guide.

## [2024] - Visualization and Performance Improvements

### Added

#### Visualization Enhancements
- **Heatmap/Contour Visualization**: New density-based visualization style showing party distributions as heatmaps/contours
- **Visualization Style Parameter**: Choose between `'centroids'` (large circles) or `'heatmap'` (density contours)
- **Individual Speech Control**: `--show_individual_speeches` flag to control visibility of individual speech points (default: False)
- **Hover Data**: Interactive tooltips showing party, speaker, and speech content (first 100 chars) for individual speeches; party and speaker for speaker centroids
- **Square Root Scaling**: Non-linear size scaling for speaker centroids to prevent overly large markers

#### Scripts
- **`scripts/regenerate_visualizations.py`**: Regenerate visualizations from cached activations without re-extraction
- **`scripts/regenerate_visualizations.sh`**: Shell script wrapper for visualization regeneration

### Changed

#### Visualization
- **Default Style**: Changed from centroids-only to heatmap style (density contours)
- **Layer Ordering**: Contours now render on top of points and speaker centroids for better visibility
- **Speaker Size Calculation**: Uses square root scaling (`base_size + sqrt(count) * scaling_factor`) instead of linear scaling
- **Individual Speech Visibility**: Hidden by default; only speakers and parties shown unless explicitly enabled

#### Activation Caching
- **`save_extracted_activations()`**: New function to save already-extracted activations without re-extraction
- **Efficient Saving**: When `--save_activations` is enabled, uses existing activations instead of re-extracting

### Fixed

1. **Double Activation Extraction**
   - **Problem**: Activations were extracted twice when `--save_activations` was enabled
   - **Solution**: Use `save_extracted_activations()` to save already-extracted activations

2. **Layer Index Mapping in LDA**
   - **Problem**: `IndexError` when accessing layers in filtered activation tensor (e.g., accessing layer 7 in tensor with only 5 layers)
   - **Solution**: Store original layer indices and map them to positions in filtered tensor during LDA fitting

3. **Layer Filtering Efficiency**
   - **Problem**: All layers were extracted then filtered, wasting computation
   - **Solution**: Filter layers during extraction by checking layer index in hook function

## [2024] - 1D Visualization and ID-Based Caching

### Added

#### 1D Distribution Visualization
- **`plot_lda_1d()`**: New 1D distribution visualization method
  - Shows single LDA dimension as distribution plot
  - Party distributions as area histograms (proportional to number of speeches)
  - Speaker centroids as aggregated square markers (below axis at -0.02)
  - Individual speech points (below axis at -0.04, optional)
  - Area plots trimmed to only show regions with values > 0.001 (configurable threshold)
- **`visualize_1d()`**: Analyzer method for 1D visualization
- **`--visualize_1d`**: Command-line flag to enable 1D visualization
- **`--lda_dimension`**: Parameter to select which LDA dimension to visualize (0-indexed, default: 0)

#### ID-Based Activation Caching
- **`cache_activations_by_id()`**: Cache activations with per-text ID tracking
  - Only extracts activations for texts not already cached
  - Supports incremental updates when parties are added/removed
  - Maintains `activation_index.json` mapping text IDs to batch files
- **`load_activations_by_id()`**: Load activations by text IDs
  - Selective loading (only requested texts)
  - Preserves DataFrame order
- **`compute_text_id()`**: Generate unique hash IDs for texts based on content and metadata
- Automatic detection and use of ID-based cache when available

#### Scripts
- **`run_cz_parlasent_1d_experiment.sh`**: Czech ParlaSent 1D visualization
- **`run_uk_parlasent_1d_experiment.sh`**: UK ParlaSent 1D visualization
- **`run_1d_visualization_experiment.sh`**: Generic 1D visualization script

### Changed

#### Script Naming
- Renamed 2D visualization scripts to include "2d" suffix:
  - `run_cz_parlasent_experiment.sh` → `run_cz_parlasent_2d_experiment.sh`
  - `run_uk_parlasent_experiment.sh` → `run_uk_parlasent_2d_experiment.sh`
  - `run_cross_model_experiment.sh` → `run_cross_model_2d_experiment.sh`
  - `run_layer_analysis_experiment.sh` → `run_layer_analysis_2d_experiment.sh`

#### Visualization
- **1D Area Plots**: Area under distribution proportional to number of speeches
- **1D Styling**: Matches 2D style (no axis labels, light gray grid/zeroline)
- **1D Positioning**: Speakers at y=-0.02, speeches at y=-0.04 (below axis)
- **Area Trimming**: Only plots regions with values > threshold (default: 0.001)

#### Activation Caching
- **ID-Based System**: New default caching system tracks individual texts by ID
- **Incremental Updates**: When party filters change, only new texts are extracted
- **Backward Compatible**: Old sequential cache format still supported
- **Automatic Detection**: Scripts automatically use ID-based cache if available

#### Experiment Script
- **1D Visualization**: Automatically disables 2D when `--visualize_1d` is specified
- **Smart Caching**: Detects ID-based cache and uses incremental updates

### Fixed

1. **Area Plot Rendering**
   - **Problem**: Area histograms not showing in 1D visualization
   - **Solution**: Fixed indentation - moved `fig.add_trace` outside conditional block

2. **Area Proportionality**
   - **Problem**: Area under distributions not proportional to number of speeches
   - **Solution**: Scale all histograms by same factor, preserving area ratios

## Future Work

### Reference-Based Framework
- Similar modular structure
- Reference corpus generation
- Reference direction optimization
- Integration with corpus-based framework

### Improvements
- Batch processing for activation extraction
- Better error handling
- Structured logging
- Unit tests
- Performance optimization

