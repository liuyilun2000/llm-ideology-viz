# Recent Modifications Summary

This document summarizes recent modifications made to the codebase for the next agent.

## 1. 1D Distribution Visualization

### Added
- **New visualization method**: `plot_lda_1d()` in `corpus_based/visualization.py`
  - Shows single LDA dimension as 1D distribution
  - Party distributions as area histograms (proportional to number of speeches)
  - Speaker centroids as square markers at y=-0.02 (below axis)
  - Individual speech points at y=-0.04 (below axis, optional)
  - Area plots trimmed to only show regions with values > 0.001

- **Analyzer integration**: `visualize_1d()` method in `corpus_based/analyzer.py`

- **Command-line support**: 
  - `--visualize_1d`: Enable 1D visualization
  - `--lda_dimension`: Select which dimension to visualize (0-indexed, default: 0)

- **Scripts**:
  - `run_cz_parlasent_1d_experiment.sh`
  - `run_uk_parlasent_1d_experiment.sh`
  - `run_1d_visualization_experiment.sh`

### Key Features
- Area under distribution proportional to number of speeches
- Styling matches 2D (no axis labels, light gray grid/zeroline)
- Automatic disabling of 2D when 1D is specified

## 2. ID-Based Activation Caching

### Added
- **`cache_activations_by_id()`**: ID-based caching system
  - Tracks each text by unique hash ID (content + metadata)
  - Only extracts activations for texts not already cached
  - Maintains `activation_index.json` for text ID → batch file mapping
  - Supports incremental updates when party filters change

- **`load_activations_by_id()`**: Selective loading by text IDs
  - Only loads activations for texts that are cached
  - Preserves DataFrame order
  - Returns activations and valid indices

- **`compute_text_id()`**: Generate unique IDs from text content and metadata

### Benefits
- **Incremental Updates**: When adding/removing parties, only new texts are extracted
- **Cross-Experiment Reuse**: Same text cached once, reused across different filters
- **Efficient**: No need to recalculate when party selection changes

### Implementation
- Automatically detects ID-based cache vs. old sequential cache
- Falls back to old format if ID-based cache not available
- Backward compatible with existing caches

## 3. Script Reorganization

### Renamed (2D scripts now have "2d" suffix)
- `run_cz_parlasent_experiment.sh` → `run_cz_parlasent_2d_experiment.sh`
- `run_uk_parlasent_experiment.sh` → `run_uk_parlasent_2d_experiment.sh`
- `run_cross_model_experiment.sh` → `run_cross_model_2d_experiment.sh`
- `run_layer_analysis_experiment.sh` → `run_layer_analysis_2d_experiment.sh`

### Created (1D scripts)
- `run_cz_parlasent_1d_experiment.sh`
- `run_uk_parlasent_1d_experiment.sh`
- `run_1d_visualization_experiment.sh`

## 4. Visualization Improvements

### 1D Visualization
- Area histograms proportional to number of speeches
- Speaker offset: -0.02 (below axis)
- Speech offset: -0.04 (below axis)
- Area trimming: Only plots regions with values > 0.001
- Styling: Matches 2D (no axis labels, light gray grid/zeroline)

### Fixed Issues
- Area plot rendering (indentation fix)
- Area proportionality (scaling fix)

## 5. Experiment Script Updates

### Modified: `scripts/run_corpus_based_experiment.py`
- Added `--visualize_1d` and `--lda_dimension` arguments
- Automatic 2D disabling when 1D is specified
- ID-based cache detection and usage
- Incremental caching support

## Files Modified

### Core Code
- `corpus_based/visualization.py`: Added `plot_lda_1d()` method
- `corpus_based/analyzer.py`: Added `visualize_1d()` method
- `data/activation_cache.py`: Added ID-based caching functions
- `data/__init__.py`: Updated exports
- `scripts/run_corpus_based_experiment.py`: Added 1D support and ID-based caching

### Scripts
- Created: `run_cz_parlasent_1d_experiment.sh`
- Created: `run_uk_parlasent_1d_experiment.sh`
- Created: `run_1d_visualization_experiment.sh`
- Renamed: All 2D scripts now have "_2d" suffix

### Documentation
- `CHANGELOG.md`: Added new section for 1D visualization and ID-based caching
- `STATUS.md`: Updated completed features
- `EXPERIMENT_SETUP.md`: Added 1D visualization examples and ID-based caching info
- `ARCHITECTURE.md`: Updated visualization and caching sections

## Usage Notes

### 1D Visualization
```bash
# Run 1D experiment
bash scripts/run_uk_parlasent_1d_experiment.sh

# Or with Python
python scripts/run_corpus_based_experiment.py \
    --visualize_1d \
    --lda_dimension 0 \
    --show_individual_speeches
```

### ID-Based Caching
- Automatically used when `--save_activations` is enabled
- First run: Extracts and caches all texts
- Subsequent runs: Only extracts new texts when party filters change
- Cache location: `data/cache/{DatasetName}/{ModelName}/{DatasetSubset}/`

## Backward Compatibility

- Old sequential cache format still supported
- Scripts automatically detect cache type
- Old scripts continue to work (renamed for clarity)

