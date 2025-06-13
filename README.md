# LLM Ideology Visualization

This project provides tools for caching model activations and visualizing political speech embeddings using LLMs.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Prepare data and cache activations:
   ```bash
   python -m data.load_dataset
   ```

3. Run visualization:
   ```bash
   python -m data.corpus_based
   ```

## Directory Structure

- `data/` - Data loading, caching, and analysis scripts
- `bak/` - Backup and experimental scripts
- `img/` - Generated images and plots
- `requirements.txt` - Python dependencies
- `.gitignore` - Files and folders to ignore in git