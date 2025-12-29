"""
Example Script: Corpus-Based Framework

This script demonstrates how to use the corpus-based framework for analyzing
ideological representations in LLMs.

Usage:
    python examples/corpus_based_example.py
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from corpus_based import CorpusBasedAnalyzer
from data import load_parlasent, filter_dataset, load_activations_to_dataframe


def main():
    """Main example function."""
    
    # Configuration
    MODEL_NAME = "meta-llama/Meta-Llama-3-8B"
    COUNTRY = "UK"
    PARTIES = [
        'Conservative',
        'Labour',
        'Liberal Democrat',
        'Scottish National Party',
    ]
    PARTY_COLORS = {
        'Conservative': '#0194E1',
        'Labour': '#DC241F',
        'Liberal Democrat': '#FAA61A',
        'Scottish National Party': '#EEE95D',
    }
    
    # Step 1: Load dataset
    print("Loading dataset...")
    df = load_parlasent(datasets=['EN'])
    
    # Step 2: Filter dataset
    print("Filtering dataset...")
    df_filtered = filter_dataset(
        df,
        country=COUNTRY,
        parties=PARTIES,
        min_speeches_per_party=1
    )
    df_filtered = df_filtered.sort_values('party')
    
    print(f"Loaded {len(df_filtered)} speeches from {len(PARTIES)} parties")
    
    # Step 3: Initialize analyzer
    analyzer = CorpusBasedAnalyzer(
        model_name=MODEL_NAME,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Step 4: Check if activations are already cached
    ACTIVATIONS_CACHE_DIR = f"data/cache/{MODEL_NAME.split('/')[-1]}/EN"
    
    if os.path.exists(ACTIVATIONS_CACHE_DIR) and len(os.listdir(ACTIVATIONS_CACHE_DIR)) > 0:
        print("Loading cached activations...")
        import torch
        from data import load_activations_to_dataframe
        
        df_with_embeddings = load_activations_to_dataframe(
            ACTIVATIONS_CACHE_DIR,
            df_filtered,
            embedding_column='embedding'
        )
        # Convert embeddings to numpy if needed
        if len(df_with_embeddings) > 0:
            first_embedding = df_with_embeddings['embedding'].iloc[0]
            if isinstance(first_embedding, torch.Tensor):
                df_with_embeddings['embedding'] = df_with_embeddings['embedding'].apply(
                    lambda x: x.cpu().numpy() if isinstance(x, torch.Tensor) else x
                )
        
        # Set pooled activations in analyzer
        analyzer.pooled_activations = np.stack(df_with_embeddings['embedding'].values)
    else:
        print("Extracting activations (this may take a while)...")
        # Extract and pool activations
        texts = df_filtered['sentence'].tolist()
        pooled_activations = analyzer.extract_and_pool(
            texts=texts,
            activation_type="mlp",
            show_progress=True
        )
        
        # Add embeddings to DataFrame
        df_with_embeddings = df_filtered.copy()
        df_with_embeddings['embedding'] = list(pooled_activations)
    
    # Step 5: Fit LDA models
    print("Fitting LDA models...")
    labels = df_with_embeddings['party'].values
    lda_results = analyzer.fit_lda(
        labels=labels,
        n_components=2,
        min_samples_per_class=1,
        layer_indices=[0, 2, 8, 16, 24]  # Analyze specific layers
    )
    
    # Step 6: Visualize results for each layer
    print("Creating visualizations...")
    for layer_idx in [0, 2, 8, 16, 24]:
        if layer_idx in analyzer.projections:
            fig = analyzer.visualize_2d(
                layer_idx=layer_idx,
                metadata=df_with_embeddings,
                party_column='party',
                speaker_column='name',
                save_path=f"output/{COUNTRY}_{MODEL_NAME.split('/')[-1]}_layer_{layer_idx}_lda_2d",
                party_colors=PARTY_COLORS
            )
            print(f"Saved visualization for layer {layer_idx}")
    
    # Step 7: Print dataset statistics
    print("\nDataset Statistics:")
    stats = analyzer.analyze_dataset_composition(df_with_embeddings)
    print(f"Total samples: {stats['total_samples']}")
    print(f"Unique parties: {stats['unique_parties']}")
    print("\nSpeeches per party:")
    for party, count in stats['parties'].items():
        print(f"  {party}: {count} ({stats['party_percentages'][party]:.1f}%)")
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    import torch
    import numpy as np
    main()

