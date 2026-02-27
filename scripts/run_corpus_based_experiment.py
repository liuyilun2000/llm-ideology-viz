#!/usr/bin/env python3
"""
Main script for running corpus-based framework experiments.

This script processes ParlaSent datasets, extracts activations, performs LDA analysis,
and generates visualizations based on parameters specified via command-line arguments.

Usage:
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
        --output_dir output/experiments
"""

import argparse
import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from corpus_based import CorpusBasedAnalyzer, MeanPooling, MaxPooling, LastTokenPooling
from data import (
    load_parlasent, 
    filter_dataset, 
    cache_activations, 
    load_activations_to_dataframe,
    cache_activations_by_id
)
from utils.env_loader import get_hf_token, load_env_to_os


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run corpus-based ideological manifold analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Dataset parameters
    parser.add_argument(
        '--dataset_name',
        type=str,
        default='ParlaSent',
        help='Name of the dataset (currently supports: ParlaSent)'
    )
    parser.add_argument(
        '--dataset_subset',
        type=str,
        required=True,
        help='Subset of the dataset (e.g., EN, BCS, CZ, SK, SL for ParlaSent)'
    )
    parser.add_argument(
        '--text_column',
        type=str,
        default='sentence',
        help='Column name containing text/sentences'
    )
    parser.add_argument(
        '--party_column',
        type=str,
        default='party',
        help='Column name containing party labels'
    )
    parser.add_argument(
        '--country',
        type=str,
        default=None,
        help='Filter by country (e.g., UK, DE)'
    )
    parser.add_argument(
        '--parties',
        type=str,
        nargs='+',
        default=None,
        help='List of parties to include in analysis'
    )
    parser.add_argument(
        '--min_speeches_per_party',
        type=int,
        default=1,
        help='Minimum number of speeches required per party'
    )
    
    # Model parameters
    parser.add_argument(
        '--model_name',
        type=str,
        required=True,
        help='Model name (e.g., meta-llama/Meta-Llama-3-8B)'
    )
    parser.add_argument(
        '--cache_dir',
        type=str,
        default=None,
        help='Directory to cache the model'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to run the model on'
    )
    
    # Activation parameters
    parser.add_argument(
        '--activation_type',
        type=str,
        default='mlp',
        choices=['mlp', 'attention', 'residual', 'hidden'],
        help='Type of activation to extract (mlp=MLP post-activation, attention=attention post-activation, residual/hidden=intermediate hidden states between blocks)'
    )
    parser.add_argument(
        '--pooling_type',
        type=str,
        default='mean',
        choices=['mean', 'max', 'last'],
        help='Pooling strategy to use'
    )
    parser.add_argument(
        '--use_cached_activations',
        action='store_true',
        help='Use cached activations if available'
    )
    parser.add_argument(
        '--activations_cache_dir',
        type=str,
        default=None,
        help='Directory to cache/load activations (auto-generated if not specified)'
    )
    
    # LDA parameters
    parser.add_argument(
        '--n_components',
        type=int,
        default=2,
        help='Number of LDA components (manifold dimension)'
    )
    parser.add_argument(
        '--layers',
        type=int,
        nargs='+',
        default=None,
        help='Specific layers to analyze (if not specified, analyzes all layers)'
    )
    parser.add_argument(
        '--min_samples_per_class',
        type=int,
        default=1,
        help='Minimum samples required per class for LDA'
    )
    
    # Output parameters
    parser.add_argument(
        '--output_dir',
        type=str,
        default='output/experiments',
        help='Directory to save outputs'
    )
    parser.add_argument(
        '--save_activations',
        action='store_true',
        help='Save extracted activations to cache'
    )
    parser.add_argument(
        '--save_projections',
        action='store_true',
        help='Save LDA projections to file'
    )
    parser.add_argument(
        '--visualize_2d',
        action='store_true',
        default=True,
        help='Generate 2D visualizations (default: True, but disabled if --visualize_1d is specified)'
    )
    parser.add_argument(
        '--visualize_3d',
        action='store_true',
        help='Generate 3D visualizations (requires n_components >= 3)'
    )
    parser.add_argument(
        '--visualize_1d',
        action='store_true',
        help='Generate 1D distribution visualizations'
    )
    parser.add_argument(
        '--lda_dimension',
        type=int,
        default=0,
        help='Which LDA dimension to visualize for 1D plots (0-indexed, default: 0 for 1st dimension)'
    )
    parser.add_argument(
        '--visualization_style',
        type=str,
        default='heatmap',
        choices=['centroids', 'heatmap'],
        help='Style of 2D visualization: "centroids" (large circles) or "heatmap" (density contours)'
    )
    parser.add_argument(
        '--show_individual_speeches',
        action='store_true',
        help='Show individual speech points in visualization (default: False, only shows speakers and parties)'
    )
    parser.add_argument(
        '--party_colors',
        type=str,
        nargs='+',
        default=None,
        help='Party colors as "Party:Color" pairs (e.g., "Conservative:#0194E1")'
    )
    
    # Other parameters
    parser.add_argument(
        '--random_seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='Batch size for activation caching (number of activations per batch file, default: 1 to minimize memory usage)'
    )
    
    return parser.parse_args()


def load_party_colors(party_colors_args):
    """Parse party colors from command-line arguments."""
    if party_colors_args is None:
        return None
    
    party_colors = {}
    for pair in party_colors_args:
        if ':' in pair:
            party, color = pair.split(':', 1)
            party_colors[party] = color
        else:
            print(f"Warning: Invalid party color format: {pair}. Expected 'Party:Color'")
    
    return party_colors if party_colors else None


def align_1d_projections_across_layers(
    analyzer,
    metadata: pd.DataFrame,
    party_column: str,
    dimension: int = 0
):
    """
    Align 1D LDA projections across layers to prevent horizontal flipping.
    
    Uses the first layer as reference. For each subsequent layer, tries both
    original and flipped versions, and chooses the one that minimizes the
    total variation in party centroids compared to the reference layer.
    
    Args:
        analyzer: CorpusBasedAnalyzer instance with fitted LDA models
        metadata: DataFrame with party labels
        party_column: Column name containing party labels
        dimension: Which LDA dimension to align (0-indexed)
    """
    if len(analyzer.projections) == 0:
        return
    
    # Get sorted layer indices
    sorted_layers = sorted(analyzer.projections.keys())
    
    if len(sorted_layers) < 2:
        # Only one layer, nothing to align
        return
    
    # Check if dimension exists in all layers
    for layer_idx in sorted_layers:
        if dimension >= analyzer.projections[layer_idx].shape[1]:
            print(f"Warning: Dimension {dimension} not available in all layers. Skipping alignment.")
            return
    
    # Get party labels
    labels = metadata[party_column].values
    unique_parties = np.unique(labels)
    
    # Reference layer: first layer
    ref_layer_idx = sorted_layers[0]
    ref_projections = analyzer.projections[ref_layer_idx]
    ref_dim_values = ref_projections[:, dimension]
    
    # Calculate reference party centroids (mean per party)
    ref_party_centroids = {}
    for party in unique_parties:
        party_mask = labels == party
        ref_party_centroids[party] = np.mean(ref_dim_values[party_mask])
    
    print(f"  Alignment reference: Layer {ref_layer_idx}")
    print(f"    Reference centroids: {', '.join([f'{p}: {v:.4f}' for p, v in sorted(ref_party_centroids.items())])}")
    
    # Align subsequent layers
    for layer_idx in sorted_layers[1:]:
        layer_projections = analyzer.projections[layer_idx]
        layer_dim_values = layer_projections[:, dimension].copy()
        
        # Calculate centroids for original orientation
        original_party_centroids = {}
        for party in unique_parties:
            party_mask = labels == party
            original_party_centroids[party] = np.mean(layer_dim_values[party_mask])
        
        # Calculate centroids for flipped orientation
        flipped_dim_values = -layer_dim_values
        flipped_party_centroids = {}
        for party in unique_parties:
            party_mask = labels == party
            flipped_party_centroids[party] = np.mean(flipped_dim_values[party_mask])
        
        # Calculate total variation (sum of squared differences) for both orientations
        original_variation = sum((original_party_centroids[party] - ref_party_centroids[party])**2 
                                 for party in unique_parties)
        flipped_variation = sum((flipped_party_centroids[party] - ref_party_centroids[party])**2 
                                for party in unique_parties)
        
        # Choose orientation with smaller variation
        if flipped_variation < original_variation:
            # Flip this dimension
            analyzer.projections[layer_idx][:, dimension] *= -1
            print(f"  Layer {layer_idx}: Flipped (variation: original={original_variation:.6f}, flipped={flipped_variation:.6f})")
        else:
            print(f"  Layer {layer_idx}: Not flipped (variation: original={original_variation:.6f}, flipped={flipped_variation:.6f})")


def main():
    """Main experiment execution function."""
    # Load environment variables from .env file
    load_env_to_os()
    
    args = parse_arguments()
    
    # If visualize_1d is specified, disable 2D visualization unless explicitly requested
    # Check if --visualize_2d was explicitly provided by checking sys.argv
    visualize_2d_explicit = '--visualize_2d' in sys.argv
    if args.visualize_1d and not visualize_2d_explicit:
        args.visualize_2d = False
    
    # Get HF token
    hf_token = get_hf_token()
    
    # Set random seed
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 80)
    print("Corpus-Based Ideological Manifold Analysis")
    print("=" * 80)
    print(f"Dataset: {args.dataset_name} / {args.dataset_subset}")
    print(f"Model: {args.model_name}")
    print(f"Activation Type: {args.activation_type}")
    print(f"Pooling Type: {args.pooling_type}")
    print(f"LDA Components: {args.n_components}")
    print(f"Layers: {args.layers if args.layers else 'All'}")
    print("=" * 80)
    
    # Step 1: Load dataset
    print("\n[1/6] Loading dataset...")
    if args.dataset_name == 'ParlaSent':
        df = load_parlasent(datasets=[args.dataset_subset])
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset_name}")
    
    print(f"Loaded {len(df)} samples from {args.dataset_subset}")
    
    # Step 2: Filter dataset
    print("\n[2/6] Filtering dataset...")
    df_filtered = filter_dataset(
        df,
        country=args.country,
        parties=args.parties,
        min_speeches_per_party=args.min_speeches_per_party
    )
    
    if len(df_filtered) == 0:
        raise ValueError("No samples remaining after filtering")
    
    print(f"Filtered to {len(df_filtered)} samples")
    print(f"Parties: {df_filtered[args.party_column].unique().tolist()}")
    
    # Step 3: Determine activation cache directory
    if args.activations_cache_dir is None:
        model_short_name = args.model_name.split('/')[-1]
        args.activations_cache_dir = f"data/cache/{args.dataset_name}/{model_short_name}/{args.dataset_subset}"
    
    os.makedirs(args.activations_cache_dir, exist_ok=True)
    
    # Step 4: Extract or load activations
    print("\n[3/6] Processing activations...")
    
    # Check for cached activations FIRST before loading model
    from data.activation_cache import check_cached_activations
    index_path = os.path.join(args.activations_cache_dir, 'activation_index.json')
    has_id_based_cache = os.path.exists(index_path)
    
    # Set up pooling strategy (needed for analyzer initialization)
    if args.pooling_type == 'mean':
        pooling_strategy = MeanPooling()
    elif args.pooling_type == 'max':
        pooling_strategy = MaxPooling()
    elif args.pooling_type == 'last':
        pooling_strategy = LastTokenPooling()
    else:
        raise ValueError(f"Unsupported pooling type: {args.pooling_type}")
    
    # Initialize analyzer with lazy model loading
    # Model will only be loaded if we need to extract activations
    analyzer = CorpusBasedAnalyzer(
        model_name=args.model_name,
        cache_dir=args.cache_dir,
        device=args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu',
        pooling_strategy=pooling_strategy,
        random_seed=args.random_seed,
        hf_token=hf_token
    )
    
    if has_id_based_cache:
        # Use ID-based caching system
        print(f"Using ID-based activation cache from {args.activations_cache_dir}...")
        
        # Try to load cached activations (only loads texts that are cached)
        try:
            df_with_embeddings = load_activations_to_dataframe(
                args.activations_cache_dir,
                df_filtered,
                embedding_column='embedding',
                text_column=args.text_column,
                use_id_based=True
            )
            
            # Check if all texts were loaded
            if len(df_with_embeddings) < len(df_filtered):
                print(f"Only {len(df_with_embeddings)} out of {len(df_filtered)} texts found in cache.")
                print("Extracting activations for missing texts...")
                
                # Get texts that weren't cached
                # df_with_embeddings has valid_indices as its index
                cached_indices = set(df_with_embeddings.index)
                all_indices = set(df_filtered.index)
                missing_indices = all_indices - cached_indices
                missing_df = df_filtered.loc[list(missing_indices)]
                
                # Cache missing texts (model will be loaded lazily when needed)
                cache_activations_by_id(
                    df=missing_df,
                    text_column=args.text_column,
                    model_name=args.model_name,
                    output_dir=args.activations_cache_dir,
                    activation_type=args.activation_type,
                    pooling_type=args.pooling_type,
                    device=args.device,
                    batch_size=args.batch_size,
                    cache_dir=None,
                    hf_token=hf_token,
                    model=analyzer.get_model()  # Reuse model to avoid double loading
                )
                
                # Reload all activations
                df_with_embeddings = load_activations_to_dataframe(
                    args.activations_cache_dir,
                    df_filtered,
                    embedding_column='embedding',
                    text_column=args.text_column,
                    use_id_based=True
                )
            
            # Convert to numpy if needed
            if len(df_with_embeddings) > 0:
                first_embedding = df_with_embeddings['embedding'].iloc[0]
                if isinstance(first_embedding, torch.Tensor):
                    df_with_embeddings['embedding'] = df_with_embeddings['embedding'].apply(
                        lambda x: x.cpu().numpy() if isinstance(x, torch.Tensor) else x
                    )
            
            # Analyzer already initialized (model won't be loaded since we have cached activations)
            analyzer.pooled_activations = np.stack(df_with_embeddings['embedding'].values)
            
            # Set layer_indices
            if args.layers is not None:
                if isinstance(args.layers, int):
                    analyzer.layer_indices = [args.layers]
                else:
                    analyzer.layer_indices = args.layers
            else:
                num_layers = analyzer.pooled_activations.shape[1]
                analyzer.layer_indices = list(range(num_layers))
            
            print(f"Loaded activations: shape {analyzer.pooled_activations.shape}")
            print(f"Layer indices: {analyzer.layer_indices}")
            
        except Exception as e:
            print(f"Error loading ID-based cache: {e}")
            print("Falling back to extraction...")
            has_id_based_cache = False
    
    if not has_id_based_cache:
        # Check old cache format or extract new
        is_cached_valid, cache_metadata = check_cached_activations(
            output_dir=args.activations_cache_dir,
            model_name=args.model_name,
            activation_type=args.activation_type,
            pooling_type=args.pooling_type,
            expected_num_samples=len(df_filtered)
        )
        
        if is_cached_valid and args.use_cached_activations:
            # Use old cache format
            print(f"Loading cached activations from {args.activations_cache_dir}...")
            df_with_embeddings = load_activations_to_dataframe(
                args.activations_cache_dir,
                df_filtered,
                embedding_column='embedding',
                use_id_based=False
            )
            
            # Convert to numpy if needed
            if len(df_with_embeddings) > 0:
                first_embedding = df_with_embeddings['embedding'].iloc[0]
                if isinstance(first_embedding, torch.Tensor):
                    df_with_embeddings['embedding'] = df_with_embeddings['embedding'].apply(
                        lambda x: x.cpu().numpy() if isinstance(x, torch.Tensor) else x
                    )
            
            # Analyzer already initialized (model won't be loaded since we have cached activations)
            analyzer.pooled_activations = np.stack(df_with_embeddings['embedding'].values)
            
            # Set layer_indices
            if args.layers is not None:
                if isinstance(args.layers, int):
                    analyzer.layer_indices = [args.layers]
                else:
                    analyzer.layer_indices = args.layers
            else:
                num_layers = analyzer.pooled_activations.shape[1]
                analyzer.layer_indices = list(range(num_layers))
            
            print(f"Loaded activations: shape {analyzer.pooled_activations.shape}")
        else:
            # Extract new activations using ID-based caching
            # Analyzer already initialized - model will be loaded lazily when needed
            if cache_metadata is not None:
                print(f"Warning: Cached activations found but parameters don't match. Re-extracting...")
            else:
                print("No valid cached activations found. Extracting activations (this may take a while)...")
            
            # Use ID-based caching for new extractions
            if args.save_activations:
                print(f"Caching activations with ID-based system to {args.activations_cache_dir}...")
                cache_activations_by_id(
                    df=df_filtered,
                    text_column=args.text_column,
                    model_name=args.model_name,
                    output_dir=args.activations_cache_dir,
                    activation_type=args.activation_type,
                    pooling_type=args.pooling_type,
                    device=args.device,
                    batch_size=args.batch_size,
                    cache_dir=None,
                    hf_token=hf_token,
                    model=analyzer.get_model()  # Reuse model to avoid double loading
                )
                
                # Load the cached activations
                df_with_embeddings = load_activations_to_dataframe(
                    args.activations_cache_dir,
                    df_filtered,
                    embedding_column='embedding',
                    text_column=args.text_column,
                    use_id_based=True
                )
                
                # Convert to numpy
                if len(df_with_embeddings) > 0:
                    first_embedding = df_with_embeddings['embedding'].iloc[0]
                    if isinstance(first_embedding, torch.Tensor):
                        df_with_embeddings['embedding'] = df_with_embeddings['embedding'].apply(
                            lambda x: x.cpu().numpy() if isinstance(x, torch.Tensor) else x
                        )
                
                analyzer.pooled_activations = np.stack(df_with_embeddings['embedding'].values)
            else:
                # Extract without caching
                texts = df_filtered[args.text_column].tolist()
                pooled_activations = analyzer.extract_and_pool(
                    texts=texts,
                    activation_type=args.activation_type,
                    layer_indices=args.layers,
                    show_progress=True
                )
                analyzer.pooled_activations = pooled_activations
                df_with_embeddings = df_filtered.copy()
                df_with_embeddings['embedding'] = list(pooled_activations)
            
            # Set layer_indices
            if args.layers is not None:
                if isinstance(args.layers, int):
                    analyzer.layer_indices = [args.layers]
                else:
                    analyzer.layer_indices = args.layers
            else:
                num_layers = analyzer.pooled_activations.shape[1]
                analyzer.layer_indices = list(range(num_layers))
            
            print(f"Extracted activations: shape {analyzer.pooled_activations.shape}")
            print(f"Layer indices: {analyzer.layer_indices}")
    
    # Step 5: Fit LDA models
    print("\n[4/6] Fitting LDA models...")
    labels = df_with_embeddings[args.party_column].values
    lda_results = analyzer.fit_lda(
        labels=labels,
        n_components=args.n_components,
        min_samples_per_class=args.min_samples_per_class,
        layer_indices=args.layers
    )
    
    print(f"Fitted LDA models for {len(lda_results)} layer(s)")
    
    # Align 1D projections across layers if 1D visualization is enabled
    if args.visualize_1d and len(analyzer.projections) > 1:
        print("\nAligning 1D projections across layers to prevent horizontal flipping...")
        align_1d_projections_across_layers(
            analyzer=analyzer,
            metadata=df_with_embeddings,
            party_column=args.party_column,
            dimension=args.lda_dimension
        )
    
    # Step 6: Generate visualizations
    print("\n[5/6] Generating visualizations...")
    party_colors = load_party_colors(args.party_colors)
    
    model_short_name = args.model_name.split('/')[-1]
    experiment_name = f"{args.dataset_subset}_{model_short_name}"
    
    for layer_idx in sorted(analyzer.projections.keys()):
        layer_output_dir = os.path.join(args.output_dir, experiment_name, f"layer_{layer_idx}")
        os.makedirs(layer_output_dir, exist_ok=True)
        
        if args.visualize_2d and analyzer.projections[layer_idx].shape[1] >= 2:
            print(f"  Creating 2D visualization for layer {layer_idx} ({args.visualization_style} style)...")
            fig = analyzer.visualize_2d(
                layer_idx=layer_idx,
                metadata=df_with_embeddings,
                party_column=args.party_column,
                speaker_column='name' if 'name' in df_with_embeddings.columns else None,
                text_column=args.text_column if args.text_column in df_with_embeddings.columns else None,
                save_path=os.path.join(layer_output_dir, 'lda_2d'),
                party_colors=party_colors,
                visualization_style=args.visualization_style,
                show_individual_speeches=args.show_individual_speeches
            )
        
        if args.visualize_3d and analyzer.projections[layer_idx].shape[1] >= 3:
            print(f"  Creating 3D visualization for layer {layer_idx}...")
            fig = analyzer.visualize_3d(
                layer_idx=layer_idx,
                metadata=df_with_embeddings,
                party_column=args.party_column,
                speaker_column='name' if 'name' in df_with_embeddings.columns else None,
                save_path=os.path.join(layer_output_dir, 'lda_3d'),
                party_colors=party_colors
            )
        
        if args.visualize_1d and analyzer.projections[layer_idx].shape[1] > args.lda_dimension:
            dim_name = f"dim_{args.lda_dimension + 1}"
            print(f"  Creating 1D visualization for layer {layer_idx}, dimension {args.lda_dimension + 1}...")
            fig = analyzer.visualize_1d(
                layer_idx=layer_idx,
                metadata=df_with_embeddings,
                dimension=args.lda_dimension,
                party_column=args.party_column,
                speaker_column='name' if 'name' in df_with_embeddings.columns else None,
                text_column=args.text_column if args.text_column in df_with_embeddings.columns else None,
                save_path=os.path.join(layer_output_dir, f'lda_1d_{dim_name}'),
                party_colors=party_colors,
                show_individual_speeches=args.show_individual_speeches
            )
    
    # Step 7: Save projections if requested
    if args.save_projections:
        print("\n[6/6] Saving projections...")
        projections_dir = os.path.join(args.output_dir, experiment_name, 'projections')
        os.makedirs(projections_dir, exist_ok=True)
        
        for layer_idx, projections in analyzer.projections.items():
            np.save(
                os.path.join(projections_dir, f'layer_{layer_idx}_projections.npy'),
                projections
            )
            # Also save metadata
            metadata_path = os.path.join(projections_dir, f'layer_{layer_idx}_metadata.csv')
            df_with_embeddings.to_csv(metadata_path, index=False)
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print("Experiment Summary")
    print("=" * 80)
    stats = analyzer.analyze_dataset_composition(df_with_embeddings)
    print(f"Total samples: {stats['total_samples']}")
    if 'unique_parties' in stats:
        print(f"Unique parties: {stats['unique_parties']}")
        print("\nSpeeches per party:")
        for party, count in sorted(stats['parties'].items(), key=lambda x: x[1], reverse=True):
            pct = stats['party_percentages'][party]
            print(f"  {party}: {count} ({pct:.1f}%)")
    
    print(f"\nOutput saved to: {args.output_dir}/{experiment_name}")
    print("=" * 80)
    print("Experiment complete!")


if __name__ == "__main__":
    main()

