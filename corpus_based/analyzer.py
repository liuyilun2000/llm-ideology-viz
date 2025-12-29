"""
Main Analyzer Class for Corpus-Based Framework

This module provides the high-level interface for the corpus-based ideological
manifold analysis, orchestrating all components of the framework.
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional, Union

from .extraction import ActivationExtractor
from .pooling import PoolingStrategy, MeanPooling
from .lda import LDAAnalyzer
from .projection import ManifoldProjector
from .visualization import CorpusVisualizer


class CorpusBasedAnalyzer:
    """
    Main analyzer class for corpus-based ideological manifold analysis.
    
    This class orchestrates the complete pipeline:
    1. Extract neural activations from LLM
    2. Apply pooling to obtain fixed-dimension representations
    3. Perform LDA to discover ideological manifolds
    4. Project activations onto discovered manifolds
    5. Provide visualization and analysis capabilities
    """
    
    def __init__(
        self,
        model_name: str,
        cache_dir: Optional[str] = None,
        device: str = "cuda",
        pooling_strategy: Optional[PoolingStrategy] = None,
        random_seed: int = 42,
        hf_token: Optional[str] = None
    ):
        """
        Initialize the corpus-based analyzer.
        
        Args:
            model_name: Name or path of the pretrained model
            cache_dir: Directory to cache the model
            device: Device to run the model on ('cpu' or 'cuda')
            pooling_strategy: Pooling strategy to use (default: MeanPooling)
            random_seed: Random seed for reproducibility
            hf_token: HuggingFace token. If None, tries to load from .env file or environment variable
        """
        self.model_name = model_name
        self.device = device
        self.random_seed = random_seed
        
        # Initialize components
        self.extractor = ActivationExtractor(
            model_name=model_name,
            cache_dir=cache_dir,
            device=device,
            hf_token=hf_token
        )
        
        self.pooling = pooling_strategy if pooling_strategy is not None else MeanPooling()
        self.lda_analyzer = LDAAnalyzer(random_seed=random_seed)
        self.visualizer = CorpusVisualizer()
        
        # Storage for results
        self.activations = None
        self.pooled_activations = None
        self.layer_indices = None  # Store which layers were extracted
        self.lda_models = {}
        self.projections = {}
        self.metadata = None
    
    def extract_and_pool(
        self,
        texts: List[str],
        activation_type: str = "mlp",
        layer_indices: Optional[Union[int, List[int]]] = None,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Extract activations and apply pooling.
        
        Args:
            texts: List of input text strings
            activation_type: Type of activation to extract ('mlp' or 'attention')
            layer_indices: Specific layer(s) to extract from
            show_progress: Whether to show progress bar
        
        Returns:
            Pooled activations of shape [N, L, D] where
            N = number of texts, L = number of layers, D = activation dimension
        """
        # Extract activations: [N, L, D] (already pooled in the hook)
        # Pooling is applied immediately in the extraction hook to handle variable-length sequences
        activations = self.extractor.extract_activations(
            texts=texts,
            activation_type=activation_type,
            layer_indices=layer_indices,
            show_progress=show_progress,
            pooling_strategy=self.pooling  # Use the analyzer's pooling strategy
        )
        
        # Convert to numpy if needed
        if isinstance(activations, torch.Tensor):
            activations = activations.cpu().numpy()
        
        # Activations should already be pooled in the extraction hook
        # Shape should be [N, L, D]
        if activations.ndim == 3:
            # Already pooled: [N, L, D]
            pooled = activations
        else:
            raise ValueError(
                f"Unexpected activation shape: {activations.shape}. "
                f"Expected 3D [N, L, D] (pooled). Activations should be pooled immediately in the extraction hook."
            )
        
        # Store layer indices for later mapping
        if layer_indices is None:
            # All layers were extracted, so indices are 0, 1, 2, ..., num_layers-1
            self.layer_indices = list(range(pooled.shape[1]))
        elif isinstance(layer_indices, int):
            self.layer_indices = [layer_indices]
        else:  # list
            self.layer_indices = layer_indices
        
        self.activations = activations
        self.pooled_activations = pooled
        
        return pooled
    
    def fit_lda(
        self,
        labels: np.ndarray,
        n_components: Optional[int] = None,
        min_samples_per_class: int = 1,
        layer_indices: Optional[Union[int, List[int]]] = None
    ) -> Dict[int, Tuple]:
        """
        Fit LDA models to discover ideological manifolds.
        
        Args:
            labels: Class labels of shape [N] (party affiliations, etc.)
            n_components: Number of LDA components (default: min(n_classes-1, D))
            min_samples_per_class: Minimum samples required per class
            layer_indices: Specific layer(s) to analyze (if None, uses all layers)
        
        Returns:
            Dictionary mapping layer indices to (LDA model, transformed activations)
        """
        if self.pooled_activations is None:
            raise ValueError("Must call extract_and_pool first")
        
        if self.layer_indices is None:
            raise ValueError("Layer indices not stored. This should not happen.")
        
        activations = self.pooled_activations  # [N, L, D]
        num_layers = activations.shape[1]
        
        # Create mapping from original layer indices to positions in the filtered tensor
        # e.g., if layer_indices=[0, 3, 7, 11, 15], then:
        #   layer 0 -> position 0, layer 3 -> position 1, layer 7 -> position 2, etc.
        layer_to_position = {orig_idx: pos for pos, orig_idx in enumerate(self.layer_indices)}
        
        # Determine which layers to process
        if layer_indices is None:
            # Process all extracted layers
            layers_to_process = self.layer_indices.copy()
        elif isinstance(layer_indices, int):
            layers_to_process = [layer_indices]
        elif isinstance(layer_indices, list):
            layers_to_process = layer_indices
        else:
            raise ValueError(f"Invalid layer_indices: {layer_indices}")
        
        # Validate that all requested layers were actually extracted
        missing_layers = [idx for idx in layers_to_process if idx not in layer_to_position]
        if missing_layers:
            raise ValueError(
                f"Requested layers {missing_layers} were not extracted. "
                f"Extracted layers: {self.layer_indices}"
            )
        
        results = {}
        
        for orig_layer_idx in layers_to_process:
            # Map original layer index to position in the filtered tensor
            position = layer_to_position[orig_layer_idx]
            layer_activations = activations[:, position, :]  # [N, D]
            
            lda_model, transformed = self.lda_analyzer.fit(
                activations=layer_activations,
                labels=labels,
                n_components=n_components,
                min_samples_per_class=min_samples_per_class
            )
            
            # Store LDA model for this layer (use original layer index, not position)
            self.lda_analyzer.lda_models[orig_layer_idx] = lda_model
            
            # Create projector
            projector = ManifoldProjector(lda_model.scalings_)
            
            results[orig_layer_idx] = (lda_model, transformed, projector)
            self.lda_models[orig_layer_idx] = lda_model
            self.projections[orig_layer_idx] = transformed
        
        return results
    
    def project_activations(
        self,
        activations: Optional[np.ndarray] = None,
        layer_idx: int = 0
    ) -> np.ndarray:
        """
        Project activations onto a previously fitted LDA manifold.
        
        Args:
            activations: Activations to project (if None, uses stored pooled activations)
            layer_idx: Layer index to use for projection
        
        Returns:
            Projected activations in LDA space
        """
        if layer_idx not in self.lda_models:
            raise ValueError(f"No LDA model found for layer {layer_idx}")
        
        if activations is None:
            if self.pooled_activations is None:
                raise ValueError("No activations available. Call extract_and_pool first.")
            activations = self.pooled_activations[:, layer_idx, :]  # [N, D]
        
        lda_model = self.lda_models[layer_idx]
        projector = ManifoldProjector(lda_model.scalings_)
        
        return projector.project(activations)
    
    def analyze_dataset_composition(self, metadata: pd.DataFrame) -> Dict:
        """
        Analyze basic composition of the dataset.
        
        Args:
            metadata: DataFrame with metadata columns
        
        Returns:
            Dictionary containing various statistics
        """
        stats = {
            'total_samples': len(metadata),
        }
        
        # Party statistics if available
        if 'party' in metadata.columns:
            stats['unique_parties'] = len(metadata['party'].unique())
            stats['parties'] = metadata['party'].value_counts().to_dict()
            stats['party_percentages'] = (
                metadata['party'].value_counts() / len(metadata) * 100
            ).round(2).to_dict()
        
        # Country statistics if available
        if 'country' in metadata.columns:
            stats['countries'] = metadata['country'].value_counts().to_dict()
            stats['parties_by_country'] = {
                country: len(metadata[metadata['country'] == country]['party'].unique())
                for country in metadata['country'].unique()
            }
        
        # Additional metadata statistics
        for col in ['gender', 'ruling']:
            if col in metadata.columns:
                stats[f'{col}_distribution'] = metadata[col].value_counts().to_dict()
        
        # Text length statistics if available
        if 'sentence' in metadata.columns:
            stats['text_length'] = {
                'mean': metadata['sentence'].str.len().mean(),
                'median': metadata['sentence'].str.len().median(),
                'min': metadata['sentence'].str.len().min(),
                'max': metadata['sentence'].str.len().max()
            }
        
        return stats
    
    def visualize_2d(
        self,
        layer_idx: int,
        metadata: pd.DataFrame,
        party_column: str = 'party',
        speaker_column: Optional[str] = None,
        text_column: Optional[str] = None,
        save_path: Optional[str] = None,
        party_colors: Optional[Dict[str, str]] = None,
        visualization_style: str = 'heatmap',
        show_individual_speeches: bool = False
    ):
        """
        Create a 2D visualization of LDA projections.
        
        Args:
            layer_idx: Layer index to visualize
            metadata: DataFrame with metadata
            party_column: Column name for party labels
            speaker_column: Optional column name for speaker labels
            text_column: Optional column name for text/speech content (for hover data)
            save_path: Optional path to save the plot
            party_colors: Optional dictionary mapping party names to colors
            visualization_style: Style of visualization - 'centroids' (large circles) or 'heatmap' (density contours)
            show_individual_speeches: Whether to show individual speech points (default: False, only shows speakers and parties)
        """
        if layer_idx not in self.projections:
            raise ValueError(f"No projections found for layer {layer_idx}")
        
        projections = self.projections[layer_idx]
        
        if projections.shape[1] < 2:
            raise ValueError("Need at least 2 LDA components for 2D visualization")
        
        # Use first 2 components
        projections_2d = projections[:, :2]
        
        if party_colors:
            self.visualizer.party_colors = party_colors
        
        return self.visualizer.plot_lda_2d(
            projections=projections_2d,
            metadata=metadata,
            party_column=party_column,
            speaker_column=speaker_column,
            text_column=text_column,
            save_path=save_path,
            title=f"Corpus-Based Ideological Manifold (Layer {layer_idx})",
            visualization_style=visualization_style,
            show_individual_speeches=show_individual_speeches
        )
    
    def visualize_3d(
        self,
        layer_idx: int,
        metadata: pd.DataFrame,
        party_column: str = 'party',
        speaker_column: Optional[str] = None,
        save_path: Optional[str] = None,
        party_colors: Optional[Dict[str, str]] = None
    ):
        """
        Create a 3D visualization of LDA projections.
        
        Args:
            layer_idx: Layer index to visualize
            metadata: DataFrame with metadata
            party_column: Column name for party labels
            speaker_column: Optional column name for speaker labels
            save_path: Optional path to save the plot
            party_colors: Optional dictionary mapping party names to colors
        """
        if layer_idx not in self.projections:
            raise ValueError(f"No projections found for layer {layer_idx}")
        
        projections = self.projections[layer_idx]
        
        if projections.shape[1] < 3:
            raise ValueError("Need at least 3 LDA components for 3D visualization")
        
        # Use first 3 components
        projections_3d = projections[:, :3]
        
        if party_colors:
            self.visualizer.party_colors = party_colors
        
        return self.visualizer.plot_lda_3d(
            projections=projections_3d,
            metadata=metadata,
            party_column=party_column,
            speaker_column=speaker_column,
            save_path=save_path,
            title=f"Corpus-Based Ideological Manifold 3D (Layer {layer_idx})"
        )
    
    def visualize_1d(
        self,
        layer_idx: int,
        metadata: pd.DataFrame,
        dimension: int = 0,
        party_column: str = 'party',
        speaker_column: Optional[str] = None,
        text_column: Optional[str] = None,
        save_path: Optional[str] = None,
        party_colors: Optional[Dict[str, str]] = None,
        show_individual_speeches: bool = False
    ):
        """
        Create a 1D distribution visualization of a single LDA dimension.
        
        Shows parties as area histograms/distributions, speakers as aggregated points,
        and individual speeches as points.
        
        Args:
            layer_idx: Layer index to visualize
            metadata: DataFrame with metadata
            dimension: Which LDA dimension to visualize (0-indexed, default: 0 for 1st dimension)
            party_column: Column name for party labels
            speaker_column: Optional column name for speaker labels
            text_column: Optional column name for text/speech content (for hover data)
            save_path: Optional path to save the plot
            party_colors: Optional dictionary mapping party names to colors
            show_individual_speeches: Whether to show individual speech points
        """
        if layer_idx not in self.projections:
            raise ValueError(f"No projections found for layer {layer_idx}")
        
        projections = self.projections[layer_idx]
        
        if dimension >= projections.shape[1]:
            raise ValueError(f"Dimension {dimension} not available. Projections have {projections.shape[1]} dimensions.")
        
        if party_colors:
            self.visualizer.party_colors = party_colors
        
        return self.visualizer.plot_lda_1d(
            projections=projections,
            metadata=metadata,
            dimension=dimension,
            party_column=party_column,
            speaker_column=speaker_column,
            text_column=text_column,
            save_path=save_path,
            title=f"Corpus-Based Ideological Manifold 1D - Dimension {dimension + 1} (Layer {layer_idx})",
            show_individual_speeches=show_individual_speeches
        )

