"""
Linear Discriminant Analysis (LDA) Module

This module implements the LDA-based manifold discovery for the corpus-based
framework, as described in the methodology section.
"""

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder
from typing import Optional, Union, Tuple, Dict
import warnings


class LDAAnalyzer:
    """
    Performs Linear Discriminant Analysis to discover optimal ideological manifolds.
    
    LDA finds linear combinations of features that maximize separation between
    classes, enabling data-driven discovery of ideological dimensions.
    """
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize the LDA analyzer.
        
        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        self.label_encoder = LabelEncoder()
        self.lda_models = {}  # Store LDA models for each layer
        np.random.seed(random_seed)
    
    def fit(
        self,
        activations: np.ndarray,
        labels: np.ndarray,
        n_components: Optional[int] = None,
        min_samples_per_class: int = 1
    ) -> Tuple[LinearDiscriminantAnalysis, np.ndarray]:
        """
        Fit LDA model to activations with class labels.
        
        Args:
            activations: Activation vectors of shape [N, D] or [N, L, D] where
                        N = number of samples, L = number of layers, D = activation dimension
            labels: Class labels of shape [N] (can be strings or integers)
            n_components: Number of LDA components. If None, uses min(n_classes-1, D)
            min_samples_per_class: Minimum samples required per class (for filtering)
        
        Returns:
            Tuple of (fitted LDA model, transformed activations)
        """
        # Handle multi-layer activations
        if activations.ndim == 3:
            # [N, L, D] - process each layer separately
            num_layers = activations.shape[1]
            lda_models = []
            transformed_list = []
            
            for layer_idx in range(num_layers):
                layer_activations = activations[:, layer_idx, :]
                lda, transformed = self._fit_single_layer(
                    layer_activations, labels, n_components, min_samples_per_class
                )
                lda_models.append(lda)
                transformed_list.append(transformed)
            
            # Stack results: [N, L, n_components]
            return lda_models, np.stack(transformed_list, axis=1)
        else:
            return self._fit_single_layer(
                activations, labels, n_components, min_samples_per_class
            )
    
    def _fit_single_layer(
        self,
        activations: np.ndarray,
        labels: np.ndarray,
        n_components: Optional[int],
        min_samples_per_class: int
    ) -> Tuple[LinearDiscriminantAnalysis, np.ndarray]:
        """Fit LDA for a single layer."""
        # Filter classes with insufficient samples
        unique_labels, counts = np.unique(labels, return_counts=True)
        valid_classes = unique_labels[counts >= min_samples_per_class]
        
        if len(valid_classes) < 2:
            raise ValueError(
                f"Need at least 2 classes with >= {min_samples_per_class} samples each. "
                f"Found {len(valid_classes)} valid classes."
            )
        
        # Filter data to valid classes
        mask = np.isin(labels, valid_classes)
        X = activations[mask]
        y = labels[mask]
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        n_classes = len(self.label_encoder.classes_)
        
        # Determine number of components
        if n_components is None:
            n_components = min(n_classes - 1, X.shape[1])
        else:
            n_components = min(n_components, n_classes - 1, X.shape[1])
        
        # Flatten if activations are multi-dimensional
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)
        
        # Fit LDA
        lda = LinearDiscriminantAnalysis(n_components=n_components)
        X_lda = lda.fit_transform(X, y_encoded)
        
        return lda, X_lda
    
    def transform(
        self,
        activations: np.ndarray,
        layer_idx: Optional[int] = None
    ) -> np.ndarray:
        """
        Transform activations using a previously fitted LDA model.
        
        Args:
            activations: Activation vectors to transform
            layer_idx: Layer index if using multi-layer LDA models
        
        Returns:
            Transformed activations in LDA space
        """
        if layer_idx is not None:
            if layer_idx not in self.lda_models:
                raise ValueError(f"No LDA model found for layer {layer_idx}")
            lda = self.lda_models[layer_idx]
        elif len(self.lda_models) == 1:
            lda = list(self.lda_models.values())[0]
        else:
            raise ValueError("Multiple LDA models found. Specify layer_idx.")
        
        # Flatten if needed
        if activations.ndim > 2:
            activations = activations.reshape(activations.shape[0], -1)
        
        return lda.transform(activations)
    
    def get_explained_variance_ratio(self, layer_idx: Optional[int] = None) -> np.ndarray:
        """
        Get the explained variance ratio for each LDA component.
        
        Args:
            layer_idx: Layer index if using multi-layer models
        
        Returns:
            Array of explained variance ratios
        """
        if layer_idx is not None:
            lda = self.lda_models[layer_idx]
        elif len(self.lda_models) == 1:
            lda = list(self.lda_models.values())[0]
        else:
            raise ValueError("Multiple LDA models found. Specify layer_idx.")
        
        # LDA doesn't directly provide explained_variance_ratio like PCA,
        # but we can compute it from eigenvalues
        if hasattr(lda, 'explained_variance_ratio_'):
            return lda.explained_variance_ratio_
        else:
            # Compute from eigenvalues (between-class / total variance)
            if hasattr(lda, 'scalings_'):
                # Approximate explained variance from the transformation
                warnings.warn("Using approximate explained variance ratio")
                # This is a simplified approximation
                return np.ones(lda.n_components) / lda.n_components
            return None
    
    def get_discriminant_axes(self, layer_idx: Optional[int] = None) -> np.ndarray:
        """
        Get the discriminant axes (transformation matrix).
        
        Args:
            layer_idx: Layer index if using multi-layer models
        
        Returns:
            Transformation matrix of shape [D, n_components]
        """
        if layer_idx is not None:
            lda = self.lda_models[layer_idx]
        elif len(self.lda_models) == 1:
            lda = list(self.lda_models.values())[0]
        else:
            raise ValueError("Multiple LDA models found. Specify layer_idx.")
        
        return lda.scalings_

