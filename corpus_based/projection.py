"""
Manifold Projection Module

This module handles projection of activations onto discovered ideological manifolds.
"""

import numpy as np
import torch
from typing import Union, Optional


class ManifoldProjector:
    """
    Projects activations onto ideological manifolds discovered via LDA.
    
    The projection operation maps high-dimensional activations onto lower-dimensional
    ideological manifolds, enabling visualization and analysis.
    """
    
    def __init__(self, transformation_matrix: np.ndarray):
        """
        Initialize the projector with a transformation matrix.
        
        Args:
            transformation_matrix: LDA transformation matrix of shape [D, k] where
                                   D is activation dimension and k is manifold dimension
        """
        self.transformation_matrix = transformation_matrix
        self.activation_dim = transformation_matrix.shape[0]
        self.manifold_dim = transformation_matrix.shape[1]
    
    def project(self, activations: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Project activations onto the ideological manifold.
        
        Args:
            activations: Activation vectors of shape [N, D] or [N, L, D] where
                        N = number of samples, L = number of layers, D = activation dimension
        
        Returns:
            Projected activations of shape [N, k] or [N, L, k] where k is manifold dimension
        """
        # Convert torch tensor to numpy if needed
        if isinstance(activations, torch.Tensor):
            activations = activations.cpu().numpy()
        
        # Handle multi-layer activations
        if activations.ndim == 3:
            # [N, L, D] -> [N, L, k]
            N, L, D = activations.shape
            if D != self.activation_dim:
                raise ValueError(
                    f"Activation dimension mismatch: expected {self.activation_dim}, got {D}"
                )
            
            # Reshape to [N*L, D], project, then reshape back
            activations_flat = activations.reshape(-1, D)
            projected_flat = activations_flat @ self.transformation_matrix
            return projected_flat.reshape(N, L, self.manifold_dim)
        elif activations.ndim == 2:
            # [N, D] -> [N, k]
            N, D = activations.shape
            if D != self.activation_dim:
                raise ValueError(
                    f"Activation dimension mismatch: expected {self.activation_dim}, got {D}"
                )
            return activations @ self.transformation_matrix
        else:
            raise ValueError(f"Unsupported activation shape: {activations.shape}")
    
    def get_manifold_basis(self) -> np.ndarray:
        """
        Get the basis vectors of the ideological manifold.
        
        Returns:
            Transformation matrix of shape [D, k]
        """
        return self.transformation_matrix.copy()
    
    def get_manifold_dimension(self) -> int:
        """Get the dimensionality of the manifold."""
        return self.manifold_dim

