"""
Pooling Operations Module

This module implements pooling strategies to convert variable-length token-level
activations into fixed-dimension representations, as described in the methodology.
"""

import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Union


class PoolingStrategy(ABC):
    """
    Abstract base class for pooling strategies.
    
    Pooling operations aggregate token-level activations into fixed-dimension
    representations suitable for downstream analysis.
    """
    
    @abstractmethod
    def pool(self, activations: torch.Tensor) -> torch.Tensor:
        """
        Apply pooling operation to activations.
        
        Args:
            activations: Tensor of shape [batch_size, seq_len, hidden_dim] or
                        [seq_len, hidden_dim] for single sequence
        
        Returns:
            Pooled tensor of shape [batch_size, hidden_dim] or [hidden_dim]
        """
        pass


class MeanPooling(PoolingStrategy):
    """
    Mean pooling strategy.
    
    Computes the average activation across the token dimension:
    a* = (1/T) * sum(t=1 to T) a_t
    
    This is the default pooling strategy used in the methodology.
    """
    
    def __init__(self, mask_padding: bool = True, padding_token_id: int = 0):
        """
        Initialize mean pooling.
        
        Args:
            mask_padding: Whether to mask padding tokens before pooling
            padding_token_id: Token ID used for padding (if mask_padding is True)
        """
        self.mask_padding = mask_padding
        self.padding_token_id = padding_token_id
    
    def pool(self, activations: torch.Tensor) -> torch.Tensor:
        """
        Apply mean pooling to activations.
        
        Args:
            activations: Tensor of shape [batch_size, seq_len, hidden_dim] or
                        [seq_len, hidden_dim] or [num_layers, seq_len, hidden_dim]
        
        Returns:
            Pooled tensor with sequence dimension removed
        """
        # Handle different input shapes
        if activations.dim() == 2:
            # [seq_len, hidden_dim] -> [hidden_dim]
            return torch.mean(activations, dim=0)
        elif activations.dim() == 3:
            # [batch_size, seq_len, hidden_dim] -> [batch_size, hidden_dim]
            # or [num_layers, seq_len, hidden_dim] -> [num_layers, hidden_dim]
            return torch.mean(activations, dim=1)
        elif activations.dim() == 4:
            # [batch_size, num_layers, seq_len, hidden_dim] -> [batch_size, num_layers, hidden_dim]
            return torch.mean(activations, dim=2)
        else:
            raise ValueError(f"Unsupported activation shape: {activations.shape}")
    
    def pool_batch(self, activations: torch.Tensor) -> torch.Tensor:
        """
        Apply mean pooling to a batch of activations.
        
        Args:
            activations: Tensor of shape [N, L, T, D] where
                        N = number of texts
                        L = number of layers
                        T = sequence length
                        D = activation dimension
        
        Returns:
            Pooled tensor of shape [N, L, D]
        """
        if activations.dim() != 4:
            raise ValueError(f"Expected 4D tensor [N, L, T, D], got shape {activations.shape}")
        
        # Mean pool over sequence dimension (dim=2)
        return torch.mean(activations, dim=2)


class MaxPooling(PoolingStrategy):
    """
    Max pooling strategy.
    
    Takes the maximum activation across the token dimension.
    """
    
    def pool(self, activations: torch.Tensor) -> torch.Tensor:
        """
        Apply max pooling to activations.
        
        Args:
            activations: Tensor of shape [batch_size, seq_len, hidden_dim] or
                        [seq_len, hidden_dim]
        
        Returns:
            Pooled tensor with sequence dimension removed
        """
        if activations.dim() == 2:
            return torch.max(activations, dim=0)[0]
        elif activations.dim() == 3:
            return torch.max(activations, dim=1)[0]
        elif activations.dim() == 4:
            return torch.max(activations, dim=2)[0]
        else:
            raise ValueError(f"Unsupported activation shape: {activations.shape}")
    
    def pool_batch(self, activations: torch.Tensor) -> torch.Tensor:
        """Apply max pooling to a batch of activations."""
        if activations.dim() != 4:
            raise ValueError(f"Expected 4D tensor [N, L, T, D], got shape {activations.shape}")
        return torch.max(activations, dim=2)[0]


class LastTokenPooling(PoolingStrategy):
    """
    Last token pooling strategy.
    
    Takes the activation of the last token in the sequence.
    """
    
    def pool(self, activations: torch.Tensor) -> torch.Tensor:
        """
        Apply last token pooling to activations.
        
        Args:
            activations: Tensor of shape [batch_size, seq_len, hidden_dim] or
                        [seq_len, hidden_dim]
        
        Returns:
            Pooled tensor with sequence dimension removed
        """
        if activations.dim() == 2:
            return activations[-1]
        elif activations.dim() == 3:
            return activations[:, -1, :]
        elif activations.dim() == 4:
            return activations[:, :, -1, :]
        else:
            raise ValueError(f"Unsupported activation shape: {activations.shape}")
    
    def pool_batch(self, activations: torch.Tensor) -> torch.Tensor:
        """Apply last token pooling to a batch of activations."""
        if activations.dim() != 4:
            raise ValueError(f"Expected 4D tensor [N, L, T, D], got shape {activations.shape}")
        return activations[:, :, -1, :]

