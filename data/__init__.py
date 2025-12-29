"""
Data Module

This module provides utilities for loading datasets and managing activation caches.
"""

from .datasets import load_parlasent, filter_dataset
from .activation_cache import (
    cache_activations, 
    load_activations, 
    load_activations_to_dataframe,
    cache_activations_by_id,
    load_activations_by_id,
    compute_text_id
)

__all__ = [
    'load_parlasent',
    'filter_dataset',
    'cache_activations',
    'load_activations',
    'load_activations_to_dataframe',
    'cache_activations_by_id',
    'load_activations_by_id',
    'compute_text_id',
]
