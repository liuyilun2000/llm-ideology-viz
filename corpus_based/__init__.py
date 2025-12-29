"""
Corpus-Based Framework for Ideological Neural Manifold Analysis

This package implements the corpus-based framework for analyzing ideological
representations in LLMs using Linear Discriminant Analysis (LDA).

The framework follows the methodology described in the thesis:
1. Extract neural activations from FFN layers
2. Apply pooling operations to obtain fixed-dimension representations
3. Perform LDA to discover optimal ideological manifolds
4. Project activations onto the discovered manifolds
5. Visualize and analyze the results

Main Components:
- extraction: Neural activation extraction from LLMs
- pooling: Pooling operations for variable-length sequences
- lda: Linear Discriminant Analysis for manifold discovery
- projection: Projection operations onto ideological manifolds
- visualization: Plotting and visualization utilities
- analyzer: High-level analyzer class orchestrating the framework
"""

from .analyzer import CorpusBasedAnalyzer
from .extraction import ActivationExtractor
from .pooling import PoolingStrategy, MeanPooling, MaxPooling, LastTokenPooling
from .lda import LDAAnalyzer
from .projection import ManifoldProjector
from .visualization import CorpusVisualizer

__all__ = [
    'CorpusBasedAnalyzer',
    'ActivationExtractor',
    'PoolingStrategy',
    'MeanPooling',
    'MaxPooling',
    'LastTokenPooling',
    'LDAAnalyzer',
    'ManifoldProjector',
    'CorpusVisualizer',
]

