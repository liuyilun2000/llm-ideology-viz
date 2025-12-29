"""
Neural Activation Extraction Module

This module handles the extraction of neural activations from LLM feed-forward
networks (FFNs) as described in the methodology section.
"""

import torch
import numpy as np
import re
from typing import List, Optional, Union
from transformer_lens import HookedTransformer
from tqdm.auto import tqdm
import sys
from pathlib import Path

# Add utils to path for env_loader
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.env_loader import get_hf_token
from .pooling import PoolingStrategy, MeanPooling


class ActivationExtractor:
    """
    Extracts neural activations from LLM feed-forward networks.
    
    Following the methodology, this class extracts post-activation outputs
    from FFN layers, which possess a privileged basis and maintain
    interpretability within the network's computation.
    """
    
    def __init__(
        self,
        model_name: str,
        cache_dir: Optional[str] = None,
        device: str = "cuda",
        trust_remote_code: bool = True,
        hf_token: Optional[str] = None
    ):
        """
        Initialize the activation extractor.
        
        Args:
            model_name: Name or path of the pretrained model
            cache_dir: Directory to cache the model
            device: Device to run the model on ('cpu' or 'cuda')
            trust_remote_code: Whether to trust remote code in model loading
            hf_token: HuggingFace token. If None, tries to load from .env file or environment variable
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.device = device
        self.model = None
        
        # Load HF token if not provided
        if hf_token is None:
            hf_token = get_hf_token()
        
        self.hf_token = hf_token
        self._load_model(trust_remote_code)
    
    def _load_model(self, trust_remote_code: bool = True):
        """Load the model for activation extraction."""
        # Set token as environment variable if available (transformer_lens reads from env)
        import os
        original_token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_TOKEN')
        if self.hf_token:
            os.environ['HF_TOKEN'] = self.hf_token
        
        try:
            # Prepare kwargs for from_pretrained
            kwargs = {
                'trust_remote_code': trust_remote_code,
            }
            
            if self.cache_dir is not None:
                kwargs['cache_dir'] = self.cache_dir
            
            self.model = HookedTransformer.from_pretrained(self.model_name, **kwargs)
        finally:
            # Restore original token if it existed
            if original_token:
                os.environ['HF_TOKEN'] = original_token
            elif 'HF_TOKEN' in os.environ and not self.hf_token:
                # Only remove if we set it and there was no original
                pass
        self.model.eval()
        self.model.to(self.device)
        torch.set_grad_enabled(False)
    
    def extract_activations(
        self,
        texts: List[str],
        activation_type: str = "mlp",
        layer_indices: Optional[Union[int, List[int]]] = None,
        batch_size: int = 1,
        show_progress: bool = True,
        pooling_strategy: Optional[PoolingStrategy] = None
    ) -> torch.Tensor:
        """
        Extract neural activations from the model for given texts.
        
        Args:
            texts: List of input text strings
            activation_type: Type of activation to extract ('mlp', 'attention', 'residual', or 'hidden')
                         - 'mlp': MLP post-activation outputs (default)
                         - 'attention': Attention post-activation outputs
                         - 'residual' or 'hidden': Intermediate hidden states in residual stream between blocks
            layer_indices: Specific layer(s) to extract from. If None, extracts from all layers.
                          Can be an int, list of ints, or None for all layers.
            batch_size: Batch size for processing (currently supports batch_size=1)
            show_progress: Whether to show progress bar
            pooling_strategy: Pooling strategy to apply immediately in the hook. If None, uses MeanPooling.
                           This pools each text's activations immediately to avoid variable-length issues.
                           Returns [N, L, D] instead of [N, L, T, D]
        
        Returns:
            torch.Tensor: Stacked activations of shape [N, L, D_act] where
                         N is number of texts, L is number of layers, D_act is activation dimension
                         (sequence dimension is pooled immediately)
        """
        if activation_type == "mlp":
            activation_filter = lambda name: ("mlp" in name) and ("hook_post" in name)
        elif activation_type == "attention":
            activation_filter = lambda name: ("attn" in name) and ("hook_post" in name)
        elif activation_type == "residual" or activation_type == "hidden":
            # Extract intermediate hidden states from residual stream between blocks
            # This captures the hidden state after each transformer block
            activation_filter = lambda name: ("blocks" in name) and ("hook_resid_post" in name)
        else:
            raise ValueError(f"activation_type must be 'mlp', 'attention', 'residual', or 'hidden', got {activation_type}")
        
        # Set up pooling strategy (default to mean pooling)
        if pooling_strategy is None:
            pooling_strategy = MeanPooling()
        
        # Determine which layers to extract from
        if layer_indices is None:
            # Extract from all layers
            target_layers = None
        elif isinstance(layer_indices, int):
            target_layers = {layer_indices}
        else:  # list
            target_layers = set(layer_indices)
        
        activations_list = []
        iterator = tqdm(texts, desc="Extracting activations") if show_progress else texts
        
        for text in iterator:
            # Use dict to store activations by layer index for efficient filtering
            activations_dict = {}
            
            def store_activation(activation, hook):
                """Hook function to store activations from specified layers only."""
                # Extract layer index from hook name
                # Hook names for residual: "blocks.{layer_idx}.hook_resid_post"
                # Hook names for MLP: "blocks.{layer_idx}.mlp.hook_post"
                # Hook names for attention: "blocks.{layer_idx}.attn.hook_post"
                hook_name = hook.name
                try:
                    # Extract layer index from hook name
                    # Format: "blocks.{layer_idx}.{rest}"
                    parts = hook_name.split('.')
                    if len(parts) >= 2 and parts[0] == 'blocks':
                        layer_idx = int(parts[1])
                    else:
                        # Fallback: try to find layer index in name
                        match = re.search(r'blocks\.(\d+)', hook_name)
                        if match:
                            layer_idx = int(match.group(1))
                        else:
                            # If we can't determine layer, skip (shouldn't happen)
                            return
                except (ValueError, IndexError):
                    # If we can't parse layer index, skip this hook
                    return
                
                # Only store if this layer is in our target layers
                if target_layers is not None and layer_idx not in target_layers:
                    return
                
                # activation[0] has shape [batch_size, seq_len, hidden_dim]
                # For single text, batch_size=1
                act = activation[0]
                
                if act.ndim == 2:
                    # Already [seq_len, hidden_dim] - pool over sequence dimension
                    pooled_act = pooling_strategy.pool(act)  # Shape: [hidden_dim]
                elif act.ndim == 3:
                    # [batch_size, seq_len, hidden_dim], take first batch item then pool
                    pooled_act = pooling_strategy.pool(act[0])  # Shape: [hidden_dim]
                else:
                    raise ValueError(f"Unexpected activation shape in hook: {act.shape}")
                
                # Store the pooled representation directly
                activations_dict[layer_idx] = pooled_act
            
            with torch.no_grad():
                self.model.run_with_hooks(
                    text,
                    return_type=None,
                    fwd_hooks=[(activation_filter, store_activation)],
                )
            
            # Convert dict to ordered list based on layer_indices or natural order
            if len(activations_dict) == 0:
                raise ValueError("No activations were extracted. Check activation_type and model configuration.")
            
            if target_layers is None:
                # Extract from all layers - sort by layer index
                sorted_layers = sorted(activations_dict.keys())
                activations = [activations_dict[idx] for idx in sorted_layers]
            elif isinstance(layer_indices, int):
                # Single layer
                if layer_indices not in activations_dict:
                    raise ValueError(
                        f"Layer index {layer_indices} not found in extracted activations. "
                        f"Model may have fewer layers than expected."
                    )
                activations = [activations_dict[layer_indices]]
            else:
                # Multiple layers - preserve order from layer_indices
                activations = []
                for idx in layer_indices:
                    if idx not in activations_dict:
                        raise ValueError(
                            f"Layer index {idx} not found in extracted activations. "
                            f"Model may have fewer layers than expected."
                        )
                    activations.append(activations_dict[idx])
            
            # Stack activations: [num_layers, hidden_dim] (already pooled)
            layer_activations = torch.stack(activations)
            
            # Verify shape after stacking
            if layer_activations.ndim != 2:
                raise ValueError(
                    f"Expected 2D tensor [L, D] after stacking (activations should be pooled), "
                    f"got shape {layer_activations.shape}"
                )
            
            activations_list.append(layer_activations)
        
        # Stack all texts: [N, L, D] where activations are already pooled
        if len(activations_list) == 0:
            raise ValueError("No activations were extracted from any texts.")
        
        stacked = torch.stack(activations_list)  # [N, L, D]
        
        # Verify final shape is correct
        if stacked.ndim != 3:
            raise ValueError(
                f"Expected 3D tensor [N, L, D] after stacking, got shape {stacked.shape}. "
                f"Activations should be pooled immediately in the hook."
            )
        
        return stacked
    
    def get_num_layers(self) -> int:
        """Get the number of layers in the model."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call _load_model first.")
        return self.model.cfg.n_layers
    
    def get_activation_dim(self, activation_type: str = "mlp") -> int:
        """
        Get the activation dimension for the model.
        
        Args:
            activation_type: Type of activation ('mlp', 'attention', 'residual', or 'hidden')
        
        Returns:
            Activation dimension
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call _load_model first.")
        
        if activation_type in ["residual", "hidden"]:
            return self.model.cfg.d_model
        elif activation_type == "attention":
            return self.model.cfg.d_model
        else:  # mlp
            return self.model.cfg.d_mlp

