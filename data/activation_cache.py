"""
Activation Caching Module

This module handles caching and loading of neural activations to avoid
recomputing them for the same texts and models.
"""

import os
import torch
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Tuple, Set
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer
import sys
import json
import hashlib
from pathlib import Path
from datetime import datetime

# Add utils to path for env_loader
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.env_loader import get_hf_token


def compute_text_id(text: str, metadata: Optional[Dict] = None) -> str:
    """
    Compute a unique ID for a text based on its content and optional metadata.
    
    Args:
        text: The text content
        metadata: Optional dictionary with metadata (e.g., {'sentence': text, 'party': 'Conservative'})
    
    Returns:
        A unique hash string identifying this text
    """
    # Create a string representation of text + metadata
    if metadata is not None:
        # Sort metadata keys for consistent hashing
        meta_str = json.dumps(metadata, sort_keys=True)
        content = f"{text}|||{meta_str}"
    else:
        content = text
    
    # Use SHA256 hash for unique ID
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


def cache_activations_by_id(
    df: pd.DataFrame,
    text_column: str,
    model_name: str,
    output_dir: str,
    activation_type: str = "mlp",
    pooling_type: str = "mean",
    device: str = "cpu",
    batch_size: int = 100,
    cache_dir: Optional[str] = None,
    hf_token: Optional[str] = None,
    id_column: Optional[str] = None
) -> None:
    """
    Cache activations for texts with per-text ID tracking.
    
    Only extracts activations for texts that aren't already cached.
    Uses text IDs to track which activations are already computed.
    
    Args:
        df: DataFrame with texts and metadata
        text_column: Column name containing the text
        model_name: Name of the pretrained model
        output_dir: Directory to save activations
        activation_type: Type of activation ('mlp', 'attention', 'residual', 'hidden')
        pooling_type: Pooling type ('mean', 'max', 'last')
        device: Device to run model on
        batch_size: Number of activations per batch file
        cache_dir: Directory to cache the model
        hf_token: HuggingFace token
        id_column: Optional column name to use as text ID (if None, computes hash)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load or create index file
    index_path = os.path.join(output_dir, 'activation_index.json')
    if os.path.exists(index_path):
        with open(index_path, 'r') as f:
            index = json.load(f)
    else:
        index = {
            'text_ids': {},  # text_id -> batch_file_index
            'batch_files': {},  # batch_file_index -> list of text_ids in order
            'model_name': model_name,
            'activation_type': activation_type,
            'pooling_type': pooling_type
        }
    
    # Compute text IDs for all texts
    text_ids = []
    texts_to_process = []
    indices_to_process = []
    
    for idx, row in df.iterrows():
        text = row[text_column]
        
        # Compute or get text ID
        if id_column and id_column in row:
            text_id = str(row[id_column])
        else:
            # Use text content + key metadata for ID
            metadata = {
                text_column: text,
                'dataset': row.get('dataset', ''),
                'country': row.get('country', ''),
            }
            text_id = compute_text_id(text, metadata)
        
        text_ids.append(text_id)
        
        # Check if already cached
        if text_id not in index['text_ids']:
            texts_to_process.append(text)
            indices_to_process.append(idx)
    
    if len(texts_to_process) == 0:
        print("All texts are already cached. No extraction needed.")
        return
    
    print(f"Found {len(texts_to_process)} new texts to process (out of {len(df)} total)")
    
    # Load HF token if not provided
    if hf_token is None:
        hf_token = get_hf_token()
    
    # Set token as environment variable
    original_token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_TOKEN')
    if hf_token:
        os.environ['HF_TOKEN'] = hf_token
    
    try:
        # Load model
        kwargs = {'trust_remote_code': True}
        if cache_dir is not None:
            kwargs['cache_dir'] = cache_dir
        
        model = HookedTransformer.from_pretrained(model_name, **kwargs)
    finally:
        if original_token:
            os.environ['HF_TOKEN'] = original_token
    
    model.eval()
    model.to(device)
    torch.set_grad_enabled(False)
    
    # Setup activation extraction
    activations = []
    
    def store_activation(activation, hook):
        """Hook function to store activations."""
        if pooling_type == "mean":
            activations.append(torch.mean(activation[0], dim=0))
        elif pooling_type == "max":
            activations.append(torch.max(activation[0], dim=0)[0])
        elif pooling_type == "last":
            activations.append(activation[0][-1])  # Last token
        else:
            raise NotImplementedError(f"Pooling type {pooling_type} not supported.")
    
    if activation_type == "mlp":
        activation_filter = lambda name: ("mlp" in name) and ("hook_post" in name)
    elif activation_type == "attention":
        activation_filter = lambda name: ("attn" in name) and ("hook_post" in name)
    elif activation_type == "residual" or activation_type == "hidden":
        activation_filter = lambda name: ("blocks" in name) and ("hook_resid_post" in name)
    else:
        raise ValueError("activation_type must be 'mlp', 'attention', 'residual', or 'hidden'")
    
    # Find next batch index
    existing_batch_files = [int(f.split('.')[0]) for f in os.listdir(output_dir) 
                           if f.endswith('.pt') and f.split('.')[0].isdigit()]
    batch_idx = max(existing_batch_files) + 1 if existing_batch_files else 0
    
    batch_activations = []
    batch_text_ids = []
    
    # Extract activations for new texts
    for text, orig_idx in zip(tqdm(texts_to_process, desc="Extracting activations"), indices_to_process):
        text_id = text_ids[orig_idx]
        activations.clear()
        
        with torch.no_grad():
            model.run_with_hooks(
                text,
                return_type=None,
                fwd_hooks=[(activation_filter, store_activation)],
            )
        
        batch_activations.append(torch.stack(activations))
        batch_text_ids.append(text_id)
        
        # Save batch when full
        if len(batch_activations) == batch_size:
            res = torch.stack(batch_activations)
            batch_file = f"{batch_idx}.pt"
            torch.save(res.cpu(), os.path.join(output_dir, batch_file))
            
            # Update index
            index['batch_files'][str(batch_idx)] = batch_text_ids.copy()
            for tid in batch_text_ids:
                index['text_ids'][tid] = batch_idx
            
            batch_idx += 1
            batch_activations = []
            batch_text_ids = []
    
    # Save remaining activations
    if len(batch_activations) > 0:
        res = torch.stack(batch_activations)
        batch_file = f"{batch_idx}.pt"
        torch.save(res.cpu(), os.path.join(output_dir, batch_file))
        
        # Update index
        index['batch_files'][str(batch_idx)] = batch_text_ids.copy()
        for tid in batch_text_ids:
            index['text_ids'][tid] = batch_idx
    
    # Save updated index
    index['last_updated'] = datetime.now().isoformat()
    with open(index_path, 'w') as f:
        json.dump(index, f, indent=2)
    
    print(f"Cached {len(texts_to_process)} new activations. Total cached: {len(index['text_ids'])}")


def load_activations_by_id(
    df: pd.DataFrame,
    text_column: str,
    output_dir: str,
    id_column: Optional[str] = None
) -> Tuple[np.ndarray, List[int]]:
    """
    Load activations for texts in DataFrame by their IDs.
    
    Args:
        df: DataFrame with texts
        text_column: Column name containing text
        output_dir: Directory where activations are cached
        id_column: Optional column name to use as text ID
    
    Returns:
        Tuple of (activations_array, valid_indices) where:
        - activations_array: numpy array of shape [N, L, D]
        - valid_indices: list of DataFrame indices that were successfully loaded
    """
    index_path = os.path.join(output_dir, 'activation_index.json')
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Activation index not found at {index_path}")
    
    with open(index_path, 'r') as f:
        index = json.load(f)
    
    # Compute text IDs and find which are cached
    text_ids = []
    cached_text_ids = []
    valid_indices = []
    
    for idx, row in df.iterrows():
        text = row[text_column]
        
        if id_column and id_column in row:
            text_id = str(row[id_column])
        else:
            metadata = {
                text_column: text,
                'dataset': row.get('dataset', ''),
                'country': row.get('country', ''),
            }
            text_id = compute_text_id(text, metadata)
        
        text_ids.append(text_id)
        
        if text_id in index['text_ids']:
            cached_text_ids.append(text_id)
            valid_indices.append(idx)
    
    if len(cached_text_ids) == 0:
        raise ValueError("No cached activations found for the provided texts")
    
    if len(cached_text_ids) < len(df):
        print(f"Warning: Only {len(cached_text_ids)} out of {len(df)} texts have cached activations")
    
    # Group by batch file for efficient loading
    batch_to_texts = {}
    batch_to_positions = {}
    
    for i, text_id in enumerate(cached_text_ids):
        batch_idx = index['text_ids'][text_id]
        if batch_idx not in batch_to_texts:
            batch_to_texts[batch_idx] = []
            batch_to_positions[batch_idx] = []
        batch_to_texts[batch_idx].append(text_id)
        batch_to_positions[batch_idx].append(i)
    
    # Load activations from batches
    # Create a mapping from text_id to activation for efficient lookup
    text_id_to_activation = {}
    
    for batch_idx, text_ids_in_batch in tqdm(batch_to_texts.items(), desc="Loading activations"):
        batch_file = os.path.join(output_dir, f"{batch_idx}.pt")
        if not os.path.exists(batch_file):
            raise FileNotFoundError(f"Batch file {batch_file} not found")
        
        batch_activations = torch.load(batch_file)  # [batch_size, L, D]
        batch_text_ids = index['batch_files'][str(batch_idx)]
        
        # Map each text_id to its activation
        for text_id in text_ids_in_batch:
            batch_pos = batch_text_ids.index(text_id)
            activation = batch_activations[batch_pos].cpu().numpy()
            text_id_to_activation[text_id] = activation
    
    # Build activations array in DataFrame order
    activations_list = []
    for text_id in cached_text_ids:
        activations_list.append(text_id_to_activation[text_id])
    
    activations_array = np.array(activations_list)
    
    return activations_array, valid_indices


# Keep old functions for backward compatibility
def cache_activations(
    sentences: List[str],
    model_name: str,
    output_dir: str,
    activation_type: str = "mlp",
    pooling_type: str = "mean",
    device: str = "cpu",
    batch_size: int = 100,
    cache_dir: Optional[str] = None,
    skip_batch: int = 0,
    hf_token: Optional[str] = None
) -> None:
    """
    Process sentences through a language model and store activations with pooling.
    
    This function extracts activations from the model's feed-forward networks,
    applies pooling, and saves them to disk for later use.
    
    Args:
        sentences: List of sentences to process
        model_name: Name of the pretrained model to use
        output_dir: Directory to save the activation tensors
        activation_type: Type of activation to capture ('mlp' or 'attention')
        pooling_type: Pooling type to apply ('mean' or 'max')
        device: Device to run the model on ('cpu' or 'cuda')
        batch_size: Number of activations to save per file
        cache_dir: Directory to cache the model
        skip_batch: Number of batches to skip (for resuming)
        hf_token: HuggingFace token. If None, tries to load from .env file or environment variable
    """
    # Load HF token if not provided
    if hf_token is None:
        hf_token = get_hf_token()
    
    # Set token as environment variable if available (transformer_lens reads from env)
    original_token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_TOKEN')
    if hf_token:
        os.environ['HF_TOKEN'] = hf_token
    
    try:
        # Load model
        kwargs = {
            'trust_remote_code': True,
        }
        if cache_dir is not None:
            kwargs['cache_dir'] = cache_dir
        
        model = HookedTransformer.from_pretrained(model_name, **kwargs)
    finally:
        # Restore original token if it existed
        if original_token:
            os.environ['HF_TOKEN'] = original_token
        elif 'HF_TOKEN' in os.environ and not hf_token:
            # Only remove if we set it and there was no original
            pass
    model.eval()
    model.to(device)
    
    torch.set_grad_enabled(False)
    
    if output_dir is None:
        output_dir = f"cache/activations/{model_name.split('/')[-1]}/{activation_type}_{pooling_type}"
    os.makedirs(output_dir, exist_ok=True)
    
    activations = []
    
    def store_activation(activation, hook):
        """Hook function to store activations."""
        if pooling_type == "mean":
            activations.append(torch.mean(activation[0], dim=0))
        elif pooling_type == "max":
            activations.append(torch.max(activation[0], dim=0)[0])
        else:
            raise NotImplementedError("Only mean and max pooling supported.")
    
    if activation_type == "mlp":
        activation_filter = lambda name: ("mlp" in name) and ("hook_post" in name)
    elif activation_type == "attention":
        activation_filter = lambda name: ("attn" in name) and ("hook_post" in name)
    elif activation_type == "residual" or activation_type == "hidden":
        # Extract intermediate hidden states from residual stream between blocks
        activation_filter = lambda name: ("blocks" in name) and ("hook_resid_post" in name)
    else:
        raise ValueError("activation_type must be 'mlp', 'attention', 'residual', or 'hidden'")
    
    batch_idx = skip_batch
    batch_activations = []
    
    for idx, text in enumerate(tqdm(sentences, desc="Extracting activations")):
        activations.clear()
        with torch.no_grad():
            model.run_with_hooks(
                text,
                return_type=None,
                fwd_hooks=[(activation_filter, store_activation)],
            )
        batch_activations.append(torch.stack(activations))
        
        if len(batch_activations) == batch_size:
            res = torch.stack(batch_activations)
            torch.save(res.cpu(), os.path.join(output_dir, f"{batch_idx}.pt"))
            batch_idx += 1
            batch_activations = []
    
    # Save remaining activations
    if len(batch_activations) > 0:
        res = torch.stack(batch_activations)
        torch.save(res.cpu(), os.path.join(output_dir, f"{batch_idx}.pt"))


def load_activations(
    output_dir: str,
    num_files: Optional[int] = None
) -> torch.Tensor:
    """
    Load saved activation tensors from a directory and stack them.
    
    Performs sanity checks on the loaded files to ensure consistency.
    
    Args:
        output_dir: Directory where activation tensors are saved
        num_files: Number of files to load (if None, loads all .pt files)
    
    Returns:
        Stacked tensor of all loaded activations of shape [N, L, D] where
        N = number of texts, L = number of layers, D = activation dimension
    """
    # Check if new ID-based index exists
    index_path = os.path.join(output_dir, 'activation_index.json')
    if os.path.exists(index_path):
        # Use new ID-based loading
        raise NotImplementedError(
            "This cache uses ID-based indexing. Use load_activations_by_id() instead."
        )
    
    # Get all .pt files in directory
    pt_files = [f for f in os.listdir(output_dir) if f.endswith('.pt')]
    pt_files.sort(key=lambda x: int(x.split('.')[0]))  # Sort numerically
    
    # Sanity check: Verify all expected files exist
    if num_files is None:
        num_files = len(pt_files)
    expected_files = set(f"{i}.pt" for i in range(num_files))
    actual_files = set(pt_files)
    missing_files = expected_files - actual_files
    
    if missing_files:
        raise FileNotFoundError(f"Missing activation files: {missing_files}")
    
    # Load and verify each file
    activations = []
    for file_name in tqdm(pt_files[:num_files], desc="Loading activations"):
        file_path = os.path.join(output_dir, file_name)
        activation = torch.load(file_path)
        
        # Sanity check: Verify tensor shape and type
        if not isinstance(activation, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(activation)} for file {file_name}")
        if len(activation.shape) != 3:  # Assuming shape is [batch_size, num_layers, hidden_size]
            raise ValueError(f"Unexpected tensor shape {activation.shape} for file {file_name}")
        
        activations.append(activation)
    
    # Stack all activations: [N, L, D]
    return torch.cat(activations, dim=0)


def load_activations_to_dataframe(
    output_dir: str,
    df: pd.DataFrame,
    embedding_column: str = 'embedding',
    num_files: Optional[int] = None,
    text_column: str = 'sentence',
    use_id_based: bool = True
) -> pd.DataFrame:
    """
    Load activations and add them as a column to a DataFrame.
    
    Args:
        output_dir: Directory where activation tensors are saved
        df: DataFrame to add activations to
        embedding_column: Name of the column to store activations
        num_files: Number of files to load (for old cache format)
        text_column: Column name containing text (for ID-based loading)
        use_id_based: Whether to use ID-based loading if available
    
    Returns:
        DataFrame with activations added as a column
    """
    # Check if ID-based index exists
    index_path = os.path.join(output_dir, 'activation_index.json')
    if use_id_based and os.path.exists(index_path):
        # Use ID-based loading
        activations, valid_indices = load_activations_by_id(
            df, text_column, output_dir
        )
        
        # Create DataFrame with activations, preserving original DataFrame indices
        # valid_indices are the original DataFrame indices that were successfully loaded
        # They come in the order they appear in df, so we can use iloc
        df_result = df.loc[valid_indices].copy()
        df_result[embedding_column] = list(activations)
        
        # Reorder to match original DataFrame order (valid_indices are already in order)
        # But we need to ensure the DataFrame is in the same order as the original
        df_result = df_result.reindex([idx for idx in df.index if idx in valid_indices])
        
        return df_result
    else:
        # Use old sequential loading
        activations = load_activations(output_dir, num_files)
        
        # Convert to numpy and store as list of arrays
        if isinstance(activations, torch.Tensor):
            activations = activations.cpu().numpy()
        
        # Check if number of activations matches DataFrame length
        if len(activations) != len(df):
            raise ValueError(
                f"Number of activations ({len(activations)}) does not match "
                f"DataFrame length ({len(df)})"
            )
        
        # Add activations as a new column
        df = df.copy()
        df[embedding_column] = list(activations)
        
        return df


def save_activation_metadata(
    output_dir: str,
    model_name: str,
    activation_type: str,
    pooling_type: str,
    num_samples: int,
    num_layers: int,
    activation_dim: int,
    batch_size: int = 100
) -> None:
    """
    Save metadata about cached activations to verify correctness.
    
    Args:
        output_dir: Directory where activations are saved
        model_name: Name of the model used
        activation_type: Type of activation (mlp, attention, residual, etc.)
        pooling_type: Type of pooling (mean, max, last)
        num_samples: Number of samples/texts
        num_layers: Number of layers
        activation_dim: Dimension of activations
        batch_size: Batch size used for saving
    """
    metadata = {
        'model_name': model_name,
        'activation_type': activation_type,
        'pooling_type': pooling_type,
        'num_samples': num_samples,
        'num_layers': num_layers,
        'activation_dim': activation_dim,
        'batch_size': batch_size,
        'saved_at': datetime.now().isoformat(),
        'num_batch_files': (num_samples + batch_size - 1) // batch_size
    }
    
    metadata_path = os.path.join(output_dir, 'activation_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved activation metadata to {metadata_path}")


def check_cached_activations(
    output_dir: str,
    model_name: str,
    activation_type: str,
    pooling_type: str,
    expected_num_samples: Optional[int] = None
) -> Tuple[bool, Optional[Dict]]:
    """
    Check if cached activations exist and are valid.
    
    Args:
        output_dir: Directory where activations should be cached
        model_name: Expected model name
        activation_type: Expected activation type
        pooling_type: Expected pooling type
        expected_num_samples: Optional expected number of samples (for validation)
    
    Returns:
        Tuple of (is_valid, metadata_dict). is_valid is True if cached activations
        exist and match the expected parameters. metadata_dict contains the metadata
        if available, None otherwise.
    """
    if not os.path.exists(output_dir):
        return False, None
    
    # Check for ID-based index
    index_path = os.path.join(output_dir, 'activation_index.json')
    if os.path.exists(index_path):
        with open(index_path, 'r') as f:
            index = json.load(f)
        
        # Check if parameters match
        if index.get('model_name') != model_name:
            return False, index
        if index.get('activation_type') != activation_type:
            return False, index
        if index.get('pooling_type') != pooling_type:
            return False, index
        
        # For ID-based cache, we don't require exact sample count match
        # since we can load subsets
        return True, index
    
    # Fall back to old metadata check
    metadata_path = os.path.join(output_dir, 'activation_metadata.json')
    if not os.path.exists(metadata_path):
        # Check if there are any .pt files (old cache without metadata)
        pt_files = [f for f in os.listdir(output_dir) if f.endswith('.pt')]
        if len(pt_files) > 0:
            print(f"Warning: Found cached activations without metadata file. "
                  f"Consider re-saving to add metadata for validation.")
            return True, None
        return False, None
    
    # Load and check metadata
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    except Exception as e:
        print(f"Warning: Could not load metadata file: {e}")
        return False, None
    
    # Check if parameters match
    if metadata.get('model_name') != model_name:
        print(f"Warning: Model name mismatch. Expected {model_name}, "
              f"found {metadata.get('model_name')}")
        return False, metadata
    
    if metadata.get('activation_type') != activation_type:
        print(f"Warning: Activation type mismatch. Expected {activation_type}, "
              f"found {metadata.get('activation_type')}")
        return False, metadata
    
    if metadata.get('pooling_type') != pooling_type:
        print(f"Warning: Pooling type mismatch. Expected {pooling_type}, "
              f"found {metadata.get('pooling_type')}")
        return False, metadata
    
    # Check if expected number of samples matches (if provided)
    if expected_num_samples is not None:
        if metadata.get('num_samples') != expected_num_samples:
            print(f"Warning: Number of samples mismatch. Expected {expected_num_samples}, "
                  f"found {metadata.get('num_samples')}")
            return False, metadata
    
    # Check if all batch files exist
    num_batch_files = metadata.get('num_batch_files', 0)
    for i in range(num_batch_files):
        batch_file = os.path.join(output_dir, f"{i}.pt")
        if not os.path.exists(batch_file):
            print(f"Warning: Missing batch file {batch_file}")
            return False, metadata
    
    return True, metadata


def save_extracted_activations(
    activations: np.ndarray,
    output_dir: str,
    model_name: str,
    activation_type: str,
    pooling_type: str,
    batch_size: int = 100
) -> None:
    """
    Save already-extracted activations to disk in batches.
    
    This function is used when activations have already been extracted
    and we just need to save them, avoiding re-extraction.
    
    Args:
        activations: Activation tensor of shape [N, L, D] where
                    N = number of texts, L = number of layers, D = activation dimension
        output_dir: Directory to save the activation tensors
        model_name: Name of the model used (for metadata)
        activation_type: Type of activation (for metadata)
        pooling_type: Type of pooling (for metadata)
        batch_size: Number of activations to save per file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to torch tensor if needed
    if isinstance(activations, np.ndarray):
        activations = torch.from_numpy(activations)
    
    num_samples = activations.shape[0]
    num_layers = activations.shape[1]
    activation_dim = activations.shape[2]
    num_batches = (num_samples + batch_size - 1) // batch_size
    batch_idx = 0
    
    for start_idx in tqdm(range(0, num_samples, batch_size), desc="Saving activations", total=num_batches):
        end_idx = min(start_idx + batch_size, num_samples)
        batch = activations[start_idx:end_idx]  # [batch_size, L, D]
        
        file_path = os.path.join(output_dir, f"{batch_idx}.pt")
        torch.save(batch.cpu(), file_path)
        batch_idx += 1
    
    # Save metadata after successful save
    save_activation_metadata(
        output_dir=output_dir,
        model_name=model_name,
        activation_type=activation_type,
        pooling_type=pooling_type,
        num_samples=num_samples,
        num_layers=num_layers,
        activation_dim=activation_dim,
        batch_size=batch_size
    )
