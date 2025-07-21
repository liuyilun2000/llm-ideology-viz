# utils.py

import os
import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder
from typing import List, Optional

from datasets import load_dataset

def load_parlasent(datasets: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Load and process the ParlaSent datasets into a single DataFrame.

    Args:
        datasets: List of dataset codes to load (default: all available)

    Returns:
        pd.DataFrame: Combined DataFrame with columns ['dataset', 'sentence', 'country', 'date', 'name', 'party', 'gender', 'birth_year', 'ruling']
    """
    if datasets is None:
        datasets = ['EN', 'EN_additional_test', 'BCS', 'BCS_additional_test', 'CZ', 'SK', 'SL']
    
    def load_and_process_dataset(dataset):
        ds = load_dataset("classla/ParlaSent", dataset)['train']
        df = pd.DataFrame(ds)
        df['dataset'] = dataset
        return df[['dataset', 'sentence', 'country', 'date', 'name', 'party', 'gender', 'birth_year', 'ruling']]
    
    dfs = [load_and_process_dataset(dataset) for dataset in tqdm(datasets, desc="Loading ParlaSent datasets")]
    df = pd.concat(dfs, ignore_index=True)
    return df


def cache_activations(
    sentences: List[str],
    model_name: str,
    output_dir: str = None,
    cache_dir: str = None,
    activation_type: str = "mlp",
    pooling_type: str = "mean",
    device: str = "cpu",
    batch_size: int = 100,
    skip_batch: int = 0,
) -> None:
    """
    Process sentences through a language model and store activations with pooling.

    Args:
        sentences (List[str]): List of sentences to process.
        model_name (str): Name of the pretrained model to use.
        output_dir (str): Directory to save the activation tensors.
        cache_dir (str): Directory to cache the model.
        activation_type (str): Type of activation to capture ('mlp').
        pooling_type (str): Pooling type to apply ('mean' or 'max').
        device (str): Device to run the model on ('cpu' or 'cuda').
        batch_size (int): Number of activations to save per file.
        skip_batch (int): Number of batches to skip (for resuming).
    """
    # Load model
    model = HookedTransformer.from_pretrained(
        model_name, 
        trust_remote_code=True,
        cache_dir=cache_dir,
    )
    model.eval()
    model.to(device)

    torch.set_grad_enabled(False)

    if output_dir is None:
        output_dir = f"cache/activations/{model_name.split('/')[-1]}/{activation_type}_{pooling_type}"
    os.makedirs(output_dir, exist_ok=True)

    activations = []
    def store_activation(activation, hook):
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
    else:
        raise ValueError("activation_type must be 'mlp' or 'attention'")

    batch_idx = 0
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
    if len(batch_activations) > 0:
        res = torch.stack(batch_activations)
        torch.save(res.cpu(), os.path.join(output_dir, f"{batch_idx}.pt"))


def load_activations(output_dir: str, num_files: Optional[int] = None) -> torch.Tensor:
    """
    Load saved activation tensors from a directory and stack them.
    Performs sanity checks on the loaded files.

    Args:
        output_dir (str): Directory where activation tensors are saved.
        num_files (int): Number of files to load.

    Returns:
        torch.Tensor: Stacked tensor of all loaded activations.
    """
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
    for file_name in tqdm(pt_files, desc="Loading activations"):
        file_path = os.path.join(output_dir, file_name)
        activation = torch.load(file_path)
        
        # Sanity check: Verify tensor shape and type
        if not isinstance(activation, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(activation)} for file {file_name}")
        if len(activation.shape) != 3:  # Assuming shape is [batch_size, num_layers, hidden_size]
            raise ValueError(f"Unexpected tensor shape {activation.shape} for file {file_name}")
            
        activations.append(activation)
    
    # Stack all activations
    return torch.cat(activations, dim=0)

# Example usage:
# process_activations(
#     sentences=df_speech_filtered_concat['Text'].tolist(),
#     model_name="meta-llama/Meta-Llama-3-8B",
#     cache_dir=TRANSFORMERS_CACHE_DIR,
#     output_dir=ACTIVATIONS_CACHE_DIR,
#     activation_type="mlp",
#     device="cuda" if torch.cuda.is_available() else "cpu",
#     batch_size=8
# )

# To load the activations later:
# all_activations = load_activations(ACTIVATIONS_CACHE_DIR, len(df_speech_filtered_concat))








def perform_lda_analysis(df: pd.DataFrame, 
                         n_components: int,
                         embedding_column: str,
                         party_column: str,
                         speaker_column: str,
                         entry_count_column: str,
                         additional_columns: List[str] = None) -> pd.DataFrame:
    """
    Perform LDA analysis on speaker embeddings across all layers.

    Args:
    df (pd.DataFrame): DataFrame containing speaker embeddings and metadata.
    n_components (int): Number of LDA components to compute.
    embedding_column (str): Name of the column containing embeddings.
    party_column (str): Name of the column containing party information.
    speaker_column (str): Name of the column containing speaker information.
    entry_count_column (str): Name of the column containing entry counts.
    additional_columns (List[str]): List of additional columns to include in the output.

    Returns:
    pd.DataFrame: DataFrame with LDA projections for each layer and speaker/party.
    """
    if additional_columns is None:
        additional_columns = []

    num_layers = df.iloc[0][embedding_column].shape[0]
    plot_data = []

    for layer in tqdm(range(num_layers), desc="Processing layers"):
        layer_plot_data = []
        
        # Prepare data for LDA
        X = np.array([row[embedding_column][layer].flatten() for _, row in df.iterrows()])
        y = df[party_column].values
        
        # Encode party labels
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        # Perform LDA
        lda = LinearDiscriminantAnalysis(n_components=n_components)
        X_lda = lda.fit_transform(X, y_encoded)
        
        # Create plot data
        for i, (_, row) in enumerate(df.iterrows()):
            data_point = {
                party_column: row[party_column],
                speaker_column: row[speaker_column],
                entry_count_column: row[entry_count_column],
                "Symbol": 'Speaker',
                "Layer": layer
            }
            for j in range(n_components):
                data_point[f"LDA{j+1}"] = X_lda[i, j]
            for col in additional_columns:
                data_point[col] = row[col]
            layer_plot_data.append(data_point)

        # Calculate centroids for each party
        agg_dict = {entry_count_column: 'sum'}
        for j in range(n_components):
            agg_dict[f"LDA{j+1}"] = 'mean'
        
        centroid_data = pd.DataFrame(layer_plot_data).groupby(party_column).agg(agg_dict).reset_index()
        centroid_data[speaker_column] = centroid_data[party_column]
        centroid_data['Symbol'] = 'Party'
        centroid_data['Layer'] = layer

        # Append data to plot_data
        plot_data.extend(layer_plot_data)
        plot_data.extend(centroid_data.to_dict('records'))

    return pd.DataFrame(plot_data)

# Example usage:
# lda_df = perform_lda_analysis(df_speaker_mean_embeddings, 
#                               n_components=3,
#                               embedding_column='Embedding',
#                               party_column='Partei',
#                               speaker_column='Speaker',
#                               entry_count_column='EntryCount',
#                               additional_columns=['SpeakerStatus', 'Religion'])
