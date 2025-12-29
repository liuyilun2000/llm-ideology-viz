"""
Dataset Loading Module

This module handles loading and preprocessing of political speech datasets.
"""

import pandas as pd
from typing import List, Optional
from tqdm.auto import tqdm
from datasets import load_dataset


def load_parlasent(
    datasets: Optional[List[str]] = None,
    columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Load and process the ParlaSent datasets into a single DataFrame.
    
    ParlaSent is a multilingual collection of manually annotated parliamentary
    speeches. This function loads one or more ParlaSent datasets and combines
    them into a single DataFrame.
    
    Args:
        datasets: List of dataset codes to load. Available codes:
                 - 'EN': English (UK Parliament)
                 - 'EN_additional_test': English additional test set
                 - 'BCS': Bosnian/Croatian/Serbian
                 - 'BCS_additional_test': BCS additional test set
                 - 'CZ': Czech
                 - 'SK': Slovak
                 - 'SL': Slovenian
                 If None, loads all available datasets.
        columns: List of columns to include. If None, includes all standard columns:
                 ['dataset', 'sentence', 'country', 'date', 'name', 'party',
                  'gender', 'birth_year', 'ruling']
    
    Returns:
        pd.DataFrame: Combined DataFrame with political speech data
    """
    if datasets is None:
        datasets = [
            'EN', 'EN_additional_test', 'BCS', 'BCS_additional_test',
            'CZ', 'SK', 'SL'
        ]
    
    if columns is None:
        columns = [
            'dataset', 'sentence', 'country', 'date', 'name', 'party',
            'gender', 'birth_year', 'ruling'
        ]
    
    def load_and_process_dataset(dataset_code: str) -> pd.DataFrame:
        """Load and process a single ParlaSent dataset."""
        try:
            ds = load_dataset("classla/ParlaSent", dataset_code)['train']
            df = pd.DataFrame(ds)
            df['dataset'] = dataset_code
            
            # Select only requested columns that exist
            available_columns = [col for col in columns if col in df.columns]
            return df[available_columns]
        except Exception as e:
            print(f"Warning: Failed to load dataset {dataset_code}: {e}")
            return pd.DataFrame()
    
    # Load all datasets
    dfs = [
        load_and_process_dataset(dataset)
        for dataset in tqdm(datasets, desc="Loading ParlaSent datasets")
    ]
    
    # Filter out empty DataFrames
    dfs = [df for df in dfs if not df.empty]
    
    if not dfs:
        raise ValueError("No datasets were successfully loaded")
    
    # Combine all DataFrames
    df = pd.concat(dfs, ignore_index=True)
    
    return df


def filter_dataset(
    df: pd.DataFrame,
    country: Optional[str] = None,
    parties: Optional[List[str]] = None,
    min_speeches_per_party: int = 0
) -> pd.DataFrame:
    """
    Filter dataset by country and/or parties.
    
    Args:
        df: Input DataFrame
        country: Optional country code to filter by
        parties: Optional list of party names to include
        min_speeches_per_party: Minimum number of speeches required per party
    
    Returns:
        Filtered DataFrame
    """
    filtered_df = df.copy()
    
    # Filter by country
    if country is not None:
        if 'country' not in filtered_df.columns:
            raise ValueError("DataFrame must have 'country' column")
        filtered_df = filtered_df[filtered_df['country'] == country]
    
    # Filter by parties
    if parties is not None:
        if 'party' not in filtered_df.columns:
            raise ValueError("DataFrame must have 'party' column")
        filtered_df = filtered_df[filtered_df['party'].isin(parties)]
    
    # Filter by minimum speeches per party
    if min_speeches_per_party > 0:
        if 'party' not in filtered_df.columns:
            raise ValueError("DataFrame must have 'party' column")
        party_counts = filtered_df['party'].value_counts()
        valid_parties = party_counts[party_counts >= min_speeches_per_party].index
        filtered_df = filtered_df[filtered_df['party'].isin(valid_parties)]
    
    return filtered_df

