#!/usr/bin/env python3
"""
Script to check party counts in ParlaSent datasets.

This script loads each ParlaSent subset and prints the number of speeches
per party, helping to determine which parties to include in experiments.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data import load_parlasent
import pandas as pd


def check_party_counts(dataset_subset: str, country: str = None):
    """Check party counts for a given dataset subset."""
    print(f"\n{'='*80}")
    print(f"Party Counts for {dataset_subset}")
    if country:
        print(f"Country: {country}")
    print(f"{'='*80}\n")
    
    # Load dataset
    df = load_parlasent(datasets=[dataset_subset])
    
    # Filter by country if specified
    if country:
        df = df[df['country'] == country]
        print(f"Total samples after country filter: {len(df)}")
    else:
        print(f"Total samples: {len(df)}")
    
    if len(df) == 0:
        print("No samples found!")
        return
    
    # Count by party
    party_counts = df['party'].value_counts().sort_values(ascending=False)
    
    print(f"\nParties (sorted by count):")
    print(f"{'Party':<40} {'Count':<10} {'Percentage':<10}")
    print("-" * 60)
    
    total = len(df)
    for party, count in party_counts.items():
        percentage = (count / total) * 100
        print(f"{party:<40} {count:<10} {percentage:>6.2f}%")
    
    print(f"\nTotal parties: {len(party_counts)}")
    print(f"Total speeches: {total}")
    
    # Suggest parties with sufficient data (>= 50 speeches)
    sufficient_parties = party_counts[party_counts >= 50]
    if len(sufficient_parties) > 0:
        print(f"\nParties with >= 50 speeches (recommended for analysis):")
        for party, count in sufficient_parties.items():
            print(f"  - {party}: {count}")
    
    return party_counts


def main():
    """Check party counts for all ParlaSent subsets."""
    subsets = {
        'EN': 'UK',
        'BCS': None,  # BCS spans multiple countries
        'CZ': 'CZ',
        'SK': 'SK',
        'SL': 'SL'
    }
    
    for subset, country in subsets.items():
        try:
            check_party_counts(subset, country)
        except Exception as e:
            print(f"\nError checking {subset}: {e}\n")
    
    # Also check BCS by country
    print(f"\n{'='*80}")
    print("BCS Dataset - Breakdown by Country")
    print(f"{'='*80}\n")
    try:
        df_bcs = load_parlasent(datasets=['BCS'])
        print(f"Total BCS samples: {len(df_bcs)}")
        
        if 'country' in df_bcs.columns:
            print("\nBy Country:")
            country_counts = df_bcs['country'].value_counts()
            for country, count in country_counts.items():
                print(f"  {country}: {count}")
            
            # Check parties for each country in BCS
            for country in df_bcs['country'].unique():
                if pd.notna(country):
                    print(f"\n--- BCS - {country} ---")
                    df_country = df_bcs[df_bcs['country'] == country]
                    party_counts = df_country['party'].value_counts().sort_values(ascending=False)
                    for party, count in party_counts.items():
                        print(f"  {party}: {count}")
    except Exception as e:
        print(f"Error checking BCS: {e}")


if __name__ == "__main__":
    main()
