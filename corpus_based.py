import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from scipy import stats
from typing import Dict, List, Tuple, Union, Optional


import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio

pio.templates.default = "plotly_white"


class PoliticalSpeechAnalyzer:
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the analyzer with a DataFrame containing political speech data.
        
        Args:
            df: DataFrame with columns including 'party', 'country', 'sentence', 'embedding'
        """
        self.df = df
        self.party_encoder = LabelEncoder()
        self.n_layers = df['embedding'].iloc[0].shape[0]
        self.filtered_df = None  # Will store filtered DataFrame
        
    def analyze_dataset_composition(self) -> Dict:
        """
        Analyze basic composition of the dataset.
        
        Returns:
            Dictionary containing various statistics about the dataset
        """
        # Basic counts
        stats = {
            'total_speeches': len(self.df),
            'unique_parties': len(self.df['party'].unique()),
            'countries': self.df['country'].value_counts().to_dict(),
            'parties_by_country': {
                country: len(self.df[self.df['country'] == country]['party'].unique())
                for country in self.df['country'].unique()
            }
        }
        
        # Party statistics
        party_stats = self.df['party'].value_counts()
        stats['speeches_per_party'] = party_stats.to_dict()
        stats['party_percentages'] = (party_stats / len(self.df) * 100).round(2).to_dict()
        
        # Additional demographics if available
        if 'gender' in self.df.columns:
            stats['gender_distribution'] = self.df['gender'].value_counts().to_dict()
        
        if 'ruling' in self.df.columns:
            stats['ruling_party_distribution'] = self.df['ruling'].value_counts().to_dict()
            
        # Speech length statistics
        stats['speech_length'] = {
            'mean': self.df['sentence'].str.len().mean(),
            'median': self.df['sentence'].str.len().median(),
            'min': self.df['sentence'].str.len().min(),
            'max': self.df['sentence'].str.len().max()
        }
        
        # Time period if dates are available
        if 'date' in self.df.columns:
            stats['time_period'] = {
                'start': self.df['date'].min(),
                'end': self.df['date'].max()
            }
        
        return stats
        
    def print_dataset_statistics(self):
        """
        Print a formatted summary of the dataset statistics with aligned columns.
        """
        stats = self.analyze_dataset_composition()
        
        print("\nDataset Statistics:")
        print("="* 80)
        print(f"Total speeches: {stats['total_speeches']:,}")
        print(f"Unique parties: {stats['unique_parties']}")
        
        print("\nSpeeches by Country:")
        print("="* 80)
        # Format: country(30) | count(10) | percentage(10) | parties(10)
        print(f"{'Country':<30} {'Speeches':>10} {'Percent':>10} {'Parties':>10}")
        print("-" * 80)
        for country, count in sorted(stats['countries'].items(), key=lambda x: x[1], reverse=True):
            print(f"{country:<30} {count:>10,d} {count/stats['total_speeches']*100:>9.1f}% {stats['parties_by_country'][country]:>10d}")
        
        print("\nSpeeches by Party:")
        print("="* 80)
        # Format: party(40) | count(10) | percentage(10)
        print(f"{'Party':<40} {'Speeches':>10} {'Percent':>10}")
        print("-" * 80)
        for party, count in sorted(stats['speeches_per_party'].items(), key=lambda x: x[1], reverse=True):
            print(f"{party:<40} {count:>10,d} {stats['party_percentages'][party]:>9.1f}%")
        
        if 'speech_length' in stats:
            print("\nSpeech Length Statistics:")
            print("="* 80)
            print(f"{'Metric':<20} {'Value':>10}")
            print("-" * 80)
            print(f"{'Mean length:':<20} {stats['speech_length']['mean']:>10.1f}")
            print(f"{'Median length:':<20} {stats['speech_length']['median']:>10.1f}")
            print(f"{'Minimum length:':<20} {stats['speech_length']['min']:>10d}")
            print(f"{'Maximum length:':<20} {stats['speech_length']['max']:>10d}")
        
        if 'time_period' in stats:
            print("\nTime Period:")
            print("="* 80)
            print(f"From: {stats['time_period']['start']}")
            print(f"To:   {stats['time_period']['end']}")
        
        if 'gender_distribution' in stats:
            print("\nGender Distribution:")
            print("="* 80)
            # Format: gender(20) | count(10) | percentage(10)
            print(f"{'Gender':<20} {'Count':>10} {'Percent':>10}")
            print("-" * 80)
            total = sum(stats['gender_distribution'].values())
            for gender, count in sorted(stats['gender_distribution'].items(), key=lambda x: x[1], reverse=True):
                print(f"{gender:<20} {count:>10,d} {count/total*100:>9.1f}%")
        
        if 'ruling_party_distribution' in stats:
            print("\nRuling Party Distribution:")
            print("="* 80)
            # Format: status(20) | count(10) | percentage(10)
            print(f"{'Status':<20} {'Count':>10} {'Percent':>10}")
            print("-" * 80)
            total = sum(stats['ruling_party_distribution'].values())
            for status, count in sorted(stats['ruling_party_distribution'].items(), key=lambda x: x[1], reverse=True):
                print(f"{status:<20} {count:>10,d} {count/total*100:>9.1f}%")
        
    def prepare_embeddings_for_lda(self, 
                                min_speeches_per_party: int = 100,
                                layers: Optional[Union[int, List[int], str]] = None
                                ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare embeddings for LDA analysis by filtering and encoding parties.
        """
        # Filter parties with sufficient data
        party_counts = self.df['party'].value_counts()
        valid_parties = party_counts[party_counts >= min_speeches_per_party].index
        
        # Filter dataframe and store it
        self.filtered_df = self.df[self.df['party'].isin(valid_parties)].copy()
        
        # Encode parties
        self.filtered_df['party_encoded'] = self.party_encoder.fit_transform(self.filtered_df['party'])
        
        # Process layer selection
        embeddings = np.stack(self.filtered_df['embedding'].values)
        
        if layers is None or layers == "all":
            X = embeddings
        elif isinstance(layers, int):
            X = embeddings[:, layers, :]
        elif isinstance(layers, list):
            X = embeddings[:, layers, :]
        elif layers == "last":
            X = embeddings[:, -1, :]
        elif layers == "mean":
            X = np.mean(embeddings, axis=1)
        else:
            raise ValueError(f"Invalid layers parameter: {layers}")
            
        y = self.filtered_df['party_encoded'].values
        
        return X, y
    
    def perform_lda(self, 
                    n_components: int = 3, 
                    min_speeches_per_party: int = 100,
                    layers: Optional[Union[int, List[int], str]] = None,
                    random_seed: int = 42
                    ) -> Tuple[LinearDiscriminantAnalysis, np.ndarray]:
        """
        Perform LDA on the embeddings.
        
        Args:
            n_components: Number of LDA components to extract
            min_speeches_per_party: Minimum speeches required for a party
            layers: Which layers to use
            random_seed: Seed for numpy's random number generator
            
        Returns:
            Tuple of (fitted LDA model, transformed embeddings)
        """
        # Set random seed
        np.random.seed(random_seed)
        
        X, y = self.prepare_embeddings_for_lda(min_speeches_per_party, layers)
        
        # Flatten embeddings if they're 3D (in case of multiple layers)
        if len(X.shape) > 2:
            print(f"X {X.shape}")
            X = X.reshape(X.shape[0], -1)
            print(f"X {X.shape}")
        
        # Perform LDA
        lda = LinearDiscriminantAnalysis(n_components=n_components)
        X_lda = lda.fit_transform(X, y)
        
        return lda, X_lda
        
    def plot_lda_results_2d(self, X_lda: np.ndarray, layer: Optional[int] = None, 
                            save_path: str = None, n_std: float = 3.0, 
                            party_colors: Optional[Dict[str, str]] = None):
        """
        Plot the LDA results in 2D using plotly express with unified scatter plot.
        """
        if X_lda.shape[1] != 2:
            raise ValueError("This plotting function requires exactly 2 LDA components")
        
        if self.filtered_df is None:
            raise ValueError("No LDA has been performed yet. Run perform_lda first.")
        
        # Create a copy of the data for outlier handling
        X_plot = X_lda.copy()
        
        # Calculate z-scores for both dimensions
        #z_scores = np.abs(stats.zscore(X_plot))
        
        # Create mask for non-outlier points
        #mask = (z_scores < n_std).all(axis=1)
        
        # Apply mask to both the LDA coordinates and the filtered DataFrame
        #X_plot = X_plot[mask]
        #plot_df = self.filtered_df[mask].copy()
        plot_df = self.filtered_df.copy()
        
        # Normalize the coordinates to center 0 and range ±1
        # First center to 0
        X_mean = X_plot.mean(axis=0)
        X_std = X_plot.std(axis=0)
        
        # Center the data
        X_plot = X_plot - X_mean
        
        # Scale the data so that n_std standard deviations = ±1
        X_plot = X_plot / (n_std * X_std)
        
        # Print outlier statistics
        #total_points = len(mask)
        #outliers = np.sum(~mask)
        #print(f"Removed {outliers} outliers out of {total_points} points ({(outliers/total_points)*100:.2f}%)")
        
        # Create plot data for individual speeches
        speech_data = []
        name_data = []
        party_data = []
        
        # First process each party
        for party in plot_df['party'].unique():
            party_mask = plot_df['party'] == party
            party_points = X_plot[party_mask]
            party_count = np.sum(party_mask)
            party_center = np.mean(party_points, axis=0)
            
            # Then process each name within the party
            for name in plot_df[party_mask]['name'].unique():
                name_mask = plot_df['name'] == name
                name_points = X_plot[name_mask]
                name_count = np.sum(name_mask)
                name_center = np.mean(name_points, axis=0)
                
                # Add name center point
                name_data.append({
                    'LDA1': name_center[0],
                    'LDA2': name_center[1],
                    'Party': party,
                    'Category': 'Speaker',
                    'size': name_count * 10,
                })
            
            # Add party center
            party_data.append({
                'LDA1': party_center[0],
                'LDA2': party_center[1],
                'Party': party,
                'Category': 'Party',
                'size': party_count * 15,
            })
        
        # Combine all data
        plot_data = pd.DataFrame(party_data + name_data)
        
        # Create plot using plotly express
        fig = px.scatter(
            plot_data, 
            x='LDA1', 
            y='LDA2',
            color='Party',
            symbol='Category',
            size='size',
            size_max=200,
            color_discrete_map=party_colors,
            symbol_map={'Speaker': 'square', 'Party': 'circle'},
            opacity=0.60,
            width=300,
            height=320
        )
        
        max_abs_val = max(
            abs(plot_data['LDA1']).max(),
            abs(plot_data['LDA2']).max()
        )
        # Add a small buffer (e.g., 10%)
        max_range = max_abs_val * 1.1
        # Add grid
        fig.update_xaxes(showticklabels=False, showgrid=True, gridwidth=1, gridcolor='LightGray')
        fig.update_yaxes(showticklabels=False, showgrid=True, gridwidth=1, gridcolor='LightGray')
        # Update layout
        fig.update_layout(
            xaxis_range=[-max_range, max_range],
            yaxis_range=[-max_range, max_range],
            margin=dict(l=20, r=20, t=20, b=20),
            font_family="Libertinus Sans",
            plot_bgcolor='white',
            #showlegend=False
            legend=dict(
                orientation="h",  # horizontal legend
                yanchor="top",
                y=-0.15,
                xanchor="left",
                x=0
            )
        )
        
        # Save if path provided
        if save_path:
            fig.write_html(f"{save_path}.html")
            fig.write_image(f"{save_path}.png", scale=4)
        
        return fig



country='UK'
parties = [
    'Conservative',
    'Labour',
    'Liberal Democrat',
    'Scottish National Party',
    #'Democratic Unionist Party',
    #'Green Party',
    #'Plaid Cymru',
    #'Ulster Unionist Party',
    #'Social Democratic & Labour Party'
]
party_colors = {
    'Conservative': '#0194E1',
    'Labour': '#DC241F',            # Labour Red
    'Liberal Democrat': '#FAA61A',   # Lib Dem Gold/Orange
    'Scottish National Party': '#EEE95D', # SNP Yellow
    'Democratic Unionist Party': '#D46A4C', # DUP Orange-Red
    'Green Party': '#6AB023',        # Green
    'Plaid Cymru': '#008142',       # Plaid Green
    'Ulster Unionist Party': '#9999FF', # UUP Blue
    'Social Democratic & Labour Party': '#99FF66' # SDLP Green
}

df_filtered = df[
    (df['country'] == country) & 
    (df['party'].isin(parties))
]
df_filtered = df_filtered.sort_values('party')
# Initialize the analyzer
analyzer = PoliticalSpeechAnalyzer(df_filtered)

# Get dataset statistics
analyzer.print_dataset_statistics()

#analyzer.prepare_embeddings_for_lda(layers=layer)
for layer in tqdm([0, 2]):#, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]):
    # Perform LDA on a single layer
    lda_model, X_lda = analyzer.perform_lda(
        n_components=2, 
        layers=layer, 
        min_speeches_per_party=1,
        random_seed=42
    )
    # Plot the results
    fig = analyzer.plot_lda_results_2d(
        X_lda, 
        party_colors=party_colors,
        save_path=f"img/{country}_{MODEL_NAME.split('/')[-1]}_layer_{layer}_lda_2d"
    )



