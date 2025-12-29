"""
Visualization Module for Corpus-Based Framework

This module provides plotting and visualization utilities for the corpus-based
ideological manifold analysis.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from typing import Dict, Optional, List
from scipy import stats
from scipy.ndimage import gaussian_filter

# Set default plotly template
pio.templates.default = "plotly_white"


class CorpusVisualizer:
    """
    Visualization utilities for corpus-based ideological manifolds.
    """
    
    def __init__(self, party_colors: Optional[Dict[str, str]] = None):
        """
        Initialize the visualizer.
        
        Args:
            party_colors: Optional dictionary mapping party names to colors
        """
        self.party_colors = party_colors or {}
    
    def plot_lda_2d(
        self,
        projections: np.ndarray,
        metadata: pd.DataFrame,
        party_column: str = 'party',
        speaker_column: Optional[str] = None,
        text_column: Optional[str] = None,
        save_path: Optional[str] = None,
        n_std: float = 3.0,
        title: Optional[str] = None,
        show_legend: bool = True,
        visualization_style: str = 'heatmap',
        show_individual_speeches: bool = False,
        show_points: bool = True,
        show_contours: bool = True,
        contour_levels: int = 4
    ) -> go.Figure:
        """
        Create a 2D plot of LDA projections.
        
        Args:
            projections: LDA projections of shape [N, 2] or [N, L, 2]
            metadata: DataFrame with metadata (party, speaker, etc.)
            party_column: Column name for party labels
            speaker_column: Optional column name for speaker labels
            text_column: Optional column name for text/speech content (for hover data)
            save_path: Optional path to save the plot (without extension)
            n_std: Number of standard deviations for normalization
            title: Optional plot title
            show_legend: Whether to show legend
            visualization_style: Style of visualization - 'centroids' (large circles) or 'heatmap' (density contours)
            show_individual_speeches: Whether to show individual speech points (default: False, only shows speakers and parties)
            show_points: Whether to show individual data points (only for heatmap style, ignored if show_individual_speeches=False)
            show_contours: Whether to show density contours (only for heatmap style)
            contour_levels: Number of contour levels to show (only for heatmap style)
        
        Returns:
            plotly Figure object
        """
        if projections.ndim == 3:
            # Use the last layer if multi-layer
            projections = projections[:, -1, :]
        
        if projections.shape[1] != 2:
            raise ValueError("This function requires exactly 2 LDA components")
        
        # Normalize coordinates
        X_plot = self._normalize_coordinates(projections, n_std)
        
        # Auto-detect text column if not provided
        if text_column is None:
            # Try common column names
            for col in ['sentence', 'text', 'speech', 'content']:
                if col in metadata.columns:
                    text_column = col
                    break
        
        # Choose visualization style
        if visualization_style == 'centroids':
            return self._plot_lda_2d_centroids(
                X_plot, metadata, party_column, speaker_column, text_column,
                save_path, title, show_legend, show_individual_speeches
            )
        elif visualization_style == 'heatmap':
            return self._plot_lda_2d_heatmap(
                X_plot, metadata, party_column, speaker_column, text_column,
                save_path, title, show_legend, show_individual_speeches, 
                show_points, show_contours, contour_levels
            )
        else:
            raise ValueError(f"Unknown visualization_style: {visualization_style}. Must be 'centroids' or 'heatmap'")
    
    def _plot_lda_2d_centroids(
        self,
        X_plot: np.ndarray,
        metadata: pd.DataFrame,
        party_column: str,
        speaker_column: Optional[str],
        text_column: Optional[str],
        save_path: Optional[str],
        title: Optional[str],
        show_legend: bool,
        show_individual_speeches: bool = False
    ) -> go.Figure:
        """
        Create a 2D scatter plot with large circles for party centroids (original style).
        """
        # Create plot data with centroids
        plot_data = self._prepare_plot_data(
            X_plot, metadata, party_column, speaker_column, show_individual_speeches, text_column
        )
        
        # Filter out individual speeches if not requested
        if not show_individual_speeches:
            plot_data = plot_data[plot_data['Category'].isin(['Party', 'Speaker'])]
        
        # Create plot using go.Figure for better hover control
        fig = go.Figure()
        
        # Group by party and category for proper styling
        for party in plot_data['Party'].unique():
            party_data = plot_data[plot_data['Party'] == party]
            party_color = self.party_colors.get(party, None)
            
            # Process each category (Party, Speaker, Speech)
            for category in party_data['Category'].unique():
                category_data = party_data[party_data['Category'] == category]
                
                # Prepare hover texts
                hover_texts = []
                for _, row in category_data.iterrows():
                    hover_parts = [f"<b>Party:</b> {row['Party']}"]
                    
                    if category == 'Speaker' and speaker_column:
                        # Get speaker name from metadata
                        speaker_name = row.get(speaker_column, 'Unknown') if speaker_column in row else 'Unknown'
                        hover_parts.append(f"<b>Speaker:</b> {speaker_name}")
                    elif category == 'Speech' and text_column:
                        # Get text from original metadata if available
                        text_content = row.get(text_column, '') if text_column in row else ''
                        if text_content:
                            text_preview = str(text_content)[:32]
                            if len(str(text_content)) > 32:
                                text_preview += "..."
                            hover_parts.append(f"<b>Speech:</b> {text_preview}")
                    
                    hover_texts.append("<br>".join(hover_parts))
                
                # Determine symbol
                symbol_map = {'Speaker': 'square', 'Party': 'circle', 'Speech': 'circle'}
                symbol = symbol_map.get(category, 'circle')
                
                fig.add_trace(
                    go.Scatter(
                        x=category_data['LDA1'],
                        y=category_data['LDA2'],
                        mode='markers',
                        name=party if category == 'Party' else '',
                        marker=dict(
                            size=category_data['size'],
                            color=party_color,
                            opacity=0.60,
                            symbol=symbol,
                            line=dict(width=1, color='white') if category == 'Speaker' else None
                        ),
                        hovertemplate='%{hovertext}<extra></extra>',
                        hovertext=hover_texts,
                        legendgroup=party,
                        showlegend=False  # Hide from legend, will add separate legend trace
                    )
                )
        
        # Add legend-only traces with solid colors and larger markers
        if show_legend:
            for party in plot_data['Party'].unique():
                party_color = self.party_colors.get(party, None)
                if party_color is None:
                    colors = list(px.colors.qualitative.Set3)
                    party_color = colors[hash(party) % len(colors)]
                
                # Place marker outside visible plot area (will be hidden but visible in legend)
                fig.add_trace(
                    go.Scatter(
                        x=[None],  # Outside plot area
                        y=[None],
                        mode='markers',
                        name=party,
                        marker=dict(
                            size=12,  # Larger marker for legend
                            color=party_color,
                            opacity=1.0,  # Solid color
                            line=dict(width=1, color='white')
                        ),
                        showlegend=True,
                        legendgroup=party
                    )
                )
        
        # Update layout
        max_abs_val = max(
            abs(plot_data['LDA1']).max(),
            abs(plot_data['LDA2']).max()
        )
        max_range = max_abs_val * 1.1
        
        fig.update_xaxes(
            showticklabels=False,
            showgrid=True,
            gridwidth=1,
            gridcolor='LightGray',
            range=[-max_range, max_range]
        )
        fig.update_yaxes(
            showticklabels=False,
            showgrid=True,
            gridwidth=1,
            gridcolor='LightGray',
            range=[-max_range, max_range]
        )
        
        fig.update_layout(
            margin=dict(l=20, r=20, t=40, b=20),
            font_family="Libertinus Sans",
            plot_bgcolor='white',
            showlegend=show_legend,
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.15,
                xanchor="left",
                x=0
            ) if show_legend else None
        )
        
        # Save if path provided
        if save_path:
            fig.write_html(f"{save_path}.html")
            try:
                fig.write_image(f"{save_path}.png", scale=4, width=600, height=600)
            except Exception as e:
                print(f"Warning: Could not save PNG image: {e}")
        
        return fig
    
    def _plot_lda_2d_heatmap(
        self,
        X_plot: np.ndarray,
        metadata: pd.DataFrame,
        party_column: str,
        speaker_column: Optional[str],
        text_column: Optional[str],
        save_path: Optional[str],
        title: Optional[str],
        show_legend: bool,
        show_individual_speeches: bool,
        show_points: bool,
        show_contours: bool,
        contour_levels: int
    ) -> go.Figure:
        """
        Create a 2D heatmap/contour plot with distribution visualization.
        """
        # Create figure
        fig = go.Figure()
        
        # Get unique parties
        parties = metadata[party_column].unique()
        
        # Calculate overall range for consistent axes
        max_abs_val = max(
            abs(X_plot[:, 0]).max(),
            abs(X_plot[:, 1]).max()
        )
        max_range = max_abs_val * 1.1
        
        # First pass: Add individual points and speaker centroids (bottom layer)
        # Process each party
        for party in parties:
            party_mask = metadata[party_column] == party
            party_points = X_plot[party_mask]
            
            if len(party_points) == 0:
                continue
            
            # Get party color
            party_color = self.party_colors.get(party, None)
            if party_color is None:
                # Generate a default color if not specified
                colors = list(px.colors.qualitative.Set3)
                party_color = colors[hash(party) % len(colors)]
            
            # Convert hex color to rgba for transparency
            if party_color.startswith('#'):
                rgb = tuple(int(party_color[i:i+2], 16) for i in (1, 3, 5))
                rgba_contour = f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, 0.3)"
                rgba_points = f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, 0.6)"
            else:
                rgba_contour = f"rgba(128, 128, 128, 0.3)"
                rgba_points = f"rgba(128, 128, 128, 0.6)"
            
            # Add individual points (only if show_individual_speeches is True)
            if show_individual_speeches and show_points:
                # Prepare hover text for individual speeches
                party_metadata_subset = metadata[party_mask]
                hover_texts = []
                
                for idx in range(len(party_points)):
                    hover_parts = [f"<b>Party:</b> {party}"]
                    
                    if speaker_column and speaker_column in party_metadata_subset.columns:
                        speaker_name = party_metadata_subset.iloc[idx][speaker_column]
                        if pd.notna(speaker_name):
                            hover_parts.append(f"<b>Speaker:</b> {speaker_name}")
                    
                    if text_column and text_column in party_metadata_subset.columns:
                        text_content = party_metadata_subset.iloc[idx][text_column]
                        if pd.notna(text_content):
                            # Truncate to first 32 characters
                            text_preview = str(text_content)[:32]
                            if len(str(text_content)) > 32:
                                text_preview += "..."
                            hover_parts.append(f"<b>Speech:</b> {text_preview}")
                    
                    hover_texts.append("<br>".join(hover_parts))
                
                fig.add_trace(
                    go.Scatter(
                        x=party_points[:, 0],
                        y=party_points[:, 1],
                        mode='markers',
                        name=party,
                        marker=dict(
                            size=4,
                            color=party_color if not party_color.startswith('rgba') else rgba_points,
                            opacity=0.2,
                            line=dict(width=0.5, color='white')
                        ),
                        legendgroup=party,
                        showlegend=False,  # Hide from legend, will add separate legend trace
                        hovertemplate='%{hovertext}<extra></extra>',
                        hovertext=hover_texts
                    )
                )
        
        # Add speaker centroids (always use original centroids style)
        if speaker_column and speaker_column in metadata.columns:
            # Calculate speaker centroids manually to ensure we have the right data
            for party in parties:
                party_mask = metadata[party_column] == party
                party_points = X_plot[party_mask]
                
                if len(party_points) == 0:
                    continue
                
                # Get party color
                party_color = self.party_colors.get(party, None)
                if party_color is None:
                    colors = list(px.colors.qualitative.Set3)
                    party_color = colors[hash(party) % len(colors)]
                
                # Process each speaker in this party
                party_metadata = metadata[party_mask]
                for speaker in party_metadata[speaker_column].unique():
                    speaker_mask = (metadata[party_column] == party) & \
                                  (metadata[speaker_column] == speaker)
                    speaker_points = X_plot[speaker_mask]
                    speaker_count = np.sum(speaker_mask)
                    
                    if speaker_count == 0:
                        continue
                    
                    # Calculate speaker centroid (averaged location)
                    speaker_center = np.mean(speaker_points, axis=0)
                    
                    # Calculate speaker size with square root scaling to prevent too large sizes
                    # This prevents speakers with many speeches from becoming linearly too large
                    base_size = 0  # Base size for speakers
                    size_scaling = np.sqrt(speaker_count) * 4  # Square root scaling
                    speaker_size = base_size + size_scaling
                    # Cap maximum size to prevent extremely large markers
                    max_speaker_size = 150
                    speaker_size = min(speaker_size, max_speaker_size)
                    
                    # Prepare hover text for speaker
                    hover_parts = [f"<b>Party:</b> {party}"]
                    if speaker_column:
                        hover_parts.append(f"<b>Speaker:</b> {speaker}")
                    hover_text = "<br>".join(hover_parts)
                    
                    # Add speaker centroid as square (size based on entry count with non-linear scaling)
                    fig.add_trace(
                        go.Scatter(
                            x=[speaker_center[0]],
                            y=[speaker_center[1]],
                            mode='markers',
                            name=f"{speaker} ({party})",
                            marker=dict(
                                size=speaker_size,
                                color=party_color,
                                opacity=0.5,
                                symbol='square',
                                line=dict(width=1, color='white')
                            ),
                            legendgroup=party,
                            showlegend=False,  # Don't show in legend to avoid clutter
                            hovertemplate='%{hovertext}<extra></extra>',
                            hovertext=[hover_text]
                        )
                    )
        
        # Second pass: Add density contours/heatmaps on top (top layer)
        for party in parties:
            party_mask = metadata[party_column] == party
            party_points = X_plot[party_mask]
            
            if len(party_points) == 0:
                continue
            
            # Get party color
            party_color = self.party_colors.get(party, None)
            if party_color is None:
                colors = list(px.colors.qualitative.Set3)
                party_color = colors[hash(party) % len(colors)]
            
            # Convert hex color to rgba for transparency
            if party_color.startswith('#'):
                rgb = tuple(int(party_color[i:i+2], 16) for i in (1, 3, 5))
                rgba_contour = f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, 0.3)"
            else:
                rgba_contour = f"rgba(128, 128, 128, 0.3)"
            
            # Add density contour/heatmap (on top)
            if show_contours and len(party_points) > 2:
                try:
                    # Create density heatmap using histogram2d
                    hist, xedges, yedges = np.histogram2d(
                        party_points[:, 0],
                        party_points[:, 1],
                        bins=50,
                        range=[[-max_range, max_range], [-max_range, max_range]]
                    )
                    
                    # Smooth the histogram slightly for better visualization
                    hist_smooth = gaussian_filter(hist, sigma=1.0)
                    
                    # Create contour plot
                    xcenters = (xedges[:-1] + xedges[1:]) / 2
                    ycenters = (yedges[:-1] + yedges[1:]) / 2
                    
                    # Normalize histogram for better contour levels
                    hist_max = hist_smooth.max()
                    if hist_max > 0:
                        hist_normalized = hist_smooth / hist_max
                        
                        fig.add_trace(
                            go.Contour(
                                x=xcenters,
                                y=ycenters,
                                z=hist_normalized.T,
                                colorscale=[[0, 'rgba(255,255,255,0)'], [0.3, rgba_contour], [1, party_color]],
                                showscale=False,
                                contours=dict(
                                    showlines=True,
                                    start=0.1,
                                    end=0.9,
                                    size=0.8 / contour_levels
                                ),
                                line=dict(width=1.5, color=party_color),
                                name=party,
                                legendgroup=party,
                                showlegend=False,
                                hoverinfo='skip'  # Disable hover on contours so underlying points are hoverable
                            )
                        )
                except Exception as e:
                    print(f"Warning: Could not create contour for {party}: {e}")
        
        # Add legend-only traces with solid colors and larger markers
        if show_legend:
            for party in parties:
                party_color = self.party_colors.get(party, None)
                if party_color is None:
                    colors = list(px.colors.qualitative.Set3)
                    party_color = colors[hash(party) % len(colors)]
                
                # Place marker outside visible plot area (will be hidden but visible in legend)
                fig.add_trace(
                    go.Scatter(
                        x=[None],  # Outside plot area
                        y=[None],
                        mode='markers',
                        name=party,
                        marker=dict(
                            size=12,  # Larger marker for legend
                            color=party_color,
                            opacity=1.0,  # Solid color
                            line=dict(width=1, color='white')
                        ),
                        showlegend=True,
                        legendgroup=party
                    )
                )
        
        # Update layout
        fig.update_xaxes(
            showticklabels=False,
            showgrid=True,
            gridwidth=1,
            gridcolor='LightGray',
            range=[-max_range, max_range],
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor='LightGray'
        )
        fig.update_yaxes(
            showticklabels=False,
            showgrid=True,
            gridwidth=1,
            gridcolor='LightGray',
            range=[-max_range, max_range],
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor='LightGray'
        )
        
        fig.update_layout(
            width=600,
            height=600,
            margin=dict(l=20, r=20, t=40, b=20),
            font_family="Libertinus Sans",
            plot_bgcolor='white',
            title=title,
            showlegend=show_legend,
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.15,
                xanchor="left",
                x=0
            ) if show_legend else None
        )
        
        # Save if path provided
        if save_path:
            fig.write_html(f"{save_path}.html")
            try:
                fig.write_image(f"{save_path}.png", scale=4, width=600, height=600)
            except Exception as e:
                print(f"Warning: Could not save PNG image: {e}")
        
        return fig
    
    def plot_lda_3d(
        self,
        projections: np.ndarray,
        metadata: pd.DataFrame,
        party_column: str = 'party',
        speaker_column: Optional[str] = None,
        save_path: Optional[str] = None,
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Create a 3D scatter plot of LDA projections.
        
        Args:
            projections: LDA projections of shape [N, 3] or [N, L, 3]
            metadata: DataFrame with metadata
            party_column: Column name for party labels
            speaker_column: Optional column name for speaker labels
            save_path: Optional path to save the plot
            title: Optional plot title
        
        Returns:
            plotly Figure object
        """
        if projections.ndim == 3:
            projections = projections[:, -1, :]
        
        if projections.shape[1] != 3:
            raise ValueError("This function requires exactly 3 LDA components")
        
        # Prepare plot data
        plot_data = self._prepare_plot_data(
            projections, metadata, party_column, speaker_column
        )
        
        # Create 3D plot
        fig = px.scatter_3d(
            plot_data,
            x='LDA1',
            y='LDA2',
            z='LDA3',
            color='Party',
            symbol='Category',
            size='size',
            size_max=200,
            color_discrete_map=self.party_colors,
            opacity=0.60,
            title=title
        )
        
        fig.update_layout(
            margin=dict(l=0, r=0, t=40, b=0),
            font_family="Libertinus Sans",
            scene=dict(
                xaxis=dict(showticklabels=False, showgrid=True),
                yaxis=dict(showticklabels=False, showgrid=True),
                zaxis=dict(showticklabels=False, showgrid=True),
            )
        )
        
        if save_path:
            fig.write_html(f"{save_path}.html")
        
        return fig
    
    def _normalize_coordinates(
        self,
        coordinates: np.ndarray,
        n_std: float = 3.0
    ) -> np.ndarray:
        """
        Normalize coordinates to center at 0 and scale by n_std standard deviations.
        
        Args:
            coordinates: Array of shape [N, k]
            n_std: Number of standard deviations for scaling
        
        Returns:
            Normalized coordinates
        """
        X_plot = coordinates.copy()
        X_mean = X_plot.mean(axis=0)
        X_std = X_plot.std(axis=0)
        
        # Center the data
        X_plot = X_plot - X_mean
        
        # Scale so that n_std standard deviations = ±1
        X_plot = X_plot / (n_std * X_std + 1e-8)  # Add small epsilon to avoid division by zero
        
        return X_plot
    
    def _prepare_plot_data(
        self,
        projections: np.ndarray,
        metadata: pd.DataFrame,
        party_column: str,
        speaker_column: Optional[str],
        include_individual_speeches: bool = False,
        text_column: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Prepare data for plotting by computing party and speaker centroids.
        
        Args:
            projections: LDA projections of shape [N, k]
            metadata: DataFrame with metadata
            party_column: Column name for party labels
            speaker_column: Optional column name for speaker labels
        
        Returns:
            DataFrame ready for plotting
        """
        plot_data = []
        n_dims = projections.shape[1]
        
        # Create column names for LDA dimensions
        dim_cols = [f'LDA{i+1}' for i in range(n_dims)]
        
        # Add dimension columns to metadata
        for i, col in enumerate(dim_cols):
            metadata = metadata.copy()
            metadata[col] = projections[:, i]
        
        # Process each party
        for party in metadata[party_column].unique():
            party_mask = metadata[party_column] == party
            party_points = projections[party_mask]
            party_count = np.sum(party_mask)
            
            # Party centroid
            party_center = np.mean(party_points, axis=0)
            party_data = {
                'Party': party,
                'Category': 'Party',
                'size': party_count * 15
            }
            for i, col in enumerate(dim_cols):
                party_data[col] = party_center[i]
            plot_data.append(party_data)
            
            # Speaker centroids (if speaker column provided)
            if speaker_column and speaker_column in metadata.columns:
                for speaker in metadata[party_mask][speaker_column].unique():
                    speaker_mask = (metadata[party_column] == party) & \
                                  (metadata[speaker_column] == speaker)
                    speaker_points = projections[speaker_mask]
                    speaker_count = np.sum(speaker_mask)
                    speaker_center = np.mean(speaker_points, axis=0)
                    
                    # Calculate speaker size with square root scaling to prevent too large sizes
                    base_size = 0
                    size_scaling = np.sqrt(speaker_count) * 4
                    speaker_size = base_size + size_scaling
                    max_speaker_size = 150
                    speaker_size = min(speaker_size, max_speaker_size)
                    
                    speaker_data = {
                        'Party': party,
                        'Category': 'Speaker',
                        'size': speaker_size
                    }
                    # Store speaker name if available
                    if speaker_column:
                        speaker_data[speaker_column] = speaker
                    for i, col in enumerate(dim_cols):
                        speaker_data[col] = speaker_center[i]
                    plot_data.append(speaker_data)
                    
                    # Add individual speeches if requested
                    if include_individual_speeches:
                        speaker_metadata_subset = metadata[speaker_mask]
                        for idx, point in enumerate(speaker_points):
                            speech_data = {
                                'Party': party,
                                'Category': 'Speech',
                                'size': 4
                            }
                            # Store speaker name and text if available
                            if speaker_column:
                                speech_data[speaker_column] = speaker
                            if text_column and text_column in speaker_metadata_subset.columns:
                                speech_data[text_column] = speaker_metadata_subset.iloc[idx][text_column]
                            for i, col in enumerate(dim_cols):
                                speech_data[col] = point[i]
                            plot_data.append(speech_data)
        
        return pd.DataFrame(plot_data)
    
    def plot_lda_1d(
        self,
        projections: np.ndarray,
        metadata: pd.DataFrame,
        dimension: int = 0,
        party_column: str = 'party',
        speaker_column: Optional[str] = None,
        text_column: Optional[str] = None,
        save_path: Optional[str] = None,
        n_std: float = 3.0,
        title: Optional[str] = None,
        show_legend: bool = True,
        show_individual_speeches: bool = False,
        bins: int = 50
    ) -> go.Figure:
        """
        Create a 1D distribution plot of a single LDA dimension.
        
        Shows parties as area histograms/distributions above the axis,
        speakers as aggregated points (square style), and individual speeches as points.
        
        Args:
            projections: LDA projections of shape [N, k] or [N, L, k]
            metadata: DataFrame with metadata (party, speaker, etc.)
            dimension: Which LDA dimension to visualize (0-indexed, default: 0 for 1st dimension)
            party_column: Column name for party labels
            speaker_column: Optional column name for speaker labels
            text_column: Optional column name for text/speech content (for hover data)
            save_path: Optional path to save the plot (without extension)
            n_std: Number of standard deviations for normalization
            title: Optional plot title
            show_legend: Whether to show legend
            show_individual_speeches: Whether to show individual speech points
            bins: Number of bins for histogram
        
        Returns:
            plotly Figure object
        """
        if projections.ndim == 3:
            # Use the last layer if multi-layer
            projections = projections[:, -1, :]
        
        if dimension >= projections.shape[1]:
            raise ValueError(f"Dimension {dimension} not available. Projections have {projections.shape[1]} dimensions.")
        
        # Extract single dimension
        X_1d = projections[:, dimension]
        
        # Normalize coordinates
        X_mean = X_1d.mean()
        X_std = X_1d.std()
        X_plot = (X_1d - X_mean) / (n_std * X_std + 1e-8)
        
        # Auto-detect text column if not provided
        if text_column is None:
            for col in ['sentence', 'text', 'speech', 'content']:
                if col in metadata.columns:
                    text_column = col
                    break
        
        # Create figure
        fig = go.Figure()
        
        # Get unique parties
        parties = metadata[party_column].unique()
        
        # Calculate overall range for consistent axis
        max_abs_val = abs(X_plot).max()
        max_range = max_abs_val * 1.1
        x_range = [-max_range, max_range]
        
        # Calculate histogram bins
        hist_bins = np.linspace(x_range[0], x_range[1], bins + 1)
        bin_centers = (hist_bins[:-1] + hist_bins[1:]) / 2
        bin_width = hist_bins[1] - hist_bins[0]
        
        # First pass: calculate total speeches and party counts for area scaling
        total_speeches = len(X_plot)
        party_counts = {}
        party_hists = {}
        
        for party in parties:
            party_mask = metadata[party_column] == party
            party_points = X_plot[party_mask]
            party_counts[party] = len(party_points)
            if len(party_points) > 0:
                hist, _ = np.histogram(party_points, bins=hist_bins)
                party_hists[party] = hist
        
        # Scale histograms so area is proportional to number of speeches
        # Since sum(hist) = count, if we scale all histograms by the same factor,
        # the area ratio will be preserved: area_A / area_B = count_A / count_B
        # Find the maximum height across all parties for normalization
        max_height = 0
        for party in parties:
            if party in party_hists:
                max_height = max(max_height, party_hists[party].max())
        
        # Process each party
        for party in parties:
            party_mask = metadata[party_column] == party
            party_points = X_plot[party_mask]
            
            if len(party_points) == 0:
                continue
            
            # Get party color
            party_color = self.party_colors.get(party, None)
            if party_color is None:
                colors = list(px.colors.qualitative.Set3)
                party_color = colors[hash(party) % len(colors)]
            
            # Convert hex color to rgba for transparency
            if party_color.startswith('#'):
                rgb = tuple(int(party_color[i:i+2], 16) for i in (1, 3, 5))
                rgba_area = f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, 0.3)"
                rgba_points = f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, 0.6)"
            else:
                rgba_area = f"rgba(128, 128, 128, 0.3)"
                rgba_points = f"rgba(128, 128, 128, 0.6)"
            
            # Get histogram
            hist = party_hists[party]
            
            # Scale all histograms by the same factor for display
            # Since sum(hist) = count, area = sum(hist_normalized) * bin_width = sum(hist * scale) * bin_width = count * scale * bin_width
            # So area_A / area_B = (count_A * scale) / (count_B * scale) = count_A / count_B ✓
            if max_height > 0:
                # Scale to reasonable display height (0.3 max height)
                # All parties scaled by same factor preserves area proportionality
                hist_normalized = hist / max_height * 0.3
            else:
                hist_normalized = np.zeros_like(hist)
            
            # Add area plot (filled area above axis)
            # Only plot where values are > 0.005 to avoid flat zero areas at edges
            if hist_normalized.max() > 0:  # Only add if there's data
                threshold = 0.001
                # Find indices where values are above threshold
                mask = hist_normalized > threshold
                
                if mask.any():
                    # Get the indices where we have data above threshold
                    indices = np.where(mask)[0]
                    
                    # Extend by one bin on each side to ensure smooth connection to zero
                    if len(indices) > 0:
                        start_idx = max(0, indices[0] - 1)
                        end_idx = min(len(hist_normalized), indices[-1] + 2)
                        
                        # Extract the relevant portion
                        x_plot = bin_centers[start_idx:end_idx]
                        y_plot = hist_normalized[start_idx:end_idx]
                        
                        # Ensure we start and end at zero for clean fill
                        if start_idx > 0:
                            x_plot = np.concatenate([[bin_centers[start_idx]], x_plot])
                            y_plot = np.concatenate([[0], y_plot])
                        if end_idx < len(hist_normalized):
                            x_plot = np.concatenate([x_plot, [bin_centers[end_idx - 1]]])
                            y_plot = np.concatenate([y_plot, [0]])
                    else:
                        x_plot = bin_centers
                        y_plot = hist_normalized
                else:
                    # If nothing above threshold, don't plot
                    x_plot = []
                    y_plot = []
                
                if len(x_plot) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=x_plot,
                            y=y_plot,
                            mode='lines',
                            name=party,
                            fill='tozeroy',
                            fillcolor=rgba_area,
                            line=dict(color=party_color, width=2),
                            legendgroup=party,
                            showlegend=show_legend,
                            hoverinfo='skip'
                        )
                    )
            
            # Add individual speech points (if requested)
            if show_individual_speeches:
                party_metadata_subset = metadata[party_mask]
                hover_texts = []
                
                for idx in range(len(party_points)):
                    hover_parts = [f"<b>Party:</b> {party}"]
                    
                    if speaker_column and speaker_column in party_metadata_subset.columns:
                        speaker_name = party_metadata_subset.iloc[idx][speaker_column]
                        if pd.notna(speaker_name):
                            hover_parts.append(f"<b>Speaker:</b> {speaker_name}")
                    
                    if text_column and text_column in party_metadata_subset.columns:
                        text_content = party_metadata_subset.iloc[idx][text_column]
                        if pd.notna(text_content):
                            text_preview = str(text_content)[:100]
                            if len(str(text_content)) > 100:
                                text_preview += "..."
                            hover_parts.append(f"<b>Speech:</b> {text_preview}")
                    
                    hover_texts.append("<br>".join(hover_parts))
                
                fig.add_trace(
                    go.Scatter(
                        x=party_points,
                        y=[-0.04] * len(party_points),  # Position below axis
                        mode='markers',
                        name=f"{party} (speeches)",
                        marker=dict(
                            size=4,
                            color=party_color if not party_color.startswith('rgba') else rgba_points,
                            opacity=0.4,
                            line=dict(width=0.5, color='white')
                        ),
                        legendgroup=party,
                        showlegend=False,
                        hovertemplate='%{hovertext}<extra></extra>',
                        hovertext=hover_texts
                    )
                )
            
            # Add speaker centroids (aggregated, square style)
            if speaker_column and speaker_column in metadata.columns:
                party_metadata = metadata[party_mask]
                for speaker in party_metadata[speaker_column].unique():
                    speaker_mask = (metadata[party_column] == party) & \
                                  (metadata[speaker_column] == speaker)
                    speaker_points = X_plot[speaker_mask]
                    speaker_count = np.sum(speaker_mask)
                    
                    if speaker_count == 0:
                        continue
                    
                    # Calculate speaker centroid
                    speaker_center = np.mean(speaker_points)
                    
                    # Calculate speaker size with square root scaling (same as 2D)
                    base_size = 0
                    size_scaling = np.sqrt(speaker_count) * 4
                    speaker_size = base_size + size_scaling
                    max_speaker_size = 150
                    speaker_size = min(speaker_size, max_speaker_size)
                    
                    # Prepare hover text
                    hover_parts = [f"<b>Party:</b> {party}"]
                    if speaker_column:
                        hover_parts.append(f"<b>Speaker:</b> {speaker}")
                    hover_text = "<br>".join(hover_parts)
                    
                    # Add speaker centroid as square (positioned below axis)
                    fig.add_trace(
                        go.Scatter(
                            x=[speaker_center],
                            y=[-0.02],  # Slightly below axis
                            mode='markers',
                            name=f"{speaker} ({party})",
                            marker=dict(
                                size=speaker_size,
                                color=party_color,
                                opacity=0.5,
                                symbol='square',
                                line=dict(width=1, color='white')
                            ),
                            legendgroup=party,
                            showlegend=False,
                            hovertemplate='%{hovertext}<extra></extra>',
                            hovertext=[hover_text]
                        )
                    )
        
        # Update layout (matching 2D style)
        fig.update_xaxes(
            showticklabels=False,
            showgrid=True,
            gridwidth=1,
            gridcolor='LightGray',
            range=x_range,
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor='LightGray'
        )
        fig.update_yaxes(
            showticklabels=False,
            showgrid=True,
            gridwidth=1,
            gridcolor='LightGray',
            range=[-0.05, 0.4],  # Room for area plots and markers below axis
            zeroline=False
        )
        
        fig.update_layout(
            width=800,
            height=400,
            margin=dict(l=60, r=20, t=40, b=40),
            font_family="Libertinus Sans",
            plot_bgcolor='white',
            title=title,
            showlegend=show_legend,
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.15,
                xanchor="left",
                x=0
            ) if show_legend else None
        )
        
        # Save if path provided
        if save_path:
            fig.write_html(f"{save_path}.html")
            try:
                fig.write_image(f"{save_path}.png", scale=4, width=800, height=400)
            except Exception as e:
                print(f"Warning: Could not save PNG image: {e}")
        
        return fig
    
    def plot_layer_comparison(
        self,
        projections_by_layer: Dict[int, np.ndarray],
        metadata: pd.DataFrame,
        party_column: str = 'party',
        selected_layers: Optional[List[int]] = None,
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Create a comparison plot showing LDA projections across multiple layers.
        
        Args:
            projections_by_layer: Dictionary mapping layer indices to projections
            metadata: DataFrame with metadata
            party_column: Column name for party labels
            selected_layers: Optional list of layers to plot (if None, plots all)
            save_path: Optional path to save the plot
        
        Returns:
            plotly Figure with subplots
        """
        if selected_layers is None:
            selected_layers = sorted(projections_by_layer.keys())
        
        from plotly.subplots import make_subplots
        
        n_layers = len(selected_layers)
        cols = min(3, n_layers)
        rows = (n_layers + cols - 1) // cols
        
        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=[f'Layer {layer}' for layer in selected_layers],
            horizontal_spacing=0.1,
            vertical_spacing=0.15
        )
        
        for idx, layer in enumerate(selected_layers):
            row = idx // cols + 1
            col = idx % cols + 1
            
            projections = projections_by_layer[layer]
            plot_data = self._prepare_plot_data(
                projections, metadata, party_column, None
            )
            
            for party in plot_data['Party'].unique():
                party_data = plot_data[plot_data['Party'] == party]
                color = self.party_colors.get(party, None)
                
                fig.add_trace(
                    go.Scatter(
                        x=party_data['LDA1'],
                        y=party_data['LDA2'],
                        mode='markers',
                        name=party if idx == 0 else '',
                        marker=dict(
                            size=party_data['size'],
                            color=color,
                            opacity=0.6
                        ),
                        showlegend=(idx == 0)
                    ),
                    row=row,
                    col=col
                )
        
        fig.update_layout(
            height=300 * rows,
            title_text="LDA Projections Across Layers",
            showlegend=True
        )
        
        if save_path:
            fig.write_html(f"{save_path}.html")
        
        return fig

