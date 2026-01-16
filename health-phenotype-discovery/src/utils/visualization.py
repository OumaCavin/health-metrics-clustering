"""Utility module for health phenotype discovery."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def set_plotting_style():
    """Set consistent plotting style."""
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")


def plot_distribution(df: pd.DataFrame, column: str, save_path: str = None):
    """
    Plot distribution of a column.
    
    Args:
        df: Input DataFrame
        column: Column to plot
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    if df[column].dtype in ['object', 'category']:
        df[column].value_counts().plot(kind='bar')
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Count')
        plt.xticks(rotation=45)
    else:
        df[column].hist(bins=30, edgecolor='black')
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.close()


def plot_correlation_matrix(df: pd.DataFrame, save_path: str = None):
    """
    Plot correlation matrix heatmap.
    
    Args:
        df: Input DataFrame
        save_path: Path to save the plot
    """
    plt.figure(figsize=(14, 12))
    
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.close()


def plot_cluster_results(df: pd.DataFrame, labels: np.ndarray, features: list, save_path: str = None):
    """
    Plot clustering results.
    
    Args:
        df: Input DataFrame
        labels: Cluster labels
        features: Features to use for plotting
        save_path: Path to save the plot
    """
    df = df.copy()
    df['Cluster'] = labels
    
    plt.figure(figsize=(12, 8))
    
    scatter = plt.scatter(df[features[0]], df[features[1]], c=df['Cluster'], cmap='viridis')
    plt.colorbar(scatter, label='Cluster')
    plt.xlabel(features[0])
    plt.ylabel(features[1])
    plt.title('Clustering Results')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.close()


def generate_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate summary statistics for the dataset.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with summary statistics
    """
    summary = df.describe(include='all').T
    summary['missing_values'] = df.isnull().sum()
    summary['missing_percentage'] = (df.isnull().sum() / len(df) * 100).round(2)
    
    return summary
