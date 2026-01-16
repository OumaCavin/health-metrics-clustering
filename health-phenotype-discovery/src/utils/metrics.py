"""Metrics module for health phenotype discovery."""

import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


def calculate_clustering_metrics(X: np.ndarray, labels: np.ndarray) -> dict:
    """
    Calculate clustering quality metrics.
    
    Args:
        X: Feature matrix
        labels: Cluster labels
        
    Returns:
        Dictionary with clustering metrics
    """
    metrics = {
        'silhouette_score': silhouette_score(X, labels),
        'calinski_harabasz_score': calinski_harabasz_score(X, labels),
        'davies_bouldin_score': davies_bouldin_score(X, labels)
    }
    
    return metrics
