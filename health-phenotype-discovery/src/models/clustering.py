"""Clustering module for health phenotype discovery."""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score


def find_optimal_clusters(X: np.ndarray, max_clusters: int = 10) -> dict:
    """
    Find optimal number of clusters using elbow method and silhouette score.
    
    Args:
        X: Feature matrix
        max_clusters: Maximum number of clusters to try
        
    Returns:
        Dictionary with optimal cluster metrics
    """
    inertias = []
    silhouette_scores = []
    calinski_scores = []
    
    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X, kmeans.labels_))
        calinski_scores.append(calinski_harabasz_score(X, kmeans.labels_))
    
    return {
        'k_range': list(range(2, max_clusters + 1)),
        'inertias': inertias,
        'silhouette_scores': silhouette_scores,
        'calinski_scores': calinski_scores
    }


def perform_kmeans_clustering(X: np.ndarray, n_clusters: int, random_state: int = 42) -> np.ndarray:
    """
    Perform K-means clustering.
    
    Args:
        X: Feature matrix
        n_clusters: Number of clusters
        random_state: Random state for reproducibility
        
    Returns:
        Cluster labels
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X)
    
    return labels


def perform_hierarchical_clustering(X: np.ndarray, n_clusters: int) -> np.ndarray:
    """
    Perform hierarchical clustering.
    
    Args:
        X: Feature matrix
        n_clusters: Number of clusters
        
    Returns:
        Cluster labels
    """
    hc = AgglomerativeClustering(n_clusters=n_clusters)
    labels = hc.fit_predict(X)
    
    return labels


def perform_dbscan_clustering(X: np.ndarray, eps: float = 0.5, min_samples: int = 5) -> np.ndarray:
    """
    Perform DBSCAN clustering.
    
    Args:
        X: Feature matrix
        eps: Maximum distance between samples to be in same neighborhood
        min_samples: Minimum number of samples in a neighborhood
        
    Returns:
        Cluster labels
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X)
    
    return labels


def get_cluster_statistics(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """
    Calculate statistics for each cluster.
    
    Args:
        df: Original DataFrame
        labels: Cluster labels
        
    Returns:
        DataFrame with cluster statistics
    """
    df = df.copy()
    df['Cluster'] = labels
    
    statistics = df.groupby('Cluster').agg(['mean', 'std', 'count'])
    
    return statistics
