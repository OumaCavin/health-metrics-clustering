#!/usr/bin/env python3
"""
DBSCAN clustering with aggressive optimization to achieve target Silhouette Score
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import os
import json

print("=" * 70)
print("AGGRESSIVE DBSCAN OPTIMIZATION")
print("=" * 70)

# Configuration
PROJECT_ROOT = os.path.abspath('.')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output_v4')

PHASE_DIRS = {
    'data': os.path.join(DATA_DIR, 'raw'),
    'processed': os.path.join(DATA_DIR, 'processed'),
}

OUTPUT_SUBDIRS = {
    'metrics': os.path.join(OUTPUT_DIR, 'metrics'),
    'predictions': os.path.join(OUTPUT_DIR, 'predictions'),
}

for dir_path in [OUTPUT_DIR, *OUTPUT_SUBDIRS.values()]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# Load data
print("\n[INFO] Loading data...")
data_path = os.path.join(PHASE_DIRS['data'], 'nhanes_health_data.csv')
df = pd.read_csv(data_path)
print(f"  Dataset: {df.shape}")

# Preprocessing - Focus on numeric features only
print("\n[INFO] Preprocessing...")
df_numeric = df.select_dtypes(include=[np.number]).copy()
print(f"  Numeric features: {df_numeric.shape[1]}")

# Handle missing values
for col in df_numeric.columns:
    if df_numeric[col].isnull().sum() > 0:
        df_numeric[col].fillna(df_numeric[col].median(), inplace=True)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_numeric)
print(f"  Scaled features: {X_scaled.shape[1]}")

# PCA for dimensionality reduction
pca = PCA(n_components=0.90, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print(f"  PCA components: {X_pca.shape[1]} ({pca.explained_variance_ratio_.sum():.1%} variance)")

# Final scaling
final_scaler = StandardScaler()
X_final = final_scaler.fit_transform(X_pca)

# Find optimal epsilon using k-distance
print("\n[INFO] Finding optimal epsilon...")
k = 5
neighbors = NearestNeighbors(n_neighbors=k)
neighbors_fit = neighbors.fit(X_final)
distances, indices = neighbors_fit.kneighbors(X_final)
k_distances = np.sort(distances[:, k-1])

# Use percentile-based epsilon selection
eps_percentile = 70  # Try different percentiles
optimal_eps = np.percentile(k_distances, eps_percentile)
print(f"  Epsilon (p{eps_percentile}): {optimal_eps:.4f}")

# =============================================================================
# AGGRESSIVE HYPERPARAMETER SEARCH
# =============================================================================
print("\n" + "=" * 70)
print("HYPERPARAMETER OPTIMIZATION")
print("=" * 70)

TARGET_SILHOUETTE = 0.87

# Use a wide range of eps values around the k-distance estimate
eps_values = np.linspace(optimal_eps * 0.3, optimal_eps * 1.5, 50)
min_samples_values = [3, 5, 7, 10, 15]

results = []
print(f"\n[INFO] Testing {len(eps_values) * len(min_samples_values)} configurations...")

for eps in eps_values:
    for min_samples in min_samples_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X_final)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        # Only consider configurations with meaningful clustering
        if n_clusters >= 2 and n_clusters <= 8:
            mask = labels != -1
            clustered_ratio = mask.sum() / len(labels)
            
            if clustered_ratio >= 0.1:  # At least 10% clustered
                X_valid = X_final[mask]
                labels_valid = labels[mask]
                
                try:
                    silhouette = silhouette_score(X_valid, labels_valid)
                    results.append({
                        'eps': eps,
                        'min_samples': min_samples,
                        'n_clusters': n_clusters,
                        'n_noise': n_noise,
                        'silhouette': silhouette,
                        'clustered_ratio': clustered_ratio
                    })
                except:
                    pass

print(f"  Valid configurations: {len(results)}")

if len(results) > 0:
    results_df = pd.DataFrame(results)
    
    # Find best configuration
    best_config = results_df.loc[results_df['silhouette'].idxmax()]
    
    print(f"\n[INFO] Best Configuration:")
    print(f"  Epsilon: {best_config['eps']:.4f}")
    print(f"  Min Samples: {best_config['min_samples']}")
    print(f"  Clusters: {int(best_config['n_clusters'])}")
    print(f"  Silhouette: {best_config['silhouette']:.4f}")
    print(f"  Clustered: {best_config['clustered_ratio']:.1%}")
    
    OPTIMAL_EPS = best_config['eps']
    OPTIMAL_MIN_SAMPLES = int(best_config['min_samples'])
    
    # =============================================================================
    # FINAL CLUSTERING
    # =============================================================================
    print("\n" + "=" * 70)
    print("FINAL DBSCAN CLUSTERING")
    print("=" * 70)
    
    final_dbscan = DBSCAN(eps=OPTIMAL_EPS, min_samples=OPTIMAL_MIN_SAMPLES)
    cluster_labels = final_dbscan.fit_predict(X_final)
    
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)
    mask = cluster_labels != -1
    X_clustered = X_final[mask]
    labels_clustered = cluster_labels[mask]
    
    final_silhouette = silhouette_score(X_clustered, labels_clustered)
    
    print(f"\n[INFO] Results:")
    print(f"  Clusters: {n_clusters}")
    print(f"  Noise: {n_noise} ({n_noise/len(cluster_labels):.1%})")
    print(f"  Silhouette Score: {final_silhouette:.4f}")
    
    # Save results
    df_clustered = df.copy()
    df_clustered['Cluster'] = cluster_labels
    df_clustered.to_csv(os.path.join(OUTPUT_SUBDIRS['predictions'], 'clustered_data.csv'), index=False)
    
    # Save summary
    summary = {
        'Silhouette_Score': f"{final_silhouette:.4f}",
        'Target_Achieved': final_silhouette >= TARGET_SILHOUETTE,
        'Epsilon': OPTIMAL_EPS,
        'Min_Samples': OPTIMAL_MIN_SAMPLES,
        'Clusters': n_clusters,
        'Noise_Ratio': f"{n_noise/len(cluster_labels):.2%}"
    }
    
    with open(os.path.join(OUTPUT_SUBDIRS['metrics'], 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n[TARGET STATUS]: {'[✓] ACHIEVED' if final_silhouette >= TARGET_SILHOUETTE else '[✗] NOT ACHIEVED'}")
else:
    print("\n[ERROR] No valid configurations found!")
