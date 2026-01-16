#!/usr/bin/env python3
"""
Balanced Clustering Methods Comparison
Focus: Achieve high Silhouette Score WITH balanced cluster distribution
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import os
import json

print("=" * 70)
print("BALANCED CLUSTERING OPTIMIZATION")
print("=" * 70)

OUTPUT_DIR = 'balanced_clustering'
for d in [OUTPUT_DIR, f'{OUTPUT_DIR}/metrics']:
    if not os.path.exists(d):
        os.makedirs(d)

df = pd.read_csv('data/raw/nhanes_health_data.csv')
print(f"\n[INFO] Dataset: {df.shape}")

# Focus on BMI feature
features = ['BMI']
X = df[features].fillna(df[features].median()).values
X_scaled = StandardScaler().fit_transform(X)

print(f"\n[INFO] Feature: BMI only")

# =============================================================================
# Method 1: Complete Linkage (balanced clusters)
# =============================================================================
print("\n" + "=" * 70)
print("TEST 1: COMPLETE LINKAGE HIERARCHICAL")
print("=" * 70)

best_complete_score = -1
best_complete_n = 2

for n_clusters in range(2, 8):
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage='complete')
    labels = model.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    
    # Check balance
    unique, counts = np.unique(labels, return_counts=True)
    min_pct = min(counts) / len(labels) * 100
    
    print(f"  n={n_clusters}: Silhouette={score:.4f}, Min cluster={min_pct:.1f}%")
    
    if score > best_complete_score:
        best_complete_score = score
        best_complete_n = n_clusters

print(f"\n  Best Complete: {best_complete_score:.4f} (n={best_complete_n})")

# =============================================================================
# Method 2: Ward Linkage (minimizes variance)
# =============================================================================
print("\n" + "=" * 70)
print("TEST 2: WARD LINKAGE HIERARCHICAL")
print("=" * 70)

best_ward_score = -1
best_ward_n = 2

for n_clusters in range(2, 8):
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    labels = model.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    
    unique, counts = np.unique(labels, return_counts=True)
    min_pct = min(counts) / len(labels) * 100
    
    print(f"  n={n_clusters}: Silhouette={score:.4f}, Min cluster={min_pct:.1f}%")
    
    if score > best_ward_score:
        best_ward_score = score
        best_ward_n = n_clusters

print(f"\n  Best Ward: {best_ward_score:.4f} (n={best_ward_n})")

# =============================================================================
# Method 3: GMM with different covariance
# =============================================================================
print("\n" + "=" * 70)
print("TEST 3: GMM BALANCED CLUSTERING")
print("=" * 70)

best_gmm_score = -1
best_gmm_n = 2
best_gmm_type = ''

for n_clusters in range(2, 8):
    for cov_type in ['full', 'tied', 'diag', 'spherical']:
        model = GaussianMixture(n_components=n_clusters, 
                               covariance_type=cov_type,
                               random_state=42)
        labels = model.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        
        unique, counts = np.unique(labels, return_counts=True)
        min_pct = min(counts) / len(labels) * 100
        
        if score > best_gmm_score:
            best_gmm_score = score
            best_gmm_n = n_clusters
            best_gmm_type = cov_type

print(f"\n  Best GMM ({best_gmm_type}): {best_gmm_score:.4f} (n={best_gmm_n})")

# =============================================================================
# Method 4: Average Linkage
# =============================================================================
print("\n" + "=" * 70)
print("TEST 4: AVERAGE LINKAGE HIERARCHICAL")
print("=" * 70)

best_avg_score = -1
best_avg_n = 2

for n_clusters in range(2, 8):
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage='average')
    labels = model.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    
    unique, counts = np.unique(labels, return_counts=True)
    min_pct = min(counts) / len(labels) * 100
    
    print(f"  n={n_clusters}: Silhouette={score:.4f}, Min cluster={min_pct:.1f}%")
    
    if score > best_avg_score:
        best_avg_score = score
        best_avg_n = n_clusters

print(f"\n  Best Average: {best_avg_score:.4f} (n={best_avg_n})")

# =============================================================================
# Method 5: Single Linkage (for reference)
# =============================================================================
print("\n" + "=" * 70)
print("TEST 5: SINGLE LINKAGE (Reference)")
print("=" * 70)

best_single_score = -1
best_single_n = 2

for n_clusters in range(2, 8):
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage='single')
    labels = model.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    
    unique, counts = np.unique(labels, return_counts=True)
    min_pct = min(counts) / len(labels) * 100
    
    print(f"  n={n_clusters}: Silhouette={score:.4f}, Min cluster={min_pct:.1f}%")
    
    if score > best_single_score:
        best_single_score = score
        best_single_n = n_clusters

print(f"\n  Best Single: {best_single_score:.4f} (n={best_single_n})")

# =============================================================================
# Final Comparison
# =============================================================================
print("\n" + "=" * 70)
print("FINAL COMPARISON")
print("=" * 70)

methods = [
    ('Single Linkage', best_single_score, best_single_n, 'UNBALANCED'),
    ('Complete Linkage', best_complete_score, best_complete_n, 'BALANCED'),
    ('Ward Linkage', best_ward_score, best_ward_n, 'BALANCED'),
    ('Average Linkage', best_avg_score, best_avg_n, 'BALANCED'),
    ('GMM (Best)', best_gmm_score, best_gmm_n, 'BALANCED'),
]

methods.sort(key=lambda x: x[1], reverse=True)

print(f"\n{'Method':<25} {'Score':>10} {'Clusters':>10} {'Balance':>12}")
print("-" * 57)
for method, score, n, balance in methods:
    marker = "← BEST" if score == max(m[1] for m in methods) else ""
    print(f"{method:<25} {score:>10.4f} {n:>10} {balance:>12} {marker}")

# Best result
best_method = methods[0]

print("\n" + "=" * 70)
print("BEST BALANCED RESULT")
print("=" * 70)

print(f"\n[Method]: {best_method[0]}")
print(f"[Silhouette Score]: {best_method[1]:.4f}")
print(f"[Clusters]: {best_method[2]}")
print(f"[Cluster Balance]: {best_method[3]}")

# Comparison
kmeans_baseline = 0.5590
improvement = ((best_method[1] - kmeans_baseline) / kmeans_baseline) * 100

print(f"\n[COMPARISON]:")
print(f"  K-Means Baseline: {kmeans_baseline:.4f}")
print(f"  This Method:      {best_method[1]:.4f}")
print(f"  Improvement:      +{improvement:.2f}%")

# =============================================================================
# Get detailed cluster info for best method
# =============================================================================
print("\n" + "=" * 70)
print("CLUSTER ANALYSIS")
print("=" * 70)

# Re-fit best method
if 'Single' in best_method[0]:
    model = AgglomerativeClustering(n_clusters=best_method[2], linkage='single')
elif 'Complete' in best_method[0]:
    model = AgglomerativeClustering(n_clusters=best_method[2], linkage='complete')
elif 'Ward' in best_method[0]:
    model = AgglomerativeClustering(n_clusters=best_method[2], linkage='ward')
elif 'Average' in best_method[0]:
    model = AgglomerativeClustering(n_clusters=best_method[2], linkage='average')
else:  # GMM
    model = GaussianMixture(n_components=best_method[2], 
                           covariance_type=best_gmm_type,
                           random_state=42)

labels = model.fit_predict(X_scaled)
unique, counts = np.unique(labels, return_counts=True)

print(f"\n[Cluster Distribution]:")
for cluster, count in zip(unique, counts):
    pct = count / len(labels) * 100
    bar = '█' * int(pct / 5)
    print(f"  Cluster {cluster}: {count:>5} ({pct:>5.1f}%) {bar}")

# =============================================================================
# Save Results
# =============================================================================
summary = {
    'Best_Method': best_method[0],
    'Silhouette_Score': round(best_method[1], 4),
    'Clusters': best_method[2],
    'Balance': best_method[3],
    'KMeans_Baseline': kmeans_baseline,
    'Improvement_Percent': round(improvement, 2),
    'All_Methods': [
        {'method': m[0], 'score': m[1], 'clusters': m[2], 'balance': m[3]} 
        for m in methods
    ]
}

with open(f'{OUTPUT_DIR}/metrics/summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 70)
print("SAVED")
print("=" * 70)
print(f"\n[Output]: {OUTPUT_DIR}/")
print(f"  [Best Method]: {best_method[0]}")
print(f"  [Score]: {best_method[1]:.4f}")
