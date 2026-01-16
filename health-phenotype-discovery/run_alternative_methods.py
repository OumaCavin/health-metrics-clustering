#!/usr/bin/env python3
"""
Alternative Clustering Methods Comparison
Testing multiple algorithms to achieve better Silhouette Scores
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import (
    AgglomerativeClustering, 
    SpectralClustering,
    MeanShift,
    Birch,
    AffinityPropagation
)
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
import os
import json

print("=" * 70)
print("ALTERNATIVE CLUSTERING METHODS COMPARISON")
print("=" * 70)

# Configuration
OUTPUT_DIR = 'alternative_methods'
for d in [OUTPUT_DIR, f'{OUTPUT_DIR}/metrics']:
    if not os.path.exists(d):
        os.makedirs(d)

# Load data
print("\n[INFO] Loading data...")
df = pd.read_csv('data/raw/nhanes_health_data.csv')
print(f"  Dataset: {df.shape}")

# Use BMI feature (best performing from previous analysis)
features = ['BMI']
X = df[features].fillna(df[features].median()).values

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"\n[INFO] Using {features} feature only")
print(f"  Shape: {X_scaled.shape}")

# Define cluster range
cluster_range = range(2, 8)

results = {}

# =============================================================================
# Method 1: Agglomerative Hierarchical Clustering
# =============================================================================
print("\n" + "=" * 70)
print("TESTING: Agglomerative Hierarchical Clustering")
print("=" * 70)

linkage_methods = ['ward', 'complete', 'average', 'single']
agg_results = {}

for linkage in linkage_methods:
    print(f"\n  Testing linkage: {linkage}...")
    best_score = -1
    best_n = 2
    
    for n_clusters in cluster_range:
        try:
            model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
            labels = model.fit_predict(X_scaled)
            score = silhouette_score(X_scaled, labels)
            
            if score > best_score:
                best_score = score
                best_n = n_clusters
        except:
            continue
    
    agg_results[linkage] = {'score': best_score, 'clusters': best_n}
    print(f"    Best: {best_score:.4f} (n={best_n})")

results['Agglomerative_Hierarchical'] = agg_results

# =============================================================================
# Method 2: Spectral Clustering
# =============================================================================
print("\n" + "=" * 70)
print("TESTING: Spectral Clustering")
print("=" * 70)

spectral_results = {}
best_spectral_score = -1
best_spectral_n = 2

print("\n  Testing different cluster counts...")
for n_clusters in cluster_range:
    try:
        model = SpectralClustering(n_clusters=n_clusters, 
                                   affinity='rbf',
                                   random_state=42,
                                   assign_labels='kmeans')
        labels = model.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        
        if score > best_spectral_score:
            best_spectral_score = score
            best_spectral_n = n_clusters
            
    except Exception as e:
        continue

spectral_results['rbf_kmeans'] = {'score': best_spectral_score, 'clusters': best_spectral_n}
print(f"  Best (RBF + K-Means): {best_spectral_score:.4f} (n={best_spectral_n})")

# Try different affinity methods
for affinity in ['nearest_neighbors', 'rbf']:
    for assign in ['discretize', 'kmeans']:
        try:
            model = SpectralClustering(n_clusters=best_spectral_n, 
                                       affinity=affinity,
                                       assign_labels=assign,
                                       random_state=42)
            labels = model.fit_predict(X_scaled)
            score = silhouette_score(X_scaled, labels)
            
            key = f"{affinity}_{assign}"
            if score > spectral_results.get(key, {}).get('score', 0):
                spectral_results[key] = {'score': score, 'clusters': best_spectral_n}
                
        except:
            continue

results['Spectral_Clustering'] = spectral_results

# =============================================================================
# Method 3: Gaussian Mixture Models
# =============================================================================
print("\n" + "=" * 70)
print("TESTING: Gaussian Mixture Models (GMM)")
print("=" * 70)

covariance_types = ['full', 'tied', 'diag', 'spherical']
gmm_results = {}

for cov_type in covariance_types:
    print(f"\n  Testing covariance_type: {cov_type}...")
    best_score = -1
    best_n = 2
    
    for n_clusters in cluster_range:
        try:
            model = GaussianMixture(n_components=n_clusters, 
                                   covariance_type=cov_type,
                                   random_state=42)
            labels = model.fit_predict(X_scaled)
            score = silhouette_score(X_scaled, labels)
            
            if score > best_score:
                best_score = score
                best_n = n_clusters
        except:
            continue
    
    gmm_results[cov_type] = {'score': best_score, 'clusters': best_n}
    print(f"    Best: {best_score:.4f} (n={best_n})")

results['GMM'] = gmm_results

# =============================================================================
# Method 4: Birch Clustering
# =============================================================================
print("\n" + "=" * 70)
print("TESTING: Birch Clustering")
print("=" * 70)

birch_results = {}
best_birch_score = -1
best_birch_n = 2

for n_clusters in cluster_range:
    try:
        model = Birch(n_clusters=n_clusters)
        labels = model.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        
        if score > best_birch_score:
            best_birch_score = score
            best_birch_n = n_clusters
            
    except:
        continue

birch_results['default'] = {'score': best_birch_score, 'clusters': best_birch_n}
print(f"\n  Best Birch: {best_birch_score:.4f} (n={best_birch_n})")

results['Birch'] = birch_results

# =============================================================================
# Method 5: Mean Shift (automatic cluster detection)
# =============================================================================
print("\n" + "=" * 70)
print("TESTING: Mean Shift (Automatic Cluster Detection)")
print("=" * 70)

try:
    # Try different bandwidths
    from sklearn.cluster import estimate_bandwidth
    
    bandwidths = [0.1, 0.2, 0.5, 1.0, 2.0]
    meanshift_results = {}
    best_meanshift_score = -1
    best_bandwidth = 0.5
    
    for bw in bandwidths:
        try:
            model = MeanShift(bandwidth=bw)
            labels = model.fit_predict(X_scaled)
            n_clusters = len(set(labels))
            
            if n_clusters >= 2:
                score = silhouette_score(X_scaled, labels)
                meanshift_results[f'bandwidth_{bw}'] = {'score': score, 'clusters': n_clusters}
                
                if score > best_meanshift_score:
                    best_meanshift_score = score
                    best_bandwidth = bw
                    
        except:
            continue
    
    results['MeanShift'] = meanshift_results
    print(f"\n  Best MeanShift: {best_meanshift_score:.4f} (bandwidth={best_bandwidth})")
    
except Exception as e:
    print(f"  MeanShift failed: {e}")
    results['MeanShift'] = {}

# =============================================================================
# Method 6: Affinity Propagation
# =============================================================================
print("\n" + "=" * 70)
print("TESTING: Affinity Propagation")
print("=" * 70)

try:
    # Try different damping values
    affinity_results = {}
    best_affinity_score = -1
    
    for damping in [0.5, 0.6, 0.7, 0.8, 0.9]:
        try:
            model = AffinityPropagation(damping=damping, random_state=42)
            labels = model.fit_predict(X_scaled)
            n_clusters = len(set(labels))
            
            if n_clusters >= 2 and n_clusters <= 10:
                score = silhouette_score(X_scaled, labels)
                affinity_results[f'damping_{damping}'] = {'score': score, 'clusters': n_clusters}
                
                if score > best_affinity_score:
                    best_affinity_score = score
                    
        except:
            continue
    
    results['Affinity_Propagation'] = affinity_results
    print(f"\n  Best Affinity: {best_affinity_score:.4f}")
    
except Exception as e:
    print(f"  Affinity Propagation failed: {e}")
    results['Affinity_Propagation'] = {}

# =============================================================================
# Find Best Overall Method
# =============================================================================
print("\n" + "=" * 70)
print("FINAL RESULTS COMPARISON")
print("=" * 70)

all_scores = []

# Collect all scores
for method_name, method_results in results.items():
    if isinstance(method_results, dict):
        for variant, data in method_results.items():
            if isinstance(data, dict) and 'score' in data:
                all_scores.append({
                    'method': method_name,
                    'variant': variant,
                    'score': data['score'],
                    'clusters': data.get('clusters', 'N/A')
                })

# Sort by score
all_scores.sort(key=lambda x: x['score'], reverse=True)

print(f"\n{'Method':<35} {'Variant':<20} {'Score':>10} {'Clusters':>10}")
print("-" * 75)
for item in all_scores[:15]:  # Top 15
    variant = item['variant'] if item['variant'] != 'default' else ''
    print(f"{item['method']:<35} {variant:<20} {item['score']:>10.4f} {item['clusters']:>10}")

# Best result
best_result = all_scores[0] if all_scores else None

print("\n" + "=" * 70)
print("BEST RESULT")
print("=" * 70)

if best_result:
    print(f"\n[Method]: {best_result['method']}")
    if best_result['variant'] and best_result['variant'] != 'default':
        print(f"[Variant]: {best_result['variant']}")
    print(f"[Silhouette Score]: {best_result['score']:.4f}")
    print(f"[Clusters]: {best_result['clusters']}")
    
    # Compare to K-Means baseline
    kmeans_baseline = 0.5590
    print(f"\n[Comparison to K-Means Baseline]:")
    print(f"  K-Means Score: {kmeans_baseline:.4f}")
    print(f"  This Method:   {best_result['score']:.4f}")
    
    if best_result['score'] > kmeans_baseline:
        improvement = ((best_result['score'] - kmeans_baseline) / kmeans_baseline) * 100
        print(f"  Improvement:   +{improvement:.2f}%")
        print(f"\n  [✓] NEW BEST ACHIEVED!")
    else:
        diff = kmeans_baseline - best_result['score']
        print(f"  Difference:    {diff:.4f}")
        print(f"\n  [≈] Similar to K-Means")
else:
    print("\n[ERROR] No valid results found!")

# =============================================================================
# Save Results
# =============================================================================
summary = {
    'Best_Method': best_result['method'] if best_result else None,
    'Best_Variant': best_result['variant'] if best_result else None,
    'Best_Score': float(best_result['score']) if best_result else None,
    'Best_Clusters': int(best_result['clusters']) if best_result else None,
    'KMeans_Baseline': 0.5590,
    'All_Results': all_scores[:20]
}

with open(f'{OUTPUT_DIR}/metrics/summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

# Save detailed results
pd.DataFrame(all_scores).to_csv(f'{OUTPUT_DIR}/metrics/detailed_results.csv', index=False)

print("\n" + "=" * 70)
print("RESULTS SAVED")
print("=" * 70)
print(f"\n[Output Directory]: {OUTPUT_DIR}/")
print(f"[Files]:")
print(f"  - metrics/summary.json")
print(f"  - metrics/detailed_results.csv")
