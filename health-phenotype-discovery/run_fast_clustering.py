#!/usr/bin/env python3
"""
Fast clustering optimization to achieve target Silhouette Score
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import os
import json

print("=" * 70)
print("FAST CLUSTERING OPTIMIZATION")
print("=" * 70)

# Configuration
OUTPUT_DIR = 'output_v6'
for d in [OUTPUT_DIR, f'{OUTPUT_DIR}/metrics', f'{OUTPUT_DIR}/predictions']:
    if not os.path.exists(d):
        os.makedirs(d)

# Load and prepare data
print("\n[INFO] Loading data...")
df = pd.read_csv('data/raw/nhanes_health_data.csv')
print(f"  Dataset: {df.shape}")

# Use only key clinical features
features = ['Age', 'Systolic_BP', 'Diastolic_BP', 'BMI', 'Waist_Circumference',
            'Blood_Glucose', 'HbA1c', 'HDL_Cholesterol', 'Total_Cholesterol']

print(f"\n[INFO] Using {len(features)} features")
X = df[features].fillna(df[features].median()).values
X_scaled = StandardScaler().fit_transform(X)

# PCA
pca = PCA(n_components=0.95, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print(f"  PCA: {X_pca.shape[1]} components")

X_final = StandardScaler().fit_transform(X_pca)

# Test different cluster counts with K-Means
print("\n[INFO] Testing cluster configurations...")

best_score = -1
best_n_clusters = 2
results = []

for n_clusters in range(2, 10):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_final)
    
    score = silhouette_score(X_final, labels)
    results.append((n_clusters, score))
    
    if score > best_score:
        best_score = score
        best_n_clusters = n_clusters

print(f"\n[INFO] Best K-Means Result:")
print(f"  Clusters: {best_n_clusters}")
print(f"  Silhouette: {best_score:.4f}")

# Final clustering
print("\n" + "=" * 70)
print("FINAL RESULTS")
print("=" * 70)

kmeans = KMeans(n_clusters=best_n_clusters, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_final)

final_silhouette = silhouette_score(X_final, cluster_labels)

print(f"\n[CLUSTERING RESULTS]:")
print(f"  Method: K-Means")
print(f"  Clusters: {best_n_clusters}")
print(f"  Silhouette Score: {final_silhouette:.4f}")
print(f"  Target: 0.87")
print(f"  Status: {'[✓] ACHIEVED' if final_silhouette >= 0.87 else '[✗] NOT ACHIEVED'}")

# Save results
df_clustered = df.copy()
df_clustered['Cluster'] = cluster_labels
df_clustered.to_csv(f'{OUTPUT_DIR}/predictions/clustered_data.csv', index=False)

summary = {
    'Method': 'K-Means',
    'Clusters': best_n_clusters,
    'Silhouette_Score': f"{final_silhouette:.4f}",
    'Target_Achieved': final_silhouette >= 0.87
}

with open(f'{OUTPUT_DIR}/metrics/summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

pd.DataFrame(results, columns=['Clusters', 'Silhouette']).to_csv(f'{OUTPUT_DIR}/metrics/all_results.csv', index=False)

print(f"\n[INFO] Results saved to {OUTPUT_DIR}/")
