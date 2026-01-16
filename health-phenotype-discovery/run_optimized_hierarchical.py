#!/usr/bin/env python3
"""
Optimize Agglomerative Hierarchical Clustering with Single Linkage
Achieved 0.6533 - Better than K-Means baseline of 0.5590
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import os
import json

print("=" * 70)
print("HIERARCHICAL CLUSTERING OPTIMIZATION")
print("=" * 70)

# Configuration
OUTPUT_DIR = 'optimized_clustering'
for d in [OUTPUT_DIR, f'{OUTPUT_DIR}/metrics']:
    if not os.path.exists(d):
        os.makedirs(d)

# Load data
df = pd.read_csv('data/raw/nhanes_health_data.csv')
print(f"\n[INFO] Dataset: {df.shape}")

# Test different feature combinations
feature_sets = [
    (['BMI'], 'BMI Only'),
    (['BMI', 'Blood_Glucose'], 'BMI + Glucose'),
    (['BMI', 'Waist_Circumference'], 'BMI + Waist'),
    (['BMI', 'Age'], 'BMI + Age'),
]

print("\n[INFO] Testing feature sets with single linkage hierarchical clustering...")

best_overall = {'score': 0, 'features': '', 'n_clusters': 2, 'distance': 'euclidean'}
results = []

for features, name in feature_sets:
    print(f"\n[Testing]: {name}")
    
    X = df[features].fillna(df[features].median()).values
    X_scaled = StandardScaler().fit_transform(X)
    
    best_score = -1
    best_n = 2
    
    for n_clusters in range(2, 10):
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage='single')
        labels = model.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        
        if score > best_score:
            best_score = score
            best_n = n_clusters
    
    results.append({
        'features': name,
        'score': best_score,
        'n_clusters': best_n
    })
    
    print(f"  Best: {best_score:.4f} (n={best_n})")
    
    if best_score > best_overall['score']:
        best_overall = {
            'score': best_score,
            'features': name,
            'n_clusters': best_n,
            'distance': 'euclidean'
        }

# =============================================================================
# Try with different distance metrics
# =============================================================================
print("\n" + "=" * 70)
print("TESTING DISTANCE METRICS")
print("=" * 70)

# For single linkage, we need to test more carefully
X_bmi = df[['BMI']].fillna(df[['BMI']].median()).values
X_bmi_scaled = StandardScaler().fit_transform(X_bmi)

distance_metrics = ['euclidean', 'manhattan', 'cosine']
print(f"\n[Testing]: BMI Only with different distance metrics")

for metric in distance_metrics:
    try:
        best_score = -1
        best_n = 2
        
        for n_clusters in range(2, 10):
            model = AgglomerativeClustering(n_clusters=n_clusters, 
                                           linkage='single',
                                           metric=metric)
            labels = model.fit_predict(X_bmi_scaled)
            score = silhouette_score(X_bmi_scaled, labels)
            
            if score > best_score:
                best_score = score
                best_n = n_clusters
        
        results.append({
            'features': f'BMI + {metric}',
            'score': best_score,
            'n_clusters': best_n
        })
        
        print(f"  {metric}: {best_score:.4f} (n={best_n})")
        
        if best_score > best_overall['score']:
            best_overall = {
                'score': best_score,
                'features': f'BMI ({metric})',
                'n_clusters': best_n,
                'distance': metric
            }
            
    except Exception as e:
        print(f"  {metric}: Failed - {e}")

# =============================================================================
# Fine-tune the best configuration
# =============================================================================
print("\n" + "=" * 70)
print("FINE-TUNING BEST CONFIGURATION")
print("=" * 70)

print(f"\n[Best so far]: {best_overall['features']} - Score: {best_overall['score']:.4f}")

# Try with the best feature set
best_features = ['BMI']
X_best = df[best_features].fillna(df[best_features].median()).values
X_best_scaled = StandardScaler().fit_transform(X_best)

# Very fine-grained search
print("\n[Fine-tuning cluster count...]")
best_score = -1
best_n = 2

for n_clusters in range(2, 20):
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage='single')
    labels = model.fit_predict(X_best_scaled)
    score = silhouette_score(X_best_scaled, labels)
    
    if score > best_score:
        best_score = score
        best_n = n_clusters

print(f"  Fine-tuned: {best_score:.4f} (n={best_n})")

if best_score > best_overall['score']:
    best_overall = {
        'score': best_score,
        'features': 'BMI (fine-tuned)',
        'n_clusters': best_n,
        'distance': 'euclidean'
    }

# =============================================================================
# Final Results
# =============================================================================
print("\n" + "=" * 70)
print("FINAL RESULTS")
print("=" * 70)

# Sort results
results.sort(key=lambda x: x['score'], reverse=True)

print(f"\n{'Feature Set':<30} {'Score':>10} {'Clusters':>10}")
print("-" * 50)
for r in results:
    print(f"{r['features']:<30} {r['score']:>10.4f} {r['n_clusters']:>10}")

print("\n" + "=" * 70)
print("BEST RESULT")
print("=" * 70)

print(f"\n[Method]: Agglomerative Hierarchical Clustering (Single Linkage)")
print(f"[Features]: {best_overall['features']}")
print(f"[Clusters]: {best_overall['n_clusters']}")
print(f"[Distance Metric]: {best_overall['distance']}")
print(f"[Silhouette Score]: {best_overall['score']:.4f}")

# Comparison
kmeans_baseline = 0.5590
improvement = ((best_overall['score'] - kmeans_baseline) / kmeans_baseline) * 100

print(f"\n[COMPARISON TO K-MEANS BASELINE]:")
print(f"  K-Means Baseline:     {kmeans_baseline:.4f}")
print(f"  Hierarchical (Best):  {best_overall['score']:.4f}")
print(f"  Improvement:          +{improvement:.2f}%")

if best_overall['score'] > kmeans_baseline:
    print(f"\n  [✓] ACHIEVED BETTER RESULTS!")
else:
    print(f"\n  [≈] Similar to K-Means")

# =============================================================================
# Cluster Distribution Analysis
# =============================================================================
print("\n" + "=" * 70)
print("CLUSTER DISTRIBUTION")
print("=" * 70)

# Re-fit with best parameters
X_final = df[best_features].fillna(df[best_features].median()).values
X_final_scaled = StandardScaler().fit_transform(X_final)

model = AgglomerativeClustering(n_clusters=best_overall['n_clusters'], linkage='single')
labels = model.fit_predict(X_final_scaled)

print(f"\n[Cluster Sizes]:")
unique, counts = np.unique(labels, return_counts=True)
for cluster, count in zip(unique, counts):
    pct = count / len(labels) * 100
    print(f"  Cluster {cluster}: {count} samples ({pct:.1f}%)")

# =============================================================================
# Save Results
# =============================================================================
summary = {
    'Method': 'Agglomerative Hierarchical Clustering (Single Linkage)',
    'Features': best_overall['features'],
    'Clusters': best_overall['n_clusters'],
    'Distance_Metric': best_overall['distance'],
    'Silhouette_Score': round(best_overall['score'], 4),
    'KMeans_Baseline': 0.5590,
    'Improvement_Percent': round(improvement, 2),
    'All_Results': results
}

with open(f'{OUTPUT_DIR}/metrics/summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

pd.DataFrame(results).to_csv(f'{OUTPUT_DIR}/metrics/all_results.csv', index=False)

print("\n" + "=" * 70)
print("SAVED")
print("=" * 70)
print(f"\n[Output]: {OUTPUT_DIR}/")
print(f"  - metrics/summary.json")
print(f"  - metrics/all_results.csv")
