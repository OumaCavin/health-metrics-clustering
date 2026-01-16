#!/usr/bin/env python3
"""
Feature-selected clustering approach with clinical focus
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import os
import json

print("=" * 70)
print("FEATURE-SELECTED CLUSTERING APPROACH")
print("=" * 70)

# Configuration
PROJECT_ROOT = os.path.abspath('.')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output_v5')

for dir_path in [OUTPUT_DIR, os.path.join(OUTPUT_DIR, 'metrics'), os.path.join(OUTPUT_DIR, 'predictions')]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# Load data
print("\n[INFO] Loading data...")
data_path = os.path.join(DATA_DIR, 'raw', 'nhanes_health_data.csv')
df = pd.read_csv(data_path)
print(f"  Dataset: {df.shape}")

# Select only clinically relevant numeric features
clinical_features = [
    'Age', 'Systolic_BP', 'Diastolic_BP', 'BMI', 'Waist_Circumference',
    'Total_Cholesterol', 'HDL_Cholesterol', 'LDL_Cholesterol', 
    'Blood_Glucose', 'HbA1c', 'Triglycerides',
    'Pulse', 'Respiratory_Rate'
]

print(f"\n[INFO] Using {len(clinical_features)} clinical features")
df_clinical = df[clinical_features].copy()

# Handle missing values
for col in df_clinical.columns:
    if df_clinical[col].isnull().sum() > 0:
        df_clinical[col].fillna(df_clinical[col].median(), inplace=True)

# Scale features
scaler = RobustScaler()
X_scaled = scaler.fit_transform(df_clinical)

# Apply PCA
pca = PCA(n_components=0.95, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print(f"  PCA: {X_pca.shape[1]} components ({pca.explained_variance_ratio_.sum():.1%} variance)")

# Final scaling
X_final = StandardScaler().fit_transform(X_pca)

# Find optimal epsilon
print("\n[INFO] Optimizing parameters...")
k = 5
neighbors = NearestNeighbors(n_neighbors=k)
neighbors_fit = neighbors.fit(X_final)
distances, indices = neighbors_fit.kneighbors(X_final)
k_distances = np.sort(distances[:, k-1])

# Try a wide range
eps_values = np.linspace(0.5, 10.0, 100)
min_samples_values = [3, 5, 7, 10]

results = []
print(f"\n[INFO] Testing DBSCAN configurations...")

for eps in eps_values:
    for min_samples in min_samples_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X_final)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        if n_clusters >= 2 and n_clusters <= 6:
            mask = labels != -1
            clustered_ratio = mask.sum() / len(labels)
            
            if clustered_ratio >= 0.1 and clustered_ratio <= 0.9:
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

# Also test K-Means for comparison
print("[INFO] Testing K-Means configurations...")
for n_clusters in range(2, 7):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_final)
    
    silhouette = silhouette_score(X_final, labels)
    results.append({
        'eps': 0,  # Mark as K-Means
        'min_samples': n_clusters,
        'n_clusters': n_clusters,
        'n_noise': 0,
        'silhouette': silhouette,
        'clustered_ratio': 1.0,
        'method': 'K-Means'
    })

# Test GMM
print("[INFO] Testing GMM configurations...")
for n_clusters in range(2, 7):
    gmm = GaussianMixture(n_components=n_clusters, random_state=42, covariance_type='full')
    labels = gmm.fit_predict(X_final)
    
    silhouette = silhouette_score(X_final, labels)
    results.append({
        'eps': 0,  # Mark as GMM
        'min_samples': n_clusters,
        'n_clusters': n_clusters,
        'n_noise': 0,
        'silhouette': silhouette,
        'clustered_ratio': 1.0,
        'method': 'GMM'
    })

results_df = pd.DataFrame(results)
print(f"\n[INFO] Total configurations tested: {len(results)}")

# Find best result
best_idx = results_df['silhouette'].idxmax()
best_result = results_df.loc[best_idx]

print(f"\n" + "=" * 70)
print("BEST RESULTS")
print("=" * 70)
print(f"\n[Method]: {best_result.get('method', 'DBSCAN')}")
print(f"[Silhouette Score]: {best_result['silhouette']:.4f}")
print(f"[Clusters]: {int(best_result['n_clusters'])}")
print(f"[Noise]: {best_result.get('n_noise', 0)} ({best_result.get('noise_ratio', 0):.1%})")

# Final clustering with best method
print("\n" + "=" * 70)
print("FINAL CLUSTERING")
print("=" * 70)

if best_result.get('method') == 'K-Means':
    print("\n[INFO] Using K-Means with best parameters...")
    kmeans = KMeans(n_clusters=int(best_result['min_samples']), random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_final)
    method_name = "K-Means"
elif best_result.get('method') == 'GMM':
    print("\n[INFO] Using GMM with best parameters...")
    gmm = GaussianMixture(n_components=int(best_result['min_samples']), random_state=42, covariance_type='full')
    cluster_labels = gmm.fit_predict(X_final)
    method_name = "GMM"
else:
    print(f"\n[INFO] Using DBSCAN with epsilon={best_result['eps']:.4f}, min_samples={int(best_result['min_samples'])}...")
    dbscan = DBSCAN(eps=best_result['eps'], min_samples=int(best_result['min_samples']))
    cluster_labels = dbscan.fit_predict(X_final)
    method_name = "DBSCAN"

final_silhouette = silhouette_score(X_final, cluster_labels)
n_clusters = len(set(cluster_labels))
n_noise = list(cluster_labels == -1).count(True)

print(f"\n[FINAL RESULTS ({method_name})]:")
print(f"  Clusters: {n_clusters}")
print(f"  Noise: {n_noise} ({n_noise/len(cluster_labels):.1%})")
print(f"  Silhouette Score: {final_silhouette:.4f}")

# Save results
df_clustered = df.copy()
df_clustered['Cluster'] = cluster_labels
df_clustered.to_csv(os.path.join(OUTPUT_DIR, 'predictions', 'clustered_data.csv'), index=False)

summary = {
    'Method': method_name,
    'Silhouette_Score': f"{final_silhouette:.4f}",
    'Target_Achieved': final_silhouette >= 0.87,
    'Clusters': n_clusters,
    'Noise_Ratio': f"{n_noise/len(cluster_labels):.2%}",
    'Best_Parameters': best_result.to_dict()
}

with open(os.path.join(OUTPUT_DIR, 'metrics', 'summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)

# Save all results for analysis
results_df.to_csv(os.path.join(OUTPUT_DIR, 'metrics', 'all_results.csv'), index=False)

print(f"\n[TARGET STATUS]: {'[✓] ACHIEVED' if final_silhouette >= 0.87 else '[✗] NOT ACHIEVED'}")
print(f"\n[Target Score]: 0.87")
print(f"[Achieved Score]: {final_silhouette:.4f}")
print(f"[Difference]: {0.87 - final_silhouette:.4f}")

print("\n" + "=" * 70)
print("PIPELINE COMPLETE")
print("=" * 70)
