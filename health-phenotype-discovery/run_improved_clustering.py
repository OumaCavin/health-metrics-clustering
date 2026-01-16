#!/usr/bin/env python3
"""
Improved DBSCAN clustering pipeline with feature engineering and dimensionality reduction
to achieve target Silhouette Score of 0.87-1.00
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from itertools import product
import os
import json

print("=" * 70)
print("IMPROVED DBSCAN CLUSTERING PIPELINE")
print("=" * 70)

# =============================================================================
# CONFIGURATION
# =============================================================================
PROJECT_ROOT = os.path.abspath('.')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output_v3')  # Use new output directory
MODELS_DIR = os.path.join(OUTPUT_DIR, 'models')
FIGURES_DIR = os.path.join(OUTPUT_DIR, 'figures')

PHASE_DIRS = {
    'data': os.path.join(DATA_DIR, 'raw'),
    'processed': os.path.join(DATA_DIR, 'processed'),
    'reports': os.path.join(OUTPUT_DIR, 'reports'),
    'logs': os.path.join(OUTPUT_DIR, 'logs'),
    'plots': os.path.join(FIGURES_DIR, 'plots')
}

MODEL_SUBDIRS = {
    'baseline': os.path.join(MODELS_DIR, 'baseline'),
    'tuned': os.path.join(MODELS_DIR, 'tuned'),
    'final': os.path.join(MODELS_DIR, 'final')
}

OUTPUT_SUBDIRS = {
    'metrics': os.path.join(OUTPUT_DIR, 'metrics'),
    'predictions': os.path.join(OUTPUT_DIR, 'predictions'),
    'cluster_profiles': os.path.join(OUTPUT_DIR, 'cluster_profiles')
}

# Create directories
all_dirs = [PROJECT_ROOT, DATA_DIR, OUTPUT_DIR, MODELS_DIR, FIGURES_DIR, 
            *PHASE_DIRS.values(), *MODEL_SUBDIRS.values(), *OUTPUT_SUBDIRS.values()]
for dir_path in all_dirs:
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

print(f"[INFO] Using output directory: {OUTPUT_DIR}")

# =============================================================================
# DATA LOADING
# =============================================================================
print("\n" + "=" * 70)
print("STEP 1: DATA LOADING")
print("=" * 70)

data_path = os.path.join(PHASE_DIRS['data'], 'nhanes_health_data.csv')
df = pd.read_csv(data_path)
print(f"[INFO] Dataset Shape: {df.shape}")

# =============================================================================
# ADVANCED DATA PREPROCESSING
# =============================================================================
print("\n" + "=" * 70)
print("STEP 2: ADVANCED DATA PREPROCESSING")
print("=" * 70)

df_processed = df.copy()

# Step 1: Handle missing values first
print("\n[STEP 1] Handling missing values...")
for col in df_processed.columns:
    if df_processed[col].isnull().sum() > 0:
        if df_processed[col].dtype in ['float64', 'int64']:
            df_processed[col].fillna(df_processed[col].median(), inplace=True)
        else:
            df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)
print("  [OK] Missing values handled!")

# Step 2: Separate numeric and categorical columns
numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()

print(f"\n[INFO] Numeric columns: {len(numeric_cols)}")
print(f"[INFO] Categorical columns: {len(categorical_cols)}")

# Step 3: One-hot encode categorical variables (better than label encoding)
print("\n[STEP 2] One-hot encoding categorical variables...")
df_encoded = pd.get_dummies(df_processed, columns=categorical_cols, drop_first=True)
print(f"  [OK] One-hot encoding complete!")
print(f"  [INFO] Features after encoding: {df_encoded.shape[1]}")

# Step 4: Apply RobustScaler (better for data with outliers)
print("\n[STEP 3] Scaling features with RobustScaler...")
scaler = RobustScaler()
X_scaled = scaler.fit_transform(df_encoded)
print(f"  [OK] Scaled {X_scaled.shape[1]} features!")

# Step 5: Feature selection - remove low variance features
print("\n[STEP 4] Feature selection (removing low variance features)...")
selector = VarianceThreshold(threshold=0.01)
X_selected = selector.fit_transform(X_scaled)
selected_features = df_encoded.columns[selector.get_support()].tolist()
print(f"  [OK] Selected {len(selected_features)} features from {X_scaled.shape[1]}")

# Step 6: Dimensionality reduction with PCA
print("\n[STEP 5] Dimensionality reduction with PCA...")
# Keep 95% of variance
pca = PCA(n_components=0.95, random_state=42)
X_pca = pca.fit_transform(X_selected)
explained_variance = pca.explained_variance_ratio_.sum()
print(f"  [OK] PCA reduced to {X_pca.shape[1]} components")
print(f"  [INFO] Explained variance: {explained_variance:.2%}")

# Step 7: Final standardization on PCA components
print("\n[STEP 6] Final standardization...")
final_scaler = StandardScaler()
X_final = final_scaler.fit_transform(X_pca)
print(f"  [OK] Final feature matrix shape: {X_final.shape}")

# Save preprocessed data
df_processed.to_csv(os.path.join(PHASE_DIRS['processed'], 'preprocessed_data.csv'), index=False)
np.save(os.path.join(PHASE_DIRS['processed'], 'X_pca.npy'), X_pca)
np.save(os.path.join(PHASE_DIRS['processed'], 'X_final.npy'), X_final)

print("\n[OK] Preprocessing complete!")

# =============================================================================
# OPTIMAL EPSILON DETERMINATION
# =============================================================================
print("\n" + "=" * 70)
print("STEP 3: OPTIMAL EPSILON DETERMINATION")
print("=" * 70)

# Try different k values
k_values = [5, 7, 10]
best_k = 5
best_eps = None
best_elbow_score = -np.inf

print("\n[INFO] Testing different k values for k-distance analysis...")
for k in k_values:
    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors_fit = neighbors.fit(X_final)
    distances, indices = neighbors_fit.kneighbors(X_final)
    
    k_distances = np.sort(distances[:, k-1])
    
    # Find elbow using second derivative
    diff1 = np.diff(k_distances)
    diff2 = np.diff(diff1)
    if len(diff2) > 0:
        elbow_idx = np.argmax(diff2) + 1
        elbow_eps = k_distances[elbow_idx]
        
        # Calculate elbow score (higher is better)
        elbow_score = k_distances[elbow_idx] - k_distances[0]
        
        print(f"  k={k}: Optimal epsilon = {elbow_eps:.4f}, Score = {elbow_score:.4f}")
        
        if elbow_score > best_elbow_score:
            best_elbow_score = elbow_score
            best_k = k
            best_eps = elbow_eps

optimal_eps = best_eps
print(f"\n[INFO] Selected: k={best_k}, Epsilon={optimal_eps:.4f}")

# =============================================================================
# ITERATIVE HYPERPARAMETER OPTIMIZATION
# =============================================================================
print("\n" + "=" * 70)
print("STEP 4: HYPERPARAMETER OPTIMIZATION")
print("=" * 70)

TARGET_SILHOUETTE = 0.87

# Phase 1: Broad search
print("\n[PHASE 1] Broad hyperparameter search...")
eps_values = np.arange(0.5, 3.0, 0.1)
min_samples_values = [3, 5, 7, 10, 15]

results = []
for eps, min_samples in product(eps_values, min_samples_values):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X_final)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    if n_clusters >= 2 and n_clusters <= 10:  # Limit cluster count for meaningful results
        mask = labels != -1
        if mask.sum() > n_clusters and len(set(labels[mask])) >= 2:
            X_valid = X_final[mask]
            labels_valid = labels[mask]
            
            silhouette = silhouette_score(X_valid, labels_valid)
            calinski = calinski_harabasz_score(X_valid, labels_valid)
            davies = davies_bouldin_score(X_valid, labels_valid)
            
            results.append({
                'eps': eps,
                'min_samples': min_samples,
                'n_clusters': n_clusters,
                'n_noise': n_noise,
                'silhouette': silhouette,
                'calinski': calinski,
                'davies': davies,
                'noise_ratio': n_noise / len(labels),
                'clustered_ratio': mask.sum() / len(labels)
            })

results_df = pd.DataFrame(results)
print(f"  Evaluated {len(results)} configurations")

# Phase 2: Refine around best configurations
print("\n[PHASE 2] Refining around best configurations...")
valid_results = results_df[
    (results_df['n_clusters'] >= 2) & 
    (results_df['n_clusters'] <= 10) &
    (results_df['noise_ratio'] < 0.5)  # Require less than 50% noise
].copy()

if len(valid_results) > 0:
    # Get top 5 configurations
    top_configs = valid_results.nlargest(5, 'silhouette')
    print(f"\n  Top 5 configurations:")
    print(top_configs[['eps', 'min_samples', 'n_clusters', 'silhouette', 'noise_ratio']].to_string(index=False))
    
    # Refine around each top config
    refined_results = []
    for _, row in top_configs.iterrows():
        base_eps = row['eps']
        base_min_samples = int(row['min_samples'])
        
        eps_range = np.arange(max(0.1, base_eps - 0.2), base_eps + 0.2, 0.02)
        min_samples_range = range(max(2, base_min_samples - 2), base_min_samples + 3)
        
        for eps, min_samples in product(eps_range, min_samples_range):
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(X_final)
            
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            
            if n_clusters >= 2 and n_clusters <= 10:
                mask = labels != -1
                if mask.sum() > n_clusters:
                    X_valid = X_final[mask]
                    labels_valid = labels[mask]
                    
                    silhouette = silhouette_score(X_valid, labels_valid)
                    calinski = calinski_harabasz_score(X_valid, labels_valid)
                    davies = davies_bouldin_score(X_valid, labels_valid)
                    
                    refined_results.append({
                        'eps': eps,
                        'min_samples': min_samples,
                        'n_clusters': n_clusters,
                        'n_noise': n_noise,
                        'silhouette': silhouette,
                        'calinski': calinski,
                        'davies': davies,
                        'noise_ratio': n_noise / len(labels),
                        'clustered_ratio': mask.sum() / len(labels)
                    })
    
    refined_df = pd.DataFrame(refined_results)
    all_results = pd.concat([results_df, refined_df], ignore_index=True)
else:
    all_results = results_df

print(f"  Total configurations evaluated: {len(all_results)}")

# =============================================================================
# OPTIMAL PARAMETER SELECTION
# =============================================================================
print("\n" + "=" * 70)
print("STEP 5: OPTIMAL PARAMETER SELECTION")
print("=" * 70)

# Filter valid results
valid_results = all_results[
    (all_results['n_clusters'] >= 2) & 
    (all_results['n_clusters'] <= 10) &
    (all_results['noise_ratio'] < 0.5)
].copy()

print(f"\n[INFO] Valid configurations: {len(valid_results)}")

# Check for target achievement
high_quality_configs = valid_results[valid_results['silhouette'] >= TARGET_SILHOUETTE].copy()
print(f"[INFO] Configurations with Silhouette Score >= {TARGET_SILHOUETTE}: {len(high_quality_configs)}")

if len(high_quality_configs) > 0:
    # Sort by silhouette score, then by noise ratio (prefer lower noise)
    high_quality_configs = high_quality_configs.sort_values(
        ['silhouette', 'noise_ratio'], ascending=[False, True]
    )
    best_config = high_quality_configs.iloc[0]
    print("\n[✓] TARGET SILHOUETTE SCORE ACHIEVED!")
else:
    print(f"\n[INFO] Target not achieved. Selecting best available configuration...")
    
    # Composite score: balance silhouette, noise ratio, and cluster count
    if len(valid_results) > 0:
        valid_results['silhouette_norm'] = (valid_results['silhouette'] - valid_results['silhouette'].min()) / \
                                           (valid_results['silhouette'].max() - valid_results['silhouette'].min() + 1e-10)
        valid_results['noise_norm'] = 1 - (valid_results['noise_ratio'])  # Lower noise is better
        valid_results['cluster_norm'] = valid_results['n_clusters'] / 10  # Normalize cluster count
        
        # Weighted combination
        valid_results['combined_score'] = (
            0.5 * valid_results['silhouette_norm'] +
            0.3 * valid_results['noise_norm'] +
            0.2 * valid_results['cluster_norm']
        )
        
        best_config = valid_results.loc[valid_results['combined_score'].idxmax()]
    else:
        best_config = all_results.loc[all_results['silhouette'].idxmax()]

OPTIMAL_EPS = best_config['eps']
OPTIMAL_MIN_SAMPLES = int(best_config['min_samples'])

print(f"\n[INFO] Selected Optimal Parameters:")
print(f"  Epsilon: {OPTIMAL_EPS:.4f}")
print(f"  Min Samples: {OPTIMAL_MIN_SAMPLES}")
print(f"  Number of Clusters: {int(best_config['n_clusters'])}")
print(f"  Silhouette Score: {best_config['silhouette']:.4f}")
print(f"  Calinski-Harabasz Index: {best_config['calinski']:.2f}")
print(f"  Davies-Bouldin Index: {best_config['davies']:.4f}")
print(f"  Noise Ratio: {best_config['noise_ratio']:.2%}")

# Save optimization results
all_results.to_csv(os.path.join(OUTPUT_SUBDIRS['metrics'], 'hyperparameter_optimization_results.csv'), index=False)

# =============================================================================
# FINAL DBSCAN CLUSTERING
# =============================================================================
print("\n" + "=" * 70)
print("STEP 6: FINAL DBSCAN CLUSTERING")
print("=" * 70)

final_dbscan = DBSCAN(eps=OPTIMAL_EPS, min_samples=OPTIMAL_MIN_SAMPLES)
cluster_labels = final_dbscan.fit_predict(X_final)

n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
n_noise = list(cluster_labels).count(-1)

print(f"\n[INFO] Clustering Results:")
print(f"  Number of clusters: {n_clusters}")
print(f"  Noise points: {n_noise} ({n_noise/len(cluster_labels):.2%})")
print(f"  Clustered points: {len(cluster_labels) - n_noise} ({(len(cluster_labels) - n_noise)/len(cluster_labels):.2%})")

# Calculate final metrics
mask = cluster_labels != -1
X_clustered = X_final[mask]
labels_clustered = cluster_labels[mask]

final_silhouette = silhouette_score(X_clustered, labels_clustered)
final_calinski = calinski_harabasz_score(X_clustered, labels_clustered)
final_davies = davies_bouldin_score(X_clustered, labels_clustered)

print(f"\n[INFO] Final Clustering Metrics:")
print(f"  Silhouette Score: {final_silhouette:.4f}")
print(f"  Calinski-Harabasz Index: {final_calinski:.2f}")
print(f"  Davies-Bouldin Index: {final_davies:.4f}")

# Add cluster labels to original dataframe
df_clustered = df.copy()
df_clustered['Cluster'] = cluster_labels
df_clustered['Is_Noise'] = cluster_labels == -1

# Save clustered data
df_clustered.to_csv(os.path.join(OUTPUT_SUBDIRS['predictions'], 'clustered_data.csv'), index=False)

print("\n[OK] Clustering complete!")

# =============================================================================
# CLUSTER PROFILE ANALYSIS
# =============================================================================
print("\n" + "=" * 70)
print("STEP 7: CLUSTER PROFILE ANALYSIS")
print("=" * 70)

cluster_stats = []
for cluster_id in sorted(set(cluster_labels)):
    if cluster_id == -1:
        continue
    
    cluster_data = df_clustered[df_clustered['Cluster'] == cluster_id]
    n_points = len(cluster_data)
    
    stats = {
        'Cluster': cluster_id,
        'Size': n_points,
        'Percentage': n_points / len(df_clustered[~df_clustered['Is_Noise']]) * 100
    }
    
    # Calculate mean for key numeric columns
    key_cols = ['Age', 'BMI', 'Systolic_BP', 'Blood_Glucose', 'HbA1c', 'HDL_Cholesterol', 'Total_Cholesterol']
    for col in key_cols:
        if col in cluster_data.columns:
            stats[f'{col}_mean'] = cluster_data[col].mean()
    
    cluster_stats.append(stats)

cluster_stats_df = pd.DataFrame(cluster_stats)

print(f"\n[INFO] Cluster Sizes:")
for _, row in cluster_stats_df.iterrows():
    print(f"  Cluster {int(row['Cluster'])}: {int(row['Size'])} samples ({row['Percentage']:.1f}%)")

# Save cluster statistics
cluster_stats_df.to_csv(os.path.join(OUTPUT_SUBDIRS['cluster_profiles'], 'cluster_statistics.csv'), index=False)

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print(f"\n[DATASET]")
print(f"  Total Samples: {len(df)}")
print(f"  Original Features: {df.shape[1]}")
print(f"  Features after encoding: {df_encoded.shape[1]}")
print(f"  Features after PCA: {X_final.shape[1]}")

print(f"\n[DBSCAN PARAMETERS]")
print(f"  Epsilon: {OPTIMAL_EPS:.4f}")
print(f"  Min Samples: {OPTIMAL_MIN_SAMPLES}")

print(f"\n[CLUSTERING RESULTS]")
print(f"  Number of Clusters: {n_clusters}")
print(f"  Noise Points: {n_noise} ({n_noise/len(cluster_labels):.2%})")

print(f"\n[QUALITY METRICS]")
print(f"  Silhouette Score: {final_silhouette:.4f}")
print(f"  Calinski-Harabasz Index: {final_calinski:.2f}")
print(f"  Davies-Bouldin Index: {final_davies:.4f}")

print(f"\n[TARGET STATUS]")
if final_silhouette >= TARGET_SILHOUETTE:
    print(f"  [✓] SUCCESS: Silhouette Score {final_silhouette:.4f} >= {TARGET_SILHOUETTE}")
    status = "ACHIEVED"
else:
    print(f"  [✗] IN PROGRESS: Silhouette Score {final_silhouette:.4f} < {TARGET_SILHOUETTE}")
    print(f"  [→] Consider trying alternative approaches:")
    print(f"      - Feature engineering (create domain-specific features)")
    print(f"      - Try different clustering algorithms (K-Means, GMM)")
    print(f"      - Use domain knowledge to select relevant features")
    status = "IN_PROGRESS"

print("\n" + "=" * 70)
print("PIPELINE COMPLETE!")
print("=" * 70)

# Save summary
summary = {
    'Dataset': {
        'Total_Samples': len(df),
        'Original_Features': df.shape[1],
        'Encoded_Features': df_encoded.shape[1],
        'PCA_Features': X_final.shape[1]
    },
    'DBSCAN_Parameters': {
        'Epsilon': OPTIMAL_EPS,
        'Min_Samples': OPTIMAL_MIN_SAMPLES
    },
    'Clustering_Results': {
        'Number_of_Clusters': n_clusters,
        'Noise_Points': n_noise,
        'Noise_Ratio': f"{n_noise/len(cluster_labels):.2%}"
    },
    'Quality_Metrics': {
        'Silhouette_Score': f"{final_silhouette:.4f}",
        'Calinski_Harabasz_Index': f"{final_calinski:.2f}",
        'Davies_Bouldin_Index': f"{final_davies:.4f}"
    },
    'Target_Status': status
}

with open(os.path.join(OUTPUT_SUBDIRS['metrics'], 'final_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\nResults saved to:")
print(f"  - {OUTPUT_SUBDIRS['metrics']}/")
print(f"  - {OUTPUT_SUBDIRS['predictions']}/")
print(f"  - {OUTPUT_SUBDIRS['cluster_profiles']}/")
