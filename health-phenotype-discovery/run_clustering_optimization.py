#!/usr/bin/env python3
"""
Execute the DBSCAN clustering pipeline and iteratively improve performance
to achieve target Silhouette Score of 0.87-1.00
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from itertools import product
import os
import sys
import json

# =============================================================================
# PROJECT CONFIGURATION
# =============================================================================
print("=" * 70)
print("DBSCAN CLUSTERING PIPELINE - ITERATIVE OPTIMIZATION")
print("=" * 70)

PROJECT_ROOT = os.path.abspath('.')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output_v2')
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

print(f"[INFO] Directory structure created")
print(f"[INFO] Project Root: {PROJECT_ROOT}")

# =============================================================================
# DATA LOADING
# =============================================================================
print("\n" + "=" * 70)
print("STEP 1: DATA LOADING")
print("=" * 70)

data_path = os.path.join(PHASE_DIRS['data'], 'nhanes_health_data.csv')
df = pd.read_csv(data_path)

print(f"[INFO] Dataset Shape: {df.shape}")
print(f"[INFO] Number of samples: {df.shape[0]}")
print(f"[INFO] Number of features: {df.shape[1]}")

# =============================================================================
# DATA EXPLORATION
# =============================================================================
print("\n" + "=" * 70)
print("DATA EXPLORATION")
print("=" * 70)

print("\n[INFO] Data Types:")
print(df.dtypes.value_counts())

print("\n[INFO] Missing Values:")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({'Missing Count': missing, 'Percentage': missing_pct})
print(missing_df[missing_df['Missing Count'] > 0] if missing.sum() > 0 else "  No missing values found!")

print("\n[INFO] Numerical Statistics:")
print(df.describe().T.round(2))

# =============================================================================
# DATA PREPROCESSING
# =============================================================================
print("\n" + "=" * 70)
print("DATA PREPROCESSING")
print("=" * 70)

df_processed = df.copy()

# Handle missing values
print("\n[STEP 1] Handling missing values...")
for col in df_processed.columns:
    if df_processed[col].isnull().sum() > 0:
        if df_processed[col].dtype in ['float64', 'int64']:
            df_processed[col].fillna(df_processed[col].median(), inplace=True)
        else:
            df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)
print("  [OK] Missing values handled!")

# Encode categorical variables
print("\n[STEP 2] Encoding categorical variables...")
label_encoders = {}
cat_columns = df_processed.select_dtypes(include=['object']).columns

for col in cat_columns:
    le = LabelEncoder()
    df_processed[col] = le.fit_transform(df_processed[col].astype(str))
    label_encoders[col] = le

print(f"  [OK] Encoded {len(cat_columns)} categorical columns!")

# Scale features
print("\n[STEP 3] Scaling features...")
scaler = StandardScaler()
feature_columns = df_processed.columns.tolist()
X_scaled = scaler.fit_transform(df_processed)
print(f"  [OK] Scaled {len(feature_columns)} features!")

print(f"\n[INFO] Preprocessed data shape: {X_scaled.shape}")

# Save preprocessed data
df_processed.to_csv(os.path.join(PHASE_DIRS['processed'], 'preprocessed_data.csv'), index=False)
np.save(os.path.join(PHASE_DIRS['processed'], 'X_scaled.npy'), X_scaled)
pd.DataFrame(feature_columns, columns=['feature']).to_csv(os.path.join(OUTPUT_SUBDIRS['metrics'], 'feature_names.csv'), index=False)

print("\n[OK] Preprocessing complete!")

# =============================================================================
# OPTIMAL EPSILON DETERMINATION
# =============================================================================
print("\n" + "=" * 70)
print("OPTIMAL EPSILON DETERMINATION")
print("=" * 70)

k = 5
neighbors = NearestNeighbors(n_neighbors=k)
neighbors_fit = neighbors.fit(X_scaled)
distances, indices = neighbors_fit.kneighbors(X_scaled)

k_distances = np.sort(distances[:, k-1])

# Find optimal epsilon using elbow method
diff1 = np.diff(k_distances)
diff2 = np.diff(diff1)
elbow_idx = np.argmax(diff2) + 1
optimal_eps = k_distances[elbow_idx]

print(f"\n[INFO] Optimal Epsilon Analysis:")
print(f"  k value used: {k}")
print(f"  Optimal epsilon: {optimal_eps:.4f}")
print(f"  Min k-distance: {k_distances.min():.4f}")
print(f"  Max k-distance: {k_distances.max():.4f}")
print(f"  Mean k-distance: {k_distances.mean():.4f}")

# =============================================================================
# HYPERPARAMETER OPTIMIZATION - ITERATIVE IMPROVEMENT
# =============================================================================
print("\n" + "=" * 70)
print("HYPERPARAMETER OPTIMIZATION - ITERATIVE APPROACH")
print("=" * 70)

TARGET_SILHOUETTE = 0.87
best_silhouette = -1
best_config = None
best_results_df = None

# Phase 1: Broad search
print("\n[PHASE 1] Broad hyperparameter search...")
eps_values = np.arange(0.3, 5.0, 0.2)
min_samples_values = [3, 5, 7, 10, 15, 20, 25, 30]

results = []
for eps, min_samples in product(eps_values, min_samples_values):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X_scaled)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    if n_clusters >= 2:
        mask = labels != -1
        if mask.sum() > n_clusters and len(set(labels[mask])) >= 2:
            X_valid = X_scaled[mask]
            labels_valid = labels[mask]
            
            silhouette = silhouette_score(X_valid, labels_valid)
            calinski = calinski_harabasz_score(X_valid, labels_valid)
            davies = davies_bouldin_score(X_valid, labels_valid)
        else:
            continue
    else:
        continue
    
    results.append({
        'eps': eps,
        'min_samples': min_samples,
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'silhouette': silhouette,
        'calinski': calinski,
        'davies': davies,
        'noise_ratio': n_noise / len(labels)
    })

results_df = pd.DataFrame(results)
print(f"[PHASE 1] Evaluated {len(results)} configurations")

# Phase 2: Refine around best configurations
print("\n[PHASE 2] Refining around best configurations...")
valid_results = results_df[results_df['n_clusters'] >= 2].copy()

if len(valid_results) > 0:
    # Get top 10 configurations by silhouette score
    top_configs = valid_results.nlargest(10, 'silhouette')
    print("\n[INFO] Top 10 configurations from Phase 1:")
    print(top_configs[['eps', 'min_samples', 'n_clusters', 'silhouette', 'calinski', 'davies']].to_string(index=False))
    
    # Refine around each top configuration
    refined_results = []
    for _, row in top_configs.iterrows():
        base_eps = row['eps']
        base_min_samples = int(row['min_samples'])
        
        # Fine-grained search around each top config
        eps_range = np.arange(max(0.1, base_eps - 0.3), base_eps + 0.3, 0.05)
        min_samples_range = range(max(2, base_min_samples - 3), base_min_samples + 4)
        
        for eps, min_samples in product(eps_range, min_samples_range):
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(X_scaled)
            
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            
            if n_clusters >= 2:
                mask = labels != -1
                if mask.sum() > n_clusters and len(set(labels[mask])) >= 2:
                    X_valid = X_scaled[mask]
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
                        'noise_ratio': n_noise / len(labels)
                    })
    
    refined_df = pd.DataFrame(refined_results)
    all_results = pd.concat([results_df, refined_df], ignore_index=True)
else:
    all_results = results_df

print(f"[PHASE 2] Total configurations evaluated: {len(all_results)}")

# =============================================================================
# OPTIMAL PARAMETER SELECTION
# =============================================================================
print("\n" + "=" * 70)
print("OPTIMAL PARAMETER SELECTION")
print("=" * 70)

valid_results = all_results[all_results['n_clusters'] >= 2].copy()
print(f"\n[INFO] Valid configurations (n_clusters >= 2): {len(valid_results)}")

# Find configurations meeting target
high_quality_configs = valid_results[valid_results['silhouette'] >= TARGET_SILHOUETTE].copy()
print(f"[INFO] Configurations with Silhouette Score >= {TARGET_SILHOUETTE}: {len(high_quality_configs)}")

if len(high_quality_configs) > 0:
    # Sort by silhouette, then by noise ratio (prefer lower noise)
    high_quality_configs = high_quality_configs.sort_values(
        ['silhouette', 'noise_ratio'], ascending=[False, True]
    )
    best_config = high_quality_configs.iloc[0]
    print("\n[SUCCESS] Target Silhouette Score achieved!")
else:
    print(f"\n[INFO] Target Silhouette Score not yet achieved. Selecting best available...")
    
    # Balance silhouette with other metrics
    valid_results['combined_score'] = (
        (valid_results['silhouette'] - valid_results['silhouette'].min()) / 
        (valid_results['silhouette'].max() - valid_results['silhouette'].min())
    ) - (
        (valid_results['davies'] - valid_results['davies'].min()) / 
        (valid_results['davies'].max() - valid_results['davies'].min())
    )
    
    best_config = valid_results.loc[valid_results['combined_score'].idxmax()]

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
print("FINAL DBSCAN CLUSTERING")
print("=" * 70)

final_dbscan = DBSCAN(eps=OPTIMAL_EPS, min_samples=OPTIMAL_MIN_SAMPLES)
cluster_labels = final_dbscan.fit_predict(X_scaled)

n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
n_noise = list(cluster_labels).count(-1)

print(f"\n[INFO] Clustering Results:")
print(f"  Number of clusters: {n_clusters}")
print(f"  Noise points: {n_noise} ({n_noise/len(cluster_labels):.2%})")
print(f"  Clustered points: {len(cluster_labels) - n_noise} ({(len(cluster_labels) - n_noise)/len(cluster_labels):.2%})")

# Calculate final metrics
mask = cluster_labels != -1
X_clustered = X_scaled[mask]
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
print("CLUSTER PROFILE ANALYSIS")
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
    
    # Calculate mean for numerical columns
    for col in df_clustered.select_dtypes(include=[np.number]).columns:
        if col != 'Cluster':
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
print(f"  Total Features: {df.shape[1]}")

print(f"\n[DBSCAN PARAMETERS]")
print(f"  Epsilon: {OPTIMAL_EPS:.4f}")
print(f"  Min Samples: {OPTIMAL_MIN_SAMPLES}")

print(f"\n[CLUSTERING RESULTS]")
print(f"  Number of Clusters: {n_clusters}")
print(f"  Noise Points: {n_noise}")

print(f"\n[QUALITY METRICS]")
print(f"  Silhouette Score: {final_silhouette:.4f}")
print(f"  Calinski-Harabasz Index: {final_calinski:.2f}")
print(f"  Davies-Bouldin Index: {final_davies:.4f}")

print(f"\n[OPTIMIZATION STATUS]")
if final_silhouette >= TARGET_SILHOUETTE:
    print(f"  [✓] TARGET ACHIEVED: Silhouette Score {final_silhouette:.4f} >= {TARGET_SILHOUETTE}")
else:
    print(f"  [✗] TARGET NOT ACHIEVED: Silhouette Score {final_silhouette:.4f} < {TARGET_SILHOUETTE}")
    print(f"  [→] Consider feature engineering or different preprocessing strategies")

print("\n" + "=" * 70)
print("PIPELINE COMPLETE!")
print("=" * 70)

# Save summary
summary = {
    'Dataset': {
        'Total_Samples': len(df),
        'Total_Features': df.shape[1]
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
    'Target_Status': 'ACHIEVED' if final_silhouette >= TARGET_SILHOUETTE else 'NOT_ACHIEVED'
}

with open(os.path.join(OUTPUT_SUBDIRS['metrics'], 'final_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\nResults saved to:")
print(f"  - {OUTPUT_SUBDIRS['metrics']}/")
print(f"  - {OUTPUT_SUBDIRS['predictions']}/")
print(f"  - {OUTPUT_SUBDIRS['cluster_profiles']}/")
