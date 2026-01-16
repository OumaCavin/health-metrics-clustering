#!/usr/bin/env python3
"""
Create phenotype-based clusters using clinical thresholds
to demonstrate achievable Silhouette Scores
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import os
import json

print("=" * 70)
print("PHENOTYPE-BASED CLUSTERING ANALYSIS")
print("=" * 70)

# Configuration
OUTPUT_DIR = 'output_v7'
for d in [OUTPUT_DIR, f'{OUTPUT_DIR}/metrics', f'{OUTPUT_DIR}/predictions']:
    if not os.path.exists(d):
        os.makedirs(d)

# Load data
print("\n[INFO] Loading data...")
df = pd.read_csv('data/raw/nhanes_health_data.csv')
print(f"  Dataset: {df.shape}")

# Create phenotype labels based on clinical guidelines
print("\n[INFO] Creating phenotype categories...")

# BMI categories
bmi_categories = pd.cut(df['BMI'], 
                        bins=[0, 18.5, 25, 30, 100], 
                        labels=['Underweight', 'Normal', 'Overweight', 'Obese'])

# Blood pressure categories  
df['BP_Category'] = 'Normal'
df.loc[df['Systolic_BP'] >= 140, 'BP_Category'] = 'High'
df.loc[(df['Systolic_BP'] >= 120) & (df['Systolic_BP'] < 140), 'BP_Category'] = 'Elevated'

# Diabetes categories
df['Glucose_Category'] = 'Normal'
df.loc[df['Blood_Glucose'] >= 126, 'Glucose_Category'] = 'Diabetic'
df.loc[(df['Blood_Glucose'] >= 100) & (df['Blood_Glucose'] < 126), 'Glucose_Category'] = 'Prediabetic'

# Create combined phenotype
df['Phenotype'] = bmi_categories.astype(str) + '_' + df['BP_Category'] + '_' + df['Glucose_Category']

print(f"  Unique phenotypes: {df['Phenotype'].nunique()}")
print(f"\n  Phenotype distribution:")
print(df['Phenotype'].value_counts().head(10))

# Prepare features for clustering
features = ['Age', 'Systolic_BP', 'Diastolic_BP', 'BMI', 'Waist_Circumference',
            'Blood_Glucose', 'HbA1c', 'HDL_Cholesterol', 'Total_Cholesterol']

X = df[features].fillna(df[features].median()).values
X_scaled = StandardScaler().fit_transform(X)

# Test 1: Using phenotype labels as clusters
print("\n" + "=" * 70)
print("ANALYSIS 1: PHENOTYPE-BASED CLUSTERS")
print("=" * 70)

# Encode phenotype labels
phenotype_map = {p: i for i, p in enumerate(df['Phenotype'].unique())}
phenotype_labels = df['Phenotype'].map(phenotype_map).values

# Calculate silhouette with phenotype labels
phenotype_silhouette = silhouette_score(X_scaled, phenotype_labels)
print(f"\n[INFO] Silhouette Score (using phenotype labels): {phenotype_silhouette:.4f}")

# Test 2: K-Means with optimal clusters
print("\n" + "=" * 70)
print("ANALYSIS 2: K-MEANS CLUSTERING")
print("=" * 70)

from sklearn.cluster import KMeans

best_score = -1
best_n = 2

for n in range(2, 15):
    kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    
    if score > best_score:
        best_score = score
        best_n = n

print(f"\n[INFO] Best K-Means:")
print(f"  Clusters: {best_n}")
print(f"  Silhouette: {best_score:.4f}")

# Test 3: Try fewer, more distinct features
print("\n" + "=" * 70)
print("ANALYSIS 3: DISTINCT FEATURE CLUSTERING")
print("=" * 70)

# Use only highly variable, clinically distinct features
distinct_features = ['BMI', 'Blood_Glucose', 'Systolic_BP']
X_distinct = df[distinct_features].fillna(df[distinct_features].median()).values
X_distinct_scaled = StandardScaler().fit_transform(X_distinct)

best_distinct_score = -1
best_distinct_n = 2

for n in range(2, 8):
    kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_distinct_scaled)
    score = silhouette_score(X_distinct_scaled, labels)
    
    if score > best_distinct_score:
        best_distinct_score = score
        best_distinct_n = n

print(f"\n[INFO] Best with {distinct_features}:")
print(f"  Clusters: {best_distinct_n}")
print(f"  Silhouette: {best_distinct_score:.4f}")

# Final comparison
print("\n" + "=" * 70)
print("FINAL COMPARISON")
print("=" * 70)

print(f"\n[METHOD COMPARISON]:")
print(f"  1. Phenotype-based:     {phenotype_silhouette:.4f}")
print(f"  2. K-Means (all):       {best_score:.4f} (n={best_n})")
print(f"  3. K-Means (distinct):  {best_distinct_score:.4f} (n={best_distinct_n})")

best_overall = max(phenotype_silhouette, best_score, best_distinct_score)
target = 0.87

print(f"\n[TARGET ANALYSIS]:")
print(f"  Target Silhouette: 0.87-1.00")
print(f"  Best Achieved: {best_overall:.4f}")
print(f"  Gap: {target - best_overall:.4f}")

if best_overall < target:
    print(f"\n[INSIGHT]:")
    print(f"  The NHANES health data represents a continuous spectrum of health states")
    print(f"  rather than discrete clusters. Achieving 0.87+ would require:")
    print(f"  - Artificially engineered categorical boundaries")
    print(f"  - Very small, homogeneous subgroups")
    print(f"  - Clinical phenotype definitions (not pure clustering)")
    
# Save best result
df_clustered = df.copy()
df_clustered['Cluster'] = phenotype_labels
df_clustered.to_csv(f'{OUTPUT_DIR}/predictions/clustered_data.csv', index=False)

summary = {
    'Phenotype_Silhouette': f"{phenotype_silhouette:.4f}",
    'KMeans_Best_Silhouette': f"{best_score:.4f}",
    'Distinct_Features_Silhouette': f"{best_distinct_score:.4f}",
    'Best_Achieved': f"{best_overall:.4f}",
    'Target': '0.87-1.00',
    'Gap': f"{target - best_overall:.4f}",
    'Analysis': 'Real health data forms continuum, not discrete clusters'
}

with open(f'{OUTPUT_DIR}/metrics/summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n[STATUS]: {'[✓] Target feasible' if best_overall >= 0.87 else '[✗] Target requires phenotype engineering'}")
print(f"\n[INFO] Results saved to {OUTPUT_DIR}/")
