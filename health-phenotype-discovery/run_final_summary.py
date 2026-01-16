#!/usr/bin/env python3
"""
Final Silhouette Score Analysis - Summary Report
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os
import json

print("=" * 70)
print("SILHOUETTE OPTIMIZATION - FINAL RESULTS")
print("=" * 70)

OUTPUT_DIR = 'output_final'
for d in [OUTPUT_DIR, f'{OUTPUT_DIR}/metrics']:
    if not os.path.exists(d):
        os.makedirs(d)

# Load data
df = pd.read_csv('data/raw/nhanes_health_data.csv')
print(f"\n[INFO] Dataset: {df.shape}")

# Comprehensive feature testing
features_list = [
    (['BMI'], 'BMI Only'),
    (['BMI', 'Blood_Glucose'], 'BMI + Glucose'),
    (['BMI', 'Blood_Glucose', 'Systolic_BP'], 'Cardiometabolic'),
    (['Age', 'BMI', 'Systolic_BP', 'Blood_Glucose', 'HDL_Cholesterol'], 'Comprehensive'),
]

results = {}

for features, name in features_list:
    print(f"\n[TESTING]: {name}")
    
    X = df[features].fillna(df[features].median()).values
    X_scaled = StandardScaler().fit_transform(X)
    
    best_score = -1
    best_n = 2
    
    for n in range(2, 8):
        kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        
        if score > best_score:
            best_score = score
            best_n = n
    
    results[name] = {'score': best_score, 'clusters': best_n}
    print(f"  Best: {best_score:.4f} (n={best_n})")

# Best overall
best_method = max(results.items(), key=lambda x: x[1]['score'])

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print(f"\n[METHOD COMPARISON]:")
for name, data in results.items():
    marker = " ← BEST" if name == best_method[0] else ""
    print(f"  {name}: {data['score']:.4f} (n={data['clusters']}){marker}")

print(f"\n[BEST RESULT]:")
print(f"  Method: {best_method[0]}")
print(f"  Silhouette Score: {best_method[1]['score']:.4f}")
print(f"  Clusters: {best_method[1]['clusters']}")

print(f"\n[TARGET ANALYSIS]:")
print(f"  Target: 0.87-1.00")
print(f"  Achieved: {best_method[1]['score']:.4f}")
print(f"  Gap: {0.87 - best_method[1]['score']:.4f}")

if best_method[1]['score'] >= 0.87:
    status = "ACHIEVED"
    print(f"\n[✓] SUCCESS: Target Silhouette Score achieved!")
else:
    status = "NOT_ACHIEVED"
    print(f"\n[ANALYSIS]:")
    print(f"  Real-world NHANES health data forms a continuous spectrum.")
    print(f"  Natural clustering achieves ~0.22-0.23 maximum Silhouette Score.")
    print(f"  The target of 0.87-1.00 would require:")
    print(f"    - Pre-defined categorical labels (not clustering)")
    print(f"    - Artificially engineered boundaries")
    print(f"    - Very small, homogeneous samples")
    print(f"    - Synthetic data with clear separations")

# Save summary
summary = {
    'Best_Method': best_method[0],
    'Best_Silhouette': f"{best_method[1]['score']:.4f}",
    'Best_Clusters': best_method[1]['clusters'],
    'Target': '0.87-1.00',
    'Gap': f"{0.87 - best_method[1]['score']:.4f}",
    'Status': status,
    'All_Results': {k: f"{v['score']:.4f} (n={v['clusters']})" for k, v in results.items()},
    'Insight': 'Natural clustering limited by continuous health data spectrum'
}

with open(f'{OUTPUT_DIR}/metrics/summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n[STATUS]: {status}")
print(f"\n[RECOMMENDATION]: To achieve target scores, use clinical phenotype")
print(f"   categories as predefined clusters rather than unsupervised clustering.")
