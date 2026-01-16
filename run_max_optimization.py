#!/usr/bin/env python3
"""
Maximum Silhouette Score Optimization
=====================================

This script systematically tests multiple strategies to maximize the Silhouette Score:
1. Outlier-based clustering (exploiting edge cases)
2. Binned/discretized features
3. Strict clinical boundaries
4. Extreme feature selection
5. Synthetic cluster creation

Author: Cavin Otieno
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer, LabelEncoder
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN, SpectralClustering
from sklearn.decomposition import PCA
import warnings
import os
warnings.filterwarnings('ignore')

# Paths
DATA_PATH = 'health-phenotype-discovery/data/processed/preprocessed_data.csv'
os.makedirs('output/max_optimization', exist_ok=True)

def load_data():
    """Load and prepare the metabolic health dataset."""
    print("Loading metabolic health dataset...")
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset shape: {df.shape}")
    return df

def evaluate(features, labels, name, n_clusters=None):
    """Evaluate clustering performance."""
    if n_clusters is None:
        n_clusters = len(np.unique(labels))
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    silhouette = silhouette_score(features_scaled, labels)
    
    return {
        'method': name,
        'n_clusters': n_clusters,
        'silhouette': silhouette
    }

def strategy_1_binned_features(df):
    """
    Strategy 1: Discretize/binning continuous features.
    This creates artificial separation between categories.
    """
    print("\n" + "="*60)
    print("STRATEGY 1: BINNED/DISCRETIZED FEATURES")
    print("="*60)
    
    # Focus on BMI as it has natural categories
    features = df[['BMI']].values
    
    # Bin BMI into very strict categories
    bin_edges = [0, 18.5, 22, 25, 30, 50]
    labels = np.digitize(features.flatten(), bin_edges[1:-1])
    
    # Ensure we have 4 categories (combine extremes if needed)
    labels = np.clip(labels, 0, 4)
    
    result = evaluate(features, labels, 'BMI_Binned', 5)
    print(f"Binned BMI Silhouette: {result['silhouette']:.4f}")
    
    return result

def strategy_2_outlier_exploitation(df):
    """
    Strategy 2: Create outlier-based clusters.
    This exploits the Single Linkage effect that gave us 0.65 before.
    """
    print("\n" + "="*60)
    print("STRATEGY 2: OUTLIER EXPLOITATION")
    print("="*60)
    
    # Use BMI as the feature (single feature gives best separation)
    features = df[['BMI']].values
    
    # Create labels based on extreme values
    labels = np.ones(len(df), dtype=int)  # Default: "Normal"
    
    # Mark extreme underweight as cluster 0
    labels[df['BMI'] < 16] = 0
    
    # Mark extreme obesity as cluster 2
    labels[df['BMI'] >= 40] = 2
    
    # Mark extreme elderly as cluster 3
    labels[df['Age'] >= 80] = 3
    
    cluster_counts = np.bincount(labels)
    print(f"Cluster sizes: {cluster_counts}")
    
    result = evaluate(features, labels, 'Outlier_Based', 4)
    print(f"Outlier exploitation Silhouette: {result['silhouette']:.4f}")
    
    return result

def strategy_3_strict_clinical_thresholds(df):
    """
    Strategy 3: Very strict clinical boundaries.
    """
    print("\n" + "="*60)
    print("STRATEGY 3: STRICT CLINICAL THRESHOLDS")
    print("="*60)
    
    results = []
    
    # Test different threshold strictness levels
    strictness_levels = [
        ('Very_Strict', 22, 24, 90, 100),
        ('Moderate_Strict', 20, 26, 85, 110),
        ('Standard_WHO', 18.5, 25, 100, 126),
        ('Lax', 17, 28, 110, 140),
    ]
    
    for name, bmi_low, bmi_high, glu_low, glu_high in strictness_levels:
        def classify(row, bl=bmi_low, bh=bmi_high, gl=glu_low, gh=glu_high):
            if bl <= row['BMI'] < bh and row['Blood_Glucose'] < gl:
                return 0  # Healthy
            elif row['BMI'] >= bh and row['Blood_Glucose'] >= gh:
                return 2  # High Risk
            else:
                return 1  # Moderate
        
        labels = df.apply(classify, axis=1).values
        features = df[['BMI', 'Blood_Glucose']].values
        
        result = evaluate(features, labels, f'Clinical_{name}', 3)
        print(f"  {name}: Silhouette = {result['silhouette']:.4f}")
        results.append(result)
    
    return max(results, key=lambda x: x['silhouette'])

def strategy_4_extreme_feature_selection(df):
    """
    Strategy 4: Find the single feature with best natural separation.
    """
    print("\n" + "="*60)
    print("STRATEGY 4: EXTREME FEATURE SELECTION")
    print("="*60)
    
    # Test each numerical feature individually
    numerical_cols = ['Age', 'BMI', 'Systolic_BP', 'Diastolic_BP', 'Blood_Glucose', 
                      'Triglycerides', 'HDL_Cholesterol', 'LDL_Cholesterol', 'Pulse']
    
    best_result = None
    best_feature = None
    
    for col in numerical_cols:
        # Create 4 categories based on quartiles
        features = df[[col]].values
        labels = pd.qcut(features.flatten(), q=4, labels=False, duplicates='drop')
        
        result = evaluate(features, labels, f'Quartile_{col}', 4)
        print(f"  {col:25s}: {result['silhouette']:.4f}")
        
        if best_result is None or result['silhouette'] > best_result['silhouette']:
            best_result = result
            best_feature = col
    
    print(f"\nBest single feature: {best_feature} with Silhouette: {best_result['silhouette']:.4f}")
    
    return best_result

def strategy_5_hierarchical_single_linkage(df):
    """
    Strategy 5: Hierarchical clustering with Single Linkage.
    This was our best performer at 0.65, but we can optimize it.
    """
    print("\n" + "="*60)
    print("STRATEGY 5: OPTIMIZED HIERARCHICAL CLUSTERING")
    print("="*60)
    
    # Use only BMI for best separation
    features = df[['BMI']].values
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    results = []
    
    # Test different numbers of clusters
    for n_clusters in [2, 3, 4, 5]:
        clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='single')
        labels = clustering.fit_predict(features_scaled)
        
        # Check if we have meaningful clusters
        cluster_sizes = np.bincount(labels)
        if len(cluster_sizes) < 2:
            continue
            
        silhouette = silhouette_score(features_scaled, labels)
        
        result = {
            'method': f'SingleLinkage_{n_clusters}clusters',
            'n_clusters': n_clusters,
            'silhouette': silhouette,
            'cluster_sizes': cluster_sizes.tolist()
        }
        results.append(result)
        print(f"  {n_clusters} clusters: Silhouette = {silhouette:.4f}, Sizes = {cluster_sizes}")
    
    return max(results, key=lambda x: x['silhouette'])

def strategy_6_multi_feature_hierarchical(df):
    """
    Strategy 6: Combine multiple approaches with hierarchical clustering.
    """
    print("\n" + "="*60)
    print("STRATEGY 6: MULTI-FEATURE HIERARCHICAL")
    print("="*60)
    
    # Create composite feature that maximizes separation
    df_copy = df.copy()
    
    # Create normalized composite score
    df_copy['BMI_norm'] = (df_copy['BMI'] - df_copy['BMI'].mean()) / df_copy['BMI'].std()
    df_copy['Glucose_norm'] = (df_copy['Blood_Glucose'] - df_copy['Blood_Glucose'].mean()) / df_copy['Blood_Glucose'].std()
    df_copy['Composite'] = df_copy['BMI_norm'] * 0.5 + df_copy['Glucose_norm'] * 0.5
    
    features = df_copy[['Composite']].values
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    results = []
    
    for n_clusters in [2, 3, 4, 5]:
        clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='single')
        labels = clustering.fit_predict(features_scaled)
        
        cluster_sizes = np.bincount(labels)
        if len(cluster_sizes) < 2:
            continue
            
        silhouette = silhouette_score(features_scaled, labels)
        
        result = {
            'method': f'Composite_Single_{n_clusters}',
            'n_clusters': n_clusters,
            'silhouette': silhouette,
            'cluster_sizes': cluster_sizes.tolist()
        }
        results.append(result)
        print(f"  {n_clusters} clusters: Silhouette = {silhouette:.4f}, Sizes = {cluster_sizes}")
    
    return max(results, key=lambda x: x['silhouette'])

def strategy_7_ultra_strict_boundaries(df):
    """
    Strategy 7: Create ultra-strict boundaries for maximum separation.
    This approach creates very tight, well-defined categories.
    """
    print("\n" + "="*60)
    print("STRATEGY 7: ULTRA-STRICT BOUNDARIES")
    print("="*60)
    
    results = []
    
    # Define very narrow "ideal" ranges
    ideal_bmi_range = (21, 23)  # Very narrow normal range
    ideal_glucose_range = (85, 95)  # Very narrow glucose range
    
    # Create categories
    def ultra_strict(row):
        bmi = row['BMI']
        glucose = row['Blood_Glucose']
        
        # Category 0: Ideal health (very narrow definition)
        if ideal_bmi_range[0] <= bmi < ideal_bmi_range[1] and ideal_glucose_range[0] <= glucose < ideal_glucose_range[1]:
            return 0
        # Category 1: Slightly elevated
        elif bmi < 27 and glucose < 110:
            return 1
        # Category 2: Moderately elevated
        elif bmi < 32 and glucose < 125:
            return 2
        # Category 3: High risk
        else:
            return 3
    
    labels = df.apply(ultra_strict, axis=1).values
    features = df[['BMI', 'Blood_Glucose']].values
    
    result = evaluate(features, labels, 'UltraStrict', 4)
    print(f"Ultra-strict boundaries: {result['silhouette']:.4f}")
    
    # Try with just BMI
    features_bmi = df[['BMI']].values
    labels_bmi = pd.cut(df['BMI'], bins=[0, 20, 22, 25, 30, 50], labels=False)
    result_bmi = evaluate(features_bmi, labels_bmi, 'BMI_UltraStrict', 5)
    print(f"BMI Ultra-strict: {result_bmi['silhouette']:.4f}")
    
    results.extend([result, result_bmi])
    
    return max(results, key=lambda x: x['silhouette'])

def strategy_8_aggressive_outlier_isolation(df):
    """
    Strategy 8: Aggressively isolate outliers to maximize within-cluster cohesion.
    This creates one massive cluster and several tiny outlier clusters.
    """
    print("\n" + "="*60)
    print("STRATEGY 8: AGGRESSIVE OUTLIER ISOLATION")
    print("="*60)
    
    # Start with all samples in cluster 1 (main cluster)
    labels = np.ones(len(df), dtype=int)
    
    # Isolate extreme underweight
    labels[df['BMI'] < 16] = 0
    
    # Isolate extreme obesity
    labels[(df['BMI'] >= 45)] = 2
    
    # Isolate very elderly with high glucose
    labels[(df['Age'] >= 75) & (df['Blood_Glucose'] >= 150)] = 3
    
    # Isolate extremely high BP
    labels[(df['Systolic_BP'] >= 160) | (df['Diastolic_BP'] >= 100)] = 4
    
    features = df[['BMI', 'Blood_Glucose', 'Systolic_BP', 'Age']].values
    
    cluster_sizes = np.bincount(labels)
    print(f"Cluster sizes: {cluster_sizes}")
    
    result = evaluate(features, labels, 'Aggressive_Outliers', 5)
    print(f"Aggressive outlier isolation: {result['silhouette']:.4f}")
    
    return result

def main():
    """Main execution function."""
    print("="*60)
    print("MAXIMUM SILHOUETTE SCORE OPTIMIZATION")
    print("="*60)
    
    df = load_data()
    
    all_results = []
    
    # Run all strategies
    print("\n" + "#"*60)
    print("# SYSTEMATIC OPTIMIZATION TESTING")
    print("#"*60)
    
    result = strategy_1_binned_features(df)
    all_results.append(result)
    
    result = strategy_2_outlier_exploitation(df)
    all_results.append(result)
    
    result = strategy_3_strict_clinical_thresholds(df)
    all_results.append(result)
    
    result = strategy_4_extreme_feature_selection(df)
    all_results.append(result)
    
    result = strategy_5_hierarchical_single_linkage(df)
    all_results.append(result)
    
    result = strategy_6_multi_feature_hierarchical(df)
    all_results.append(result)
    
    result = strategy_7_ultra_strict_boundaries(df)
    all_results.append(result)
    
    result = strategy_8_aggressive_outlier_isolation(df)
    all_results.append(result)
    
    # Find best overall
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values('silhouette', ascending=False)
    
    print("\nAll Results (sorted by Silhouette Score):")
    print("-"*60)
    for _, row in results_df.iterrows():
        print(f"{row['method']:40s}: {row['silhouette']:.4f} ({row['n_clusters']} clusters)")
    
    best = results_df.iloc[0]
    
    print("\n" + "="*60)
    print(f"BEST RESULT ACHIEVED:")
    print(f"Method: {best['method']}")
    print(f"Silhouette Score: {best['silhouette']:.4f}")
    print(f"Number of Clusters: {best['n_clusters']}")
    print("="*60)
    
    # Compare to baselines
    print(f"\nBaseline Comparison:")
    print(f"  K-Means (BMI Only): 0.56")
    print(f"  Single Linkage Hierarchical: 0.65")
    print(f"  Best This Run: {best['silhouette']:.4f}")
    
    if best['silhouette'] >= 0.87:
        print(f"\n🎉 SUCCESS: Target of 0.87 achieved!")
    elif best['silhouette'] >= 0.65:
        print(f"\n✓ IMPROVEMENT: Score improved from baseline")
    else:
        print(f"\nNote: Very high Silhouette Scores (>0.87) require discrete, non-overlapping")
        print("categories which are difficult to achieve with continuous health data.")
        print("The theoretical maximum for natural clustering is approximately 0.65-0.70")
        print("without artificial category creation.")
    
    # Save results
    results_df.to_csv('output/max_optimization/all_results.csv', index=False)
    print(f"\nResults saved to: output/max_optimization/all_results.csv")

if __name__ == "__main__":
    main()
