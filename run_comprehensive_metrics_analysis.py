#!/usr/bin/env python3
"""
Comprehensive Health Metrics Clustering Analysis
================================================

This script systematically tests all numerical health metrics with multiple
clustering algorithms to identify the best combinations for metabolic health
phenotype discovery.

Author: Cavin Otieno
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN, SpectralClustering
import warnings
import os
warnings.filterwarnings('ignore')

# Paths
DATA_PATH = 'health-phenotype-discovery/data/processed/preprocessed_data.csv'
OUTPUT_DIR = 'output/metrics_comparison/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data():
    """Load and prepare the metabolic health dataset."""
    print("Loading metabolic health dataset...")
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset shape: {df.shape}")
    return df

def get_all_numerical_features(df):
    """Extract all numerical health metrics from the dataset."""
    numerical_cols = [
        'Age',
        'BMI', 
        'Systolic_BP',
        'Diastolic_BP', 
        'Blood_Glucose',
        'Triglycerides',
        'HDL_Cholesterol',
        'LDL_Cholesterol',
        'Pulse',
        'Waist_Circumference',
        'Total_Cholesterol',
        'Hemoglobin',
        'HbA1c',
        'Creatinine',
        'BUN',
        'WBC',
        'Platelets',
        'Vitamin_D',
        'Respiratory_Rate'
    ]
    
    # Filter to only columns that exist in the dataset
    available_cols = [col for col in numerical_cols if col in df.columns]
    print(f"\nAvailable numerical health metrics ({len(available_cols)}):")
    for col in available_cols:
        print(f"  - {col}")
    
    return available_cols

def evaluate_clustering(features, labels, name, n_clusters=None):
    """Evaluate clustering performance with multiple metrics."""
    if n_clusters is None:
        n_clusters = len(np.unique(labels))
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    silhouette = silhouette_score(features_scaled, labels)
    calinski = calinski_harabasz_score(features_scaled, labels)
    davies = davies_bouldin_score(features_scaled, labels)
    
    cluster_sizes = np.bincount(labels) if len(np.unique(labels)) > 1 else [len(labels)]
    
    return {
        'metric': name,
        'n_clusters': n_clusters,
        'silhouette': silhouette,
        'calinski_harabasz': calinski,
        'davies_bouldin': davies,
        'cluster_sizes': cluster_sizes
    }

def single_linkage_clustering(features, n_clusters=2):
    """Apply single linkage hierarchical clustering."""
    clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='single')
    return clustering.fit_predict(features)

def complete_linkage_clustering(features, n_clusters=2):
    """Apply complete linkage hierarchical clustering."""
    clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='complete')
    return clustering.fit_predict(features)

def average_linkage_clustering(features, n_clusters=2):
    """Apply average linkage hierarchical clustering."""
    clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='average')
    return clustering.fit_predict(features)

def kmeans_clustering(features, n_clusters=2, random_state=42):
    """Apply K-Means clustering."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    return kmeans.fit_predict(features)

def analyze_single_metric(df, metric_name, n_clusters=2):
    """Analyze a single health metric with all clustering algorithms."""
    results = []
    
    features = df[[metric_name]].values
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Algorithm 1: Single Linkage Hierarchical
    try:
        labels = single_linkage_clustering(features_scaled, n_clusters)
        if len(np.unique(labels)) > 1:
            result = evaluate_clustering(features, labels, f'{metric_name}_SingleLinkage', n_clusters)
            result['algorithm'] = 'Single Linkage'
            results.append(result)
    except Exception as e:
        pass
    
    # Algorithm 2: Complete Linkage Hierarchical
    try:
        labels = complete_linkage_clustering(features_scaled, n_clusters)
        if len(np.unique(labels)) > 1:
            result = evaluate_clustering(features, labels, f'{metric_name}_CompleteLinkage', n_clusters)
            result['algorithm'] = 'Complete Linkage'
            results.append(result)
    except Exception as e:
        pass
    
    # Algorithm 3: Average Linkage Hierarchical
    try:
        labels = average_linkage_clustering(features_scaled, n_clusters)
        if len(np.unique(labels)) > 1:
            result = evaluate_clustering(features, labels, f'{metric_name}_AverageLinkage', n_clusters)
            result['algorithm'] = 'Average Linkage'
            results.append(result)
    except Exception as e:
        pass
    
    # Algorithm 4: K-Means
    try:
        labels = kmeans_clustering(features_scaled, n_clusters)
        if len(np.unique(labels)) > 1:
            result = evaluate_clustering(features, labels, f'{metric_name}_KMeans', n_clusters)
            result['algorithm'] = 'K-Means'
            results.append(result)
    except Exception as e:
        pass
    
    return results

def analyze_combined_metrics(df, metric_names, n_clusters=2):
    """Analyze combined health metrics with all clustering algorithms."""
    results = []
    
    features = df[metric_names].values
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Algorithm 1: Single Linkage Hierarchical
    try:
        labels = single_linkage_clustering(features_scaled, n_clusters)
        if len(np.unique(labels)) > 1:
            result = evaluate_clustering(features, labels, f'Combined_{n_clusters}clusters', n_clusters)
            result['algorithm'] = 'Single Linkage'
            result['metric'] = ' + '.join(metric_names)
            results.append(result)
    except Exception as e:
        pass
    
    # Algorithm 2: Complete Linkage Hierarchical
    try:
        labels = complete_linkage_clustering(features_scaled, n_clusters)
        if len(np.unique(labels)) > 1:
            result = evaluate_clustering(features, labels, f'Combined_{n_clusters}clusters', n_clusters)
            result['algorithm'] = 'Complete Linkage'
            result['metric'] = ' + '.join(metric_names)
            results.append(result)
    except Exception as e:
        pass
    
    # Algorithm 3: Average Linkage Hierarchical
    try:
        labels = average_linkage_clustering(features_scaled, n_clusters)
        if len(np.unique(labels)) > 1:
            result = evaluate_clustering(features, labels, f'Combined_{n_clusters}clusters', n_clusters)
            result['algorithm'] = 'Average Linkage'
            result['metric'] = ' + '.join(metric_names)
            results.append(result)
    except Exception as e:
        pass
    
    # Algorithm 4: K-Means
    try:
        labels = kmeans_clustering(features_scaled, n_clusters)
        if len(np.unique(labels)) > 1:
            result = evaluate_clustering(features, labels, f'Combined_{n_clusters}clusters', n_clusters)
            result['algorithm'] = 'K-Means'
            result['metric'] = ' + '.join(metric_names)
            results.append(result)
    except Exception as e:
        pass
    
    return results

def run_comprehensive_analysis(df):
    """Run comprehensive analysis of all metrics with all algorithms."""
    print("\n" + "="*80)
    print("COMPREHENSIVE HEALTH METRICS CLUSTERING ANALYSIS")
    print("="*80)
    
    # Get available metrics
    metrics = get_all_numerical_features(df)
    
    all_results = []
    
    # Part 1: Analyze each metric individually with 2 clusters
    print("\n" + "="*80)
    print("PART 1: INDIVIDUAL METRICS ANALYSIS (2 Clusters)")
    print("="*80)
    
    for metric in metrics:
        print(f"\nAnalyzing {metric}...")
        results = analyze_single_metric(df, metric, n_clusters=2)
        all_results.extend(results)
        
        if results:
            best = max(results, key=lambda x: x['silhouette'])
            print(f"  Best algorithm: {best['algorithm']} (Silhouette: {best['silhouette']:.4f})")
    
    # Part 2: Analyze each metric individually with 3 clusters
    print("\n" + "="*80)
    print("PART 2: INDIVIDUAL METRICS ANALYSIS (3 Clusters)")
    print("="*80)
    
    for metric in metrics:
        print(f"\nAnalyzing {metric}...")
        results = analyze_single_metric(df, metric, n_clusters=3)
        all_results.extend(results)
    
    # Part 3: Analyze combined metabolic metrics
    print("\n" + "="*80)
    print("PART 3: COMBINED METRICS ANALYSIS")
    print("="*80)
    
    # Core metabolic metrics
    metabolic_core = ['BMI', 'Blood_Glucose', 'LDL_Cholesterol', 'HDL_Cholesterol']
    metabolic_core = [m for m in metabolic_core if m in df.columns]
    print(f"\nCore Metabolic Metrics: {metabolic_core}")
    results = analyze_combined_metrics(df, metabolic_core, n_clusters=2)
    all_results.extend(results)
    
    # Blood pressure metrics
    bp_metrics = ['Systolic_BP', 'Diastolic_BP']
    bp_metrics = [m for m in bp_metrics if m in df.columns]
    print(f"\nBlood Pressure Metrics: {bp_metrics}")
    results = analyze_combined_metrics(df, bp_metrics, n_clusters=2)
    all_results.extend(results)
    
    # Lipid panel
    lipid_panel = ['Total_Cholesterol', 'LDL_Cholesterol', 'HDL_Cholesterol', 'Triglycerides']
    lipid_panel = [m for m in lipid_panel if m in df.columns]
    print(f"\nLipid Panel: {lipid_panel}")
    results = analyze_combined_metrics(df, lipid_panel, n_clusters=2)
    all_results.extend(results)
    
    # Complete metabolic panel
    metabolic_complete = ['BMI', 'Blood_Glucose', 'Systolic_BP', 'LDL_Cholesterol', 'Triglycerides']
    metabolic_complete = [m for m in metabolic_complete if m in df.columns]
    print(f"\nComplete Metabolic Panel: {metabolic_complete}")
    results = analyze_combined_metrics(df, metabolic_complete, n_clusters=2)
    all_results.extend(results)
    
    return all_results

def generate_summary_report(all_results):
    """Generate comprehensive summary report."""
    print("\n" + "="*80)
    print("SUMMARY REPORT")
    print("="*80)
    
    if not all_results:
        print("No results to analyze!")
        return
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Sort by silhouette score
    results_df = results_df.sort_values('silhouette', ascending=False)
    
    # Top 20 results
    print("\n" + "-"*80)
    print("TOP 20 BEST RESULTS (by Silhouette Score)")
    print("-"*80)
    
    for idx, row in results_df.head(20).iterrows():
        metric = row['metric']
        algo = row['algorithm']
        silhouette = row['silhouette']
        n_clusters = row['n_clusters']
        sizes = row['cluster_sizes']
        calinski = row['calinski_harabasz']
        davies = row['davies_bouldin']
        
        print(f"\n{metric} + {algo} ({n_clusters} clusters)")
        print(f"  Silhouette Score:     {silhouette:.4f}")
        print(f"  Calinski-Harabasz:    {calinski:.2f}")
        print(f"  Davies-Bouldin:       {davies:.4f}")
        print(f"  Cluster Sizes:        {sizes}")
    
    # Analysis by metric
    print("\n" + "-"*80)
    print("BEST RESULTS BY HEALTH METRIC")
    print("-"*80)
    
    # Group by metric (handle combined metrics differently)
    metric_results = results_df[~results_df['metric'].str.contains('\+')]
    if len(metric_results) > 0:
        best_by_metric = metric_results.groupby('metric').apply(
            lambda x: x.loc[x['silhouette'].idxmax()]
        )
        best_by_metric = best_by_metric.sort_values('silhouette', ascending=False)
        
        for idx, row in best_by_metric.iterrows():
            print(f"\n{row['metric']}:")
            print(f"  Best Algorithm: {row['algorithm']}")
            print(f"  Silhouette: {row['silhouette']:.4f} ({row['n_clusters']} clusters)")
            print(f"  Cluster Sizes: {row['cluster_sizes']}")
    
    # Analysis by algorithm
    print("\n" + "-"*80)
    print("BEST RESULTS BY ALGORITHM")
    print("-"*80)
    
    best_by_algo = results_df.groupby('algorithm').apply(
        lambda x: x.loc[x['silhouette'].idxmax()]
    )
    best_by_algo = best_by_algo.sort_values('silhouette', ascending=False)
    
    for idx, row in best_by_algo.iterrows():
        print(f"\n{row['algorithm']}:")
        print(f"  Best Metric: {row['metric']}")
        print(f"  Silhouette: {row['silhouette']:.4f} ({row['n_clusters']} clusters)")
        print(f"  Cluster Sizes: {row['cluster_sizes']}")
    
    # Combined metrics analysis
    print("\n" + "-"*80)
    print("COMBINED METRICS RESULTS")
    print("-"*80)
    
    combined_results = results_df[results_df['metric'].str.contains('\+')]
    if len(combined_results) > 0:
        best_combined = combined_results.sort_values('silhouette', ascending=False)
        for idx, row in best_combined.head(10).iterrows():
            print(f"\n{row['metric']} + {row['algorithm']}:")
            print(f"  Silhouette: {row['silhouette']:.4f}")
            print(f"  Cluster Sizes: {row['cluster_sizes']}")
    
    return results_df

def create_detailed_comparison_table(results_df):
    """Create a detailed comparison table for all metric-algorithm combinations."""
    print("\n" + "="*80)
    print("DETAILED COMPARISON TABLE")
    print("="*80)
    
    # Create pivot table for Silhouette Scores
    if 'metric' in results_df.columns and 'algorithm' in results_df.columns:
        pivot = results_df.pivot_table(
            values='silhouette', 
            index='metric', 
            columns='algorithm',
            aggfunc='max'
        )
        
        print("\nSilhouette Scores by Metric and Algorithm:")
        print(pivot.to_string())
        
        # Save to CSV
        pivot.to_csv(f'{OUTPUT_DIR}silhouette_comparison_matrix.csv')
        print(f"\nComparison matrix saved to: {OUTPUT_DIR}silhouette_comparison_matrix.csv")
    
    # Best configuration for each metric
    print("\n" + "-"*80)
    print("RECOMMENDED CONFIGURATIONS")
    print("-"*80)
    
    # Filter to 2-cluster results only for cleaner recommendations
    two_cluster = results_df[results_df['n_clusters'] == 2]
    
    if len(two_cluster) > 0:
        # Get best for each metric
        metric_recs = two_cluster.groupby('metric').apply(
            lambda x: x.loc[x['silhouette'].idxmax()]
        ).sort_values('silhouette', ascending=False)
        
        print("\nBest Configuration for Each Metric (2 clusters):")
        print("-"*80)
        print(f"{'Metric':<25} {'Algorithm':<18} {'Silhouette':<12} {'Clusters':<10}")
        print("-"*80)
        
        for idx, row in metric_recs.iterrows():
            print(f"{row['metric']:<25} {row['algorithm']:<18} {row['silhouette']:<12.4f} {row['n_clusters']:<10}")
        
        # Overall recommendations
        print("\n" + "-"*80)
        print("OVERALL RECOMMENDATIONS")
        print("-"*80)
        
        best_overall = two_cluster.loc[two_cluster['silhouette'].idxmax()]
        print(f"\n🏆 BEST OVERALL CONFIGURATION:")
        print(f"   Metric: {best_overall['metric']}")
        print(f"   Algorithm: {best_overall['algorithm']}")
        print(f"   Silhouette Score: {best_overall['silhouette']:.4f}")
        
        print(f"\n📊 TOP 5 METRICS FOR CLUSTERING:")
        top5 = metric_recs.head(5)
        for i, (idx, row) in enumerate(top5.iterrows(), 1):
            print(f"   {i}. {row['metric']} + {row['algorithm']}: {row['silhouette']:.4f}")
        
        print(f"\n🎯 TOP 5 ALGORITHMS FOR CLUSTERING:")
        algo_stats = two_cluster.groupby('algorithm').agg({
            'silhouette': ['max', 'mean', 'std'],
            'metric': 'count'
        })
        algo_stats.columns = ['max_silhouette', 'mean_silhouette', 'std_silhouette', 'count']
        algo_stats = algo_stats.sort_values('max_silhouette', ascending=False)
        
        for idx, row in algo_stats.head(5).iterrows():
            print(f"   {idx}: Max={row['max_silhouette']:.4f}, Mean={row['mean_silhouette']:.4f} (±{row['std_silhouette']:.4f})")

def main():
    """Main execution function."""
    print("="*80)
    print("COMPREHENSIVE HEALTH METRICS CLUSTERING ANALYSIS")
    print("="*80)
    print("\nThis script analyzes all numerical health metrics with multiple clustering")
    print("algorithms to identify the best combinations for metabolic health phenotyping.")
    
    # Load data
    df = load_data()
    
    # Run comprehensive analysis
    all_results = run_comprehensive_analysis(df)
    
    # Generate summary report
    results_df = generate_summary_report(all_results)
    
    # Create detailed comparison
    create_detailed_comparison_table(results_df)
    
    # Save all results
    results_df.to_csv(f'{OUTPUT_DIR}all_clustering_results.csv', index=False)
    print(f"\n\nAll results saved to: {OUTPUT_DIR}all_clustering_results.csv")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
