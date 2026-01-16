#!/usr/bin/env python3
"""
Main analysis script for health phenotype discovery project.
This script runs the complete analysis pipeline on NHANES data.
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.load_data import load_data
from data.preprocess import preprocess_data
from models.clustering import perform_kmeans_clustering, find_optimal_clusters
from models.classification import train_random_forest, evaluate_model, prepare_data_for_classification
from models.interpretability import get_feature_importance, plot_feature_importance
from utils.visualization import set_plotting_style, generate_summary_statistics
import yaml


def main():
    """Run the complete analysis pipeline."""
    
    print("=" * 60)
    print("Health Phenotype Discovery - Analysis Pipeline")
    print("=" * 60)
    
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'default_config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set plotting style
    set_plotting_style()
    
    # Load data
    print("\n[1/5] Loading NHANES data...")
    data_path = config['data']['raw_path']
    df = load_data(data_path)
    print(f"    Loaded {len(df)} samples with {len(df.columns)} features")
    
    # Generate summary statistics
    print("\n[2/5] Generating summary statistics...")
    summary = generate_summary_statistics(df)
    print(f"    Summary statistics generated for {len(summary)} variables")
    
    # Preprocess data
    print("\n[3/5] Preprocessing data...")
    df_processed = preprocess_data(df, config['preprocessing'])
    print(f"    Preprocessed data shape: {df_processed.shape}")
    
    # Clustering analysis
    print("\n[4/5] Performing clustering analysis...")
    X = df_processed.values
    
    # Find optimal number of clusters
    cluster_metrics = find_optimal_clusters(X, max_clusters=10)
    optimal_k = cluster_metrics['k_range'][cluster_metrics['silhouette_scores'].index(max(cluster_metrics['silhouette_scores']))]
    print(f"    Optimal number of clusters: {optimal_k}")
    
    # Perform clustering with optimal k
    cluster_labels = perform_kmeans_clustering(X, n_clusters=optimal_k)
    df_processed['Cluster'] = cluster_labels
    print(f"    Clustering complete with {optimal_k} clusters")
    
    # Classification analysis
    print("\n[5/5] Performing classification analysis...")
    # Use a binary target (e.g., Diabetes status) if available
    target_column = 'Diabetes'
    
    if target_column in df_processed.columns:
        X_train, X_test, y_train, y_test = prepare_data_for_classification(
            df_processed, target_column, 
            test_size=config['data']['test_size'],
            random_state=config['data']['random_state']
        )
        
        # Train model
        model = train_random_forest(X_train, y_train)
        
        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test)
        print(f"    Classification Accuracy: {metrics['accuracy']:.4f}")
        
        # Feature importance
        feature_importance = get_feature_importance(model, df_processed.columns.tolist())
        plot_feature_importance(feature_importance, top_n=20, save_path='reports/feature_importance.png')
        print(f"    Feature importance plot saved to reports/feature_importance.png")
    
    print("\n" + "=" * 60)
    print("Analysis Pipeline Complete!")
    print("=" * 60)
    print("\nResults saved to:")
    print("  - reports/feature_importance.png")
    print("  - data/processed/ (processed datasets)")


if __name__ == "__main__":
    main()
