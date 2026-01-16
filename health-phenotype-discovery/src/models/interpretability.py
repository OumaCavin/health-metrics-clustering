"""Interpretability module for health phenotype discovery."""

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt


def get_feature_importance(model, feature_names: list) -> pd.DataFrame:
    """
    Get feature importance from model.
    
    Args:
        model: Trained model
        feature_names: List of feature names
        
    Returns:
        DataFrame with feature importances
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_)
    else:
        raise ValueError("Model does not have feature importances")
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    return importance_df


def calculate_shap_values(model, X_sample: pd.DataFrame) -> np.ndarray:
    """
    Calculate SHAP values for model explanation.
    
    Args:
        model: Trained model
        X_sample: Sample of features for SHAP calculation
        
    Returns:
        SHAP values array
    """
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        
        # For binary classification, take the positive class SHAP values
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        return shap_values
    except Exception as e:
        print(f"SHAP calculation error: {e}")
        return None


def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 20, save_path: str = None):
    """
    Plot feature importance.
    
    Args:
        importance_df: DataFrame with feature importances
        top_n: Number of top features to display
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 8))
    
    top_features = importance_df.head(top_n)
    
    plt.barh(range(len(top_features)), top_features['importance'].values)
    plt.yticks(range(len(top_features)), top_features['feature'].values)
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title(f'Top {top_n} Feature Importances')
    plt.gca().invert_yaxis()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.close()


def plot_shap_summary(shap_values: np.ndarray, feature_names: list, save_path: str = None):
    """
    Plot SHAP summary.
    
    Args:
        shap_values: SHAP values array
        feature_names: List of feature names
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 8))
    
    shap.summary_plot(shap_values, feature_names=feature_names, show=False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.close()
