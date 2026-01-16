"""Classification module for health phenotype discovery."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from xgboost import XGBClassifier


def prepare_data_for_classification(df: pd.DataFrame, target_column: str, test_size: float = 0.2, random_state: int = 42):
    """
    Prepare data for classification.
    
    Args:
        df: Input DataFrame
        target_column: Name of target variable
        test_size: Proportion of test set
        random_state: Random state for reproducibility
        
    Returns:
        Train/test split data
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test


def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series, params: dict = None) -> RandomForestClassifier:
    """
    Train Random Forest classifier.
    
    Args:
        X_train: Training features
        y_train: Training labels
        params: Hyperparameters
        
    Returns:
        Trained Random Forest model
    """
    default_params = {
        'n_estimators': 100,
        'max_depth': 10,
        'random_state': 42
    }
    params = {**default_params, **(params or {})}
    
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    
    return model


def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series, params: dict = None) -> XGBClassifier:
    """
    Train XGBoost classifier.
    
    Args:
        X_train: Training features
        y_train: Training labels
        params: Hyperparameters
        
    Returns:
        Trained XGBoost model
    """
    default_params = {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'random_state': 42,
        'use_label_encoder': False,
        'eval_metric': 'logloss'
    }
    params = {**default_params, **(params or {})}
    
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    
    return model


def train_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series, params: dict = None) -> LogisticRegression:
    """
    Train Logistic Regression classifier.
    
    Args:
        X_train: Training features
        y_train: Training labels
        params: Hyperparameters
        
    Returns:
        Trained Logistic Regression model
    """
    default_params = {
        'max_iter': 1000,
        'random_state': 42
    }
    params = {**default_params, **(params or {})}
    
    model = LogisticRegression(**params)
    model.fit(X_train, y_train)
    
    return model


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Evaluate classification model.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        
    Returns:
        Dictionary with evaluation metrics
    """
    y_pred = model.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
    }
    
    return metrics


def cross_validate_model(model, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> dict:
    """
    Perform cross-validation on model.
    
    Args:
        model: Model to evaluate
        X: Feature matrix
        y: Labels
        cv: Number of folds
        
    Returns:
        Dictionary with CV results
    """
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    
    return {
        'cv_mean': scores.mean(),
        'cv_std': scores.std()
    }
