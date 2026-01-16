"""Data preprocessing module for health phenotype discovery."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder


def handle_missing_values(df: pd.DataFrame, strategy: str = 'median') -> pd.DataFrame:
    """
    Handle missing values in the dataset.
    
    Args:
        df: Input DataFrame
        strategy: Strategy for imputation ('mean', 'median', 'mode', 'drop')
        
    Returns:
        DataFrame with missing values handled
    """
    df = df.copy()
    
    for column in df.columns:
        if df[column].isnull().sum() > 0:
            if strategy == 'mean':
                df[column].fillna(df[column].mean(), inplace=True)
            elif strategy == 'median':
                df[column].fillna(df[column].median(), inplace=True)
            elif strategy == 'mode':
                df[column].fillna(df[column].mode()[0], inplace=True)
            elif strategy == 'drop':
                df.dropna(subset=[column], inplace=True)
    
    return df


def encode_categorical(df: pd.DataFrame, encoding: str = 'label') -> pd.DataFrame:
    """
    Encode categorical variables.
    
    Args:
        df: Input DataFrame
        encoding: Encoding method ('label', 'onehot')
        
    Returns:
        DataFrame with encoded categorical variables
    """
    df = df.copy()
    
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    
    for column in categorical_columns:
        if encoding == 'label':
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column].astype(str))
        elif encoding == 'onehot':
            dummies = pd.get_dummies(df[column], prefix=column)
            df = pd.concat([df, dummies], axis=1)
            df.drop(column, axis=1, inplace=True)
    
    return df


def scale_features(df: pd.DataFrame, exclude: list = None) -> pd.DataFrame:
    """
    Scale numerical features.
    
    Args:
        df: Input DataFrame
        exclude: Columns to exclude from scaling
        
    Returns:
        DataFrame with scaled features
    """
    df = df.copy()
    exclude = exclude or []
    
    numerical_columns = df.select_dtypes(include=[np.number]).columns
    columns_to_scale = [col for col in numerical_columns if col not in exclude]
    
    scaler = StandardScaler()
    df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
    
    return df


def preprocess_data(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Complete preprocessing pipeline.
    
    Args:
        df: Raw input DataFrame
        config: Preprocessing configuration
        
    Returns:
        Preprocessed DataFrame
    """
    # Handle missing values
    df = handle_missing_values(df, strategy=config.get('missing_value_strategy', 'median'))
    
    # Encode categorical variables
    df = encode_categorical(df, encoding=config.get('categorical_encoding', 'onehot'))
    
    # Scale features
    df = scale_features(df)
    
    return df
