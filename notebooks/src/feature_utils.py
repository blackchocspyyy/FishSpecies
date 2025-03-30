import pandas as pd
import numpy as np
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Set

def select_frequency_columns(df: pd.DataFrame) -> List[str]:
    """
    Select columns starting with F and followed by a number (frequency responses).

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        List[str]: Frequency-related column names.
    """
    return [col for col in df.columns if re.match(r'^F\d+(\.\d+)?$', col)]


def reduce_features_by_variance_and_correlation(
    df: pd.DataFrame, freq_cols: List[str], var_thresh: float = 0.01, corr_thresh: float = 0.95
) -> Tuple[Set[str], pd.DataFrame]:
    """
    Remove low-variance and highly correlated features.

    Args:
        df (pd.DataFrame): Input DataFrame.
        freq_cols (List[str]): List of frequency feature names.
        var_thresh (float): Variance threshold.
        corr_thresh (float): Correlation threshold.

    Returns:
        Tuple[Set[str], pd.DataFrame]: Selected feature names, correlation matrix.
    """
    variance = df[freq_cols].var()
    low_variance_features = variance[variance < var_thresh].index

    corr_matrix = df[freq_cols].corr().abs()
    high_corr_pairs = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > corr_thresh:
                high_corr_pairs.add((corr_matrix.columns[i], corr_matrix.columns[j]))

    selected = set(freq_cols) - set(low_variance_features)
    for f1, f2 in high_corr_pairs:
        if f2 in selected:
            selected.remove(f2)

    return selected, corr_matrix


def get_top_features_by_random_forest(df: pd.DataFrame, features: List[str], target_col: str, top_k: int = 30) -> List[str]:
    """
    Identify most important features using Random Forest algorithm.
    
    Trains a classifier and ranks features by their predictive importance.

    Args:
        df (pd.DataFrame): Input DataFrame.
        features (List[str]): Feature columns.
        target_col (str): Column with class labels.
        top_k (int): Number of top features to return.

    Returns:
        List[str]: List of top K feature names.
    """
    X = df[features]
    y = df[target_col]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_scaled, y)

    importance = pd.Series(rf.feature_importances_, index=features)
    return importance.nlargest(top_k).index.tolist()


def apply_pca(X_scaled: np.ndarray, n_components: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reduce feature dimensionality using Principal Component Analysis (PCA).

    Args:
        X_scaled (np.ndarray): Standardized features.
        n_components (int): Number of components.

    Returns:
        Tuple[np.ndarray, np.ndarray]: PCA-transformed data, explained variance.
    """
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    return X_pca, pca.explained_variance_ratio_
