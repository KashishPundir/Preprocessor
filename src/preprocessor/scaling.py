import numpy as np
import pandas as pd
from scipy.stats import skew
from typing import List, Optional


TREE_MODELS = {
    "decision_tree",
    "random_forest",
    "xgboost",
    "lightgbm",
    "catboost"
}

SCALE_SENSITIVE_MODELS = {
    "linear_regression",
    "logistic_regression",
    "ridge",
    "lasso",
    "elasticnet",
    "svm",
    "svc",
    "svr",
    "knn",
    "kmeans",
    "dbscan",
    "mlp",
    "neural_network"
}


def detect_outliers_iqr(series: pd.Series) -> float:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return ((series < lower) | (series > upper)).mean()


def suggest_scaling(
    X: pd.DataFrame,
    model_name: str,
    original_numeric_cols: Optional[List[str]] = None
):
    """
    Suggest scaling ONLY for original continuous numeric features.
    """

    result = {
        "scale": False,
        "scaler": None,
        "columns_scaled": [],
        "reason": "",
        "code": ""
    }

    model_name = model_name.lower()

    # -------- Model-based early exit --------
    if model_name in TREE_MODELS:
        result["reason"] = (
            "Tree-based models are scale-invariant. "
            "Scaling is not required."
        )
        return result

    if model_name not in SCALE_SENSITIVE_MODELS:
        result["reason"] = (
            "Model not recognized as scale-sensitive. "
            "Scaling left to user discretion."
        )
        return result

    # -------- Identify scale-eligible columns --------
    if original_numeric_cols is None:
        original_numeric_cols = X.select_dtypes(include=np.number).columns.tolist()

    # Exclude binary numeric features
    continuous_numeric_cols = [
        c for c in original_numeric_cols
        if c in X.columns and X[c].nunique() > 2
    ]

    if not continuous_numeric_cols:
        result["reason"] = "No continuous numeric features eligible for scaling."
        return result

    numeric_data = X[continuous_numeric_cols]

    outlier_ratios = numeric_data.apply(detect_outliers_iqr)
    skewness = numeric_data.apply(skew)

    heavy_outliers = (outlier_ratios > 0.05).any()

    result["scale"] = True
    result["columns_scaled"] = continuous_numeric_cols

    # -------- Choose scaler --------
    if heavy_outliers:
        result["scaler"] = "RobustScaler"
        result["reason"] = (
            "Heavy outliers detected in continuous numeric features. "
            "RobustScaler is recommended."
        )
        result["code"] = f"""
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
X[{continuous_numeric_cols}] = scaler.fit_transform(X[{continuous_numeric_cols}])
"""
    else:
        result["scaler"] = "StandardScaler"
        result["reason"] = (
            "No significant outliers detected. "
            "StandardScaler is recommended."
        )
        result["code"] = f"""
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X[{continuous_numeric_cols}] = scaler.fit_transform(X[{continuous_numeric_cols}])
"""

    return result
