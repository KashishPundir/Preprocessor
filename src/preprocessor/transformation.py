import pandas as pd
import numpy as np
from scipy.stats import skew
from typing import Optional
from collections import defaultdict


# -------------------- MODEL GROUPS --------------------

TRANSFORM_SENSITIVE_MODELS = {
    "linearregression",
    "logisticregression",
    "ridge",
    "lasso",
    "elasticnet",
    "svm",
    "svc",
    "svr",
    "knn",
    "kmeans",
    "pca",
    "mlp",
    "neuralnetwork",
}

TRANSFORM_INSENSITIVE_MODELS = {
    "decisiontree",
    "randomforest",
    "xgboost",
    "lightgbm",
    "catboost",
}


# -------------------- CORE FUNCTION --------------------

def suggest_transformations(
    df: pd.DataFrame,
    model: Optional[str] = None,
    target: Optional[str] = None,
    skew_threshold: float = 1.0
) -> dict:
    """
    Expert-level transformation suggestion engine.
    Groups columns by transformation type and
    generates clean, scalable code.
    """

    numeric_cols = df.select_dtypes(include="number").columns
    model_name = model.lower().replace(" ", "") if model else None

    # --------- Model-based early exit ---------
    if model_name in TRANSFORM_INSENSITIVE_MODELS:
        return {
            "message": (
                f"ℹ️ Model '{model}' is tree-based and "
                "does not require feature transformations."
            )
        }

    strategies = {}
    grouped_cols = defaultdict(list)
    reasons = []

    # -------------------- DETECTION LOOP --------------------
    for col in numeric_cols:

        if target and col == target:
            continue

        data = df[col].dropna()
        if data.empty:
            continue

        if data.nunique() <= 2:
            continue

        if data.nunique() / len(data) > 0.95:
            continue

        col_skew = skew(data)

        if abs(col_skew) <= 0.5:
            continue

        # --------- Decide transformation ---------
        if col_skew > skew_threshold:
            if (data <= 0).any():
                method = "Yeo-Johnson Transformation"
            else:
                method = "Log Transformation (log1p)"

        elif col_skew < -skew_threshold:
            method = "Yeo-Johnson Transformation"

        else:
            method = "Square Root Transformation"

        strategies[col] = method
        grouped_cols[method].append(col)

        reasons.append(
            f"{col} has skewness = {round(col_skew, 2)}. "
            "Transformation improves variance stability and model reliability."
        )

    # -------------------- NO TRANSFORM CASE --------------------
    if not strategies:
        return {
            "message": (
                "✅ No features require transformation based on "
                "distribution shape, data type, and selected model."
            )
        }

    # -------------------- GROUPED CODE GENERATION --------------------
    code_blocks = []

    if "Square Root Transformation" in grouped_cols:
        cols = grouped_cols["Square Root Transformation"]
        code_blocks.append(
            "import numpy as np\n"
            f"sqrt_cols = {cols}\n"
            "for col in sqrt_cols:\n"
            "    df[f'{col}_sqrt'] = np.sqrt(df[col])"
        )

    if "Log Transformation (log1p)" in grouped_cols:
        cols = grouped_cols["Log Transformation (log1p)"]
        code_blocks.append(
            "import numpy as np\n"
            f"log_cols = {cols}\n"
            "for col in log_cols:\n"
            "    df[f'{col}_log'] = np.log1p(df[col])"
        )

    if "Yeo-Johnson Transformation" in grouped_cols:
        cols = grouped_cols["Yeo-Johnson Transformation"]
        code_blocks.append(
            "from sklearn.preprocessing import PowerTransformer\n"
            "pt = PowerTransformer(method='yeo-johnson')\n"
            f"yj_cols = {cols}\n"
            "for col in yj_cols:\n"
            "    df[f'{col}_yj'] = pt.fit_transform(df[[col]])"
        )

    # -------------------- FINAL RETURN --------------------
    return {
        "strategy": strategies,
        "reason": (
            f"Transformations are suggested because the model '{model}' "
            "is sensitive to feature distribution and scale.\n\n"
            + "\n".join(reasons)
        ),
        "code": "\n\n".join(code_blocks),
    }
