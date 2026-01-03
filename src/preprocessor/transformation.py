import pandas as pd
import numpy as np
from scipy.stats import skew
from typing import Optional


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
    Suggests transformations only when statistically
    and model-wise justified.
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
    reasons = []
    code_blocks = []

    for col in numeric_cols:

        # Skip target variable
        if target and col == target:
            continue

        data = df[col].dropna()
        if data.empty:
            continue

        # Skip binary numeric features
        if data.nunique() <= 2:
            continue

        # Skip ID-like columns
        if data.nunique() / len(data) > 0.95:
            continue

        col_skew = skew(data)

        # Nearly symmetric → no transform
        if abs(col_skew) <= 0.5:
            continue

        # ---------- Decide transformation ----------
        if col_skew > skew_threshold:

            if (data <= 0).any():
                method = "Yeo-Johnson Transformation"
                code = (
                    "from sklearn.preprocessing import PowerTransformer\n"
                    "pt = PowerTransformer(method='yeo-johnson')\n"
                    f"df['{col}_yj'] = pt.fit_transform(df[['{col}']])"
                )

            else:
                method = "Log Transformation (log1p)"
                code = (
                    "import numpy as np\n"
                    f"df['{col}_log'] = np.log1p(df['{col}'])"
                )

        elif col_skew < -skew_threshold:
            method = "Yeo-Johnson Transformation"
            code = (
                "from sklearn.preprocessing import PowerTransformer\n"
                "pt = PowerTransformer(method='yeo-johnson')\n"
                f"df['{col}_yj'] = pt.fit_transform(df[['{col}']])"
            )

        else:
            method = "Square Root Transformation"
            code = (
                "import numpy as np\n"
                f"df['{col}_sqrt'] = np.sqrt(df['{col}'])"
            )

        strategies[col] = method
        reasons.append(
            f"{col} has skewness = {round(col_skew, 2)}. "
            "Transformation improves variance stability and model reliability."
        )
        code_blocks.append(f"# {col}\n{code}")

    if not strategies:
        return {
            "message": (
                "✅ No features require transformation based on "
                "distribution shape, data type, and selected model."
            )
        }

    return {
        "strategy": strategies,
        "reason": (
            f"Transformations are suggested because the model '{model}' "
            "is sensitive to feature distribution and scale.\n\n"
            + "\n".join(reasons)
        ),
        "code": "\n\n".join(code_blocks),
    }
