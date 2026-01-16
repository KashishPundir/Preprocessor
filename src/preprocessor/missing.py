
import pandas as pd

def outlier_percent(series):
    series = series.dropna()

    if series.empty:
        return 0.0

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1

    if iqr == 0:
        return 0.0

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    return ((series < lower) | (series > upper)).mean() * 100


def suggest_missing_strategy(X: pd.DataFrame):
    strategies = {}
    reasons = {}
    code_blocks = []

    for col in X.columns:
        missing_ratio = X[col].isna().mean()

        if missing_ratio == 0:
            continue

        # Drop column if too many missing values
        if missing_ratio > 0.4:
            strategies[col] = "Drop Column"
            reasons[col] = "More than 40% missing values."
            code_blocks.append(f"X = X.drop(columns=['{col}'])")
            continue

        # Numeric columns
        if pd.api.types.is_numeric_dtype(X[col]):
            unique_vals = set(X[col].dropna().unique())

            # Binary numeric feature
            if unique_vals.issubset({0, 1}):
                strategies[col] = "Mode Imputation (Binary Feature)"
                reasons[col] = "Binary feature (0/1) stored as numeric."
                code_blocks.append(
                    f"X['{col}'] = X['{col}'].fillna(X['{col}'].mode()[0])"
                )
                continue

            # Outlier-based decision
            p = outlier_percent(X[col])

            if p <= 5:
                strategies[col] = "Mean Imputation"
                reasons[col] = "Outliers ≤5%. Mean remains stable and efficient."
                code_blocks.append(
                    f"X['{col}'] = X['{col}'].fillna(X['{col}'].mean())"
                )

            elif p <= 30:
                strategies[col] = "Median Imputation"
                reasons[col] = "Outliers >5%. Median is more robust than mean."
                code_blocks.append(
                    f"X['{col}'] = X['{col}'].fillna(X['{col}'].median())"
                )

            else:
                strategies[col] = "Model-Based / Drop Column"
                reasons[col] = "Extreme outliers (>30%). Simple statistics unreliable."
                code_blocks.append(
                    f"# Consider KNN/MICE or dropping '{col}'"
                )

        # Categorical columns
        else:
            strategies[col] = "Most Frequent"
            reasons[col] = "Categorical feature."
            code_blocks.append(
                f"X['{col}'] = X['{col}'].fillna(X['{col}'].mode()[0])"
            )

    if not strategies:
        return {"message": "No missing values exist in the dataset."}

    return {
        "strategy": strategies,
        "reason": reasons,
        "code": "\n".join(code_blocks)
    }
