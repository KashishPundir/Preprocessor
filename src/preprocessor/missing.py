import pandas as pd

def suggest_missing_strategy(X: pd.DataFrame):
    """
    Suggest strategies for handling missing values in a DataFrame.
    """

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

            code_blocks.append(
                f"X = X.drop(columns=['{col}'])"
            )
            continue

        # Numerical columns
        if pd.api.types.is_numeric_dtype(X[col]):
            if abs(X[col].skew()) > 1:
                strategies[col] = "Median Imputation"
                reasons[col] = "Skewed numerical distribution."

                code_blocks.append(
                    f"X['{col}'] = X['{col}'].fillna(X['{col}'].median())"
                )
            else:
                strategies[col] = "Mean Imputation"
                reasons[col] = "Approximately symmetric numerical distribution."

                code_blocks.append(
                    f"X['{col}'] = X['{col}'].fillna(X['{col}'].mean())"
                )

        # Categorical columns
        else:
            strategies[col] = "Most Frequent"
            reasons[col] = "Categorical feature."

            code_blocks.append(
                f"X['{col}'] = X['{col}'].fillna(X['{col}'].mode()[0])"
            )

    # ✅ Handle NO missing values case
    if not strategies:
        return {
            "message": "No missing values exist in the dataset."
        }

    return {
        "strategy": strategies,
        "reason": reasons,
        "code": "\n".join(code_blocks)
    }
