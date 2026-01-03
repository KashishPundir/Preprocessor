import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from typing import Optional


def suggest_encoding(
    X: pd.DataFrame,
    model_name: str,
    target: Optional[str] = None
):
    """
    Suggest safe categorical encoding strategies.
    Does NOT encode numeric features or target variable.
    """

    result = {
        "encodings": {},
        "reason": {},
        "code": ""
    }

    categorical_cols = X.select_dtypes(include=["object", "category"]).columns

    if target:
        categorical_cols = [c for c in categorical_cols if c != target]

    if len(categorical_cols) == 0:
        return {"message": "No categorical features detected."}

    code_blocks = []

    for col in categorical_cols:
        unique_count = X[col].nunique(dropna=True)

        # Binary categorical
        if unique_count == 2:
            result["encodings"][col] = "LabelEncoder"
            result["reason"][col] = "Binary categorical feature."

            code_blocks.append(f"""
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X['{col}'] = le.fit_transform(X['{col}'])
""")

        # Low-cardinality nominal
        elif unique_count <= 10:
            result["encodings"][col] = "OneHotEncoder"
            result["reason"][col] = "Low-cardinality nominal feature."

            code_blocks.append(f"""
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False, drop='first', handle_unknown='ignore')
encoded = ohe.fit_transform(X[['{col}']])
encoded_df = pd.DataFrame(
    encoded,
    columns=ohe.get_feature_names_out(['{col}']),
    index=X.index
)
X = pd.concat([X.drop(columns=['{col}']), encoded_df], axis=1)
""")

        # High-cardinality categorical
        else:
            result["encodings"][col] = "Frequency Encoding"
            result["reason"][col] = "High-cardinality categorical feature."

            code_blocks.append(f"""
freq_map = X['{col}'].value_counts(normalize=True)
X['{col}'] = X['{col}'].map(freq_map)
""")

    result["code"] = "\n".join(code_blocks)
    return result
