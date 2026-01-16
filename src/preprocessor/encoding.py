

import pandas as pd
from typing import Optional


def suggest_encoding(
    X: pd.DataFrame,
    model_name: str = "tree",
    target: Optional[str] = None
):
    """
    Suggest intelligent encoding strategies with grouped executable code.
    """

    result = {
        "encodings": {},
        "reason": {},
        "code": ""
    }

    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if target and target in categorical_cols:
        categorical_cols.remove(target)

    if not categorical_cols:
        return {"message": "No categorical features detected."}

    # ---------- GROUP CONTAINERS ----------
    onehot_cols = []
    freq_cols = []
    numeric_extract_cols = []
    binary_cols = {}
    ordinal_cols = {}
    generation_cols = []
    warranty_cols = []

    for col in categorical_cols:
        unique_count = X[col].nunique(dropna=True)
        col_lower = col.lower()
        sample = X[col].dropna().astype(str).head(10).tolist()

        # Binary
        if unique_count == 2:
            result["encodings"][col] = "Binary Encoding (0/1)"
            result["reason"][col] = "Binary categorical feature."
            vals = X[col].dropna().unique().tolist()
            binary_cols[col] = {vals[0]: 0, vals[1]: 1}

        # Processor generation
        elif col_lower in ["processor_gnrtn", "processor_generation"]:
            result["encodings"][col] = "Ordinal Numeric Extraction"
            result["reason"][col] = "CPU generation follows numeric order."
            generation_cols.append(col)

        # Warranty
        elif "warranty" in col_lower:
            result["encodings"][col] = "Ordinal Numeric Extraction"
            result["reason"][col] = "Warranty duration represents ordered time."
            warranty_cols.append(col)

        # Rating
        elif "rating" in col_lower:
            result["encodings"][col] = "Ordinal Encoding"
            result["reason"][col] = "Ratings represent ordered quality levels."
            ordinal_cols[col] = "extract"

        # Numeric disguised as text
        elif any(("gb" in v.lower() or "tb" in v.lower()) for v in sample):
            result["encodings"][col] = "Numeric Extraction"
            result["reason"][col] = "Numeric feature stored as text."
            numeric_extract_cols.append(col)

        # Low-cardinality nominal
        elif unique_count <= 10:
            result["encodings"][col] = "OneHotEncoder"
            result["reason"][col] = "Low-cardinality nominal feature."
            onehot_cols.append(col)

        # High-cardinality
        else:
            result["encodings"][col] = "Frequency Encoding"
            result["reason"][col] = "High-cardinality categorical feature."
            freq_cols.append(col)

    # ---------- CODE GENERATION (GROUPED) ----------
    code = []

    if onehot_cols:
        code.append(f"""
from sklearn.preprocessing import OneHotEncoder

onehot_cols = {onehot_cols}
ohe = OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore')

encoded = ohe.fit_transform(X[onehot_cols])
encoded_df = pd.DataFrame(
    encoded,
    columns=ohe.get_feature_names_out(onehot_cols),
    index=X.index
)

X = pd.concat([X.drop(columns=onehot_cols), encoded_df], axis=1)
""")

    if freq_cols:
        code.append(f"""
# Frequency Encoding
for col in {freq_cols}:
    freq_map = X[col].value_counts(normalize=True)
    X[col] = X[col].map(freq_map)
""")

    if numeric_extract_cols:
        code.append(f"""
# Numeric Extraction (GB / TB)
for col in {numeric_extract_cols}:
    X[col] = (
        X[col].astype(str)
        .str.extract(r'(\\d+)')
        .astype(float)
    )
""")

    if generation_cols:
        code.append(f"""
# Processor Generation
for col in {generation_cols}:
    X[col] = (
        X[col].astype(str)
        .str.extract(r'(\\d+)')
        .astype(float)
    )
""")

    if warranty_cols:
        code.append(f"""
# Warranty (years)
for col in {warranty_cols}:
    X[col] = (
        X[col].astype(str)
        .str.extract(r'(\\d+)')
        .astype(float)
        .fillna(0)
    )
""")

    if binary_cols:
        code.append(f"""
# Binary Encoding
binary_maps = {binary_cols}
for col, mapping in binary_maps.items():
    X[col] = X[col].map(mapping)
""")

    result["code"] = "\n".join(code)
    return result
