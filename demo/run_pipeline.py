import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
sys.path.append(SRC_PATH)

import pandas as pd
from preprocessor import auto_preprocess, pretty_print

df = pd.DataFrame({
    "age": [22, 25, None, 30, 28],
    "salary": [40000, 50000, 60000, None, 52000],
    "gender": ["M", "F", "F", None, "M"],
    "city": ["Delhi", "Mumbai", "Delhi", "Chennai", "Mumbai"]
})

report = auto_preprocess(df, model_name="logistic_regression")
text = pretty_print(report)
print(text)

