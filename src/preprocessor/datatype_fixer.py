import pandas as pd
import numpy as np

class DataTypeFixer:
    def __init__(self, df: pd.DataFrame, threshold: float = 0.7):
        self.df = df.copy()
        self.threshold = threshold

    def _is_numeric(self, series):
        coerced = pd.to_numeric(series, errors="coerce")
        return coerced.notna().mean()

    def _is_integer(self, series):
        coerced = pd.to_numeric(series, errors="coerce")
        if coerced.notna().mean() == 0:
            return 0
        return (coerced.dropna() % 1 == 0).mean()

    def _is_datetime(self, series):
        coerced = pd.to_datetime(series, errors="coerce")
        return coerced.notna().mean()


    def detect_best_dtype(self, series):
        scores = {
            "int": self._is_integer(series),
            "float": self._is_numeric(series),
            "datetime": self._is_datetime(series)
        }

        best_type = max(scores, key=scores.get)
        best_score = scores[best_type]

        if best_score >= self.threshold:
            return best_type, best_score
        return "object", best_score

    # ✅ NOW INSIDE THE CLASS
    def get_required_dtype_changes(self):
        required = {}
        reasons = {}

        for col in self.df.columns:
            current_dtype = str(self.df[col].dtype)

            if self.df[col].isna().all():
                continue

            suggested, confidence = self.detect_best_dtype(self.df[col])

            # ❌ Suppress float → int suggestions
            if current_dtype.startswith("float") and suggested == "int":
                continue

            if suggested != "object" and suggested not in current_dtype:
                required[col] = f"{current_dtype} → {suggested}"
                reasons[col] = f"{round(confidence * 100, 1)}% values match {suggested}"

        if not required:
            return {}, {}, "No misleading datatypes detected"

        return required, reasons, None


