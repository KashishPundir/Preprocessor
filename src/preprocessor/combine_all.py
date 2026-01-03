from preprocessor.scaling import suggest_scaling
from preprocessor.encoding import suggest_encoding
from preprocessor.missing import suggest_missing_strategy
from .transformation import suggest_transformations


def auto_preprocess(X, model_name: str):
    """
    Run all preprocessing advisors and return a unified report.
    """
    return {
        "missing_values": suggest_missing_strategy(X),
        "encoding": suggest_encoding(X, model_name),
        "scaling": suggest_scaling(X, model_name),
        "transformation": suggest_transformations(X, model_name)
    }
