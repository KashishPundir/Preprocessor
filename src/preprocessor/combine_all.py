from preprocessor.scaling import suggest_scaling
from preprocessor.encoding import suggest_encoding
from preprocessor.missing import suggest_missing_strategy
from .transformation import suggest_transformations
from .datatype_fixer import DataTypeFixer


def auto_preprocess(X, model_name: str):
    dtype_fixer = DataTypeFixer(X)

    required_changes, reasons, message = dtype_fixer.get_required_dtype_changes()

    # Build datatype section cleanly
    if message:
        datatype_section = {
            "message": message
        }
    else:
        datatype_section = {
            "required_dtype_change": required_changes,
            "reason": reasons,
        }

    return {
        "datatype": datatype_section,
        "missing_values": suggest_missing_strategy(X),
        "encoding": suggest_encoding(X, model_name),
        "scaling": suggest_scaling(X, model_name),
        "transformation": suggest_transformations(X, model_name),
    }
