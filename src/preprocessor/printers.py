from IPython.display import display, HTML

def pretty_print(result: dict, *, return_output: bool = False):
    lines = []


    display(HTML("""
    <h1 style="
        text-align: center;
        font-size: 20px;
        font-weight: bold;
    ">
    Data Preprocessing Suggestion Report
    </h1>
    """))


    # ✅ Correct & complete pipeline order
    execution_order = [
        "datatype",
        "missing_values",
        "outliers",
        "transformation",
        "encoding",
        "scaling",
    ]

    # Normalize keys (case-insensitive safety)
    normalized = {k.lower(): k for k in result.keys()}
    step_number = 1
    printed_sections = set()

    # --------- ORDERED PIPELINE SECTIONS ---------
    for key in execution_order:
        if key not in normalized:
            continue

        section = normalized[key]
        details = result[section]
        printed_sections.add(section)

        lines.append(f"\n{step_number}️⃣ {section.upper()}")
        step_number += 1

        # --------- MESSAGE-ONLY SECTIONS ---------
        if isinstance(details, dict) and "message" in details:
            lines.append(details["message"])
            continue

        # --------- DATATYPE SECTION (NO CODE) ---------
        if section.lower() == "datatype":
            changes = details.get("required_dtype_change", {})
            reasons = details.get("reason", {})

            if not changes:
                lines.append("No misleading datatypes detected.")
                continue

            for col, change in changes.items():
                reason = reasons.get(col, "")
                lines.append(f"• {col} → {change} ({reason})")

            continue

        # --------- TRANSFORMATION SECTION (GROUPED CODE) ---------
        if section.lower() == "transformation":
            strategies = details.get("strategy", {})
            code = details.get("code", "")

            if not strategies:
                lines.append("No transformations required.")
                continue

            for col, method in strategies.items():
                lines.append(f"• {col} → {method}")

            if code:
                lines.append("\nSuggested Code:")
                lines.append("-" * 40)
                lines.append(code.strip())
                lines.append("-" * 40)

            continue

        # --------- ENCODING SECTION ---------
        if section.lower() == "encoding":
            encodings = details.get("encodings", {})
            reasons = details.get("reason", {})
            code = details.get("code", "")

            for col, method in encodings.items():
                reason = reasons.get(col, "")
                lines.append(f"• {col} → {method} ({reason})")

            if code:
                lines.append("\nSuggested Code:")
                lines.append("-" * 100)
                lines.append(code.strip())
                lines.append("-" * 100)

            continue

        # --------- OTHER STRATEGY SECTIONS (scaling, missing, outliers) ---------
        if isinstance(details, dict):
            for col, method in details.get("strategy", {}).items():
                lines.append(f"• {col} → {method}")

            if details.get("scaler"):
                lines.append(f"Scaler → {details['scaler']}")

            if details.get("code"):
                lines.append("\nSuggested Code:")
                lines.append("-" * 40)
                lines.append(details["code"].strip())
                lines.append("-" * 40)

    # --------- EXTRA / NON-PIPELINE SECTIONS ---------
    for section, details in result.items():
        if section in printed_sections:
            continue

        lines.append(f"\n🔹 {section.upper()}")

        if isinstance(details, dict) and "message" in details:
            lines.append(details["message"])

    output = "\n".join(lines)

    if return_output:
        return output
    else:
        print(output)


