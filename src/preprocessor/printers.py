def pretty_print(result: dict, *, return_output: bool = False):
    lines = []
    lines.append("\n🧠 Intelligent Data Preprocessing Report")
    lines.append("📌 Recommended execution order to avoid errors:\n")

    # Strict preprocessing execution order (NON-NEGOTIABLE)
    execution_order = [
        "missing_values",
        "outliers",
        "transformation",
        "encoding",
        "scaling",
    ]

    # Normalize keys once (safe & deterministic)
    normalized = {k.lower(): k for k in result.keys()}

    step_number = 1
    printed_sections = set()

    # --------- MAIN ORDERED SECTIONS ---------
    for key in execution_order:
        if key not in normalized:
            continue

        section = normalized[key]
        details = result[section]
        printed_sections.add(section)

        lines.append(f"\n{step_number}️⃣ {section.upper()}")
        step_number += 1

        # Message-only section (e.g., no missing values)
        if isinstance(details, dict) and "message" in details:
            lines.append(details["message"])
            continue

        # Strategy-based sections
        if isinstance(details, dict):

            for col, method in details.get("strategy", {}).items():
                lines.append(f"• {col} → {method}")

            for col, method in details.get("encodings", {}).items():
                lines.append(f"• {col} → {method}")

            if details.get("scaler"):
                lines.append(f"Scaler → {details['scaler']}")

            if details.get("reason"):
                lines.append("\nReason:")
                lines.append(str(details["reason"]))

            if details.get("code"):
                lines.append("\nSuggested Code:")
                lines.append("-" * 40)
                lines.append(details["code"].strip())
                lines.append("-" * 40)

    # --------- NON-PIPELINE / EXTRA SECTIONS ---------
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
