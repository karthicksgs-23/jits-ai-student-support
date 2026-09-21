import re


# =========================================================
# BASIC INPUT GUARDRAILS
# =========================================================

def validate_user_query(question: str) -> tuple[bool, str]:
    """
    Validate the incoming customer query.

    Returns:
        (True, "") if query is allowed
        (False, reason) if query should be rejected
    """

    if question is None:
        return False, "Question cannot be empty."

    question = question.strip()

    # -----------------------------------------------------
    # 1. Empty query
    # -----------------------------------------------------

    if not question:
        return False, "Question cannot be empty."


    # -----------------------------------------------------
    # 2. Extremely long query
    # -----------------------------------------------------

    if len(question) > 2000:
        return (
            False,
            "Question is too long. Please ask a shorter question."
        )


    # -----------------------------------------------------
    # 3. Prompt injection patterns
    # -----------------------------------------------------

    injection_patterns = [

        r"ignore\s+(all\s+)?previous\s+instructions",

        r"ignore\s+(all\s+)?instructions",

        r"reveal\s+(your\s+)?system\s+prompt",

        r"show\s+(your\s+)?system\s+prompt",

        r"print\s+(your\s+)?system\s+prompt",

        r"developer\s+message",

        r"hidden\s+instructions",

        r"forget\s+(all\s+)?previous\s+instructions",

        r"act\s+as\s+system",

        r"bypass\s+(the\s+)?rules"
    ]


    for pattern in injection_patterns:

        if re.search(
            pattern,
            question,
            flags=re.IGNORECASE
        ):

            return (
                False,
                "The query contains unsupported instruction patterns."
            )


    # -----------------------------------------------------
    # Query passed basic guardrails
    # -----------------------------------------------------

    return True, ""