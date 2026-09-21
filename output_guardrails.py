import re


# =========================================================
# OUTPUT GUARDRAILS
# =========================================================

def validate_agent_answer(
    answer: str
) -> tuple[bool, str]:
    """
    Validate the final answer produced by Agent 1 or Agent 2.

    Returns:
        (True, "") if answer is acceptable
        (False, reason) if answer should be blocked
    """

    if answer is None:
        return False, "Empty answer produced."

    answer = answer.strip()

    # -----------------------------------------------------
    # 1. Empty answer
    # -----------------------------------------------------

    if not answer:
        return False, "Empty answer produced."


    # -----------------------------------------------------
    # 2. Prevent accidental internal prompt leakage
    # -----------------------------------------------------

    blocked_patterns = [

        r"system prompt",

        r"developer message",

        r"hidden instructions",

        r"internal instructions",

        r"chain of thought",

        r"private reasoning"
    ]


    for pattern in blocked_patterns:

        if re.search(
            pattern,
            answer,
            flags=re.IGNORECASE
        ):

            return (
                False,
                "The generated answer contains unsupported internal information."
            )


    # -----------------------------------------------------
    # 3. Detect raw internal routing signals
    # -----------------------------------------------------

    internal_signals = [

        "SELF_RAG_INSUFFICIENT",

        "RAG_NOT_SUFFICIENT",

        "WEBSITE_NOT_SUFFICIENT",

        "WEBSITE_SEARCH_NO_RESULTS",

        "SERPER_ERROR"
    ]


    for signal in internal_signals:

        if signal in answer:

            return (
                False,
                "The generated answer contains an internal routing signal."
            )


    # -----------------------------------------------------
    # Answer passed output guardrails
    # -----------------------------------------------------

    return True, ""