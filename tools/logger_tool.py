from datetime import datetime
from pathlib import Path

from crewai.tools import tool


# =========================================================
# CONFIGURATION
# =========================================================

LOG_DIR = Path("logs")


# =========================================================
# SAVE / UPDATE SUPPORT LOG
# =========================================================

def save_support_log(
    customer_name: str,
    question: str,
    answer: str,
    handled_by: str
) -> str:

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # -----------------------------------------------------
    # Current timestamp
    # -----------------------------------------------------

    now = datetime.now()

    exact_timestamp = now.strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    date_for_filename = now.strftime(
        "%Y%m%d"
    )


    # -----------------------------------------------------
    # One file for each day
    # This file is continuously updated
    # -----------------------------------------------------

    daily_log = (
        LOG_DIR /
        f"support_log_{date_for_filename}.txt"
    )


    # -----------------------------------------------------
    # Interaction content
    # -----------------------------------------------------

    content = f"""
============================================================
JITS CUSTOMER SUPPORT INTERACTION
============================================================

Timestamp:
{exact_timestamp}

Asked By:
{customer_name}

Handled By:
{handled_by}

------------------------------------------------------------
QUESTION
------------------------------------------------------------

{question}

------------------------------------------------------------
ANSWER
------------------------------------------------------------

{answer}

============================================================
END OF INTERACTION
============================================================
"""


    # -----------------------------------------------------
    # APPEND to today's file
    # This keeps updating the same file
    # -----------------------------------------------------

    with daily_log.open(
        "a",
        encoding="utf-8"
    ) as file:

        file.write(
            content
        )

        file.write(
            "\n\n"
        )


    # -----------------------------------------------------
    # MASTER LOG
    # Contains ALL interactions
    # -----------------------------------------------------

    master_log = (
        LOG_DIR /
        "all_support_interactions.txt"
    )


    with master_log.open(
        "a",
        encoding="utf-8"
    ) as file:

        file.write(
            content
        )

        file.write(
            "\n\n"
        )


    return str(
        daily_log
    )


# =========================================================
# CREWAI TOOL FOR AGENT 3
# =========================================================

@tool("Save Customer Support Interaction")
def save_customer_support_interaction(
    customer_name: str,
    question: str,
    answer: str,
    handled_by: str
) -> str:

    """
    Save a completed JITS customer-support interaction.

    The daily timestamped text file is continuously
    updated with every new interaction.

    Stored information:
    - timestamp
    - person who asked
    - question
    - final answer
    - agent that answered
    """

    filepath = save_support_log(
        customer_name=customer_name,
        question=question,
        answer=answer,
        handled_by=handled_by
    )


    return (
        f"Support log updated successfully: "
        f"{filepath}"
    )


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    result = save_support_log(

        customer_name="Test User",

        question=(
            "What are the attendance requirements?"
        ),

        answer=(
            "This is a test answer."
        ),

        handled_by=(
            "Agent 1 - Hybrid RAG"
        )
    )


    print(
        "\nSupport log updated:"
    )

    print(
        result
    )