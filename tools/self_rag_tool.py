import json
import re

from crewai import Agent, Task, Crew, Process
from dotenv import load_dotenv


load_dotenv()


# =========================================================
# SELF-RAG EVIDENCE VALIDATOR
# =========================================================

def check_rag_evidence(
    question: str,
    context: str
) -> bool:
    """
    Decide whether retrieved Academic Regulations evidence
    is sufficient to answer the student's question.

    Returns:
        True  -> Agent 1 may answer
        False -> Route to Agent 2
    """

    # -----------------------------------------------------
    # BASIC SAFETY CHECKS
    # -----------------------------------------------------

    if not question or not question.strip():
        return False

    if not context or not context.strip():
        return False

    if context.strip() == "SELF_RAG_INSUFFICIENT":
        return False


    # =====================================================
    # SELF-RAG GRADER AGENT
    # =====================================================

    grader_agent = Agent(

        role="JITS Self-RAG Evidence Validator",

        goal=(
            "Strictly determine whether the retrieved JITS Academic "
            "Regulations evidence directly and completely supports "
            "an accurate answer to the student's question."
        ),

        backstory=(
            "You are an evidence validator, not a customer-support agent. "
            "Your job is to prevent weakly related retrieval results from "
            "being treated as sufficient evidence. "
            "You must reject evidence whenever there is uncertainty."
        ),

        verbose=False,
        allow_delegation=False
    )


    # =====================================================
    # GRADING TASK
    # =====================================================

    grader_task = Task(

        description="""
Evaluate whether the retrieved Academic Regulations evidence
is sufficient to answer the student's question.

STUDENT QUESTION:

{question}


RETRIEVED EVIDENCE:

{context}


You are NOT answering the student's question.

You are only validating whether the evidence is good enough.


Evaluate these four criteria independently:


1. DIRECT_RELEVANCE

Does the evidence directly address the actual intent of the question?

Do NOT approve merely because similar words appear.

Example:

Question:
"What courses are offered at JITS?"

Evidence:
"Basic Sciences, Core Courses, Professional Electives"

This is NOT directly relevant because these are curriculum
categories, not the list of courses/programmes offered by JITS.


2. ANSWERABILITY

Can the student's question actually be answered from the
retrieved evidence alone?

The evidence must contain the factual information required
for the answer.


3. SUFFICIENCY

Is there enough information to provide a complete and
non-misleading answer?

If an important condition, threshold, exception, consequence,
fee, eligibility rule, or qualification is missing, mark this false.


4. CONSISTENCY

Do the retrieved chunks support one coherent interpretation?

If the chunks appear contradictory, ambiguous, or could easily
lead to the wrong interpretation, mark this false.


IMPORTANT RULES:

- Similar wording is NOT sufficient.
- Topic similarity is NOT sufficient.
- A partial answer is NOT sufficient if it could mislead.
- Do not use outside knowledge.
- Do not infer missing facts.
- Do not assume what the regulations probably mean.
- If uncertain, reject the evidence.
- The decision should be SUFFICIENT only when all four criteria are true.
- Confidence must reflect how certain you are that the evidence
  directly supports the intended answer.


ATTENDANCE EXAMPLE:

Question:
"What are the attendance requirements?"

Evidence:
- normal requirement is 75%
- 65% to below 75% may be condoned
- below 65% cannot be condoned

This is sufficient because the evidence establishes the
requirement, exception range, and lower threshold.


BAD ATTENDANCE EXAMPLE:

Question:
"What is the minimum attendance requirement?"

Evidence:
- attendance between 65% and below 75% may be condoned

This alone is insufficient because 65% could incorrectly be
described as the normal attendance requirement.


CONTACT EXAMPLE:

Question:
"What is the JITS contact number?"

Evidence:
The regulations document explicitly provides the official
JITS phone number.

This CAN be sufficient because the evidence directly answers
the question, even though it is not an academic-rule question.


Return ONLY valid JSON in this exact structure:

{
    "direct_relevance": true,
    "answerability": true,
    "sufficiency": true,
    "consistency": true,
    "confidence": 0.95,
    "decision": "SUFFICIENT",
    "reason": "short explanation"
}

decision must be either:

"SUFFICIENT"

or

"INSUFFICIENT"
""",

        expected_output=(
            "Valid JSON containing the four evidence checks, "
            "confidence, decision, and short reason."
        ),

        agent=grader_agent
    )


    # =====================================================
    # RUN SELF-RAG GRADER
    # =====================================================

    try:

        crew = Crew(
            agents=[grader_agent],
            tasks=[grader_task],
            process=Process.sequential,
            verbose=False
        )


        result = crew.kickoff(
            inputs={
                "question": question,
                "context": context
            }
        )


        raw_output = result.raw.strip()


        # =================================================
        # REMOVE POSSIBLE MARKDOWN CODE FENCES
        # =================================================

        raw_output = re.sub(
            r"^```(?:json)?\s*",
            "",
            raw_output,
            flags=re.IGNORECASE
        )

        raw_output = re.sub(
            r"\s*```$",
            "",
            raw_output
        )


        # =================================================
        # PARSE JSON
        # =================================================

        result_data = json.loads(
            raw_output
        )


        direct_relevance = bool(
            result_data.get(
                "direct_relevance",
                False
            )
        )

        answerability = bool(
            result_data.get(
                "answerability",
                False
            )
        )

        sufficiency = bool(
            result_data.get(
                "sufficiency",
                False
            )
        )

        consistency = bool(
            result_data.get(
                "consistency",
                False
            )
        )

        confidence = float(
            result_data.get(
                "confidence",
                0
            )
        )

        decision = str(
            result_data.get(
                "decision",
                "INSUFFICIENT"
            )
        ).upper()


        # =================================================
        # STRICT FINAL APPROVAL
        # =================================================

        approved = (

            direct_relevance

            and answerability

            and sufficiency

            and consistency

            and confidence >= 0.75

            and decision == "SUFFICIENT"
        )


        # =================================================
        # TERMINAL DEBUGGING
        # =================================================

        print("\n")
        print("=" * 60)
        print("SELF-RAG VALIDATION")
        print("=" * 60)

        print(
            f"Direct relevance : {direct_relevance}"
        )

        print(
            f"Answerability    : {answerability}"
        )

        print(
            f"Sufficiency     : {sufficiency}"
        )

        print(
            f"Consistency     : {consistency}"
        )

        print(
            f"Confidence      : {confidence:.2f}"
        )

        print(
            f"Decision        : {decision}"
        )

        print(
            f"Final approval  : "
            f"{'APPROVED' if approved else 'REJECTED'}"
        )

        print(
            f"Reason          : "
            f"{result_data.get('reason', '')}"
        )


        return approved


    # =====================================================
    # INVALID GRADER OUTPUT
    # =====================================================

    except json.JSONDecodeError as error:

        print(
            "\nSelf-RAG returned invalid JSON."
        )

        print(
            f"Error: {error}"
        )

        # Fail closed
        return False


    # =====================================================
    # ANY OTHER FAILURE
    # =====================================================

    except Exception as error:

        print(
            "\nSelf-RAG validation failed."
        )

        print(
            f"Error: {error}"
        )

        # Never allow uncertain evidence through
        return False