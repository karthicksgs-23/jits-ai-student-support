from crewai import Agent, Task, Crew, Process
from dotenv import load_dotenv


load_dotenv()


# =========================================================
# SELF-RAG EVIDENCE GRADER
# =========================================================

def check_rag_evidence(
    question: str,
    context: str
) -> bool:

    """
    Check whether the retrieved RAG evidence is
    relevant and sufficient for the user's actual intent.

    Returns:
        True  -> Agent 1 may answer
        False -> Send query to Agent 2
    """

    grader_agent = Agent(

        role="Self-RAG Evidence Grader",

        goal=(
            "Determine whether retrieved JITS Academic "
            "Regulations evidence directly and sufficiently "
            "answers the customer's intended question."
        ),

        backstory=(
            "You are a strict evidence evaluator. "
            "You do not answer customer questions. "
            "You only decide whether the retrieved evidence "
            "is relevant and sufficient to answer accurately."
        ),

        verbose=False,

        allow_delegation=False
    )


    # =====================================================
    # GRADING TASK
    # =====================================================

    grader_task = Task(

        description="""
Evaluate the retrieved evidence against the customer's
actual question.

CUSTOMER QUESTION:

{question}


RETRIEVED RAG EVIDENCE:

{context}


Evaluate the following:

1. RELEVANCE
   Does the evidence directly relate to what the
   customer is asking?

2. INTENT
   Does the evidence match the customer's intended
   meaning, rather than merely sharing similar words?

3. SUFFICIENCY
   Does the evidence contain enough information to
   answer accurately?

4. SPECIFICITY
   Is the evidence specific enough to support a
   factual answer?

5. MISLEADING RISK
   Would answering from this evidence risk giving
   the customer a misleading answer?


IMPORTANT EXAMPLE:

Question:

"What courses are offered at JITS?"

Evidence:

"Basic Sciences, Professional Electives,
Open Electives, Core Courses..."

This evidence is INSUFFICIENT because it describes
curriculum course categories and does not answer which
degree programs or branches JITS offers.


Another example:

Question:

"What are the attendance requirements for B.Tech students?"

Evidence:

"Attendance shortage from 65% to below 75% may be
condoned... attendance below 65% cannot be condoned..."

This evidence is relevant to the attendance question.


Return ONLY one word:

SUFFICIENT

or

INSUFFICIENT

Do not provide an explanation.
""",

        expected_output="""
Exactly one word:

SUFFICIENT

or

INSUFFICIENT
""",

        agent=grader_agent
    )


    # =====================================================
    # RUN GRADER
    # =====================================================

    crew = Crew(

        agents=[
            grader_agent
        ],

        tasks=[
            grader_task
        ],

        process=Process.sequential,

        verbose=False
    )


    result = crew.kickoff(

        inputs={
            "question": question,
            "context": context
        }

    )


    decision = (
        result.raw
        .strip()
        .upper()
    )


    print(
        f"\nSelf-RAG decision: {decision}"
    )


    return (
        decision == "SUFFICIENT"
    )