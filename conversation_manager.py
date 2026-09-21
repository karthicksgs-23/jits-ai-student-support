from crewai import Agent, Task, Crew, Process
from dotenv import load_dotenv


load_dotenv()


# =========================================================
# CONVERSATION CONTEXT MANAGER
# =========================================================

def contextualize_question(
    question: str,
    conversation_history: list
) -> str:
    """
    Convert a conversational follow-up question into a
    standalone question using recent conversation history.

    Example:

    Previous:
        User: What is the attendance requirement?
        Assistant: Minimum attendance is 75%...

    Current:
        What happens if it is 68%?

    Output:
        What happens if a B.Tech student's attendance is 68%?
    """


    # -----------------------------------------------------
    # No previous conversation
    # -----------------------------------------------------

    if not conversation_history:

        return question.strip()


    # -----------------------------------------------------
    # Build recent conversation context
    # -----------------------------------------------------

    recent_history = conversation_history[-6:]


    history_text = ""


    for turn in recent_history:

        role = turn.get(
            "role",
            "unknown"
        )

        content = turn.get(
            "content",
            ""
        )

        history_text += (
            f"{role.upper()}: "
            f"{content}\n"
        )


    # -----------------------------------------------------
    # CONTEXT AGENT
    # -----------------------------------------------------

    context_agent = Agent(

        role=(
            "JITS Conversation Context Agent"
        ),

        goal=(
            "Understand follow-up customer questions "
            "using the recent conversation and convert "
            "them into clear standalone questions."
        ),

        backstory=(
            "You are responsible only for resolving "
            "conversation context. You do not answer "
            "the customer's question. You rewrite the "
            "latest question only when previous context "
            "is necessary."
        ),

        verbose=False,

        allow_delegation=False
    )


    # -----------------------------------------------------
    # CONTEXT TASK
    # -----------------------------------------------------

    context_task = Task(

        description="""
Use the recent conversation to understand the customer's
latest question.

RECENT CONVERSATION:

{history}

LATEST CUSTOMER QUESTION:

{question}


Your job is NOT to answer the question.

Your job is to produce one standalone question that can be
sent to the JITS support system.


RULES:

1. Preserve the customer's original intention.

2. Resolve references such as:

   - it
   - that
   - this
   - then
   - they
   - those
   - what about
   - how about

3. Use previous conversation only when required.

4. Do not introduce facts that the customer did not ask about.

5. Do not answer the question.

6. Do not add explanations.

7. If the current question is already standalone, return it
   essentially unchanged.

8. Return ONLY the standalone question.


Example:

Conversation:
USER: What is the attendance requirement?
ASSISTANT: Students normally require 75% attendance.

Question:
What if it is 68%?

Return:
What happens if a B.Tech student's attendance is 68%?


Example:

Conversation:
USER: What courses are offered at JITS?

Question:
What is the attendance requirement?

Return:
What is the attendance requirement for B.Tech students?
""",

        expected_output=(
            "One standalone customer question only."
        ),

        agent=context_agent
    )


    # -----------------------------------------------------
    # RUN CONTEXT CREW
    # -----------------------------------------------------

    crew = Crew(

        agents=[
            context_agent
        ],

        tasks=[
            context_task
        ],

        process=Process.sequential,

        verbose=False
    )


    result = crew.kickoff(

        inputs={

            "history":
                history_text,

            "question":
                question
        }
    )


    standalone_question = (
        result.raw.strip()
    )


    # -----------------------------------------------------
    # FALLBACK
    # -----------------------------------------------------

    if not standalone_question:

        return question.strip()


    return standalone_question