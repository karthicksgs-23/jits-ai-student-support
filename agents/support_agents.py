from crewai import Agent, Task, Crew, Process
from dotenv import load_dotenv

from tools.rag_tool import hybrid_rag_search
from tools.website_search_tool import jits_website_search
from tools.logger_tool import save_customer_support_interaction


# ---------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------

load_dotenv()


# =========================================================
# AGENT 1
# Hybrid RAG Agent
# =========================================================

def run_rag_agent(question: str) -> str:

    # =====================================================
    # AGENT 1
    # HYBRID RAG SUPPORT AGENT
    # =====================================================

    rag_agent = Agent(

        role="JITS Academic Regulations Support Agent",

        goal=(
            "Answer customer questions accurately using "
            "the official J-25 B.Tech Academic Regulations "
            "retrieved through Hybrid RAG."
        ),

        backstory=(
            "You are the first-level support agent for JITS. "
            "You answer questions using the official J-25 "
            "B.Tech Academic Regulations knowledge base. "
            "The knowledge base uses FAISS semantic search "
            "combined with BM25 keyword search. "
            "You must provide specific evidence-based answers "
            "and must never invent regulations."
        ),

        tools=[
            hybrid_rag_search
        ],

        verbose=True,

        allow_delegation=False,

        max_iter=5
    )


    # =====================================================
    # TASK
    # =====================================================

    rag_task = Task(

        description="""
A customer asked the following question:

{question}

You MUST use the JITS Hybrid RAG Search tool first.

The knowledge base contains the official J-25 B.Tech
Academic Regulations.

Read ALL retrieved document chunks carefully.

If the retrieved evidence contains the answer, provide
a COMPLETE factual answer using the specific details
found in the regulations.


IMPORTANT ANSWERING RULES:

1. Always use the JITS Hybrid RAG Search tool.

2. Answer only from the retrieved Academic Regulations.

3. Do NOT give vague answers such as:

   "Students must satisfy the attendance requirements."

4. If the regulations contain percentages, credit
   requirements, conditions, exceptions, fees,
   penalties, eligibility criteria, semester rules,
   promotion conditions, or other numerical values,
   include those specific details.

5. Combine relevant information from multiple retrieved
   chunks when necessary.

6. Preserve important conditions and exceptions.

7. Prefer specific facts over general summaries.

8. Do not invent information.

9. Do not use outside knowledge.

10. If the retrieved evidence genuinely does not contain
    enough information to answer accurately, return
    exactly:

    RAG_NOT_SUFFICIENT

11. Carefully distinguish between:

    - the normal attendance requirement,
    - a condonation range,
    - and an absolute minimum threshold.

    Do not describe a condonation threshold as the normal
    attendance requirement.

12. When percentages appear in several related clauses,
    interpret those clauses together before answering.

    For example, if one clause describes attendance
    between 65% and below 75% as eligible for possible
    condonation, do not state that 65% is the normal
    attendance requirement.

13. Preserve the distinction between:
    - requirement,
    - exception,
    - condonation,
    - consequence,
    - eligibility condition.

14. If the retrieved chunks do not clearly establish the
    normal requirement, do not infer it. Return:

    RAG_NOT_SUFFICIENT

15. If the JITS Hybrid RAG Search tool returns:

    SELF_RAG_INSUFFICIENT

    then do not attempt to answer the question.

    Return exactly:

    RAG_NOT_SUFFICIENT
""",

        expected_output="""
Return either:

A clear, complete, factual answer based only on the
retrieved J-25 Academic Regulations.

The answer should include relevant percentages,
conditions, exceptions, penalties and numerical
requirements whenever they are present in the evidence.

OR exactly:

RAG_NOT_SUFFICIENT
""",

        agent=rag_agent
    )


    # =====================================================
    # CREW
    # =====================================================

    crew = Crew(

        agents=[
            rag_agent
        ],

        tasks=[
            rag_task
        ],

        process=Process.sequential,

        verbose=True
    )


    # =====================================================
    # RUN AGENT 1
    # =====================================================

    result = crew.kickoff(

        inputs={
            "question": question
        }

    )


    return result.raw.strip()

# =========================================================
# AGENT 2
# Live Website Search Agent
# =========================================================
def run_website_agent(question: str) -> str:

    # =====================================================
    # AGENT 2
    # SERPER WEBSITE FALLBACK AGENT
    # =====================================================

    website_agent = Agent(

        role="JITS Live Website Support Agent",

        goal=(
            "Answer customer questions using information "
            "retrieved from the official JITS website when "
            "the local Hybrid RAG system cannot answer."
        ),

        backstory=(
            "You are the second-level customer support agent "
            "for JITS. "
            "You are activated only when Agent 1 cannot answer "
            "from the local Academic Regulations RAG system. "
            "You search the official JITS website using Serper "
            "and use the actual official page content as evidence."
        ),

        tools=[
            jits_website_search
        ],

        verbose=True,

        allow_delegation=False,

        max_iter=5
    )


    # =====================================================
    # AGENT 2 TASK
    # =====================================================

    website_task = Task(

        description="""
Agent 1 could not sufficiently answer the customer's question.

Customer question:

{question}

You MUST use the JITS Serper Website Search tool.

The tool searches:

site:jits.ac.in

It may return:
- Serper search results
- Official JITS URLs
- Actual content fetched from official JITS webpages

Read the retrieved evidence carefully.

If the official JITS website contains enough information,
provide a clear factual answer.

If the website does not contain sufficient evidence,
return exactly:

WEBSITE_NOT_SUFFICIENT


IMPORTANT RULES:

1. Always use the JITS Serper Website Search tool.

2. Use only information returned by the tool.

3. Use only official JITS website evidence.

4. Do not invent information.

5. Do not add courses, departments, policies, dates,
   fees, regulations, or other facts unless they are
   explicitly supported by the retrieved evidence.

6. Do not infer facts from general knowledge.

7. Do not infer information from similar colleges.

8. When listing courses or departments, include only
   items explicitly present in the retrieved official
   JITS evidence.

9. Prefer the ACTUAL PAGE CONTENT over the Serper
   search snippet when both are available.

10. If the search snippet and actual webpage disagree,
    use the actual official webpage content.

11. If the evidence is incomplete or ambiguous, return:

    WEBSITE_NOT_SUFFICIENT

12. Include the official source URL in the answer
    when available.

13. Keep the answer clear and customer-friendly.
""",

        expected_output="""
Return either:

A clear factual answer based only on official
JITS website evidence.

OR exactly:

WEBSITE_NOT_SUFFICIENT
""",

        agent=website_agent
    )


    # =====================================================
    # CREW
    # =====================================================

    crew = Crew(

        agents=[
            website_agent
        ],

        tasks=[
            website_task
        ],

        process=Process.sequential,

        verbose=True
    )


    # =====================================================
    # RUN AGENT 2
    # =====================================================

    result = crew.kickoff(

        inputs={
            "question": question
        }

    )


    return result.raw.strip()

# =========================================================
# TEST
# Currently testing Agent 2
# =========================================================
# =========================================================
# AGENT 3
# LOCAL LOGGING AGENT
# =========================================================

def run_logger_agent(
    customer_name: str,
    question: str,
    answer: str,
    handled_by: str
) -> str:

    # =====================================================
    # CREATE AGENT 3
    # =====================================================

    logger_agent = Agent(

        role="JITS Customer Support Logging Agent",

        goal=(
            "Store completed customer support interactions "
            "accurately on the local disk."
        ),

        backstory=(
            "You are the final agent in the JITS customer "
            "support workflow. "
            "After Agent 1 or Agent 2 produces the final answer, "
            "you save the interaction to a timestamped local "
            "text file for auditing and support records."
        ),

        tools=[
            save_customer_support_interaction
        ],

        verbose=True,

        allow_delegation=False,

        max_iter=3
    )


    # =====================================================
    # LOGGER TASK
    # =====================================================

    logger_task = Task(

        description="""
A customer-support interaction has been completed.

You MUST save the interaction using the
Save Customer Support Interaction tool.

Use these values exactly as provided.

Customer Name:
{customer_name}

Question:
{question}

Answer:
{answer}

Handled By:
{handled_by}


IMPORTANT RULES:

1. Always use the Save Customer Support Interaction tool.

2. Do not modify the customer name.

3. Do not rewrite or summarize the question.

4. Do not rewrite or summarize the answer.

5. Do not change the handled-by value.

6. Save the values exactly as they were provided.

7. The logging tool will automatically create the timestamp.

8. After saving, return the confirmation message from the tool.
""",

        expected_output="""
A confirmation that the customer-support interaction
was successfully saved to the local disk.
""",

        agent=logger_agent
    )


    # =====================================================
    # CREW
    # =====================================================

    crew = Crew(

        agents=[
            logger_agent
        ],

        tasks=[
            logger_task
        ],

        process=Process.sequential,

        verbose=False
    )


    # =====================================================
    # RUN AGENT 3
    # =====================================================

    result = crew.kickoff(

        inputs={
            "customer_name": customer_name,
            "question": question,
            "answer": answer,
            "handled_by": handled_by
        }

    )


    return result.raw.strip()

if __name__ == "__main__":

    print("\n================================")
    print("JITS CUSTOMER SUPPORT")
    print("AGENT 3 TEST")
    print("================================")

    result = run_logger_agent(

        customer_name="Karthick",

        question=(
            "What are the attendance requirements "
            "for B.Tech students?"
        ),

        answer=(
            "Students must satisfy the attendance "
            "requirements specified in the academic "
            "regulations."
        ),

        handled_by=(
            "Agent 1 - Hybrid RAG"
        )
    )

    print("\n================================")
    print("AGENT 3 RESPONSE")
    print("================================")

    print(result)