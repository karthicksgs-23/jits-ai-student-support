from dotenv import load_dotenv

from agents.support_agents import (
    run_rag_agent,
    run_website_agent,
    run_logger_agent,
)

from guardrails import validate_user_query
from output_guardrails import validate_agent_answer
from conversation_manager import contextualize_question


# =========================================================
# LOAD ENVIRONMENT
# =========================================================

load_dotenv()


# =========================================================
# CUSTOMER SUPPORT WORKFLOW
# =========================================================

def run_customer_support(
    customer_name: str,
    question: str,
    original_question: str = None
):

    # Keep the exact customer message for logging
    if original_question is None:
        original_question = question


    # =====================================================
    # INPUT GUARDRAIL
    # =====================================================

    allowed, reason = validate_user_query(
        original_question
    )

    if not allowed:

        return {
            "customer_name": customer_name,
            "question": original_question,
            "answer": (
                "I can help with JITS-related questions. "
                "Please rephrase your request."
            ),
            "handled_by": "Input Guardrail",
            "log": None
        }


    # =====================================================
    # AGENT 1
    # HYBRID RAG + SELF-RAG
    # =====================================================

    rag_answer = run_rag_agent(
        question
    ).strip()


    # =====================================================
    # ROUTING
    # =====================================================

    if rag_answer.upper() == "RAG_NOT_SUFFICIENT":

        # =================================================
        # AGENT 2
        # WEBSITE FALLBACK
        # =================================================

        website_answer = run_website_agent(
            question
        ).strip()


        if website_answer.upper() == "WEBSITE_NOT_SUFFICIENT":

            final_answer = (
                "I could not find sufficient information "
                "in the J-25 Academic Regulations or the "
                "official JITS website to answer that accurately."
            )

            handled_by = (
                "Agent 2 - Serper Website Fallback "
                "(No sufficient answer found)"
            )

        else:

            final_answer = website_answer

            handled_by = (
                "Agent 2 - Serper Website Fallback"
            )

    else:

        final_answer = rag_answer

        handled_by = (
            "Agent 1 - Hybrid RAG "
            "(FAISS + BM25 + Self-RAG)"
        )


    # =====================================================
    # OUTPUT GUARDRAIL
    # =====================================================

    output_allowed, output_reason = validate_agent_answer(
        final_answer
    )

    if not output_allowed:

        final_answer = (
            "I could not provide a reliable answer "
            "for that question."
        )

        handled_by = (
            handled_by
            + " | Output Guardrail"
        )


    # =====================================================
    # AGENT 3
    # LOG ORIGINAL CUSTOMER MESSAGE
    # =====================================================

    try:

        log_result = run_logger_agent(

            customer_name=customer_name,

            question=original_question,

            answer=final_answer,

            handled_by=handled_by
        )

    except Exception as error:

        log_result = (
            f"Logging failed: {error}"
        )


    # =====================================================
    # RETURN
    # =====================================================

    return {

        "customer_name": customer_name,

        "question": original_question,

        "standalone_question": question,

        "answer": final_answer,

        "handled_by": handled_by,

        "log": log_result
    }


# =========================================================
# CONVERSATIONAL CHAT
# =========================================================

def start_chat():

    print()
    print("=" * 60)
    print("         JITS AI CUSTOMER SUPPORT")
    print("=" * 60)

    print(
        "\nHello! I am the JITS AI Support Assistant."
    )

    print(
        "You can ask me questions about JITS, "
        "academic regulations, attendance, courses, "
        "and related information."
    )

    print(
        "\nType 'exit' to end the conversation."
    )


    # =====================================================
    # CUSTOMER NAME
    # =====================================================

    customer_name = input(
        "\nYour name: "
    ).strip()


    if not customer_name:

        customer_name = "Guest"


    # =====================================================
    # SESSION CONVERSATION HISTORY
    # =====================================================

    conversation_history = []


    print(
        f"\nAssistant: Hello {customer_name}! "
        "How can I help you today?"
    )


    # =====================================================
    # CONTINUOUS CHAT LOOP
    # =====================================================

    while True:

        print()

        user_question = input(
            "You: "
        ).strip()


        # =================================================
        # EMPTY MESSAGE
        # =================================================

        if not user_question:

            print(
                "Assistant: Please enter a question."
            )

            continue


        # =================================================
        # EXIT
        # =================================================

        if user_question.lower() in {
            "exit",
            "quit",
            "bye",
            "goodbye"
        }:

            print(
                f"\nAssistant: Thank you, {customer_name}. "
                "Have a great day!"
            )

            break


        # =================================================
        # INPUT GUARDRAIL BEFORE CONTEXT MANAGER
        # =================================================

        allowed, reason = validate_user_query(
            user_question
        )


        if not allowed:

            answer = (
                "I can help with JITS-related questions. "
                "Please rephrase your request."
            )

            print(
                f"\nAssistant: {answer}"
            )

            continue


        # =================================================
        # CONVERSATION CONTEXT
        # =================================================

        standalone_question = contextualize_question(

            question=user_question,

            conversation_history=conversation_history
        )


        # =================================================
        # RUN SUPPORT SYSTEM
        # =================================================

        result = run_customer_support(

            customer_name=customer_name,

            question=standalone_question,

            original_question=user_question
        )


        final_answer = result[
            "answer"
        ]


        # =================================================
        # CUSTOMER-FACING RESPONSE
        # =================================================

        print(
            f"\nAssistant: {final_answer}"
        )


        # =================================================
        # UPDATE SESSION MEMORY
        # =================================================

        conversation_history.append({

            "role": "user",

            "content": user_question
        })


        conversation_history.append({

            "role": "assistant",

            "content": final_answer
        })


        # =================================================
        # LIMIT HISTORY SIZE
        # =================================================

        if len(conversation_history) > 12:

            conversation_history = (
                conversation_history[-12:]
            )


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    start_chat()