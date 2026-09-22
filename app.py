import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

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
# ENVIRONMENT
# =========================================================

load_dotenv()


# =========================================================
# SYSTEM LOGGING
# =========================================================

LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

system_logger = logging.getLogger(
    "jits_student_support"
)

system_logger.setLevel(
    logging.INFO
)


if not system_logger.handlers:

    try:

        file_handler = RotatingFileHandler(
            LOG_DIR / "system_errors.log",
            maxBytes=1_000_000,
            backupCount=3,
            encoding="utf-8"
        )

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s"
        )

        file_handler.setFormatter(
            formatter
        )

        system_logger.addHandler(
            file_handler
        )

    except Exception:

        # Logging itself must never crash the app
        pass


# =========================================================
# SAFE ERROR LOGGING
# =========================================================

def log_system_error(
    stage: str,
    error: Exception
):
    """
    Record backend failures without exposing technical
    details to the student.
    """

    try:

        system_logger.exception(
            f"Failure at stage '{stage}': {error}"
        )

    except Exception:

        pass


# =========================================================
# SAFE INPUT VALIDATION
# =========================================================

def safe_validate_input(
    question: str
):
    """
    Run the input guardrail safely.

    Fail closed if the guardrail itself crashes.
    """

    try:

        return validate_user_query(
            question
        )

    except Exception as error:

        log_system_error(
            "input_guardrail",
            error
        )

        return (
            False,
            "The request could not be validated."
        )


# =========================================================
# SAFE OUTPUT VALIDATION
# =========================================================

def safe_validate_output(
    answer: str
):
    """
    Run the output guardrail safely.

    Fail closed if validation fails.
    """

    try:

        return validate_agent_answer(
            answer
        )

    except Exception as error:

        log_system_error(
            "output_guardrail",
            error
        )

        return (
            False,
            "Output validation failed."
        )


# =========================================================
# SAFE CONVERSATION CONTEXT
# =========================================================

def safe_contextualize_question(
    question: str,
    conversation_history: list
):
    """
    Convert follow-up questions into standalone questions.

    If the context manager fails, use the student's original
    question instead of crashing the chat.
    """

    try:

        standalone_question = (
            contextualize_question(
                question=question,
                conversation_history=conversation_history
            )
        )

        if not standalone_question:

            return question.strip()

        return standalone_question.strip()


    except Exception as error:

        log_system_error(
            "conversation_manager",
            error
        )

        return question.strip()


# =========================================================
# SAFE AGENT 1
# =========================================================

def safe_run_rag_agent(
    question: str
):
    """
    Run Agent 1.

    Returns:
        (answer, error)

    If Agent 1 fails technically, the controller can still
    attempt Agent 2.
    """

    try:

        answer = run_rag_agent(
            question
        )

        if answer is None:

            raise ValueError(
                "Agent 1 returned None."
            )

        answer = answer.strip()

        if not answer:

            raise ValueError(
                "Agent 1 returned an empty response."
            )

        return answer, None


    except Exception as error:

        log_system_error(
            "agent_1",
            error
        )

        return None, error


# =========================================================
# SAFE AGENT 2
# =========================================================

def safe_run_website_agent(
    question: str
):
    """
    Run Agent 2 website fallback safely.
    """

    try:

        answer = run_website_agent(
            question
        )

        if answer is None:

            raise ValueError(
                "Agent 2 returned None."
            )

        answer = answer.strip()

        if not answer:

            raise ValueError(
                "Agent 2 returned an empty response."
            )

        return answer, None


    except Exception as error:

        log_system_error(
            "agent_2",
            error
        )

        return None, error


# =========================================================
# SAFE LOGGER AGENT
# =========================================================

def safe_log_interaction(
    customer_name: str,
    question: str,
    answer: str,
    handled_by: str
):
    """
    Logging failure must never stop a student from receiving
    an otherwise valid answer.
    """

    try:

        result = run_logger_agent(
            customer_name=customer_name,
            question=question,
            answer=answer,
            handled_by=handled_by
        )

        if result is None:

            return "Logging completed."

        return str(result).strip()


    except Exception as error:

        log_system_error(
            "agent_3_logger",
            error
        )

        return (
            "Interaction logging was unavailable."
        )


# =========================================================
# MAIN STUDENT SUPPORT WORKFLOW
# =========================================================

def run_customer_support(
    customer_name: str,
    question: str,
    original_question: str = None
):
    """
    Main JITS student-support orchestration.

    The student receives a safe response even if one backend
    component fails.
    """

    # -----------------------------------------------------
    # NORMALISE INPUTS
    # -----------------------------------------------------

    customer_name = (
        str(customer_name).strip()
        if customer_name
        else "Guest"
    )

    question = (
        str(question).strip()
        if question
        else ""
    )


    if original_question is None:

        original_question = question

    else:

        original_question = (
            str(original_question).strip()
        )


    # =====================================================
    # INPUT GUARDRAIL
    # =====================================================

    allowed, reason = safe_validate_input(
        original_question
    )


    if not allowed:

        final_answer = (
            "I can help with JITS-related student questions. "
            "Please rephrase your request."
        )

        return {
            "customer_name": customer_name,
            "question": original_question,
            "standalone_question": question,
            "answer": final_answer,
            "handled_by": "Input Guardrail",
            "log": None,
            "status": "rejected"
        }


    # =====================================================
    # AGENT 1
    # HYBRID RAG + SELF-RAG
    # =====================================================

    rag_answer, rag_error = (
        safe_run_rag_agent(
            question
        )
    )


    # =====================================================
    # DECIDE WHETHER AGENT 2 IS NEEDED
    # =====================================================

    use_agent_2 = False


    # Agent 1 technical failure
    if rag_error is not None:

        use_agent_2 = True


    # Agent 1 intentionally rejected the evidence
    elif (
        rag_answer
        and rag_answer.upper()
        == "RAG_NOT_SUFFICIENT"
    ):

        use_agent_2 = True


    # =====================================================
    # AGENT 2 FALLBACK
    # =====================================================

    if use_agent_2:

        website_answer, website_error = (
            safe_run_website_agent(
                question
            )
        )


        # -------------------------------------------------
        # AGENT 2 TECHNICAL FAILURE
        # -------------------------------------------------

        if website_error is not None:

            final_answer = (
                "I'm temporarily unable to retrieve enough "
                "reliable JITS information for that question. "
                "Please try again shortly."
            )

            handled_by = (
                "Agent 2 - Website Fallback "
                "(Technical failure)"
            )


        # -------------------------------------------------
        # AGENT 2 FOUND NO SUFFICIENT EVIDENCE
        # -------------------------------------------------

        elif (
            website_answer.upper()
            == "WEBSITE_NOT_SUFFICIENT"
        ):

            final_answer = (
                "I could not find sufficient information in "
                "the JITS Academic Regulations or the official "
                "JITS website to answer that accurately."
            )

            handled_by = (
                "Agent 2 - Website Fallback "
                "(No sufficient evidence)"
            )


        # -------------------------------------------------
        # AGENT 2 SUCCESS
        # -------------------------------------------------

        else:

            final_answer = (
                website_answer
            )

            handled_by = (
                "Agent 2 - Official Website Fallback"
            )


    # =====================================================
    # AGENT 1 SUCCESS
    # =====================================================

    else:

        final_answer = (
            rag_answer
        )

        handled_by = (
            "Agent 1 - Hybrid RAG "
            "(FAISS + BM25 + Self-RAG)"
        )


    # =====================================================
    # FINAL ANSWER SANITY CHECK
    # =====================================================

    if (
        final_answer is None
        or not str(final_answer).strip()
    ):

        final_answer = (
            "I couldn't generate a reliable answer for "
            "that question. Please try again."
        )

        handled_by = (
            handled_by
            + " | Empty Response Protection"
        )


    final_answer = str(
        final_answer
    ).strip()


    # =====================================================
    # OUTPUT GUARDRAIL
    # =====================================================

    output_allowed, output_reason = (
        safe_validate_output(
            final_answer
        )
    )


    if not output_allowed:

        system_logger.warning(
            "Final response blocked by output guardrail."
        )

        final_answer = (
            "I could not provide a reliable answer "
            "for that question. Please try rephrasing it."
        )

        handled_by = (
            handled_by
            + " | Output Guardrail"
        )


    # =====================================================
    # AGENT 3 LOGGING
    # =====================================================

    log_result = safe_log_interaction(

        customer_name=customer_name,

        question=original_question,

        answer=final_answer,

        handled_by=handled_by
    )


    # =====================================================
    # FINAL RESULT
    # =====================================================

    return {

        "customer_name":
            customer_name,

        "question":
            original_question,

        "standalone_question":
            question,

        "answer":
            final_answer,

        "handled_by":
            handled_by,

        "log":
            log_result,

        "status":
            "success"
    }


# =========================================================
# TERMINAL CONVERSATIONAL CHAT
# =========================================================

def start_chat():

    print()
    print("=" * 60)
    print("          JITS AI STUDENT SUPPORT")
    print("=" * 60)

    print(
        "\nHello! I am the JITS AI Student Support Assistant."
    )

    print(
        "You can ask about academic regulations, attendance, "
        "courses, examinations, and other JITS information."
    )

    print(
        "\nType 'exit' to end the conversation."
    )


    # =====================================================
    # STUDENT NAME
    # =====================================================

    try:

        student_name = input(
            "\nYour name: "
        ).strip()

    except (
        EOFError,
        KeyboardInterrupt
    ):

        print(
            "\nSession ended."
        )

        return


    if not student_name:

        student_name = "Guest"


    # =====================================================
    # SESSION HISTORY
    # =====================================================

    conversation_history = []


    print(
        f"\nAssistant: Hello {student_name}! "
        "How can I help you today?"
    )


    # =====================================================
    # CHAT LOOP
    # =====================================================

    while True:

        try:

            user_question = input(
                "\nYou: "
            ).strip()


        except KeyboardInterrupt:

            print(
                "\n\nAssistant: Session ended. "
                "Have a great day!"
            )

            break


        except EOFError:

            print(
                "\nAssistant: Session ended."
            )

            break


        # -------------------------------------------------
        # EMPTY INPUT
        # -------------------------------------------------

        if not user_question:

            print(
                "Assistant: Please enter a question."
            )

            continue


        # -------------------------------------------------
        # EXIT
        # -------------------------------------------------

        if user_question.lower() in {
            "exit",
            "quit",
            "bye",
            "goodbye"
        }:

            print(
                f"\nAssistant: Thank you, {student_name}. "
                "Have a great day!"
            )

            break


        # =================================================
        # INPUT GUARDRAIL
        # =================================================

        allowed, reason = safe_validate_input(
            user_question
        )


        if not allowed:

            print(
                "\nAssistant: I can help with JITS-related "
                "student questions. Please rephrase your request."
            )

            continue


        # =================================================
        # CONVERSATION CONTEXT
        # =================================================

        standalone_question = (
            safe_contextualize_question(

                question=user_question,

                conversation_history=conversation_history
            )
        )


        # =================================================
        # RUN SUPPORT
        # =================================================

        try:

            result = run_customer_support(

                customer_name=student_name,

                question=standalone_question,

                original_question=user_question
            )


            final_answer = result.get(
                "answer",
                "I could not generate a reliable answer."
            )


        except Exception as error:

            # Absolute final safety net
            log_system_error(
                "main_workflow",
                error
            )

            final_answer = (
                "I encountered a temporary problem while "
                "processing your question. Please try again."
            )


        # =================================================
        # DISPLAY
        # =================================================

        print(
            f"\nAssistant: {final_answer}"
        )


        # =================================================
        # UPDATE MEMORY
        # =================================================

        conversation_history.append(
            {
                "role": "user",
                "content": user_question
            }
        )


        conversation_history.append(
            {
                "role": "assistant",
                "content": final_answer
            }
        )


        if len(
            conversation_history
        ) > 12:

            conversation_history = (
                conversation_history[-12:]
            )


# =========================================================
# APPLICATION ENTRY POINT
# =========================================================

if __name__ == "__main__":

    try:

        start_chat()

    except Exception as error:

        log_system_error(
            "application_entry_point",
            error
        )

        print(
            "\nThe student support system encountered an "
            "unexpected problem. Please try again."
        )