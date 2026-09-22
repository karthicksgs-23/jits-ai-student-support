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

# =========================================================
# CONVERSATION MANAGER IMPORT WITH FALLBACK
# =========================================================

try:
    from conversation_manager import contextualize_question

except ImportError as error:

    logging.getLogger(__name__).warning(
        "Conversation manager could not be imported: %s. "
        "Falling back to the original student question.",
        error
    )

    def contextualize_question(
        question: str,
        conversation_history: list
    ) -> str:
        """
        Fallback used when the conversation manager
        cannot be imported.
        """
        return question.strip()



# =========================================================
# ENVIRONMENT
# =========================================================

try:
    load_dotenv()

except (OSError, UnicodeError) as error:
    raise RuntimeError(
        "Failed to load environment configuration."
    ) from error

# =========================================================
# CONSTANTS
# =========================================================

RAG_NOT_SUFFICIENT = "RAG_NOT_SUFFICIENT"
SELF_RAG_INSUFFICIENT = "SELF_RAG_INSUFFICIENT"
WEBSITE_NOT_SUFFICIENT = "WEBSITE_NOT_SUFFICIENT"

SYSTEM_ERROR_MESSAGE = (
    "I'm temporarily unable to process that request. "
    "Please try again shortly."
)

NO_INFORMATION_MESSAGE = (
    "I could not find sufficient information in the "
    "JITS Academic Regulations or the official JITS website "
    "to answer that accurately."
)

BLOCKED_OUTPUT_MESSAGE = (
    "I could not provide a reliable answer for that question. "
    "Please try rephrasing it."
)

REJECTED_INPUT_MESSAGE = (
    "I can help with JITS-related student questions. "
    "Please rephrase your request."
)


# =========================================================
# SYSTEM ERROR LOGGER
# =========================================================

LOG_DIR = Path("logs")

LOG_DIR.mkdir(
    parents=True,
    exist_ok=True
)

logger = logging.getLogger(
    "jits_student_support"
)

logger.setLevel(
    logging.INFO
)

logger.propagate = False


if not logger.handlers:

    file_handler = RotatingFileHandler(
        LOG_DIR / "system_errors.log",
        maxBytes=1_000_000,
        backupCount=3,
        encoding="utf-8"
    )

    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | "
            "%(name)s | %(message)s"
        )
    )

    logger.addHandler(
        file_handler
    )


# =========================================================
# ERROR LOGGING
# =========================================================

def log_backend_error(
    stage: str,
    error: Exception
):
    """
    Store technical details in the backend log.

    Technical errors are never shown directly to students.
    """

    logger.exception(
        "Stage failed: %s | %s",
        stage,
        error
    )


# =========================================================
# RESULT BUILDER
# =========================================================

def build_result(
    customer_name: str,
    question: str,
    standalone_question: str,
    answer: str,
    handled_by: str,
    status: str,
    log_result=None
):
    """
    Return a consistent result structure to the terminal
    app and Streamlit UI.
    """

    return {
        "customer_name": customer_name,
        "question": question,
        "standalone_question": standalone_question,
        "answer": answer,
        "handled_by": handled_by,
        "status": status,
        "log": log_result
    }


# =========================================================
# MAIN SUPPORT WORKFLOW
# =========================================================

def run_customer_support(
    customer_name: str,
    question: str,
    original_question: str = None
):
    """
    Main JITS Student Support controller.

    Routing:

        Input Guardrail
              ↓
           Agent 1
              ↓
       sufficient?
        /        \
      yes        no/error
       │             │
       │          Agent 2
       │             │
       └──────┬──────┘
              ↓
       Output Guardrail
              ↓
           Agent 3
              ↓
          Final answer
    """

    # =====================================================
    # NORMALISE INPUT
    # =====================================================

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

    original_question = (
        str(original_question).strip()
        if original_question is not None
        else question
    )


    # =====================================================
    # INPUT GUARDRAIL
    # =====================================================

    try:

        allowed, reason = validate_user_query(
            original_question
        )

    except Exception as error:

        log_backend_error(
            "input_guardrail",
            error
        )

        return build_result(
            customer_name=customer_name,
            question=original_question,
            standalone_question=question,
            answer=SYSTEM_ERROR_MESSAGE,
            handled_by="System - Input Guardrail Error",
            status="error"
        )


    if not allowed:

        logger.warning(
            "Input rejected by guardrail."
        )

        return build_result(
            customer_name=customer_name,
            question=original_question,
            standalone_question=question,
            answer=REJECTED_INPUT_MESSAGE,
            handled_by="Input Guardrail",
            status="rejected"
        )


    # =====================================================
    # AGENT 1
    # =====================================================

    rag_answer = None
    fallback_reason = None


    try:

        rag_answer = run_rag_agent(
            question
        )

        if rag_answer is None:

            raise ValueError(
                "Agent 1 returned None."
            )

        rag_answer = rag_answer.strip()

        if not rag_answer:

            raise ValueError(
                "Agent 1 returned an empty answer."
            )


    except Exception as error:

        log_backend_error(
            "agent_1",
            error
        )

        fallback_reason = (
            "Agent 1 technical failure"
        )


    # =====================================================
    # DETERMINE ROUTE
    # =====================================================

    use_agent_2 = False


    if rag_answer is None:

        use_agent_2 = True


    elif rag_answer.upper() in {
        RAG_NOT_SUFFICIENT,
        SELF_RAG_INSUFFICIENT
    }:

        use_agent_2 = True

        fallback_reason = (
            "Academic Regulations evidence insufficient"
        )


    # =====================================================
    # AGENT 1 SUCCESS
    # =====================================================

    if not use_agent_2:

        final_answer = rag_answer

        handled_by = (
            "Agent 1 - Hybrid RAG "
            "(FAISS + BM25 + Self-RAG)"
        )

        status = "success"


    # =====================================================
    # AGENT 2 FALLBACK
    # =====================================================

    else:

        logger.info(
            "Routing to Agent 2 | reason=%s",
            fallback_reason
        )


        try:

            website_answer = run_website_agent(
                question
            )

            if website_answer is None:

                raise ValueError(
                    "Agent 2 returned None."
                )

            website_answer = (
                website_answer.strip()
            )

            if not website_answer:

                raise ValueError(
                    "Agent 2 returned an empty answer."
                )


        except Exception as error:

            log_backend_error(
                "agent_2",
                error
            )

            final_answer = (
                SYSTEM_ERROR_MESSAGE
            )

            handled_by = (
                "Agent 2 - Website Fallback "
                "(Technical Failure)"
            )

            status = "error"


        else:

            website_upper = (
                website_answer.upper()
            )


            # ---------------------------------------------
            # NO SUFFICIENT WEBSITE EVIDENCE
            # ---------------------------------------------

            if (
                website_upper
                == WEBSITE_NOT_SUFFICIENT
                or website_upper
                == "WEBSITE_SEARCH_NO_RESULTS"
                or website_upper.startswith(
                    "SERPER_ERROR"
                )
            ):

                final_answer = (
                    NO_INFORMATION_MESSAGE
                )

                handled_by = (
                    "Agent 2 - Website Fallback "
                    "(No Sufficient Evidence)"
                )

                status = "no_answer"


            # ---------------------------------------------
            # AGENT 2 SUCCESS
            # ---------------------------------------------

            else:

                final_answer = (
                    website_answer
                )

                handled_by = (
                    "Agent 2 - Official Website Fallback"
                )

                status = "success"


    # =====================================================
    # OUTPUT GUARDRAIL
    # =====================================================

    try:

        output_allowed, output_reason = (
            validate_agent_answer(
                final_answer
            )
        )

    except Exception as error:

        log_backend_error(
            "output_guardrail",
            error
        )

        final_answer = (
            BLOCKED_OUTPUT_MESSAGE
        )

        handled_by += (
            " | Output Guardrail Error"
        )

        status = "error"


    else:

        if not output_allowed:

            logger.warning(
                "Output blocked | reason=%s",
                output_reason
            )

            final_answer = (
                BLOCKED_OUTPUT_MESSAGE
            )

            handled_by += (
                " | Output Guardrail"
            )

            status = "blocked"


    # =====================================================
    # AGENT 3 LOGGING
    # =====================================================

    log_result = None


    try:

        log_result = run_logger_agent(

            customer_name=customer_name,

            question=original_question,

            answer=final_answer,

            handled_by=handled_by
        )


    except Exception as error:

        # Logging failure must NEVER change the student answer.

        log_backend_error(
            "agent_3_logging",
            error
        )

        log_result = (
            "Logging unavailable."
        )


    # =====================================================
    # RETURN
    # =====================================================

    return build_result(

        customer_name=customer_name,

        question=original_question,

        standalone_question=question,

        answer=final_answer,

        handled_by=handled_by,

        status=status,

        log_result=log_result
    )


# =========================================================
# TERMINAL CHAT
# =========================================================

def start_chat():

    print()
    print("=" * 60)
    print("          JITS AI STUDENT SUPPORT")
    print("=" * 60)

    print(
        "\nAsk about JITS academic regulations, "
        "attendance, examinations, courses, "
        "and student information."
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
    # CONVERSATION HISTORY
    # =====================================================

    history = []


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

        except (
            KeyboardInterrupt,
            EOFError
        ):

            print(
                "\n\nAssistant: Session ended."
            )

            break


        # -------------------------------------------------
        # EMPTY
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
                f"\nAssistant: Thank you, "
                f"{student_name}. Have a great day!"
            )

            break


        # =================================================
        # CONVERSATION CONTEXT
        # =================================================

        try:

            standalone_question = (
                contextualize_question(

                    question=user_question,

                    conversation_history=history
                )
            )

        except Exception as error:

            log_backend_error(
                "conversation_manager",
                error
            )

            # Context failure should not end the conversation.
            standalone_question = (
                user_question
            )


        # =================================================
        # SUPPORT WORKFLOW
        # =================================================

        try:

            result = run_customer_support(

                customer_name=student_name,

                question=standalone_question,

                original_question=user_question
            )

            final_answer = (
                result["answer"]
            )


        except Exception as error:

            # Absolute last-resort application protection.

            log_backend_error(
                "main_controller",
                error
            )

            final_answer = (
                SYSTEM_ERROR_MESSAGE
            )


        # =================================================
        # DISPLAY
        # =================================================

        print(
            f"\nAssistant: {final_answer}"
        )


        # =================================================
        # MEMORY
        # =================================================

        history.append(
            {
                "role": "user",
                "content": user_question
            }
        )

        history.append(
            {
                "role": "assistant",
                "content": final_answer
            }
        )


        if len(history) > 12:

            history = history[-12:]


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    start_chat()