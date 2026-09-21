import streamlit as st

from app import run_customer_support
from conversation_manager import contextualize_question
from guardrails import validate_user_query


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="JITS AI Student Support",
    page_icon="🎓",
    layout="centered"
)


# =========================================================
# TITLE
# =========================================================

st.title("🎓 JITS AI Student Support")

st.caption(
    "Ask questions about JITS academic regulations, "
    "attendance, courses, examinations, and student-related information."
)


# =========================================================
# EXAMPLE QUESTIONS
# =========================================================

st.markdown(
    """
**You can ask things like:**

- What is the minimum attendance requirement?
- What happens if my attendance is below 65%?
- What courses are offered at JITS?
- What are the promotion rules?
"""
)


# =========================================================
# RELIABILITY NOTE
# =========================================================

st.info(
    "Answers are generated using official JITS Academic Regulations "
    "and information available on the official JITS website."
)


# =========================================================
# SESSION STATE
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "customer_name" not in st.session_state:
    st.session_state.customer_name = ""


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("Student")

    student_name = st.text_input(
        "Your name",
        value=st.session_state.customer_name,
        placeholder="Enter your name"
    )

    if student_name.strip():

        st.session_state.customer_name = (
            student_name.strip()
        )


    st.divider()


    # =====================================================
    # CLEAR CONVERSATION
    # =====================================================

    if st.button(
        "🗑️ Clear Conversation",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()


# =========================================================
# WELCOME MESSAGE
# =========================================================

if not st.session_state.messages:

    with st.chat_message("assistant"):

        if st.session_state.customer_name:

            st.write(
                f"Hello {st.session_state.customer_name}! "
                "How can I help you with your JITS student queries today?"
            )

        else:

            st.write(
                "Hello! How can I help you with "
                "your JITS student queries today?"
            )


# =========================================================
# DISPLAY PREVIOUS CHAT
# =========================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# =========================================================
# CHAT INPUT
# =========================================================

user_question = st.chat_input(
    "Ask a question about JITS..."
)


# =========================================================
# PROCESS STUDENT QUESTION
# =========================================================

if user_question:

    # =====================================================
    # DISPLAY USER MESSAGE
    # =====================================================

    with st.chat_message("user"):

        st.markdown(
            user_question
        )


    # =====================================================
    # INPUT GUARDRAIL
    # =====================================================

    allowed, reason = validate_user_query(
        user_question
    )


    # =====================================================
    # QUERY REJECTED
    # =====================================================

    if not allowed:

        final_answer = (
            "I can help with JITS-related student questions. "
            "Please rephrase your request."
        )


        # -------------------------------------------------
        # STORE USER MESSAGE
        # -------------------------------------------------

        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_question
            }
        )


        # -------------------------------------------------
        # STORE ASSISTANT MESSAGE
        # -------------------------------------------------

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": final_answer
            }
        )


        # -------------------------------------------------
        # DISPLAY RESPONSE
        # -------------------------------------------------

        with st.chat_message("assistant"):

            st.markdown(
                final_answer
            )


    # =====================================================
    # QUERY ALLOWED
    # =====================================================

    else:

        # =================================================
        # BUILD EXISTING CONVERSATION HISTORY
        # =================================================

        conversation_history = (
            st.session_state.messages
        )


        # =================================================
        # CONVERSATION CONTEXT MANAGER
        # =================================================

        try:

            standalone_question = (
                contextualize_question(

                    question=user_question,

                    conversation_history=conversation_history
                )
            )

        except Exception:

            # If conversation rewriting fails,
            # use the original question.
            standalone_question = user_question


        # =================================================
        # STUDENT NAME
        # =================================================

        student = (
            st.session_state.customer_name
            if st.session_state.customer_name
            else "Guest"
        )


        # =================================================
        # RUN MULTI-AGENT STUDENT SUPPORT SYSTEM
        # =================================================

        with st.chat_message("assistant"):

            with st.spinner(
                "Finding the best answer..."
            ):

                try:

                    result = run_customer_support(

                        customer_name=student,

                        question=standalone_question,

                        original_question=user_question
                    )


                    final_answer = result[
                        "answer"
                    ]


                except Exception as error:

                    final_answer = (
                        "I encountered a temporary problem while "
                        "processing your question. Please try again."
                    )

                    # Optional terminal debugging
                    print(
                        f"Student support error: {error}"
                    )


            # =================================================
            # DISPLAY FINAL ANSWER
            # =================================================

            st.markdown(
                final_answer
            )


        # =================================================
        # STORE USER MESSAGE
        # =================================================

        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_question
            }
        )


        # =================================================
        # STORE ASSISTANT RESPONSE
        # =================================================

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": final_answer
            }
        )


    # =====================================================
    # LIMIT SESSION CONVERSATION MEMORY
    # =====================================================

    if len(
        st.session_state.messages
    ) > 20:

        st.session_state.messages = (
            st.session_state.messages[-20:]
        )