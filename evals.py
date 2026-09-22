import json
import logging
import time

from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

from dotenv import load_dotenv

from agents.support_agents import (
    run_rag_agent,
    run_website_agent,
)


# =========================================================
# ENVIRONMENT
# =========================================================

load_dotenv()


# =========================================================
# PATHS
# =========================================================

EVAL_FILE = Path("eval_cases.json")

RESULT_FILE = Path("eval_results.json")

LOG_DIR = Path("logs")

EVAL_HISTORY_DIR = LOG_DIR / "eval_runs"


LOG_DIR.mkdir(
    parents=True,
    exist_ok=True
)

EVAL_HISTORY_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# CONSTANTS
# =========================================================

RAG_NOT_SUFFICIENT = "RAG_NOT_SUFFICIENT"

SELF_RAG_INSUFFICIENT = "SELF_RAG_INSUFFICIENT"

WEBSITE_NOT_SUFFICIENT = "WEBSITE_NOT_SUFFICIENT"

WEBSITE_SEARCH_NO_RESULTS = "WEBSITE_SEARCH_NO_RESULTS"


# =========================================================
# LOGGER
# =========================================================

logger = logging.getLogger(
    "jits_evals"
)

logger.setLevel(
    logging.INFO
)

logger.propagate = False


if not logger.handlers:

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )


    # -----------------------------------------------------
    # FILE LOGGER
    # -----------------------------------------------------

    file_handler = RotatingFileHandler(
        LOG_DIR / "evals.log",
        maxBytes=2_000_000,
        backupCount=5,
        encoding="utf-8"
    )

    file_handler.setFormatter(
        formatter
    )


    # -----------------------------------------------------
    # TERMINAL LOGGER
    # -----------------------------------------------------

    console_handler = logging.StreamHandler()

    console_handler.setFormatter(
        formatter
    )


    logger.addHandler(
        file_handler
    )

    logger.addHandler(
        console_handler
    )


# =========================================================
# KEYWORD / PHRASE MATCHING
# =========================================================

def evaluate_expected_keywords(
    answer: str,
    expected_keywords: list
):
    """
    Evaluate expected answer content.

    Each expected item can be:

    1. A string:
       "attendance"

       The exact phrase must appear.

    2. A list of acceptable alternatives:
       [
           "not condoned",
           "cannot be condoned"
       ]

       At least one phrase must appear.

    Returns:
        found_keywords,
        missing_keywords,
        keyword_pass
    """

    answer_lower = (
        answer.lower()
        if answer
        else ""
    )

    found_keywords = []

    missing_keywords = []


    for expected in expected_keywords:

        # =================================================
        # ALTERNATIVE VALID PHRASES
        # =================================================

        if isinstance(expected, list):

            matched_phrase = next(
                (
                    phrase
                    for phrase in expected
                    if isinstance(phrase, str)
                    and phrase.lower() in answer_lower
                ),
                None
            )


            if matched_phrase:

                found_keywords.append(
                    matched_phrase
                )

            else:

                readable_requirement = (
                    " OR ".join(
                        str(item)
                        for item in expected
                    )
                )

                missing_keywords.append(
                    readable_requirement
                )


        # =================================================
        # NORMAL REQUIRED PHRASE
        # =================================================

        elif isinstance(expected, str):

            if expected.lower() in answer_lower:

                found_keywords.append(
                    expected
                )

            else:

                missing_keywords.append(
                    expected
                )


        # =================================================
        # INVALID EVAL CONFIGURATION
        # =================================================

        else:

            missing_keywords.append(
                f"INVALID_EXPECTATION:{expected}"
            )


    keyword_pass = (
        len(missing_keywords) == 0
    )


    return (
        found_keywords,
        missing_keywords,
        keyword_pass
    )


# =========================================================
# RUN ONE EVALUATION CASE
# =========================================================

def run_eval_case(
    case: dict,
    case_number: int = 1,
    total_cases: int = 1
):
    """
    Execute one evaluation case.

    Evaluates:

    - routing
    - expected answer phrases
    - agent execution failures
    - execution time
    """

    started_at = (
        datetime.now()
        .astimezone()
        .isoformat(
            timespec="seconds"
        )
    )

    start_time = time.perf_counter()


    # =====================================================
    # READ CASE
    # =====================================================

    question = case["question"]

    expected_route = case["expected_route"]

    expected_keywords = case.get(
        "expected_keywords",
        []
    )


    # =====================================================
    # LOG CASE START
    # =====================================================

    logger.info(
        "=" * 70
    )

    logger.info(
        "CASE %d/%d",
        case_number,
        total_cases
    )

    logger.info(
        "Question: %s",
        question
    )

    logger.info(
        "Expected route: %s",
        expected_route
    )

    logger.info(
        "Expected keywords: %s",
        expected_keywords
    )


    # =====================================================
    # DEFAULT VALUES
    # =====================================================

    actual_route = None

    answer = ""

    rag_answer = None

    execution_error = None

    fallback_reason = None


    # =====================================================
    # AGENT 1
    # =====================================================

    try:

        logger.info(
            "Running Agent 1..."
        )


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
                "Agent 1 returned empty output."
            )


        logger.info(
            "Agent 1 completed."
        )


    except Exception as error:

        logger.exception(
            "Agent 1 execution failed."
        )


        rag_answer = None

        execution_error = (
            f"Agent 1 error: {error}"
        )

        fallback_reason = (
            "Agent 1 technical failure"
        )


    # =====================================================
    # ROUTING DECISION
    # =====================================================

    needs_agent_2 = False


    if rag_answer is None:

        needs_agent_2 = True


    elif rag_answer.upper() in {
        RAG_NOT_SUFFICIENT,
        SELF_RAG_INSUFFICIENT
    }:

        needs_agent_2 = True

        fallback_reason = (
            "Agent 1 evidence insufficient"
        )


    # =====================================================
    # AGENT 2 FALLBACK
    # =====================================================

    if needs_agent_2:

        actual_route = "agent_2"


        logger.info(
            "Routing decision: Agent 2"
        )


        if fallback_reason:

            logger.info(
                "Fallback reason: %s",
                fallback_reason
            )


        try:

            logger.info(
                "Running Agent 2..."
            )


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
                    "Agent 2 returned empty output."
                )


            logger.info(
                "Agent 2 completed."
            )


        except Exception as error:

            logger.exception(
                "Agent 2 execution failed."
            )


            answer = ""


            if execution_error:

                execution_error += (
                    f" | Agent 2 error: {error}"
                )

            else:

                execution_error = (
                    f"Agent 2 error: {error}"
                )


    # =====================================================
    # AGENT 1 ANSWER
    # =====================================================

    else:

        actual_route = "agent_1"

        answer = rag_answer


        logger.info(
            "Routing decision: Agent 1"
        )


    # =====================================================
    # DETECT INTERNAL FAILURE MARKERS
    # =====================================================

    internal_answer_marker = None


    if answer:

        answer_upper = answer.upper()


        if answer_upper == WEBSITE_NOT_SUFFICIENT:

            internal_answer_marker = (
                WEBSITE_NOT_SUFFICIENT
            )


        elif answer_upper == WEBSITE_SEARCH_NO_RESULTS:

            internal_answer_marker = (
                WEBSITE_SEARCH_NO_RESULTS
            )


        elif answer_upper.startswith(
            "SERPER_ERROR"
        ):

            internal_answer_marker = (
                answer
            )


    # =====================================================
    # ROUTE EVALUATION
    # =====================================================

    route_pass = (
        actual_route
        == expected_route
    )


    # =====================================================
    # KEYWORD / MEANING EVALUATION
    # =====================================================

    (
        found_keywords,
        missing_keywords,
        keyword_pass
    ) = evaluate_expected_keywords(

        answer=answer,

        expected_keywords=expected_keywords
    )


    # =====================================================
    # OVERALL RESULT
    # =====================================================

    overall_pass = (

        route_pass

        and keyword_pass

        and execution_error is None

        and internal_answer_marker is None
    )


    # =====================================================
    # TIMING
    # =====================================================

    elapsed_seconds = round(
        time.perf_counter()
        - start_time,
        3
    )


    # =====================================================
    # DETAILED LOGGING
    # =====================================================

    logger.info(
        "Actual route: %s",
        actual_route
    )


    logger.info(
        "Route result: %s",
        (
            "PASS"
            if route_pass
            else "FAIL"
        )
    )


    logger.info(
        "Found keywords: %s",
        found_keywords
    )


    logger.info(
        "Missing keywords: %s",
        missing_keywords
    )


    logger.info(
        "Keyword result: %s",
        (
            "PASS"
            if keyword_pass
            else "FAIL"
        )
    )


    logger.info(
        "Answer length: %d characters",
        len(answer)
    )


    logger.info(
        "Answer:\n%s",
        answer
    )


    if fallback_reason:

        logger.info(
            "Fallback reason: %s",
            fallback_reason
        )


    if internal_answer_marker:

        logger.warning(
            "Internal failure marker returned: %s",
            internal_answer_marker
        )


    if execution_error:

        logger.error(
            "Execution error: %s",
            execution_error
        )


    logger.info(
        "Elapsed time: %.3f seconds",
        elapsed_seconds
    )


    logger.info(
        "CASE RESULT: %s",
        (
            "PASS"
            if overall_pass
            else "FAIL"
        )
    )


    # =====================================================
    # RETURN RESULT
    # =====================================================

    return {

        "timestamp":
            started_at,

        "question":
            question,

        "expected_route":
            expected_route,

        "actual_route":
            actual_route,

        "route_pass":
            route_pass,

        "expected_keywords":
            expected_keywords,

        "found_keywords":
            found_keywords,

        "missing_keywords":
            missing_keywords,

        "keyword_pass":
            keyword_pass,

        "fallback_reason":
            fallback_reason,

        "internal_answer_marker":
            internal_answer_marker,

        "execution_error":
            execution_error,

        "elapsed_seconds":
            elapsed_seconds,

        "answer":
            answer,

        "overall_pass":
            overall_pass
    }


# =========================================================
# VALIDATE EVALUATION DATA
# =========================================================

def validate_eval_cases(
    cases
):
    """
    Validate eval_cases.json before running expensive tests.
    """

    if not isinstance(
        cases,
        list
    ):

        raise ValueError(
            "eval_cases.json must contain a JSON list."
        )


    if not cases:

        raise ValueError(
            "eval_cases.json contains no evaluation cases."
        )


    for index, case in enumerate(
        cases,
        start=1
    ):

        if not isinstance(
            case,
            dict
        ):

            raise ValueError(
                f"Evaluation case {index} must be an object."
            )


        if not case.get(
            "question"
        ):

            raise ValueError(
                f"Evaluation case {index} has no question."
            )


        expected_route = case.get(
            "expected_route"
        )


        if expected_route not in {
            "agent_1",
            "agent_2"
        }:

            raise ValueError(
                f"Evaluation case {index} has invalid "
                f"expected_route: {expected_route}"
            )


        expected_keywords = case.get(
            "expected_keywords",
            []
        )


        if not isinstance(
            expected_keywords,
            list
        ):

            raise ValueError(
                f"Evaluation case {index}: "
                "expected_keywords must be a list."
            )


# =========================================================
# RUN ALL EVALUATIONS
# =========================================================

def run_evaluations():

    logger.info("")

    logger.info(
        "#" * 70
    )

    logger.info(
        "STARTING JITS EVALUATION RUN"
    )

    logger.info(
        "#" * 70
    )


    run_started = (
        datetime.now()
        .astimezone()
    )


    # =====================================================
    # LOAD EVALUATION CASES
    # =====================================================

    if not EVAL_FILE.exists():

        raise FileNotFoundError(
            f"{EVAL_FILE} does not exist."
        )


    try:

        cases = json.loads(
            EVAL_FILE.read_text(
                encoding="utf-8"
            )
        )


    except json.JSONDecodeError:

        logger.exception(
            "Invalid JSON in eval_cases.json."
        )

        raise


    # =====================================================
    # VALIDATE EVALUATION CASES
    # =====================================================

    validate_eval_cases(
        cases
    )


    logger.info(
        "Loaded %d evaluation cases.",
        len(cases)
    )


    # =====================================================
    # RUN CASES
    # =====================================================

    results = []


    for index, case in enumerate(
        cases,
        start=1
    ):

        result = run_eval_case(

            case=case,

            case_number=index,

            total_cases=len(cases)
        )


        results.append(
            result
        )


    # =====================================================
    # SUMMARY CALCULATIONS
    # =====================================================

    total = len(results)


    passed = sum(
        1
        for item in results
        if item["overall_pass"]
    )


    failed = (
        total - passed
    )


    route_correct = sum(
        1
        for item in results
        if item["route_pass"]
    )


    keyword_correct = sum(
        1
        for item in results
        if item["keyword_pass"]
    )


    execution_errors = sum(
        1
        for item in results
        if item["execution_error"]
    )


    routing_accuracy = (
        (route_correct / total) * 100
        if total
        else 0
    )


    keyword_accuracy = (
        (keyword_correct / total) * 100
        if total
        else 0
    )


    overall_pass_rate = (
        (passed / total) * 100
        if total
        else 0
    )


    total_duration = round(
        (
            datetime.now().astimezone()
            - run_started
        ).total_seconds(),
        3
    )


    average_duration = round(
        (
            sum(
                item["elapsed_seconds"]
                for item in results
            )
            / total
        ),
        3
    ) if total else 0


    # =====================================================
    # SUMMARY OBJECT
    # =====================================================

    summary = {

        "total_tests":
            total,

        "passed":
            passed,

        "failed":
            failed,

        "execution_errors":
            execution_errors,

        "routing_accuracy":
            round(
                routing_accuracy,
                2
            ),

        "keyword_accuracy":
            round(
                keyword_accuracy,
                2
            ),

        "overall_pass_rate":
            round(
                overall_pass_rate,
                2
            ),

        "total_duration_seconds":
            total_duration,

        "average_case_duration_seconds":
            average_duration
    }


    # =====================================================
    # LOG SUMMARY
    # =====================================================

    logger.info("")

    logger.info(
        "=" * 70
    )

    logger.info(
        "EVALUATION SUMMARY"
    )

    logger.info(
        "=" * 70
    )


    logger.info(
        "Total tests: %d",
        total
    )


    logger.info(
        "Passed: %d",
        passed
    )


    logger.info(
        "Failed: %d",
        failed
    )


    logger.info(
        "Execution Errors: %d",
        execution_errors
    )


    logger.info(
        "Routing Accuracy: %.2f%%",
        routing_accuracy
    )


    logger.info(
        "Keyword Accuracy: %.2f%%",
        keyword_accuracy
    )


    logger.info(
        "Overall Eval Pass Rate: %.2f%%",
        overall_pass_rate
    )


    logger.info(
        "Average Case Duration: %.3f seconds",
        average_duration
    )


    logger.info(
        "Total Duration: %.3f seconds",
        total_duration
    )


    # =====================================================
    # FAILED CASE SUMMARY
    # =====================================================

    failed_results = [

        item

        for item in results

        if not item["overall_pass"]
    ]


    if failed_results:

        logger.warning("")

        logger.warning(
            "FAILED CASES"
        )


        for item in failed_results:

            logger.warning(
                "Question: %s",
                item["question"]
            )


            logger.warning(
                "Expected route: %s",
                item["expected_route"]
            )


            logger.warning(
                "Actual route: %s",
                item["actual_route"]
            )


            logger.warning(
                "Route pass: %s",
                item["route_pass"]
            )


            logger.warning(
                "Missing keywords: %s",
                item["missing_keywords"]
            )


            logger.warning(
                "Execution error: %s",
                item["execution_error"]
            )


            logger.warning(
                "Answer: %s",
                item["answer"]
            )


    # =====================================================
    # SAVE RESULTS
    # =====================================================

    run_timestamp = (
        run_started.strftime(
            "%Y%m%d_%H%M%S"
        )
    )


    payload = {

        "run_timestamp":
            run_started.isoformat(
                timespec="seconds"
            ),

        "summary":
            summary,

        "results":
            results
    }


    # =====================================================
    # SAVE LATEST RESULT
    # =====================================================

    RESULT_FILE.write_text(

        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False
        ),

        encoding="utf-8"
    )


    # =====================================================
    # SAVE HISTORICAL RESULT
    # =====================================================

    history_file = (

        EVAL_HISTORY_DIR

        / f"eval_run_{run_timestamp}.json"
    )


    history_file.write_text(

        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False
        ),

        encoding="utf-8"
    )


    logger.info(
        "Latest results saved: %s",
        RESULT_FILE
    )


    logger.info(
        "Historical results saved: %s",
        history_file
    )


    return results


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    try:

        run_evaluations()


    except KeyboardInterrupt:

        logger.warning(
            "Evaluation run stopped by user."
        )


    except Exception:

        logger.exception(
            "Evaluation run terminated unexpectedly."
        )

        raise