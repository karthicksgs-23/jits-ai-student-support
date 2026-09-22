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

EVAL_FILE = Path(
    "eval_cases.json"
)

RESULT_FILE = Path(
    "eval_results.json"
)

LOG_DIR = Path(
    "logs"
)

EVAL_HISTORY_DIR = (
    LOG_DIR / "eval_runs"
)

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

RAG_NOT_SUFFICIENT = (
    "RAG_NOT_SUFFICIENT"
)

SELF_RAG_INSUFFICIENT = (
    "SELF_RAG_INSUFFICIENT"
)

WEBSITE_NOT_SUFFICIENT = (
    "WEBSITE_NOT_SUFFICIENT"
)


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

    console_handler = (
        logging.StreamHandler()
    )

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
# RUN ONE CASE
# =========================================================

def run_eval_case(
    case: dict,
    case_number: int = 1,
    total_cases: int = 1
):
    """
    Execute one evaluation case and return a detailed result.
    """

    started_at = (
        datetime.now()
        .astimezone()
        .isoformat(
            timespec="seconds"
        )
    )

    start_time = (
        time.perf_counter()
    )


    question = (
        case["question"]
    )

    expected_route = (
        case["expected_route"]
    )

    expected_keywords = (
        case.get(
            "expected_keywords",
            []
        )
    )


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
    # DEFAULT RESULT VALUES
    # =====================================================

    actual_route = None

    answer = ""

    execution_error = None


    # =====================================================
    # AGENT 1
    # =====================================================

    try:

        logger.info(
            "Running Agent 1..."
        )

        rag_answer = (
            run_rag_agent(
                question
            )
        )


        if rag_answer is None:

            raise ValueError(
                "Agent 1 returned None."
            )


        rag_answer = (
            rag_answer.strip()
        )


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


    # =====================================================
    # ROUTING
    # =====================================================

    needs_agent_2 = (

        rag_answer is None

        or (
            rag_answer
            and rag_answer.upper()
            in {
                RAG_NOT_SUFFICIENT,
                SELF_RAG_INSUFFICIENT
            }
        )
    )


    # =====================================================
    # AGENT 2
    # =====================================================

    if needs_agent_2:

        actual_route = (
            "agent_2"
        )


        logger.info(
            "Routing decision: Agent 2"
        )


        try:

            logger.info(
                "Running Agent 2..."
            )


            answer = (
                run_website_agent(
                    question
                )
            )


            if answer is None:

                raise ValueError(
                    "Agent 2 returned None."
                )


            answer = (
                answer.strip()
            )


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

        actual_route = (
            "agent_1"
        )

        answer = (
            rag_answer
        )


        logger.info(
            "Routing decision: Agent 1"
        )


    # =====================================================
    # ROUTE EVALUATION
    # =====================================================

    route_pass = (
        actual_route
        == expected_route
    )


    # =====================================================
    # KEYWORD EVALUATION
    # =====================================================

    answer_lower = (
        answer.lower()
    )


    found_keywords = []

    missing_keywords = []


    for keyword in expected_keywords:

        if (
            keyword.lower()
            in answer_lower
        ):

            found_keywords.append(
                keyword
            )

        else:

            missing_keywords.append(
                keyword
            )


    keyword_pass = (
        len(
            missing_keywords
        )
        == 0
    )


    # =====================================================
    # OVERALL PASS
    # =====================================================

    overall_pass = (

        route_pass

        and keyword_pass

        and execution_error is None
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
    # RESULT
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
# RUN ALL EVALUATIONS
# =========================================================

def run_evaluations():

    logger.info(
        ""
    )

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
    # LOAD CASES
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


    if not isinstance(
        cases,
        list
    ):

        raise ValueError(
            "eval_cases.json must contain a JSON list."
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
    # SUMMARY
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


    summary = {

        "total_tests":
            total,

        "passed":
            passed,

        "failed":
            failed,

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
            total_duration
    }


    # =====================================================
    # LOG SUMMARY
    # =====================================================

    logger.info(
        ""
    )

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
        "Total Duration: %.3f seconds",
        total_duration
    )


    # =====================================================
    # FAILED CASE SUMMARY
    # =====================================================

    failed_results = [

        item

        for item in results

        if not item[
            "overall_pass"
        ]
    ]


    if failed_results:

        logger.warning(
            ""
        )

        logger.warning(
            "FAILED CASES"
        )


        for item in failed_results:

            logger.warning(
                "Question: %s",
                item["question"]
            )

            logger.warning(
                "Expected=%s | Actual=%s | "
                "Missing=%s | Error=%s",
                item["expected_route"],
                item["actual_route"],
                item["missing_keywords"],
                item["execution_error"]
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


    # -----------------------------------------------------
    # LATEST RESULTS
    # -----------------------------------------------------

    RESULT_FILE.write_text(

        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False
        ),

        encoding="utf-8"
    )


    # -----------------------------------------------------
    # HISTORICAL RUN
    # -----------------------------------------------------

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

    except Exception:

        logger.exception(
            "Evaluation run terminated unexpectedly."
        )

        raise