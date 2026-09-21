import json
from pathlib import Path

from dotenv import load_dotenv

from agents.support_agents import (
    run_rag_agent,
    run_website_agent,
)


# =========================================================
# CONFIGURATION
# =========================================================

load_dotenv()

EVAL_FILE = Path("eval_cases.json")


# =========================================================
# RUN ONE EVALUATION CASE
# =========================================================

def run_eval_case(case: dict) -> dict:

    question = case["question"]

    expected_route = case["expected_route"]

    expected_keywords = case.get(
        "expected_keywords",
        []
    )


    print("\n")
    print("=" * 70)
    print("EVALUATION CASE")
    print("=" * 70)

    print(
        f"\nQuestion:\n{question}"
    )

    print(
        f"\nExpected route: "
        f"{expected_route}"
    )


    # =====================================================
    # AGENT 1
    # =====================================================

    rag_answer = run_rag_agent(
        question
    ).strip()


    # =====================================================
    # ROUTING
    # =====================================================

    if (
        rag_answer.upper()
        == "RAG_NOT_SUFFICIENT"
    ):

        actual_route = "agent_2"

        answer = run_website_agent(
            question
        ).strip()

    else:

        actual_route = "agent_1"

        answer = rag_answer


    # =====================================================
    # ROUTING EVALUATION
    # =====================================================

    route_pass = (
        actual_route
        == expected_route
    )


    # =====================================================
    # KEYWORD EVALUATION
    # =====================================================

    answer_lower = answer.lower()

    found_keywords = []

    missing_keywords = []


    for keyword in expected_keywords:

        if keyword.lower() in answer_lower:

            found_keywords.append(
                keyword
            )

        else:

            missing_keywords.append(
                keyword
            )


    keyword_pass = (
        len(missing_keywords) == 0
    )


    # =====================================================
    # OVERALL RESULT
    # =====================================================

    overall_pass = (
        route_pass
        and keyword_pass
    )


    return {

        "question":
            question,

        "expected_route":
            expected_route,

        "actual_route":
            actual_route,

        "route_pass":
            route_pass,

        "found_keywords":
            found_keywords,

        "missing_keywords":
            missing_keywords,

        "keyword_pass":
            keyword_pass,

        "overall_pass":
            overall_pass,

        "answer":
            answer
    }


# =========================================================
# RUN ALL EVALUATIONS
# =========================================================

def run_evaluations():

    if not EVAL_FILE.exists():

        raise FileNotFoundError(
            f"{EVAL_FILE} not found."
        )


    cases = json.loads(
        EVAL_FILE.read_text(
            encoding="utf-8"
        )
    )


    results = []


    for index, case in enumerate(
        cases,
        start=1
    ):

        print("\n")
        print("#" * 70)

        print(
            f"RUNNING TEST "
            f"{index}/{len(cases)}"
        )

        print("#" * 70)


        result = run_eval_case(
            case
        )


        results.append(
            result
        )


        print("\n")
        print("-" * 70)

        print(
            f"Actual route: "
            f"{result['actual_route']}"
        )

        print(
            f"Route test: "
            f"{'PASS' if result['route_pass'] else 'FAIL'}"
        )

        print(
            f"Keyword test: "
            f"{'PASS' if result['keyword_pass'] else 'FAIL'}"
        )


        if result[
            "missing_keywords"
        ]:

            print(
                "Missing keywords: "
                + ", ".join(
                    result[
                        "missing_keywords"
                    ]
                )
            )


        print(
            f"Overall: "
            f"{'PASS' if result['overall_pass'] else 'FAIL'}"
        )


    # =====================================================
    # FINAL SUMMARY
    # =====================================================

    total = len(results)

    passed = sum(
        1
        for result in results
        if result["overall_pass"]
    )

    failed = total - passed


    route_correct = sum(
        1
        for result in results
        if result["route_pass"]
    )


    print("\n")
    print("=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)


    print(
        f"\nTotal tests: {total}"
    )

    print(
        f"Passed: {passed}"
    )

    print(
        f"Failed: {failed}"
    )


    if total > 0:

        routing_accuracy = (
            route_correct
            / total
        ) * 100

        overall_accuracy = (
            passed
            / total
        ) * 100

    else:

        routing_accuracy = 0

        overall_accuracy = 0


    print(
        f"\nRouting Accuracy: "
        f"{routing_accuracy:.2f}%"
    )

    print(
        f"Overall Eval Pass Rate: "
        f"{overall_accuracy:.2f}%"
    )


    return results


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    run_evaluations()