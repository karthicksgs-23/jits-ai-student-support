import os
import requests

from dotenv import load_dotenv
from crewai.tools import tool
from scrapling.fetchers import Fetcher


# =========================================================
# ENVIRONMENT
# =========================================================

load_dotenv()

SERPER_API_KEY = os.getenv(
    "SERPER_API_KEY"
)

SERPER_URL = (
    "https://google.serper.dev/search"
)

MAX_RESULTS = 3
MAX_PAGE_CHARS = 12000


# =========================================================
# FETCH ACTUAL JITS PAGE
# =========================================================

def fetch_jits_page(url: str) -> str:

    try:

        print(
            f"\nFetching actual JITS page:\n{url}"
        )

        page = Fetcher.get(
            url
        )

        # Try cleaner page extraction first
        content = page.markdown(
            main_content_only=True
        )

        # Fallback
        if (
            not content
            or len(content.strip()) < 200
        ):

            content = page.markdown()

        if not content:

            return ""

        content = content.strip()

        return content[
            :MAX_PAGE_CHARS
        ]

    except Exception as error:

        print(
            f"Could not fetch page: {error}"
        )

        return ""


# =========================================================
# SERPER SEARCH
# =========================================================

def search_jits_with_serper(
    question: str,
    num_results: int = MAX_RESULTS
) -> str:

    if not SERPER_API_KEY:

        return (
            "SERPER_ERROR: "
            "SERPER_API_KEY is not configured."
        )


    # -----------------------------------------------------
    # Search only official JITS website
    # -----------------------------------------------------

    search_query = (
        f"site:jits.ac.in {question}"
    )


    print("\n==========================================")
    print("AGENT 2 - SERPER FALLBACK")
    print("==========================================")

    print(
        f"\nQuestion:\n{question}"
    )

    print(
        f"\nSerper query:\n{search_query}"
    )


    headers = {

        "X-API-KEY":
            SERPER_API_KEY,

        "Content-Type":
            "application/json"
    }


    payload = {

        "q":
            search_query,

        "num":
            num_results
    }


    try:

        # =================================================
        # STEP 1
        # Search with Serper
        # =================================================

        response = requests.post(

            SERPER_URL,

            headers=headers,

            json=payload,

            timeout=20
        )


        print(
            f"\nSerper HTTP Status: "
            f"{response.status_code}"
        )


        if response.status_code != 200:

            return (
                "SERPER_ERROR: "
                f"HTTP {response.status_code}"
            )


        data = response.json()


        organic_results = data.get(
            "organic",
            []
        )


        if not organic_results:

            return (
                "WEBSITE_SEARCH_NO_RESULTS"
            )


        final_results = []


        # =================================================
        # STEP 2
        # Process top official JITS results
        # =================================================

        result_number = 0


        for result in organic_results:

            link = result.get(
                "link",
                ""
            )

            title = result.get(
                "title",
                ""
            )

            snippet = result.get(
                "snippet",
                ""
            )


            # ---------------------------------------------
            # Official JITS only
            # ---------------------------------------------

            if "jits.ac.in" not in link:

                continue


            # For now focus on HTML pages
            if link.lower().endswith(
                ".pdf"
            ):

                continue


            result_number += 1


            print("\n------------------------------------------")

            print(
                f"Result {result_number}"
            )

            print(
                f"Title: {title}"
            )

            print(
                f"URL: {link}"
            )


            # =================================================
            # STEP 3
            # Fetch ACTUAL PAGE CONTENT
            # =================================================

            actual_content = fetch_jits_page(
                link
            )


            if not actual_content:

                actual_content = (
                    "Full page could not be fetched.\n\n"
                    "Search snippet:\n"
                    + snippet
                )


            final_results.append(
                f"""
============================================================
OFFICIAL JITS SOURCE {result_number}
============================================================

TITLE:
{title}

URL:
{link}

SERPER SNIPPET:
{snippet}

------------------------------------------------------------
ACTUAL PAGE CONTENT
------------------------------------------------------------

{actual_content}
"""
            )


            if result_number >= num_results:

                break


        # =================================================
        # NO RESULTS
        # =================================================

        if not final_results:

            return (
                "WEBSITE_SEARCH_NO_RESULTS"
            )


        return "\n\n".join(
            final_results
        )


    except requests.exceptions.Timeout:

        return (
            "SERPER_ERROR: "
            "Serper request timed out."
        )


    except requests.exceptions.RequestException as error:

        return (
            "SERPER_ERROR: "
            + str(error)
        )


    except Exception as error:

        return (
            "SERPER_ERROR: "
            + str(error)
        )


# =========================================================
# CREWAI TOOL
# =========================================================

@tool("JITS Serper Website Search")
def jits_website_search(
    question: str
) -> str:

    """
    Fallback website search for JITS customer support.

    Use only when Agent 1 returns RAG_NOT_SUFFICIENT.

    The tool:

    1. Searches Google using Serper.
    2. Restricts results to jits.ac.in.
    3. Identifies relevant official JITS pages.
    4. Fetches the actual page using Scrapling.
    5. Returns the page content as evidence.

    Answer only from the returned official evidence.
    """

    return search_jits_with_serper(
        question
    )


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    question = input(
        "\nEnter search question: "
    )


    result = search_jits_with_serper(
        question
    )


    print("\n")
    print("=" * 70)
    print("SERPER + ACTUAL WEBSITE RESULTS")
    print("=" * 70)

    print(result)