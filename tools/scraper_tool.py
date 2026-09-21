from io import BytesIO
from pathlib import Path

import pymupdf
import pytesseract

from PIL import Image
from scrapling.fetchers import Fetcher


# =========================================================
# CONFIGURATION
# =========================================================

PDF_URL = (
    "https://jits.ac.in/wp-content/uploads/"
    "2026/08/B.Tech-J-25-Academic-Regulations.pdf"
)

DATA_DIR = Path("data")

OUTPUT_FILE = DATA_DIR / "academic_regulations.txt"

PDF_BACKUP_FILE = DATA_DIR / "academic_regulations.pdf"


# ---------------------------------------------------------
# Explicit Windows Tesseract path
# ---------------------------------------------------------

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


# =========================================================
# FETCH + OCR
# =========================================================

def fetch_academic_regulations():

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\n==========================================")
    print("FETCHING J-25 ACADEMIC REGULATIONS")
    print("==========================================")

    print(
        f"\nURL:\n{PDF_URL}"
    )


    # =====================================================
    # STEP 1
    # Fetch PDF using Scrapling
    # =====================================================

    print(
        "\n[1/4] Downloading PDF using Scrapling..."
    )

    response = Fetcher.get(
        PDF_URL
    )

    print(
        f"HTTP Status: {response.status}"
    )

    if response.status != 200:

        raise RuntimeError(
            f"Failed to fetch PDF. "
            f"HTTP status: {response.status}"
        )


    pdf_bytes = response.body


    if not pdf_bytes:

        raise RuntimeError(
            "Scrapling returned empty PDF content."
        )


    print(
        f"Downloaded: {len(pdf_bytes):,} bytes"
    )


    # =====================================================
    # STEP 2
    # Save PDF backup
    # =====================================================

    PDF_BACKUP_FILE.write_bytes(
        pdf_bytes
    )

    print(
        f"\nPDF backup saved to: "
        f"{PDF_BACKUP_FILE}"
    )


    # =====================================================
    # STEP 3
    # Open PDF with PyMuPDF
    # =====================================================

    print(
        "\n[2/4] Opening PDF..."
    )


    document = pymupdf.open(
        stream=pdf_bytes,
        filetype="pdf"
    )


    total_pages = document.page_count


    print(
        f"Total PDF pages: {total_pages}"
    )


    # =====================================================
    # STEP 4
    # OCR every page
    # =====================================================

    print(
        "\n[3/4] Starting OCR..."
    )


    extracted_pages = []


    for page_index in range(
        total_pages
    ):

        page_number = (
            page_index + 1
        )


        print(
            f"\nOCR page "
            f"{page_number}/{total_pages}..."
        )


        page = document.load_page(
            page_index
        )


        # -------------------------------------------------
        # Render PDF page to high-resolution image
        # -------------------------------------------------

        pixmap = page.get_pixmap(
            dpi=250,
            alpha=False
        )


        image_bytes = (
            pixmap.tobytes("png")
        )


        image = Image.open(
            BytesIO(image_bytes)
        )


        # -------------------------------------------------
        # Convert to grayscale to improve OCR
        # -------------------------------------------------

        image = image.convert(
            "L"
        )


        # -------------------------------------------------
        # Perform OCR
        # -------------------------------------------------

        text = pytesseract.image_to_string(
            image,
            lang="eng",
            config="--oem 3 --psm 6"
        )


        text = text.strip()


        if text:

            print(
                f"Extracted "
                f"{len(text):,} characters"
            )


            page_content = f"""
============================================================
PAGE {page_number}
============================================================

{text}
"""


            extracted_pages.append(
                page_content.strip()
            )


        else:

            print(
                "WARNING: OCR returned no text."
            )


    document.close()


    # =====================================================
    # STEP 5
    # Combine OCR text
    # =====================================================

    print(
        "\n[4/4] Saving OCR text..."
    )


    full_text = "\n\n".join(
        extracted_pages
    )


    if not full_text.strip():

        raise RuntimeError(
            "OCR did not extract any text."
        )


    OUTPUT_FILE.write_text(
        full_text,
        encoding="utf-8"
    )


    # =====================================================
    # SUMMARY
    # =====================================================

    print(
        "\n=========================================="
    )

    print(
        "OCR EXTRACTION COMPLETE"
    )

    print(
        "=========================================="
    )


    print(
        f"\nSaved text to:\n"
        f"{OUTPUT_FILE}"
    )


    print(
        f"\nTotal extracted characters: "
        f"{len(full_text):,}"
    )


    print(
        f"\nPages processed: "
        f"{total_pages}"
    )


    return full_text


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    fetch_academic_regulations()