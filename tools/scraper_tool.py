from io import BytesIO
from pathlib import Path


import pymupdf
import pytesseract
from PIL import Image, UnidentifiedImageError
from scrapling.fetchers import Fetcher


# =========================================================
# CONFIGURATION
# =========================================================

PDF_URL = (
    "https://jits.ac.in/wp-content/uploads/2026/08/"
    "B.Tech-J-25-Academic-Regulations.pdf"
)

DATA_DIR = Path("data")

OUTPUT_FILE = DATA_DIR / "academic_regulations.txt"

BACKUP_PDF = DATA_DIR / "academic_regulations.pdf"


TESSERACT_PATH = Path(
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


# =========================================================
# CUSTOM EXCEPTION
# =========================================================

class ScraperError(Exception):
    """Raised when the Academic Regulations scraper fails."""

    pass


# =========================================================
# CHECK TESSERACT
# =========================================================

def configure_tesseract():
    """
    Verify that Tesseract OCR is installed and configure pytesseract.
    """

    if not TESSERACT_PATH.exists():

        raise ScraperError(
            "Tesseract OCR was not found at: "
            f"{TESSERACT_PATH}"
        )


    pytesseract.pytesseract.tesseract_cmd = str(
        TESSERACT_PATH
    )


# =========================================================
# DOWNLOAD PDF
# =========================================================

def download_pdf() -> bytes:
    """
    Download the official JITS Academic Regulations PDF.

    Returns:
        PDF file contents as bytes.

    Raises:
        ScraperError if download fails.
    """

    try:

        print(
            f"Downloading Academic Regulations from:\n"
            f"{PDF_URL}"
        )

        response = Fetcher.get(
    PDF_URL,
    timeout=120,
    retries=3,
    retry_delay=2
)

    except Exception as error:

        raise ScraperError(
            "Unable to connect to the JITS website. "
            f"Original error: {error}"
        ) from error


    # -----------------------------------------------------
    # STATUS CHECK
    # -----------------------------------------------------

    status = getattr(
        response,
        "status",
        None
    )


    if status != 200:

        raise ScraperError(
            "JITS server returned an unexpected HTTP status: "
            f"{status}"
        )


    # -----------------------------------------------------
    # BODY CHECK
    # -----------------------------------------------------

    pdf_bytes = getattr(
        response,
        "body",
        None
    )


    if not pdf_bytes:

        raise ScraperError(
            "The JITS server returned an empty response."
        )


    if not isinstance(
        pdf_bytes,
        (bytes, bytearray)
    ):

        raise ScraperError(
            "The downloaded response is not valid binary data."
        )


    # -----------------------------------------------------
    # PDF SIGNATURE CHECK
    # -----------------------------------------------------

    if not pdf_bytes.startswith(
        b"%PDF"
    ):

        raise ScraperError(
            "The downloaded file does not appear to be a PDF. "
            "The website may have returned an HTML error page."
        )


    # -----------------------------------------------------
    # FILE SIZE CHECK
    # -----------------------------------------------------

    if len(pdf_bytes) < 1000:

        raise ScraperError(
            "The downloaded PDF is unexpectedly small and may "
            "be incomplete."
        )


    print(
        f"PDF downloaded successfully "
        f"({len(pdf_bytes):,} bytes)."
    )


    return bytes(pdf_bytes)


# =========================================================
# SAVE PDF BACKUP
# =========================================================

def save_pdf_backup(
    pdf_bytes: bytes
):
    """
    Save a local backup copy of the downloaded PDF.
    """

    try:

        DATA_DIR.mkdir(
            parents=True,
            exist_ok=True
        )


        BACKUP_PDF.write_bytes(
            pdf_bytes
        )


        print(
            f"Backup PDF saved to: "
            f"{BACKUP_PDF}"
        )


    except OSError as error:

        raise ScraperError(
            "Unable to save the Academic Regulations PDF backup. "
            f"Original error: {error}"
        ) from error


# =========================================================
# OPEN PDF
# =========================================================

def open_pdf(
    pdf_bytes: bytes
):
    """
    Open downloaded PDF using PyMuPDF.
    """

    try:

        document = pymupdf.open(
            stream=pdf_bytes,
            filetype="pdf"
        )


    except Exception as error:

        raise ScraperError(
            "The downloaded file could not be opened as a PDF. "
            f"Original error: {error}"
        ) from error


    if document.page_count == 0:

        document.close()

        raise ScraperError(
            "The Academic Regulations PDF contains no pages."
        )


    print(
        f"PDF opened successfully. "
        f"Pages: {document.page_count}"
    )


    return document


# =========================================================
# OCR ONE PAGE
# =========================================================

def ocr_page(
    page,
    page_number: int
) -> str:
    """
    Render one PDF page and perform OCR.

    If one page fails, return an error marker instead of
    crashing the entire scraper.
    """

    try:

        pixmap = page.get_pixmap(
            dpi=250,
            alpha=False
        )


        png_bytes = pixmap.tobytes(
            "png"
        )


        image = Image.open(
            BytesIO(
                png_bytes
            )
        )


        image = image.convert(
            "L"
        )


        text = pytesseract.image_to_string(

            image,

            lang="eng",

            config="--oem 3 --psm 6"
        )


        text = text.strip()


        if not text:

            print(
                f"Warning: No text detected on page "
                f"{page_number}."
            )

            return (
                f"[OCR WARNING: No text detected on "
                f"page {page_number}]"
            )


        print(
            f"OCR completed for page "
            f"{page_number}."
        )


        return text


    except pytesseract.TesseractNotFoundError as error:

        raise ScraperError(
            "Tesseract OCR is not available. "
            "Please verify the Tesseract installation."
        ) from error


    except pytesseract.TesseractError as error:

        print(
            f"Warning: OCR failed on page "
            f"{page_number}: {error}"
        )

        return (
            f"[OCR ERROR: Page {page_number} "
            f"could not be processed]"
        )


    except UnidentifiedImageError as error:

        print(
            f"Warning: Page {page_number} could "
            f"not be converted into an image: {error}"
        )

        return (
            f"[IMAGE ERROR: Page {page_number} "
            f"could not be rendered]"
        )


    except Exception as error:

        print(
            f"Warning: Unexpected error while processing "
            f"page {page_number}: {error}"
        )

        return (
            f"[PAGE ERROR: Page {page_number} "
            f"could not be processed]"
        )


# =========================================================
# OCR COMPLETE PDF
# =========================================================

def extract_pdf_text(
    pdf_bytes: bytes
) -> str:
    """
    OCR all pages in the Academic Regulations PDF.
    """

    configure_tesseract()


    document = open_pdf(
        pdf_bytes
    )


    extracted_pages = []

    successful_pages = 0


    try:

        for index in range(
            document.page_count
        ):

            page_number = index + 1


            print(
                f"\nProcessing page "
                f"{page_number}/{document.page_count}..."
            )


            try:

                page = document.load_page(
                    index
                )


                text = ocr_page(
                    page,
                    page_number
                )


                if not text.startswith(
                    "["
                ):

                    successful_pages += 1


                extracted_pages.append(

                    f"\n{'=' * 60}\n"
                    f"PAGE {page_number}\n"
                    f"{'=' * 60}\n\n"
                    f"{text}"
                )


            except Exception as error:

                print(
                    f"Warning: Unable to load page "
                    f"{page_number}: {error}"
                )


                extracted_pages.append(

                    f"\n{'=' * 60}\n"
                    f"PAGE {page_number}\n"
                    f"{'=' * 60}\n\n"
                    f"[PAGE ERROR: Could not load page]"
                )


    finally:

        document.close()


    # -----------------------------------------------------
    # ENSURE OCR ACTUALLY WORKED
    # -----------------------------------------------------

    if successful_pages == 0:

        raise ScraperError(
            "OCR failed for every page in the PDF. "
            "No usable Academic Regulations text was extracted."
        )


    print(
        f"\nOCR completed successfully for "
        f"{successful_pages}/{len(extracted_pages)} pages."
    )


    return "\n".join(
        extracted_pages
    )


# =========================================================
# SAVE EXTRACTED TEXT
# =========================================================

def save_extracted_text(
    text: str
):
    """
    Safely save OCR text.

    Uses a temporary file first so an existing good RAG
    source is not destroyed if writing fails halfway.
    """

    if not text.strip():

        raise ScraperError(
            "Cannot save empty OCR output."
        )


    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    temporary_file = (
        OUTPUT_FILE.with_suffix(
            ".tmp"
        )
    )


    try:

        temporary_file.write_text(

            text,

            encoding="utf-8"
        )


        # Replace existing file only after successful write
        temporary_file.replace(
            OUTPUT_FILE
        )


    except OSError as error:

        # Clean temporary file if possible
        try:

            if temporary_file.exists():

                temporary_file.unlink()

        except OSError:

            pass


        raise ScraperError(
            "Unable to save extracted Academic Regulations text. "
            f"Original error: {error}"
        ) from error


    print(
        f"\nAcademic Regulations text saved to:\n"
        f"{OUTPUT_FILE}"
    )


# =========================================================
# MAIN SCRAPER
# =========================================================

def scrape_academic_regulations() -> str:
    """
    Complete Academic Regulations download + OCR workflow.

    Returns:
        Path to generated text file.

    Raises:
        ScraperError when the workflow cannot produce a
        usable output file.
    """

    try:

        # -------------------------------------------------
        # 1. Download
        # -------------------------------------------------

        pdf_bytes = download_pdf()


        # -------------------------------------------------
        # 2. Backup
        # -------------------------------------------------

        save_pdf_backup(
            pdf_bytes
        )


        # -------------------------------------------------
        # 3. OCR
        # -------------------------------------------------

        extracted_text = extract_pdf_text(
            pdf_bytes
        )


        # -------------------------------------------------
        # 4. Save
        # -------------------------------------------------

        save_extracted_text(
            extracted_text
        )


        print(
            "\nAcademic Regulations scraper "
            "completed successfully."
        )


        return str(
            OUTPUT_FILE
        )


    except ScraperError:

        # Preserve our clean custom errors
        raise


    except KeyboardInterrupt:

        print(
            "\nScraper stopped by user."
        )

        raise


    except Exception as error:

        raise ScraperError(
            "An unexpected scraper error occurred. "
            f"Original error: {error}"
        ) from error


# =========================================================
# RUN DIRECTLY
# =========================================================

if __name__ == "__main__":

    try:

        output = scrape_academic_regulations()


        print(
            f"\nSuccess: {output}"
        )


    except ScraperError as error:

        print("\n")
        print("=" * 70)
        print("SCRAPER ERROR")
        print("=" * 70)

        print(
            f"\n{error}"
        )