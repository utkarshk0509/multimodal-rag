import os
import logging
import pymupdf as fitz  # PyMuPDF 2.0+: fitz is the legacy alias

logger = logging.getLogger(__name__)

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
DEFAULT_PDF_PATH = os.path.join(_PROJECT_ROOT, "data", "SAMPLE.pdf")


def test_read_pdf(file_path: str = DEFAULT_PDF_PATH) -> None:
    """Quick sanity-check: open the PDF and print the first 500 chars of page 1."""
    logger.info("Attempting to open: %s", file_path)

    try:
        doc = fitz.open(file_path)
    except fitz.FileNotFoundError:
        logger.error("File not found: %s", file_path)
        return
    except fitz.EmptyFileError:
        logger.error("File is empty or not a valid PDF: %s", file_path)
        return

    logger.info("Document has %d page(s).", len(doc))

    first_page_text = doc[0].get_text()
    doc.close()

    print("\n--- PREVIEW OF PAGE 1 ---")
    print(first_page_text[:500])
    print("-------------------------\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    test_read_pdf()