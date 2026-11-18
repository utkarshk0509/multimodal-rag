# Quick check that PyMuPDF can open the sample PDF and read its text.
import os
import sys
import pymupdf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DEFAULT_PDF_PATH


def preview_pdf(file_path=DEFAULT_PDF_PATH):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    try:
        doc = pymupdf.open(file_path)
    except pymupdf.EmptyFileError:
        print(f"File is empty or not a valid PDF: {file_path}")
        return

    print(f"Document has {len(doc)} page(s)")
    first_page_text = doc[0].get_text()
    doc.close()

    print("\n--- PREVIEW OF PAGE 1 ---")
    print(first_page_text[:500])
    print("-------------------------\n")


if __name__ == "__main__":
    preview_pdf()
