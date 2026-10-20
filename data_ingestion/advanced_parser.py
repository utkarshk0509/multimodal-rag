import os
import logging
import pymupdf as fitz  # PyMuPDF 2.0+: fitz is the legacy alias
import json

# Module-level logger — caller can configure level from outside
logger = logging.getLogger(__name__)

# Paths resolved relative to this file so the script works from any working directory
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
DEFAULT_PDF_PATH  = os.path.join(_PROJECT_ROOT, "data", "SAMPLE.pdf")
DEFAULT_IMAGE_DIR = os.path.join(_PROJECT_ROOT, "data", "extracted_images")
DEFAULT_CHUNKS_PATH = os.path.join(_PROJECT_ROOT, "data", "narrative_chunks.json")


def parse_satellite_pdf(file_path: str,
                        image_out_dir: str = DEFAULT_IMAGE_DIR,
                        chunks_out_path: str = DEFAULT_CHUNKS_PATH) -> list[str]:
    """
    Extracts text blocks, tables, and images from a satellite research PDF.

    Images are saved to `image_out_dir`.
    Text chunks are written to `chunks_out_path` as JSON and also returned.

    Args:
        file_path:       Path to the source PDF.
        image_out_dir:   Where to save extracted images.
        chunks_out_path: Where to write the narrative_chunks.json output.

    Returns:
        List of extracted text strings (one entry per meaningful text block).
    """
    logger.info("Opening document: %s", file_path)
    os.makedirs(image_out_dir, exist_ok=True)

    doc = fitz.open(file_path)
    logger.info("Loaded %d page(s).", len(doc))

    text_block_count = 0
    table_count = 0
    image_count = 0
    extracted_chunks: list[str] = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        logger.debug("Processing page %d", page_num + 1)

        # ── Tables ────────────────────────────────────────────────────────────
        found_tables = page.find_tables().tables
        table_count += len(found_tables)

        for idx, table in enumerate(found_tables):
            data = table.extract()
            header = data[0] if data else "Empty"
            logger.debug("  Page %d — Table [%d]: %s", page_num + 1, idx, header)

        # ── Text blocks ───────────────────────────────────────────────────────
        # Ignore very short fragments (axis labels, page numbers, etc.)
        page_blocks = 0
        for block in page.get_text("blocks"):
            text = block[4].strip()
            if len(text) > 10:
                extracted_chunks.append(text)
                text_block_count += 1
                page_blocks += 1

        logger.debug("  Page %d — %d text block(s).", page_num + 1, page_blocks)

        # ── Images / Diagrams ─────────────────────────────────────────────────
        images = page.get_images(full=True)
        image_count += len(images)

        for img_idx, img in enumerate(images):
            xref = img[0]
            raw = doc.extract_image(xref)
            img_filename = f"page{page_num+1}_img{img_idx+1}.{raw['ext']}"
            img_path = os.path.join(image_out_dir, img_filename)

            with open(img_path, "wb") as f:
                f.write(raw["image"])
            logger.debug("  Saved image: %s", img_filename)

    doc.close()

    logger.info(
        "Parse complete — %d text chunks, %d tables, %d images.",
        text_block_count, table_count, image_count
    )

    # Persist text chunks for the vector-store ingestion step
    with open(chunks_out_path, "w", encoding="utf-8") as f:
        json.dump(extracted_chunks, f, ensure_ascii=False, indent=4)
    logger.info("Saved %d chunks → %s", len(extracted_chunks), chunks_out_path)

    return extracted_chunks


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not os.path.exists(DEFAULT_PDF_PATH):
        logger.error("File not found: %s", DEFAULT_PDF_PATH)
    else:
        parse_satellite_pdf(DEFAULT_PDF_PATH)