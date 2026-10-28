import os
import sys
import time
import glob
import json
import logging
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Allow imports from the project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import (
    DATA_DIR, IMAGE_DIR, CHUNKS_JSON_PATH, CHROMA_DB_DIR,
    COLLECTION_NAME, SOURCE_PDF_NAME, EMBEDDING_MODEL,
    EMBED_BATCH_SIZE, EMBED_BATCH_WAIT
)

logger = logging.getLogger(__name__)


def load_narrative_chunks() -> list[str]:
    """Reads the text chunks produced by advanced_parser.py from JSON."""
    if not os.path.exists(CHUNKS_JSON_PATH):
        raise FileNotFoundError(
            f"Narrative chunks file not found: {CHUNKS_JSON_PATH}\n"
            "Run advanced_parser.py first to generate it."
        )
    with open(CHUNKS_JSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def build_documents() -> list[Document]:
    """
    Assembles all documents (visual profiles + narrative chunks) into a
    unified list ready for embedding.
    """
    documents: list[Document] = []

    # ── Visual profiles (chart descriptions from vision_extractor.py) ─────────
    txt_files = glob.glob(os.path.join(IMAGE_DIR, "*.txt"))
    if not txt_files:
        logger.warning("No chart description .txt files found in %s", IMAGE_DIR)

    for file_path in txt_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        documents.append(Document(
            page_content=content,
            metadata={
                "source": os.path.basename(file_path),
                "type": "visual_profile",
                "content_type": "satellite_trajectory_chart",
            }
        ))
    logger.info("Loaded %d chart description(s).", len(txt_files))

    # ── Narrative text chunks from the PDF ────────────────────────────────────
    narrative_chunks = load_narrative_chunks()
    for i, chunk_text in enumerate(narrative_chunks):
        documents.append(Document(
            page_content=chunk_text,
            metadata={
                "source": SOURCE_PDF_NAME,
                "type": "narrative_chunk",
                "chunk_index": i,
            }
        ))
    logger.info("Loaded %d text chunk(s).", len(narrative_chunks))

    return documents


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    load_dotenv()

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("No API key found. Set GOOGLE_API_KEY or GEMINI_API_KEY in your .env file.")
        return

    # RETRIEVAL_DOCUMENT task type produces better embeddings for stored content
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        task_type="RETRIEVAL_DOCUMENT"
    )

    documents = build_documents()
    total = len(documents)

    if total == 0:
        logger.error("No documents to embed — aborting.")
        return

    logger.info("Embedding %d documents into ChromaDB at %s...", total, CHROMA_DB_DIR)

    vector_store = Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )

    # Process in batches to stay within the API's free-tier rate limits
    for i in range(0, total, EMBED_BATCH_SIZE):
        batch = documents[i: i + EMBED_BATCH_SIZE]
        end = min(i + EMBED_BATCH_SIZE, total)
        logger.info("  Uploading documents %d–%d of %d...", i + 1, end, total)
        vector_store.add_documents(batch)

        # Don't hammer the API — wait between batches if more remain
        if end < total:
            logger.info("  Batch done. Waiting %ds to stay within rate limits...", EMBED_BATCH_WAIT)
            time.sleep(EMBED_BATCH_WAIT)

    logger.info("Done! %d documents stored in %s", total, DB_DIR)


if __name__ == "__main__":
    main()