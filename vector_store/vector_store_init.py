import os
import sys
import time
import glob
import json
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import (
    IMAGE_DIR, CHUNKS_JSON_PATH, CHROMA_DB_DIR, COLLECTION_NAME, SOURCE_PDF_NAME,
    EMBEDDING_MODEL, EMBED_BATCH_SIZE, EMBED_BATCH_WAIT, get_api_key
)


def build_documents():
    # Combines the chart descriptions (from vision_extractor.py) and the
    # text chunks (from advanced_parser.py) into one list of Documents.
    documents = []

    txt_files = glob.glob(os.path.join(IMAGE_DIR, "*.txt"))
    if not txt_files:
        print(f"Warning: no chart descriptions found in {IMAGE_DIR}")
    for file_path in txt_files:
        with open(file_path, "r", encoding="utf-8") as f:
            documents.append(Document(
                page_content=f.read(),
                metadata={"source": os.path.basename(file_path), "type": "visual_profile"}
            ))
    print(f"Loaded {len(txt_files)} chart description(s)")

    if not os.path.exists(CHUNKS_JSON_PATH):
        print(f"{CHUNKS_JSON_PATH} not found, run advanced_parser.py first")
        return documents
    with open(CHUNKS_JSON_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    for i, text in enumerate(chunks):
        documents.append(Document(
            page_content=text,
            metadata={"source": SOURCE_PDF_NAME, "type": "narrative_chunk", "chunk_index": i}
        ))
    print(f"Loaded {len(chunks)} text chunk(s)")

    return documents


def main():
    load_dotenv()
    api_key = get_api_key()
    if not api_key:
        print("No API key found. Set GOOGLE_API_KEY or GEMINI_API_KEY in your .env file.")
        return

    # RETRIEVAL_DOCUMENT is the task type meant for stored content
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        task_type="RETRIEVAL_DOCUMENT",
        google_api_key=api_key
    )

    documents = build_documents()
    total = len(documents)
    if total == 0:
        print("Nothing to embed")
        return

    vector_store = Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )

    # Upload in batches so we don't hit the free tier rate limit
    for start in range(0, total, EMBED_BATCH_SIZE):
        end = min(start + EMBED_BATCH_SIZE, total)
        print(f"Embedding documents {start + 1}-{end} of {total}")
        vector_store.add_documents(documents[start:end])

        if end < total:
            print(f"Waiting {EMBED_BATCH_WAIT}s before the next batch")
            time.sleep(EMBED_BATCH_WAIT)

    print(f"Done, {total} documents stored in {CHROMA_DB_DIR}")


if __name__ == "__main__":
    main()
