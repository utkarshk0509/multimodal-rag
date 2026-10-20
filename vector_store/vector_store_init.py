import os
import time
import glob
import json
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))


def load_narrative_chunks():
    """Reads the text chunks produced by advanced_parser.py from JSON."""
    json_path = os.path.join(PROJECT_ROOT, "data", "narrative_chunks.json")
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    load_dotenv()

    # RETRIEVAL_DOCUMENT task type produces better embeddings for stored content
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        task_type="RETRIEVAL_DOCUMENT"
    )

    documents = []

    # --- Visual profiles (chart descriptions from vision_extractor.py) ---
    vision_dir = os.path.join(PROJECT_ROOT, "data", "extracted_images")
    txt_files = glob.glob(os.path.join(vision_dir, "*.txt"))

    for file_path in txt_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        documents.append(Document(
            page_content=content,
            metadata={
                "source": os.path.basename(file_path),
                "type": "visual_profile",
                "content_type": "satellite_trajectory_chart"
            }
        ))

    print(f"Loaded {len(txt_files)} chart description(s).")

    # --- Narrative text chunks from the PDF ---
    narrative_chunks = load_narrative_chunks()

    for i, chunk_text in enumerate(narrative_chunks):
        documents.append(Document(
            page_content=chunk_text,
            metadata={
                "source": "SAMPLE.pdf",
                "type": "narrative_chunk",
                "chunk_index": i
            }
        ))

    print(f"Loaded {len(narrative_chunks)} text chunk(s).")
    print(f"Embedding {len(documents)} documents total into ChromaDB...")

    db_path = os.path.join(PROJECT_ROOT, "chroma_db")
    vector_store = Chroma(
        persist_directory=db_path,
        embedding_function=embeddings,
        collection_name="satellite_rag_collection"
    )

    # Process in batches of 50 to stay within the API's free-tier rate limits
    batch_size = 50
    total = len(documents)

    for i in range(0, total, batch_size):
        batch = documents[i: i + batch_size]
        print(f"  Uploading documents {i+1}–{min(i + batch_size, total)} of {total}...")
        vector_store.add_documents(batch)

        # Don't hammer the API — wait between batches if more remain
        if i + batch_size < total:
            print("  Batch done. Waiting 60s to stay within rate limits...")
            time.sleep(60)

    print(f"\nDone! {total} documents stored in {db_path}")


if __name__ == "__main__":
    main()