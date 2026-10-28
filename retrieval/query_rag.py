import os
import sys
import logging
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Allow imports from the project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import CHROMA_DB_DIR, COLLECTION_NAME, EMBEDDING_MODEL

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    load_dotenv()

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("No API key found. Set GOOGLE_API_KEY or GEMINI_API_KEY in your .env file.")
        return

    # Use the query task type so embeddings are optimised for search, not storage
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        task_type="RETRIEVAL_QUERY"
    )

    if not os.path.exists(CHROMA_DB_DIR):
        logger.error("ChromaDB not found at %s — run vector_store_init.py first.", CHROMA_DB_DIR)
        return

    logger.info("Connecting to vector database at: %s", CHROMA_DB_DIR)
    vector_store = Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )

    # We only want results from the visual charts — filter out plain text chunks
    retriever = vector_store.as_retriever(
        search_kwargs={
            "k": 3,
            "filter": {"type": "visual_profile"}
        }
    )

    query = "Describe the curve and axes shown in the SGP4 error trajectory chart."

    logger.info("Query: %s", query)
    results = retriever.invoke(query)

    if not results:
        logger.warning("No results returned — the collection may be empty or the filter too strict.")
        return

    for i, doc in enumerate(results):
        source = doc.metadata.get('source', 'Unknown')
        content_type = doc.metadata.get('type', 'Unknown')
        print(f"--- Result {i+1} ---")
        print(f"Source: {source} | Type: {content_type}")
        print(f"Content:\n{doc.page_content}\n")


if __name__ == "__main__":
    main()