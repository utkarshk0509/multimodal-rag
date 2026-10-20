import os
import logging
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

logger = logging.getLogger(__name__)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    load_dotenv()

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("No API key found. Set GOOGLE_API_KEY or GEMINI_API_KEY in your .env file.")
        return

    # Use the query task type so embeddings are optimised for search, not storage
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        task_type="RETRIEVAL_QUERY"
    )

    db_path = os.path.join(PROJECT_ROOT, "chroma_db")

    if not os.path.exists(db_path):
        logger.error("ChromaDB not found at %s — run vector_store_init.py first.", db_path)
        return

    logger.info("Connecting to vector database at: %s", db_path)
    vector_store = Chroma(
        persist_directory=db_path,
        embedding_function=embeddings,
        collection_name="satellite_rag_collection"
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