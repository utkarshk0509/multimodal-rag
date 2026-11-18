# Searches the vector store directly (no LLM) to check what retrieval returns.
import os
import sys
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import CHROMA_DB_DIR, COLLECTION_NAME, EMBEDDING_MODEL, get_api_key


def main():
    load_dotenv()
    api_key = get_api_key()
    if not api_key:
        print("No API key found. Set GOOGLE_API_KEY or GEMINI_API_KEY in your .env file.")
        return

    if not os.path.exists(CHROMA_DB_DIR):
        print(f"No database at {CHROMA_DB_DIR}, run vector_store_init.py first")
        return

    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        task_type="RETRIEVAL_QUERY",
        google_api_key=api_key
    )
    vector_store = Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )

    query = "Describe the curve and axes shown in the SGP4 error trajectory chart."
    print(f"Query: {query}\n")

    # Only look at the chart descriptions, not the plain text chunks
    results = vector_store.similarity_search(query, k=3, filter={"type": "visual_profile"})
    if not results:
        print("No results, the collection may be empty")
        return

    for i, doc in enumerate(results, start=1):
        print(f"--- Result {i} ---")
        print(f"Source: {doc.metadata.get('source')} | Type: {doc.metadata.get('type')}")
        print(f"{doc.page_content}\n")


if __name__ == "__main__":
    main()
