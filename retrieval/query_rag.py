import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))


def main():
    load_dotenv()

    # Use the query task type so embeddings are optimised for search, not storage
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        task_type="RETRIEVAL_QUERY"
    )

    db_path = os.path.join(PROJECT_ROOT, "chroma_db")
    print(f"Connecting to vector database at: {db_path}...")

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

    print(f"\nQuerying: '{query}'\n")
    results = retriever.invoke(query)

    for i, doc in enumerate(results):
        source = doc.metadata.get('source', 'Unknown')
        content_type = doc.metadata.get('type', 'Unknown')
        print(f"--- Result {i+1} ---")
        print(f"Source: {source} | Type: {content_type}")
        print(f"Content:\n{doc.page_content}\n")


if __name__ == "__main__":
    main()