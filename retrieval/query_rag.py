import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))

def main():
    load_dotenv()

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        task_type="RETRIEVAL_QUERY" 
    )

    persist_dir = os.path.join(PROJECT_ROOT, "chroma_db")
    
    print(f"🔗 Connecting to database at: {persist_dir}...")
    vector_store = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
        collection_name="satellite_rag_collection"
    )


    retriever = vector_store.as_retriever(
        search_kwargs={
            "k": 3, 
            "filter": {"type": "visual_profile"} # Forces retrieval ONLY from charts
        }
    )

    query = "Describe the curve and axes shown in the SGP4 error trajectory chart."
    
    print(f"\nQuerying: '{query}'\n")
    results = retriever.invoke(query)

    for i, doc in enumerate(results):
        print(f"--- Result {i+1} ---")
        print(f"Source: {doc.metadata.get('source', 'Unknown')} | Type: {doc.metadata.get('type', 'Unknown')}")
        print(f"Content:\n{doc.page_content}\n")

if __name__ == "__main__":
    main()