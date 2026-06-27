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

def main():
    load_dotenv()
    
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001", 
        task_type="RETRIEVAL_DOCUMENT" 
    )
    
    documents = []
    
    vision_dir = os.path.join(PROJECT_ROOT, "data", "extracted_images")
    txt_files = glob.glob(os.path.join(vision_dir, "*.txt"))
    
    for file_path in txt_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            doc = Document(
                page_content=content,
                metadata={
                    "source": os.path.basename(file_path),
                    "type": "visual_profile",
                    "content_type": "satellite_trajectory_chart"
                }
            )
            documents.append(doc)
            
    print(f"Loaded {len(txt_files)} visual text profiles.")
    
    narrative_chunks = get_narrative_chunks() 
    
    for i, chunk_text in enumerate(narrative_chunks):
        doc = Document(
            page_content=chunk_text,
            metadata={
                "source": "SAMPLE.pdf",
                "type": "narrative_chunk",
                "chunk_index": i
            }
        )
        documents.append(doc)
        
    print(f"Loaded {len(narrative_chunks)} narrative text chunks.")
    
    total_docs = len(documents)
    print(f"Embedding and storing {total_docs} total documents into ChromaDB...")
    
    persist_dir = os.path.join(PROJECT_ROOT, "chroma_db") 
    
    vector_store = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
        collection_name="satellite_rag_collection"
    )

    batch_size = 50 
    
    for i in range(0, total_docs, batch_size):
        batch = documents[i : i + batch_size]
        print(f"Processing batch {i + 1} to {min(i + batch_size, total_docs)} of {total_docs}...")
        
        # Add this specific batch to the database
        vector_store.add_documents(batch)
        
        # If we still have more documents left to process, pause to reset the API quota
        if i + batch_size < total_docs:
            print("Batch complete. Sleeping for 60 seconds to respect Google's Free Tier limits...")
            time.sleep(60)
            
    print(f"Success! All {total_docs} documents embedded and persisted to {persist_dir}")

def get_narrative_chunks():
    # Bulletproof path to your newly generated JSON
    json_path = os.path.join(PROJECT_ROOT, "data", "narrative_chunks.json")
    
    with open(json_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
        
    return chunks

if __name__ == "__main__":
    main()