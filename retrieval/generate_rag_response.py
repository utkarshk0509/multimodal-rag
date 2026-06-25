import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- ROBUST PATH RESOLUTION ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))

def format_docs(docs):
    """Combines all retrieved text chunks into a single string for the LLM context."""
    formatted = []
    for i, doc in enumerate(docs):
        source_type = doc.metadata.get('type', 'Unknown')
        source_name = doc.metadata.get('source', 'Unknown')
        formatted.append(f"--- Context Block {i+1} [Type: {source_type} | Source: {source_name}] ---\n{doc.page_content}")
    return "\n\n".join(formatted)

def main():
    load_dotenv()

    # 1. Initialize the Embedding Model
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        task_type="RETRIEVAL_QUERY"
    )

    # 2. Connect to the Local Vector Store
    persist_dir = os.path.join(PROJECT_ROOT, "chroma_db")
    vector_store = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
        collection_name="satellite_rag_collection"
    )

    # 3. Define the Retriever (Using our verified depth of 5 chunks)
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    # 4. Initialize the LLM Generator
    # We use gemini-1.5-flash as it is lightning fast and highly precise at context tracking
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0.2 # Low temperature keeps the model grounded strictly in the context
    )

    # 5. Construct a Strict RAG System Prompt
    # This guides the model to behave like a precise space systems research assistant
    prompt_template = """You are an advanced Space Systems and Satellite Navigation AI Assistant. 
Your objective is to provide professional, mathematically accurate, and highly structured answers based strictly on the provided technical context.

Instructions:
1. Base your answer ONLY on the retrieved Context Blocks provided below. Do not use outside knowledge.
2. If the context does not contain the answer, state clearly that the information is not present in the loaded documentation.
3. Use a clear layout with bullet points or headings if explaining complex technical concepts, charts, or error trends.

Retrieved Context:
{context}

Question: 
{question}

Answer:"""

    prompt = ChatPromptTemplate.from_template(prompt_template)

    # 6. Build the LCEL (LangChain Expression Language) Chain
    # This smoothly pipes user_query -> database lookup -> text formatting -> prompt -> LLM -> clean string
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # 7. Execute a Live End-to-End Query
    query = "What does the trajectory chart indicate about the satellite's orbit and SGP4 error behavior?"
    
    print("Processing query through the complete Multimodal RAG Pipeline...")
    print(f"Query: {query}\n")
    print("="*60 + "\n")
    
    try:
        response = rag_chain.invoke(query)
        print(response)
    except Exception as e:
        print(f"An execution error occurred: {e}")
        
    print("\n" + "="*60)

if __name__ == "__main__":
    main()