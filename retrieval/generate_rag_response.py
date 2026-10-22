import os
import sys
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Allow imports from the project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import CHROMA_DB_DIR, COLLECTION_NAME, EMBEDDING_MODEL, LLM_MODEL, LLM_TEMPERATURE, RETRIEVER_TOP_K


def format_docs(docs):
    """
    Stitches retrieved chunks into one big context string for the LLM.
    Each block gets labeled with its source file and content type so the
    model knows where the info came from.
    """
    blocks = []
    for i, doc in enumerate(docs):
        content_type = doc.metadata.get('type', 'Unknown')
        source_file = doc.metadata.get('source', 'Unknown')
        header = f"--- Context Block {i+1} [Type: {content_type} | Source: {source_file}] ---"
        blocks.append(f"{header}\n{doc.page_content}")
    return "\n\n".join(blocks)


def main():
    load_dotenv()

    # Set up the embedding model — same one used during ingestion so vectors match
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        task_type="RETRIEVAL_QUERY"
    )

    # Hook up to our local ChromaDB that was built during the ingestion phase
    vector_store = Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )

    # Pull the top most relevant chunks for any given query
    retriever = vector_store.as_retriever(search_kwargs={"k": RETRIEVER_TOP_K})

    # Low temperature = stick to the facts, don't hallucinate
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE
    )

    # The prompt keeps the model grounded — answer from context only, no winging it
    prompt_template = """You are a Space Systems and Satellite Navigation expert assistant.
Give accurate, well-structured answers based strictly on the provided context below.

Rules:
1. Only use information from the Context Blocks below. Do not rely on outside knowledge.
2. If the answer isn't in the context, say so clearly — don't guess.
3. Use bullet points or headings when explaining complex topics, charts, or error patterns.

Retrieved Context:
{context}

Question:
{question}

Answer:"""

    prompt = ChatPromptTemplate.from_template(prompt_template)

    # Wire everything together: retriever → prompt → LLM → plain text output
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Test query — swap this out for anything you want to ask the knowledge base
    query = "What does the trajectory chart indicate about the satellite's orbit and SGP4 error behavior?"

    print("Running query through the Multimodal RAG pipeline...")
    print(f"Query: {query}\n")
    print("=" * 60 + "\n")

    try:
        response = rag_chain.invoke(query)
        print(response)
    except Exception as e:
        print(f"Something went wrong: {e}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()