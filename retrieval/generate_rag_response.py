# Runs one question through the full pipeline: retrieve chunks, then ask Gemini.
import os
import sys
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import (
    CHROMA_DB_DIR, COLLECTION_NAME, EMBEDDING_MODEL, LLM_MODEL,
    LLM_TEMPERATURE, RETRIEVER_TOP_K, get_api_key
)

PROMPT_TEMPLATE = """You are a satellite navigation expert assistant.
Answer the question using only the context below.

Rules:
1. Only use information from the context. Do not use outside knowledge.
2. If the answer isn't in the context, say so instead of guessing.
3. Use bullet points or headings for longer explanations.

Context:
{context}

Question:
{question}

Answer:"""


def format_docs(docs):
    # Label each chunk so the model knows whether it came from a chart or the text
    blocks = []
    for i, doc in enumerate(docs, start=1):
        header = f"--- Block {i} [Type: {doc.metadata.get('type')} | Source: {doc.metadata.get('source')}] ---"
        blocks.append(f"{header}\n{doc.page_content}")
    return "\n\n".join(blocks)


def main():
    load_dotenv()
    api_key = get_api_key()
    if not api_key:
        print("No API key found. Set GOOGLE_API_KEY or GEMINI_API_KEY in your .env file.")
        return

    # Must be the same embedding model that was used in vector_store_init.py
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
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE, google_api_key=api_key)
    chain = ChatPromptTemplate.from_template(PROMPT_TEMPLATE) | llm | StrOutputParser()

    question = "What does the trajectory chart indicate about the satellite's orbit and SGP4 error behavior?"
    print(f"Question: {question}\n")
    print("=" * 60 + "\n")

    try:
        docs = vector_store.similarity_search(question, k=RETRIEVER_TOP_K)
        answer = chain.invoke({"context": format_docs(docs), "question": question})
        print(answer)
    except Exception as e:
        print(f"Something went wrong: {e}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
