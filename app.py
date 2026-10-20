import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from streamlit_mic_recorder import speech_to_text

# ── Setup ────────────────────────────────────────────────────────────────────
load_dotenv()
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="Multimodal RAG Engine", layout="wide")
st.title("Universal Technical Document RAG Engine")
st.write(
    "Upload any technical PDF or scanned research paper to build a searchable "
    "knowledge base, then ask questions by text or voice."
)

# Single shared embedding model for both ingestion and querying
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    task_type="RETRIEVAL_QUERY"
)


# ── PDF Processing ────────────────────────────────────────────────────────────
def process_pdf(uploaded_file):
    """
    Pulls text out of every page of a PDF.
    For pages that are purely images (scanned docs), falls back to Gemini Vision
    to OCR the page and describe any diagrams.
    Returns a list of dicts with 'text' and 'metadata' keys.
    """
    import google.generativeai as genai
    genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
    vision_model = genai.GenerativeModel('gemini-2.5-flash-lite')

    pages = []

    # Write the upload to a temp file so PyMuPDF can open it
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        with fitz.open(tmp_path) as doc:
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text().strip()

                # No native text means it's probably a scanned image — use OCR
                if not text:
                    st.toast(f"Running Vision OCR on Page {page_num + 1}...")
                    pix = page.get_pixmap(dpi=150)
                    image_data = {
                        "mime_type": "image/jpeg",
                        "data": pix.tobytes("jpeg")
                    }
                    ocr_prompt = (
                        "You are an OCR engine. Extract all readable text from this page. "
                        "If there are charts or diagrams, describe them in detail."
                    )
                    text = vision_model.generate_content([ocr_prompt, image_data]).text

                if text and text.strip():
                    pages.append({
                        "text": text,
                        "metadata": {"source": uploaded_file.name, "page": page_num + 1}
                    })
    finally:
        # Always clean up the temp file, even if something above crashed
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except PermissionError:
                pass  # Windows sometimes holds the lock briefly; not critical

    return pages


def format_docs(docs):
    """Formats retrieved chunks into a single context block for the LLM."""
    return "\n\n".join(
        f"[Source: {d.metadata.get('source')} | Page: {d.metadata.get('page')}]\n{d.page_content}"
        for d in docs
    )


# ── Sidebar: Upload & Build ───────────────────────────────────────────────────
with st.sidebar:
    st.header("Document Ingestion")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_file:
        # Sanitise the filename so it can safely be used as a collection name
        safe_name = "".join(c if c.isalnum() else "_" for c in uploaded_file.name).lower()[:30].strip("_")
        collection_name = f"col_{safe_name or 'default_doc'}"
        st.info(f"Collection: `{collection_name}`")

        if st.button("Build Knowledge Base", type="primary"):
            with st.spinner(f"Extracting content from {uploaded_file.name}..."):
                raw_pages = process_pdf(uploaded_file)

            if not raw_pages:
                st.error("Couldn't extract any content from this document.")
            else:
                with st.spinner("Chunking and embedding..."):
                    documents = [
                        Document(page_content=p["text"], metadata=p["metadata"])
                        for p in raw_pages
                    ]
                    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    chunks = splitter.split_documents(documents)

                    db_path = os.path.join(PROJECT_ROOT, "chroma_db")
                    Chroma.from_documents(
                        documents=chunks,
                        embedding=embeddings,
                        persist_directory=db_path,
                        collection_name=collection_name
                    )

                    st.session_state["active_collection"] = collection_name
                    st.success(f"Done! {len(chunks)} chunks stored.")


# ── Main Panel: Chat ──────────────────────────────────────────────────────────
if "active_collection" in st.session_state:
    st.subheader(f"Chatting with: {uploaded_file.name}")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Replay the conversation history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input row: mic on the left, text box on the right
    st.write("---")
    col_mic, col_text = st.columns([1, 15])

    with col_mic:
        voice_input = speech_to_text(
            language='en',
            start_prompt="Record",
            stop_prompt="Stop",
            just_once=True,
            key='voice_input'
        )

    with col_text:
        text_input = st.text_input(
            "Ask a question...",
            label_visibility="collapsed",
            placeholder="Type your question or click Record to dictate..."
        )

    # Voice takes priority over typed input if both arrive somehow
    user_query = voice_input if voice_input else text_input

    if user_query:
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})

        # Build the retrieval chain on demand (keeps things stateless across reruns)
        db_path = os.path.join(PROJECT_ROOT, "chroma_db")
        vector_store = Chroma(
            persist_directory=db_path,
            embedding_function=embeddings,
            collection_name=st.session_state["active_collection"]
        )

        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.2)

        prompt = ChatPromptTemplate.from_template(
            """You are an expert research assistant. Answer the question using only the provided context.

Context:
{context}

Question: {question}

Give a clear, direct, well-structured answer:"""
        )

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = rag_chain.invoke(user_query)
                st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

        # Reset voice state so the user can record another question
        if voice_input:
            st.rerun()

else:
    st.info("Upload a PDF in the sidebar to get started.")