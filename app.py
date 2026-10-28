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
from streamlit_mic_recorder import speech_to_text

# --- INITIAL SETUP & GLOBAL CONFIG ---
load_dotenv()
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="Universal Multimodal RAG Engine", layout="wide")
st.title("Universal Technical Document RAG Engine")
st.write("Upload any technical document or scanned research paper to build a dedicated vector knowledge base, then query via text or voice.")

# Initialize the global embedding model
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    task_type="RETRIEVAL_QUERY"
)

# --- HELPER FUNCTIONS ---
def process_pdf(uploaded_file):
    """Extracts text from a PDF, falling back to Vision OCR for image-only/scanned pages."""
    import google.generativeai as genai
    genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
    native_model = genai.GenerativeModel('gemini-2.5-flash-lite-preview-06-17')
    
    text_chunks = []
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        # Context manager ensures the file lock is safely released under any condition
        with fitz.open(tmp_path) as doc:
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text().strip()
                
                # Scanned PDF / Vision OCR Fallback
                if not text:
                    st.toast(f"Running Vision OCR on Page {page_num + 1}...")
                    pix = page.get_pixmap(dpi=150)
                    image_data = {
                        "mime_type": "image/jpeg",
                        "data": pix.tobytes("jpeg")
                    }
                    
                    prompt = "You are an expert OCR engine. Extract all readable text from this page. If there are diagrams, describe them in detail."
                    response = native_model.generate_content([prompt, image_data])
                    text = response.text
                    
                if text and text.strip():
                    text_chunks.append({
                        "text": text,
                        "metadata": {"source": uploaded_file.name, "page": page_num + 1}
                    })
    finally:
        # Safely clear the temporary file from disk
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except PermissionError:
                pass
                
    return text_chunks

# --- SIDEBAR: DOCUMENT INGESTION ---
with st.sidebar:
    st.header("Document Ingestion Control")
    uploaded_file = st.file_uploader("Upload a PDF File", type=["pdf"])
    
    if uploaded_file:
        safe_filename = "".join([c if c.isalnum() else "_" for c in uploaded_file.name]).lower()[:30].strip("_")
        if not safe_filename:
            safe_filename = "default_doc"
            
        collection_name = f"col_{safe_filename}"
        st.info(f"Target Collection: `{collection_name}`")
        
        if st.button("Build Knowledge Base", type="primary"):
            with st.spinner(f"Extracting content from {uploaded_file.name}..."):
                raw_pages = process_pdf(uploaded_file)
                
            if not raw_pages:
                st.error("No extractable content found in the document.")
            else:
                with st.spinner("Chunking content and generating vector embeddings..."):
                    from langchain_core.documents import Document
                    documents = [Document(page_content=p["text"], metadata=p["metadata"]) for p in raw_pages]
                    
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    split_docs = text_splitter.split_documents(documents)
                    
                    persist_dir = os.path.join(PROJECT_ROOT, "chroma_db")
                    vector_store = Chroma.from_documents(
                        documents=split_docs,
                        embedding=embeddings,
                        persist_directory=persist_dir,
                        collection_name=collection_name
                    )
                    
                    st.session_state["active_collection"] = collection_name
                    st.success(f"Context built successfully! {len(split_docs)} chunks persisted.")

# --- MAIN PANEL: CHAT INTERFACE ---
if "active_collection" in st.session_state:
    st.subheader(f"Query Session: {uploaded_file.name}")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # --- SIDE-BY-SIDE INPUT ROW ---
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

    # Resolve query priority
    user_query = voice_input if voice_input else text_input

    if user_query:
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        persist_dir = os.path.join(PROJECT_ROOT, "chroma_db")
        vector_store = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
            collection_name=st.session_state["active_collection"]
        )
        
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite-preview-06-17", temperature=0.2)
        
        prompt_template = """You are an advanced expert research assistant. Analyze the retrieved documentation context carefully to answer the user request.
        
        Context:
        {context}
        
        Question: {question}
        
        Answer professionally, directly, and maintain structural clarity:"""
        
        prompt = ChatPromptTemplate.from_template(prompt_template)
        
        def format_docs(docs):
            return "\n\n".join([f"[Source: {d.metadata.get('source')} | Page: {d.metadata.get('page')}]\n{d.page_content}" for d in docs])

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        with st.chat_message("assistant"):
            with st.spinner("Analyzing data vector space..."):
                response = rag_chain.invoke(user_query)
                st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Clear the audio transcription state buffer safely to allow consecutive queries
        if voice_input:
            st.rerun()
else:
    st.info("Please upload a technical PDF document via the sidebar to initialize the interactive retrieval engine.")