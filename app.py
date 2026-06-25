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

# --- INITIAL SETUP & GLOBAL CONFIG ---
load_dotenv()
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="Universal Multimodal RAG Engine", page_icon="🚀", layout="wide")
st.title("Universal Technical Document RAG Engine")
st.write("Upload any research paper or technical manual to instantiate a dedicated local vector knowledge base.")

# FIX: Removed the 'models/gemini-' prefix to pass LangChain's internal validation schema
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    task_type="RETRIEVAL_QUERY"
)

# --- HELPER FUNCTIONS ---
def process_pdf(uploaded_file):
    """Extracts text from an uploaded PDF file using PyMuPDF."""
    text_chunks = []
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        doc = fitz.open(tmp_path)
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            if text.strip():
                text_chunks.append({
                    "text": text,
                    "metadata": {"source": uploaded_file.name, "page": page_num + 1}
                })
        doc.close()
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
    return text_chunks

# --- SIDEBAR: DOCUMENT INGESTION ---
# --- SIDEBAR: DOCUMENT INGESTION ---
with st.sidebar:
    st.header("Document Ingestion Control")
    uploaded_file = st.file_uploader("Upload a technical PDF", type=["pdf"])
    
    if uploaded_file:
        # Generate a clean, unique collection name based on the filename
        safe_filename = "".join([c if c.isalnum() else "_" for c in uploaded_file.name]).lower()[:30]
        
        # FIX: Strip any trailing underscores so it ends in a letter/number for ChromaDB
        safe_filename = safe_filename.strip("_")
        
        # Fallback just in case the name was entirely special characters
        if not safe_filename:
            safe_filename = "default_doc"
            
        collection_name = f"col_{safe_filename}"
        
        st.info(f"Target Collection: `{collection_name}`")
        
        if st.button("Build Knowledge Base", type="primary"):
            with st.spinner("Extracting structural layout and content..."):
                raw_pages = process_pdf(uploaded_file)
# ... (Leave the rest of the file exactly as it is) ...
                
            if not raw_pages:
                st.error("No extractable text found in the document.")
            else:
                with st.spinner("Chunking document and generating vector embeddings..."):
                    # Transform raw pages to LangChain Document structures
                    from langchain_core.documents import Document
                    documents = [Document(page_content=p["text"], metadata=p["metadata"]) for p in raw_pages]
                    
                    # Split documents cleanly
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    split_docs = text_splitter.split_documents(documents)
                    
                    # Create a dedicated Chroma instance for this file
                    persist_dir = os.path.join(PROJECT_ROOT, "chroma_db")
                    vector_store = Chroma.from_documents(
                        documents=split_docs,
                        embedding=embeddings,  # Using the validated global embeddings object
                        persist_directory=persist_dir,
                        collection_name=collection_name
                    )
                    
                    st.session_state["active_collection"] = collection_name
                    st.success(f"Context built successfully! {len(split_docs)} chunks persisted.")

# --- MAIN PANEL: CHAT INTERFACE ---
if "active_collection" in st.session_state:
    st.subheader(f" Query Session: {uploaded_file.name}")
    
    # Initialize message history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle fresh user input
    if user_query := st.chat_input("Ask a question about the uploaded document..."):
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        # Connect to your local database securely using the global embeddings object
        persist_dir = os.path.join(PROJECT_ROOT, "chroma_db")
        vector_store = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
            collection_name=st.session_state["active_collection"]
        )
        
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.2)
        
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
else:
    st.info("Please upload a technical PDF document via the sidebar to initialize the interactive retrieval engine.")