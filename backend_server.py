import os
import tempfile
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

import pymupdf                                         # replaces deprecated `import fitz`
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from google import genai as google_genai               # new google-genai SDK (replaces google-generativeai)
from google.genai import types as genai_types

load_dotenv()

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Use system temp dir — writable on both Vercel (/tmp) and local Windows/Mac dev.
# tempfile.gettempdir() resolves to /tmp on Linux/Vercel, %TEMP% on Windows.
CHROMA_DIR = os.path.join(tempfile.gettempdir(), "chroma_db")
UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "uploaded_docs")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)

app = FastAPI(
    title="Multimodal RAG API",
    description="Backend API powering the modern Multimodal Technical Document RAG Application",
    version="1.0.0"
)

# Enable CORS for local development or hosted UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_api_key(provided_key: Optional[str] = None) -> str:
    key = (provided_key or "").strip() or os.getenv("GOOGLE_API_KEY", "").strip()
    if not key:
        raise HTTPException(
            status_code=400,
            detail="Google API Key is missing. Please provide it in the request or set GOOGLE_API_KEY environment variable."
        )
    return key

def get_embeddings(api_key: str):
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        task_type="RETRIEVAL_QUERY",
        google_api_key=api_key
    )

def process_pdf_content(file_path: str, filename: str, api_key: str):
    """Extract text from PDF pages with Vision OCR fallback for image-only/scanned pages."""
    client = google_genai.Client(api_key=api_key)

    text_chunks = []
    with pymupdf.open(file_path) as doc:
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text().strip()

            # Scanned PDF / Vision OCR Fallback
            is_scanned = False
            if not text:
                pix = page.get_pixmap(dpi=150)
                prompt = (
                    "You are an expert OCR engine. Extract all readable text from this page. "
                    "If there are diagrams, charts or formulas, describe and transcribe them clearly."
                )
                try:
                    response = client.models.generate_content(
                        model="gemini-2.5-flash-lite-preview-06-17",
                        contents=[
                            prompt,
                            genai_types.Part.from_bytes(
                                data=pix.tobytes("jpeg"),
                                mime_type="image/jpeg"
                            )
                        ]
                    )
                    text = (response.text or "").strip()
                    is_scanned = True
                except Exception as e:
                    print(f"Vision OCR error on page {page_num + 1}: {e}")
                    text = ""

            if text:
                text_chunks.append({
                    "text": text,
                    "metadata": {
                        "source": filename,
                        "page": page_num + 1,
                        "is_scanned": is_scanned
                    }
                })
    return text_chunks


class IngestResponse(BaseModel):
    message: str
    collection_name: str
    total_pages: int
    chunks_indexed: int

class QueryRequest(BaseModel):
    question: str
    collection_name: str
    api_key: Optional[str] = None
    top_k: Optional[int] = 5

class SourceItem(BaseModel):
    source: str
    page: Optional[int] = None
    content: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceItem]


@app.get("/api/health")
def health_check():
    has_env_key = bool(os.getenv("GOOGLE_API_KEY"))
    return {
        "status": "healthy",
        "has_env_api_key": has_env_key
    }

@app.get("/api/collections")
def list_collections(api_key: Optional[str] = Query(None)):
    """List existing collections in ChromaDB."""
    try:
        import chromadb
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        collections = client.list_collections()
        result = []
        for col in collections:
            count = col.count()
            result.append({
                "name": col.name,
                "count": count,
                "metadata": col.metadata or {}
            })
        return {"collections": result}
    except Exception as e:
        return {"collections": [], "error": str(e)}

@app.post("/api/ingest", response_model=IngestResponse)
async def ingest_document(
    file: UploadFile = File(...),
    api_key: Optional[str] = Form(None)
):
    valid_key = get_api_key(api_key)

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are currently supported.")

    # Generate sanitized collection name
    clean_name = "".join([c if c.isalnum() else "_" for c in file.filename]).lower()[:30].strip("_")
    if not clean_name:
        clean_name = "default_doc"
    collection_name = f"col_{clean_name}"

    # Save uploaded file to temp dir — writable on Vercel (/tmp) and local dev
    saved_path = os.path.join(UPLOAD_DIR, file.filename)
    content = await file.read()
    with open(saved_path, "wb") as f_out:
        f_out.write(content)

    try:
        raw_pages = process_pdf_content(saved_path, file.filename, valid_key)
        if not raw_pages:
            raise HTTPException(status_code=400, detail="No extractable text or content found in the uploaded PDF.")

        from langchain_core.documents import Document
        documents = [Document(page_content=p["text"], metadata=p["metadata"]) for p in raw_pages]

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = text_splitter.split_documents(documents)

        embeddings_model = get_embeddings(valid_key)

        vector_store = Chroma.from_documents(
            documents=split_docs,
            embedding=embeddings_model,
            persist_directory=CHROMA_DIR,
            collection_name=collection_name
        )

        return IngestResponse(
            message=f"Document '{file.filename}' processed and indexed successfully.",
            collection_name=collection_name,
            total_pages=len(raw_pages),
            chunks_indexed=len(split_docs)
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")

@app.post("/api/query", response_model=QueryResponse)
def query_rag(req: QueryRequest):
    valid_key = get_api_key(req.api_key)

    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        embeddings_model = get_embeddings(valid_key)

        vector_store = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings_model,
            collection_name=req.collection_name
        )

        k = req.top_k or 5
        retriever = vector_store.as_retriever(search_kwargs={"k": k})
        docs = retriever.invoke(req.question)

        if not docs:
            return QueryResponse(
                answer="No relevant content was found in the indexed collection for your query.",
                sources=[]
            )

        sources = [
            SourceItem(
                source=d.metadata.get("source", "Unknown"),
                page=d.metadata.get("page"),
                content=d.page_content.strip()
            )
            for d in docs
        ]

        context_str = "\n\n".join([
            f"[Source: {d.metadata.get('source')} | Page: {d.metadata.get('page')}]\n{d.page_content}"
            for d in docs
        ])

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite-preview-06-17",
            temperature=0.2,
            google_api_key=valid_key
        )

        prompt_template = """You are an advanced expert research assistant. Analyze the retrieved documentation context carefully to answer the user request.

Context:
{context}

Question: {question}

Answer professionally, directly, and maintain structural clarity. Cite specific pages/sections when available."""

        prompt = ChatPromptTemplate.from_template(prompt_template)
        rag_chain = prompt | llm | StrOutputParser()

        answer = rag_chain.invoke({
            "context": context_str,
            "question": req.question
        })

        return QueryResponse(
            answer=answer,
            sources=sources
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


# Serve static files and frontend web application
STATIC_DIR = os.path.join(PROJECT_ROOT, "static")
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    async def serve_index():
        return FileResponse(os.path.join(STATIC_DIR, "index.html"))

    @app.get("/{full_path:path}")
    async def serve_static_or_index(full_path: str):
        # Prevent interfering with /api routes
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="Endpoint not found")
        file_path = os.path.join(STATIC_DIR, full_path)
        if os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(STATIC_DIR, "index.html"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend_server:app", host="127.0.0.1", port=8000, reload=True)
