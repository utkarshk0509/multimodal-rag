import os
import mimetypes
import tempfile
from typing import List, Optional

import chromadb
import pymupdf
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai
from google.genai import types
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config

load_dotenv()

# Force correct MIME types for frontend assets. On Windows, the registry can map
# .js to text/plain, which makes browsers refuse ES-module scripts (motion.js).
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("application/javascript", ".mjs")
mimetypes.add_type("text/css", ".css")
mimetypes.add_type("image/svg+xml", ".svg")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# The temp dir is writable on Vercel (/tmp) as well as on local Windows/Mac
CHROMA_DIR = os.path.join(tempfile.gettempdir(), "chroma_db")
UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "uploaded_docs")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)

app = FastAPI(title="Multimodal RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_api_key(provided_key: Optional[str] = None) -> str:
    # A key typed in the UI wins over the one in .env
    key = (provided_key or "").strip() or (config.get_api_key() or "").strip()
    if not key:
        raise HTTPException(
            status_code=400,
            detail="Google API Key is missing. Enter it in the UI or set GOOGLE_API_KEY in the environment."
        )
    return key

def get_embeddings(api_key: str):
    return GoogleGenerativeAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        task_type="RETRIEVAL_QUERY",
        google_api_key=api_key
    )

def extract_pages(file_path: str, filename: str, api_key: str):
    # Returns one entry per page that has text. Pages with no text layer
    # (scanned pages) are sent to Gemini Vision for OCR instead.
    client = genai.Client(api_key=api_key)

    pages = []
    with pymupdf.open(file_path) as doc:
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text().strip()

            is_scanned = False
            if not text:
                pix = page.get_pixmap(dpi=150)
                prompt = (
                    "You are an expert OCR engine. Extract all readable text from this page. "
                    "If there are diagrams, charts or formulas, describe and transcribe them clearly."
                )
                try:
                    response = client.models.generate_content(
                        model=config.VISION_MODEL,
                        contents=[
                            prompt,
                            types.Part.from_bytes(
                                data=pix.tobytes("jpeg"),
                                mime_type="image/jpeg"
                            )
                        ]
                    )
                    text = (response.text or "").strip()
                    is_scanned = True
                except Exception as e:
                    print(f"Vision OCR failed on page {page_num + 1}: {e}")
                    text = ""

            if text:
                pages.append({
                    "text": text,
                    "metadata": {
                        "source": filename,
                        "page": page_num + 1,
                        "is_scanned": is_scanned
                    }
                })
    return pages


class IngestResponse(BaseModel):
    message: str
    collection_name: str
    total_pages: int
    chunks_indexed: int

class QueryRequest(BaseModel):
    question: str
    collection_name: str
    api_key: Optional[str] = None
    top_k: Optional[int] = config.RETRIEVER_TOP_K

class SourceItem(BaseModel):
    source: str
    page: Optional[int] = None
    content: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceItem]


@app.get("/api/health")
def health_check():
    return {"status": "healthy", "has_env_api_key": bool(config.get_api_key())}

@app.get("/api/collections")
def list_collections():
    try:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        result = []
        for col in client.list_collections():
            result.append({
                "name": col.name,
                "count": col.count(),
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

    # Chroma collection names only allow a limited set of characters
    clean_name = "".join([c if c.isalnum() else "_" for c in file.filename]).lower()[:30].strip("_")
    if not clean_name:
        clean_name = "default_doc"
    collection_name = f"col_{clean_name}"

    # basename() so a filename like "../../x.pdf" can't escape UPLOAD_DIR
    saved_path = os.path.join(UPLOAD_DIR, os.path.basename(file.filename))
    content = await file.read()
    with open(saved_path, "wb") as f_out:
        f_out.write(content)

    try:
        pages = extract_pages(saved_path, file.filename, valid_key)
        if not pages:
            raise HTTPException(status_code=400, detail="No extractable text or content found in the uploaded PDF.")

        documents = [Document(page_content=p["text"], metadata=p["metadata"]) for p in pages]

        splitter = RecursiveCharacterTextSplitter(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
        split_docs = splitter.split_documents(documents)

        Chroma.from_documents(
            documents=split_docs,
            embedding=get_embeddings(valid_key),
            persist_directory=CHROMA_DIR,
            collection_name=collection_name
        )

        return IngestResponse(
            message=f"Document '{file.filename}' processed and indexed successfully.",
            collection_name=collection_name,
            total_pages=len(pages),
            chunks_indexed=len(split_docs)
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process document: {e}")

@app.post("/api/query", response_model=QueryResponse)
def query_rag(req: QueryRequest):
    valid_key = get_api_key(req.api_key)

    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        vector_store = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=get_embeddings(valid_key),
            collection_name=req.collection_name
        )

        k = req.top_k or config.RETRIEVER_TOP_K
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

        # Label each chunk with its page so the model can cite it
        context = "\n\n".join([
            f"[Source: {d.metadata.get('source')} | Page: {d.metadata.get('page')}]\n{d.page_content}"
            for d in docs
        ])

        llm = ChatGoogleGenerativeAI(
            model=config.LLM_MODEL,
            temperature=config.LLM_TEMPERATURE,
            google_api_key=valid_key
        )

        prompt_template = """You are a research assistant. Answer the question using the context retrieved from the document.

Context:
{context}

Question: {question}

Answer clearly and directly. Cite page numbers when they are available."""

        prompt = ChatPromptTemplate.from_template(prompt_template)
        rag_chain = prompt | llm | StrOutputParser()

        answer = rag_chain.invoke({"context": context, "question": req.question})
        return QueryResponse(answer=answer, sources=sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")


# The frontend in static/ is served by the same app
STATIC_DIR = os.path.join(PROJECT_ROOT, "static")
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    async def serve_index():
        return FileResponse(os.path.join(STATIC_DIR, "index.html"))

    @app.get("/{full_path:path}")
    async def serve_static_or_index(full_path: str):
        # Unknown /api paths should 404 instead of returning index.html
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="Endpoint not found")
        file_path = os.path.realpath(os.path.join(STATIC_DIR, full_path))
        # Only serve files that resolve inside STATIC_DIR (blocks ../ traversal)
        inside_static = os.path.commonpath([file_path, os.path.realpath(STATIC_DIR)]) == os.path.realpath(STATIC_DIR)
        if inside_static and os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(STATIC_DIR, "index.html"))

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    host = "0.0.0.0" if os.getenv("PORT") else "127.0.0.1"
    uvicorn.run("backend_server:app", host=host, port=port, reload=True)
