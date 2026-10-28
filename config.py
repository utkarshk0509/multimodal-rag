"""
config.py — Central configuration for the Multimodal RAG project.

Import constants from here instead of hardcoding them in individual scripts.
This ensures all modules stay in sync when names or paths change.
"""
import os

# ── Path resolution ───────────────────────────────────────────────────────────
# PROJECT_ROOT is the directory containing this file (i.e. multimodal-rag/)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ── Directory paths ───────────────────────────────────────────────────────────
DATA_DIR        = os.path.join(PROJECT_ROOT, "data")
IMAGE_DIR       = os.path.join(DATA_DIR, "extracted_images")
CHROMA_DB_DIR   = os.path.join(PROJECT_ROOT, "chroma_db")

# ── File paths ────────────────────────────────────────────────────────────────
CHUNKS_JSON_PATH = os.path.join(DATA_DIR, "narrative_chunks.json")
SOURCE_PDF_NAME  = "SAMPLE.pdf"
DEFAULT_PDF_PATH = os.path.join(DATA_DIR, SOURCE_PDF_NAME)

# ── ChromaDB collection ───────────────────────────────────────────────────────
# All scripts that read/write the vector store must use this constant so that
# ingestion and retrieval always point to the same collection.
COLLECTION_NAME = "satellite_rag_collection"

# ── Model names ───────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "models/gemini-embedding-001"
LLM_MODEL       = "gemini-3.5-flash-lite"
VISION_MODEL    = "gemini-3.5-flash-lite"

# ── Retrieval settings ────────────────────────────────────────────────────────
RETRIEVER_TOP_K  = 5
LLM_TEMPERATURE  = 0.2

# ── Ingestion settings ────────────────────────────────────────────────────────
CHUNK_SIZE       = 1000
CHUNK_OVERLAP    = 200
EMBED_BATCH_SIZE = 50        # Stay within Gemini free-tier rate limits
EMBED_BATCH_WAIT = 60        # Seconds to wait between embedding batches
