# Shared settings for the web server and the CLI scripts.
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Paths used by the CLI pipeline
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
IMAGE_DIR = os.path.join(DATA_DIR, "extracted_images")
CHROMA_DB_DIR = os.path.join(PROJECT_ROOT, "chroma_db")
CHUNKS_JSON_PATH = os.path.join(DATA_DIR, "narrative_chunks.json")
SOURCE_PDF_NAME = "SAMPLE.pdf"
DEFAULT_PDF_PATH = os.path.join(DATA_DIR, SOURCE_PDF_NAME)

# Ingestion and retrieval scripts must use the same collection name
COLLECTION_NAME = "satellite_rag_collection"

# Gemini models
EMBEDDING_MODEL = "models/gemini-embedding-001"
LLM_MODEL = "gemini-3.5-flash-lite"
VISION_MODEL = "gemini-3.5-flash-lite"

RETRIEVER_TOP_K = 5
LLM_TEMPERATURE = 0.2

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBED_BATCH_SIZE = 50   # free tier rate limit
EMBED_BATCH_WAIT = 60   # seconds between batches


def get_api_key():
    # Either variable works, GOOGLE_API_KEY is checked first
    return os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
