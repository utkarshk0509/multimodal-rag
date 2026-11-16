# Multimodal RAG Engine

A **Retrieval-Augmented Generation (RAG)** application that understands both text and visuals from technical PDFs. Upload any research paper or scanned document and ask questions by typing or speaking.

---

## Features

- **PDF Parsing** — extracts text from native PDFs using PyMuPDF
- **Vision OCR** — falls back to Gemini Vision for scanned/image-only pages
- **Chart Understanding** — uses Gemini to describe charts and diagrams
- **Semantic Search** — ChromaDB + Gemini Embeddings for vector retrieval
- **Voice Input** — ask questions via microphone (speech-to-text)
- **Chat UI** — Streamlit-based conversational interface with source citations

---

## Architecture

```
PDF Upload
    │
    ▼
pdf_parser / advanced_parser   ← extracts text blocks, tables, images
    │
    ▼
vision_extractor               ← Gemini Vision describes each chart image
    │
    ▼
vector_store_init              ← embeds all chunks into ChromaDB
    │
    ▼
app.py (Streamlit)             ← retrieves top-k chunks → Gemini LLM → answer
```

---

## Project Structure

```
multimodal-rag/
├── backend_server.py             # FastAPI app: /api/* + serves the web UI
├── static/
│   ├── index.html                # Web UI
│   ├── css/style.css
│   └── js/
│       ├── app.js                # UI logic (upload, collections, chat)
│       └── motion.js             # Animation layer (Motion / Framer Motion engine)
├── Procfile, render.yaml         # Render deployment
├── vercel.json                   # Vercel deployment
├── requirements.txt
├── .env.example                  # Copy to .env and fill in your API key
├── data/
│   ├── SAMPLE.pdf                # Place your PDF here (for the CLI pipeline)
│   ├── extracted_images/         # Auto-generated: images extracted from PDF
│   └── narrative_chunks.json     # Auto-generated: text chunks from PDF
├── data_ingestion/
│   ├── pdf_parser.py             # Basic PDF text extractor
│   ├── advanced_parser.py        # Full parser: text + tables + images
│   └── vision_extractor.py       # Gemini Vision chart analyser
├── vector_store/
│   └── vector_store_init.py      # Embeds documents into ChromaDB
└── retrieval/
    ├── query_rag.py              # CLI: query the vector store directly
    └── generate_rag_response.py  # CLI: full RAG query → LLM response
```

---

## Quickstart

### 1. Clone and set up environment

```bash
git clone <repo-url>
cd multimodal-rag

python -m venv rag_env
# Windows
rag_env\Scripts\activate
# macOS / Linux
source rag_env/bin/activate

pip install -r requirements.txt
```

### 2. Configure API key

```bash
cp .env.example .env
# Edit .env and add your Google / Gemini API key
```

### 3. Run the web app

```bash
python backend_server.py
# or: uvicorn backend_server:app --reload
```

Open http://127.0.0.1:8000, upload a PDF, click **Build Knowledge Base**, and start chatting.

---

## Website Animations

The UI uses [Motion](https://motion.dev), the vanilla-JS edition of Framer Motion
(same spring physics and engine), loaded from jsDelivr as an ES module in
`static/js/motion.js`. No build step is required.

Add animation to any element in `index.html` declaratively, Framer-style:

| Attribute | Effect |
|---|---|
| `data-animate="fade-up"` | Enter on scroll into view. Variants: `fade-up`, `fade-down`, `fade`, `scale`, `blur`, `slide-left`, `slide-right` |
| `data-delay="0.2"` | Delay (seconds) for `data-animate` / `data-stagger` / `data-split` |
| `data-stagger` | Children enter one after another (`data-stagger-variant` picks the variant) |
| `data-split` | Word-by-word headline reveal |
| `data-count="83" data-suffix="%"` | Count-up number (`data-decimals`, `data-prefix` optional) |
| `data-tilt` | 3D pointer tilt with glare (mouse/trackpad only) |
| `data-magnetic` | Element drifts toward the cursor |
| `data-press="0.95"` | Spring press feedback (works on dynamically created elements) |

From `app.js`, call animations through `fx(name, ...args)`, e.g. `fx('enter', el, 'scale')`,
`fx('message', row, 'user')`, `fx('shake', el)`. These are no-ops when Motion is
unavailable or the visitor has **prefers-reduced-motion** enabled, and content
is shown anyway if the CDN can't be reached within 3 seconds.

---

## CLI Pipeline (for pre-processing a specific PDF)

If you want to pre-process a PDF before running the app, use the CLI pipeline in order:

```bash
# Step 1 — Extract text, tables, and images from the PDF
python data_ingestion/advanced_parser.py

# Step 2 — Describe chart images with Gemini Vision
python data_ingestion/vision_extractor.py

# Step 3 — Embed everything into ChromaDB
python vector_store/vector_store_init.py

# Step 4 — Query the vector store (optional test)
python retrieval/query_rag.py

# Step 5 — Run a full RAG response (optional test)
python retrieval/generate_rag_response.py
```

Place your PDF at `data/SAMPLE.pdf` before running Step 1.

---

## Requirements

- Python 3.10+
- A Google AI / Gemini API key ([get one here](https://aistudio.google.com/app/apikey))

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `GOOGLE_API_KEY` | Google Gemini API key (used by LangChain integrations) |
| `GEMINI_API_KEY` | Gemini API key (used by the `google-genai` SDK directly) |

Either one (or both) can be set — the app checks both.
