# Retail RAG Chatbot

A production-ready Retrieval-Augmented Generation (RAG) chatbot that answers user questions about mall shops and tenants. Built with FastAPI, Qdrant, and OpenAI, it demonstrates a complete pipeline: data ingestion with intelligent cleaning and LLM-assisted normalization, hybrid lexical-plus-vector retrieval, grounded LLM generation, real OpenAI moderation, transparent source attribution, and a Gradio-based user interface — all containerized with Docker.

---

## Table of Contents

- [Retail RAG Chatbot](#retail-rag-chatbot)
	- [Table of Contents](#table-of-contents)
	- [Architecture Overview](#architecture-overview)
	- [Tech Stack](#tech-stack)
		- [Why Qdrant over alternatives](#why-qdrant-over-alternatives)
	- [Project Structure](#project-structure)
	- [Setup Instructions](#setup-instructions)
		- [Prerequisites](#prerequisites)
		- [API Key Configuration](#api-key-configuration)
		- [Local Development Setup](#local-development-setup)
		- [Docker Setup (Recommended)](#docker-setup-recommended)
	- [Running the Application](#running-the-application)
		- [Quick smoke test](#quick-smoke-test)
		- [Ingestion script flags](#ingestion-script-flags)
	- [API Reference](#api-reference)
		- [`POST /chat` — Ask a question](#post-chat--ask-a-question)
		- [`GET /health` — Service health check](#get-health--service-health-check)
		- [Request tracking](#request-tracking)
		- [Auto-generated docs](#auto-generated-docs)
	- [Data Processing Pipeline](#data-processing-pipeline)
		- [Source Data](#source-data)
		- [Data Cleaning](#data-cleaning)
		- [Intelligent Name Normalization](#intelligent-name-normalization)
		- [Chunking Strategy](#chunking-strategy)
		- [Embedding Strategy](#embedding-strategy)
	- [RAG Pipeline](#rag-pipeline)
		- [Retrieval](#retrieval)
		- [Prompt Engineering](#prompt-engineering)
		- [Generation](#generation)
		- [Guardrails](#guardrails)
			- [Layer 1 — Input guardrails (`app/guardrails/input_guard.py`)](#layer-1--input-guardrails-appguardrailsinput_guardpy)
			- [Layer 2 — Prompt-level guardrails](#layer-2--prompt-level-guardrails)
			- [Layer 3 — Output guardrails (`app/guardrails/output_guard.py`)](#layer-3--output-guardrails-appguardrailsoutput_guardpy)
			- [Layer 4 — Source transparency](#layer-4--source-transparency)
	- [Gradio UI](#gradio-ui)
		- [Chat Tab](#chat-tab)
		- [Normalization Review Tab (Admin)](#normalization-review-tab-admin)
	- [Testing](#testing)
		- [Test coverage by module](#test-coverage-by-module)
	- [Configuration Reference](#configuration-reference)

---

## Architecture Overview

The system is split into two main flows — **ingestion** (offline, one-time) and **query** (online, per-request):

```
                          ┌─────────────────────────────────────┐
                          │         INGESTION FLOW              │
                          │                                     │
  shops.csv ──► Loader ──► Cleaner ──► Normalizer ──► Chunker  │
                          │                    │                │
                          │          name_mappings.json         │
                          │                    │                │
                          │              Embedder               │
                          │                    │                │
                          │              Qdrant                 │
                          └─────────────────────────────────────┘

                          ┌─────────────────────────────────────┐
                          │           QUERY FLOW                │
                          │                                     │
  User Query ──► Input Guardrails ──► Query Analyzer            │
                          │                │                    │
                          │          Embed Query                │
                          │                │                    │
                          │          Hybrid Retrieval           │
                          │          (metadata filters +        │
                          │           Qdrant vector search +    │
                          │           lexical reranking)        │
                          │                │                    │
                          │          Prompt Builder             │
                          │          (system + context + query) │
                          │                │                    │
                          │          OpenAI Generation          │
                          │                │                    │
                          │          Output Guardrails          │
                          │          (grounding + confidence)   │
                          │                │                    │
                          │          Answer + Sources +         │
                          │          Guardrail Metadata         │
                          └─────────────────────────────────────┘
```

Every response includes the source chunks used to generate the answer, so users can verify information themselves.

---

## Tech Stack

| Component       | Choice                                    | Rationale                                                                                         |
|-----------------|-------------------------------------------|---------------------------------------------------------------------------------------------------|
| Language        | Python 3.11+                              | Industry standard for AI/ML workloads                                                             |
| API Framework   | FastAPI                                   | Async, auto-generated OpenAPI docs, Pydantic-native request/response validation                   |
| Vector DB       | **Qdrant**                                | Production-grade (Rust), payload-based filtering, single-container deployment, scales to millions  |
| Embeddings      | OpenAI `text-embedding-3-small`           | High quality, cost-effective, 1536 dimensions                                                     |
| Local Embeddings| `sentence-transformers/all-MiniLM-L6-v2`  | Used only for name normalization clustering — avoids API cost for internal preprocessing           |
| LLM             | OpenAI `gpt-4o-mini`                      | Strong reasoning with low cost, well-suited for grounded Q&A                                      |
| GUI             | Gradio                                    | Quick to build, built-in chat UI, admin review panel                                              |
| Testing         | pytest                                    | Standard Python testing with async support                                                        |
| Logging         | structlog                                 | Structured JSON logging with request correlation IDs                                              |
| Config          | pydantic-settings                         | Type-safe, validated configuration from environment variables and `.env` files                     |
| Containers      | Docker + docker-compose                   | Reproducible multi-service deployment                                                             |

### Why Qdrant over alternatives

1. **Production-grade architecture** — Written in Rust with REST + gRPC APIs. Unlike ChromaDB (prototyping-focused) or FAISS (a library, not a DB), Qdrant is a full database with persistence, health checks, and first-class filtering.
2. **Payload-based filtering** — Queries like "sports shops in Siam Center" need combined semantic search + metadata filters on `mall_name` and `category`. Qdrant supports rich filter expressions natively alongside vector similarity.
3. **Operational simplicity** — Single Docker container, no Kubernetes requirement (unlike distributed Milvus). Fits naturally into the Docker Compose stack.
4. **Scalability** — HNSW indexing, sharding, and replication mean the architecture does not need to change if the dataset grows from 12 to millions of records.

---

## Project Structure

```
spw_retail-rag/
├── app/
│   ├── config.py              # pydantic-settings configuration (all env vars)
│   ├── main.py                # FastAPI app factory with async lifespan
│   ├── api/
│   │   ├── middleware.py      # X-Request-ID tracking, global error handler
│   │   ├── models.py         # Pydantic request/response schemas
│   │   └── routes.py         # POST /chat, GET /health
│   ├── generation/
│   │   ├── llm.py            # OpenAI chat completions wrapper with retry logic
│   │   └── prompts.py        # System prompt, context builder, message assembly
│   ├── guardrails/
│   │   ├── input_guard.py    # Content moderation + topical scope check
│   │   ├── openai_moderation.py # Real OpenAI moderation client
│   │   └── output_guard.py   # Grounding verification + confidence scoring
│   ├── session_memory.py     # In-memory session store for recent conversation turns
│   ├── ingestion/
│   │   ├── loader.py         # CSV loading with column normalization
│   │   ├── cleaner.py        # Whitespace, time format, missing value handling
│   │   ├── normalizer.py     # Embedding-based clustering + LLM review pipeline
│   │   └── openai_reviewer.py # LLM-backed normalization reviewer
│   │   ├── chunker.py        # Single and hierarchical chunking strategies
│   ├── rag/
│   │   ├── pipeline.py       # End-to-end RAG orchestration
│   │   └── query_analyzer.py # Query-to-metadata filter inference
│   └── retrieval/
│       ├── embeddings.py     # OpenAI embedding client
│       ├── hybrid.py         # Hybrid lexical + vector reranking
│       └── vector_store.py   # Qdrant wrapper (upsert, search, health)
├── data/
│   ├── shops.csv             # Source dataset (12 shops across 3 malls)
│   └── name_mappings.json    # Persisted normalization mappings
├── scripts/
│   └── ingest.py             # CLI ingestion script (--auto, --recreate)
├── tests/
│   ├── test_api.py           # API endpoint and middleware tests
│   ├── test_chunker.py       # Chunking strategy tests
│   ├── test_cleaner.py       # Data cleaning and time normalization tests
│   ├── test_generation.py    # LLM wrapper and prompt builder tests
│   ├── test_guardrails.py    # Input and output guardrail tests
│   ├── test_normalizer.py    # Normalization clustering and mapping tests
│   ├── test_pipeline.py      # End-to-end RAG pipeline tests
│   ├── test_session_memory.py # Session-memory retention tests
│   └── test_vector_store.py  # Qdrant operations and embedding tests
├── ui/
│   └── gradio_app.py         # Gradio chat UI + normalization review panel
├── .env.example              # All configuration variables with defaults
├── Dockerfile                # Python 3.11 slim image, multi-service capable
├── docker-compose.yml        # Qdrant + API + UI + ingestion services
├── requirements.txt          # Pinned dependency ranges
└── README.md
```

---

## Setup Instructions

### Prerequisites

- **Python 3.11+** (for local development)
- **Docker** and **Docker Compose** (for containerized deployment)
- **OpenAI API key** with access to `text-embedding-3-small` and `gpt-4o-mini`

### API Key Configuration

1. Copy the example environment file:

   ```bash
   cp .env.example .env
   ```

2. Open `.env` and set your OpenAI API key:

   ```env
   OPENAI_API_KEY=sk-your-actual-api-key-here
   ```

   All other variables have sensible defaults and can be left as-is for a standard setup. See the [Configuration Reference](#configuration-reference) for the full list.

### Local Development Setup

```bash
# 1. Create and activate a virtual environment
python -m venv .venv

# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment (see API Key Configuration above)
cp .env.example .env
# Edit .env and set OPENAI_API_KEY

# 4. Start Qdrant (requires Docker)
docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant:v1.17.1

# 5. Run data ingestion
python scripts/ingest.py --auto --recreate

# 6. Start the API server
uvicorn app.main:app --reload

# 7. (Optional) Start the Gradio UI in a separate terminal
python ui/gradio_app.py
```

The API will be available at `http://localhost:8000` and the Gradio UI at `http://localhost:7860`.

### Docker Setup (Recommended)

Docker Compose manages all services — no local Python environment needed.

```bash
# 1. Configure environment
cp .env.example .env
# Edit .env and set OPENAI_API_KEY

# 2. Build and start all services (Qdrant, API, UI)
docker compose up --build -d

# 3. Run the one-off ingestion job
docker compose --profile ingest run --rm ingest

# 4. Verify everything is running
docker compose ps
```

| Service  | URL                        | Purpose                            |
|----------|----------------------------|------------------------------------|
| API      | http://localhost:8000      | FastAPI backend (chat + health)    |
| UI       | http://localhost:7860      | Gradio web interface               |
| Qdrant   | http://localhost:6333      | Vector database dashboard          |

The compose file uses Qdrant `v1.17.1` to match the pinned `qdrant-client` version. A fresh named Docker volume is used for that upgraded server version so older local Qdrant storage created with earlier versions does not break container startup.

To stop all services:

```bash
docker compose down
```

To stop and remove stored vector data:

```bash
docker compose down -v
```

---

## Running the Application

### Quick smoke test

After setup, verify the system works end-to-end:

```bash
# Health check — should return {"status":"ok","checks":{"qdrant":true,"openai":true}}
curl http://localhost:8000/health

# Chat query
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Where is Nike?"}'
```

### Ingestion script flags

| Flag          | Effect                                                                 |
|---------------|------------------------------------------------------------------------|
| `--auto`      | Skip human review of normalization — auto-approve all suggested mappings |
| `--recreate`  | Drop and recreate the Qdrant collection before upserting               |

Without `--auto`, the script prints unknown name clusters as JSON for manual review before proceeding.

---

## API Reference

### `POST /chat` — Ask a question

**Request body:**

```json
{
  "query": "What sports shops are in ICONSIAM?",
  "session_id": "c2db21ff-1cb6-4d6b-bff5-50b5e3fced1d",
  "metadata_filters": {
    "mall_name": "ICONSIAM",
    "category": "Sports"
  }
}
```

- `query` (string, required): The user's natural-language question. Minimum 1 character.
- `session_id` (string, optional): A client-managed conversation id. When provided, the API reuses recent turns from that session to resolve short follow-ups like `okay`, `that one`, or `ต้องการ`.
- `metadata_filters` (object, optional): Key-value pairs to filter retrieval results by metadata fields (`mall_name`, `category`, `floor`, etc.).

**Response body:**

```json
{
  "answer": "Nike is a Sports shop located on floor 1 of ICONSIAM, offering athletic footwear and apparel. It's open from 10:00 to 22:00.",
  "session_id": "c2db21ff-1cb6-4d6b-bff5-50b5e3fced1d",
  "sources": [
    {
      "chunk_id": "shop-1-summary",
      "chunk_text": "Nike is a Sports shop located on floor 1 of ICONSIAM. Athletic footwear and apparel. Open from 10:00 to 22:00.",
      "relevance_score": 0.92,
      "shop_name": "Nike",
      "mall_name": "ICONSIAM",
      "floor": "1",
      "category": "Sports",
      "open_time": "10:00",
      "close_time": "22:00",
      "chunk_type": "summary",
      "parent_chunk_id": ""
    }
  ],
  "guardrails": {
    "input_flagged": false,
    "input_in_scope": true,
    "grounding_verified": true,
    "confidence": "high",
    "reason": "ok"
  }
}
```

- `answer`: The generated response, grounded in the retrieved sources.
- `session_id`: The conversation id that the API used for session memory. The Gradio UI stores and reuses this automatically per browser chat session.
- `sources`: The exact data chunks used to produce the answer, including full metadata and relevance scores.
- `guardrails`: A transparency object describing what safety checks were performed and their results.
- `retrieval_debug`: Reviewer-facing retrieval diagnostics including inferred filters, tried filter plans, candidate scores, and reranking decisions.

Example `retrieval_debug` payload:

```json
{
  "query": "Where can I buy Nike shoes?",
  "explicit_filters": {},
  "inferred_filters": {
    "shop_name": "Nike",
    "category": "Sports"
  },
  "merged_filters": {
    "shop_name": "Nike",
    "category": "Sports"
  },
  "filter_plans": [
    {"shop_name": "Nike", "category": "Sports"},
    {"shop_name": "Nike"},
    {"category": "Sports"},
    {}
  ],
  "candidates": [
    {
      "rank": 1,
      "selected": true,
      "chunk_id": "shop-0-summary",
      "shop_name": "Nike",
      "mall_name": "ICONSIAM",
      "category": "Sports",
      "vector_score": 0.4658,
      "lexical_score": 0.95,
      "metadata_boost": 0.32,
      "hybrid_score": 0.9558
    }
  ]
}
```

### `GET /health` — Service health check

**Response:**

```json
{
  "status": "ok",
  "checks": {
    "qdrant": true,
    "openai": true
  }
}
```

Returns `"degraded"` if either dependency is unreachable.

### Request tracking

Every response includes an `X-Request-ID` header (UUID). If a client sends this header, the server echoes it back; otherwise the server generates one. This ID appears in structured log entries for traceability.

### Auto-generated docs

FastAPI provides interactive API documentation:

- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

---

## Data Processing Pipeline

### Source Data

The provided `shops.csv` contains **12 shop records** across **3 Bangkok malls** (ICONSIAM, Siam Center, Siam Paragon). The data intentionally includes quality issues that the ingestion pipeline must handle:

| Issue                     | Examples in the CSV                                  |
|---------------------------|------------------------------------------------------|
| Inconsistent mall names   | `"Icon Siam"`, `"icon-siam"`, `"ICONSIAM"`          |
|                           | `"Siam Center"`, `"Siam-Center"`                    |
|                           | `"Siam Paragon"`, `"SiamParagon"`                   |
| Mixed time formats        | `"10:00 AM"`, `"10am"`, `"08:00"`, `"0900"`, `"9.30"`, `"10.00"` |
| Varying capitalization    | Present across all text columns                      |

### Data Cleaning

The cleaner (`app/ingestion/cleaner.py`) addresses these issues systematically:

1. **Schema validation** — Verifies all required columns are present: `mall_name`, `shop_name`, `category`, `floor`, `description`, `open_time`, `close_time`.
2. **Whitespace normalization** — Strips leading/trailing whitespace and collapses internal runs on all text columns.
3. **Missing value handling** — Fills empty `description` fields with `"No description available."` to ensure every chunk has readable text.
4. **Time standardization** — Converts all time values to `HH:MM` 24-hour format. Supported input formats:

   | Input      | Output  |
   |------------|---------|
   | `10:00 AM` | `10:00` |
   | `10am`     | `10:00` |
   | `8:00 PM`  | `20:00` |
   | `08:00`    | `08:00` |
   | `0900`     | `09:00` |
   | `9.30`     | `09:30` |
   | `10.00`    | `10:00` |

### Intelligent Name Normalization

Rather than hardcoding a name mapping (brittle, fails on unseen variants), the system uses a multi-stage normalization pipeline (`app/ingestion/normalizer.py`) designed to handle new data without code changes:

**Stage 1 — Embedding-based clustering:**
- Compute text similarity (Python `SequenceMatcher`) on normalized forms of each unique name (lowercased, alphanumeric only).
- Optionally use cosine similarity on sentence-transformer embeddings for semantic matching.
- Use a **Union-Find** algorithm to group names exceeding a configurable similarity threshold (default 0.75).
- Example result: `{"ICONSIAM": ["Icon Siam", "icon-siam", "ICONSIAM"]}`

**Stage 2 — LLM semantic review:**
- An OpenAI-backed reviewer validates candidate clusters and chooses canonical names.
- The reviewer is prompted to approve only true same-mall variants and to return standardized customer-facing names as JSON.
- If the LLM review fails for any reason, the ingestion flow falls back to the deterministic default reviewer so ingestion still completes safely.

**Stage 3 — Human review (Gradio UI):**
- The Gradio admin panel ("Normalization Review" tab) displays suggested clusters.
- Users can edit, approve, or reject groupings before applying them.
- This provides a human-in-the-loop safety net for production data quality.

**Stage 4 — Persistent mapping store:**
- Approved mappings are saved to `data/name_mappings.json`.
- On subsequent ingestion runs, known variants are auto-corrected from the stored mapping.
- New, unknown names are flagged for another review cycle.
- This creates a **self-improving** normalization system over time.

### Chunking Strategy

The chunker (`app/ingestion/chunker.py`) converts cleaned shop records into `ChunkDocument` objects ready for embedding and storage.

**Current approach (small dataset):**

Each shop row becomes **one self-contained natural-language chunk**:

```
Nike is a Sports shop located on floor 1 of ICONSIAM.
Athletic footwear and apparel. Open from 10:00 to 22:00.
```

This is the `"single"` strategy — appropriate when each record is short enough to fit in a single embedding window without information loss.

**Built-in hierarchical mode (for larger datasets):**

The chunker also supports a `"hierarchical"` strategy with:

- A **summary chunk** for the shop overview (always generated).
- **Detail chunks** from longer descriptions using a sliding token window (`max_chunk_tokens`, `overlap_tokens`).
- **Parent-child relationships** — detail chunks reference their parent summary chunk via `parent_chunk_id`.
- Full metadata preserved on every chunk (mall name, floor, category, times).

Switching between strategies requires only changing the `CHUNK_STRATEGY` environment variable — no code changes needed.

**Metadata on every chunk:**

Each `ChunkDocument` carries: `chunk_id`, `text`, `mall_name`, `shop_name`, `category`, `floor`, `open_time`, `close_time`, `chunk_type` ("summary" or "detail"), and `parent_chunk_id`.

### Embedding Strategy

- **Document and query embeddings** both use OpenAI `text-embedding-3-small` (1536 dimensions) to ensure retrieval consistency.
- **Name normalization clustering** uses the local `sentence-transformers/all-MiniLM-L6-v2` model — this avoids API costs for internal preprocessing and keeps the normalization pipeline independent of the production embedding model.
- Embeddings are generated in batch during ingestion and individually per query at request time.

---

## RAG Pipeline

The end-to-end pipeline (`app/rag/pipeline.py`) orchestrates every step of the query flow:

```
1. Input Guardrails  →  Block harmful or off-topic queries
2. Session Memory    →  Load recent turns for the current `session_id`
3. Resolve Follow-up →  Rewrite short/ambiguous replies into a standalone request when needed
4. Embed Query       →  OpenAI text-embedding-3-small
5. Retrieve          →  Qdrant semantic search (top-k, score threshold, metadata filters)
6. No Sources?       →  Return warm concierge fallback
7. Build Prompt      →  System prompt + recent conversation + source context + user question
8. Generate          →  OpenAI gpt-4o-mini completion
9. Output Guardrails →  Verify grounding + assess confidence
10. Ungrounded?      →  Replace answer with safe fallback
11. Return           →  Answer + sources + guardrail metadata + `session_id`
```

### Retrieval

The retrieval path now combines Qdrant vector search with query understanding and lexical reranking:

- **Session-aware follow-up resolution** rewrites short messages such as `โอเค ไป Zara` or `ต้องการ` into standalone retail queries using the recent conversation before retrieval runs.

- **Query analyzer** infers metadata filters from the user query, including:
  - `shop_name` alias matching such as `applestore` -> `Apple Store`
  - `mall_name` alias matching using normalized historical variants
  - `category` inference from product intent, such as `watch` -> `Jewelry`, `shoes` -> `Sports`, and Thai fashion-intent phrases like `ชุดลำลอง` -> `Fashion`
- **Metadata filtering** is applied in Qdrant using payload filters when explicit or inferred metadata is available.
- **Hybrid retrieval** collects vector candidates from one or more filter plans and reranks them using:
  - vector similarity,
  - lexical token overlap,
  - exact shop-name and mall-name phrase matches,
  - inferred metadata match boosts.
- **Fallback filter plans** progressively relax inferred filters only when needed, improving recall without dropping explicit user-supplied filters.

This hybrid approach fixed weak-product-intent queries that pure vector retrieval missed, such as `Where can I buy Nike shoes?`, `Where can I buy a watch?`, and Thai fashion-intent queries like `อยากได้ชุดลำลอง`.

### Prompt Engineering

The prompt system (`app/generation/prompts.py`) is designed to prevent hallucination, use recent chat history safely, and keep answers grounded:

**System prompt:**

```
You are a friendly and helpful shopping-mall concierge.
Answer only using the provided retrieved context.
Do not invent shop names, opening hours, categories, mall names, or floor information.
When the request is broad, suggest matching shops from the context and ask a brief follow-up question.
When the user wants to go to a specific shop, give the grounded location details and ask whether they want directions.
If the context is insufficient, say so warmly and suggest what you can still help with.
Respond conversationally and match the user's language when possible.
```

**Context injection:**

Retrieved chunks are formatted as numbered source blocks injected verbatim into the prompt:

```
[Source 1]
Shop: Nike | Mall: ICONSIAM | Floor: 1 | Category: Sports
Content: Nike is a Sports shop located on floor 1 of ICONSIAM.
Athletic footwear and apparel. Open from 10:00 to 22:00.
```

**User prompt wrapper:**

The user's question is wrapped with grounding plus concierge behavior instructions, and the recent conversation is included when a `session_id` is in play:

```
Use the retrieved context below to answer the user question.
If you can answer, also suggest a helpful follow-up.
If the context is insufficient, say so warmly and suggest what you can help with.

Recent conversation:
User: ...
Assistant: ...

Retrieved context:
{numbered source blocks}

User question: {user's query}
```

For short follow-ups, the system also runs a small rewrite prompt first so messages like `ต้องการ` become a standalone request for retrieval while the final answer still uses the original conversational turn.

### Generation

The LLM client (`app/generation/llm.py`) wraps OpenAI's Chat Completions API with:

- **Configurable model** — defaults to `gpt-4o-mini` with `temperature=0.1` for deterministic, factual answers.
- **Token limit** — `max_tokens=1024` (configurable via `LLM_MAX_TOKENS`).
- **Retry logic** — up to 2 retries with a 1-second delay on transient API errors, preventing single-request failures from surfacing to users.

### Guardrails

The system uses **four layers of guardrails** instead of relying on a single prompt instruction:

#### Layer 1 — Input guardrails (`app/guardrails/input_guard.py`)

| Check                | Mechanism                                                                                                   |
|----------------------|-------------------------------------------------------------------------------------------------------------|
| Content moderation   | Real OpenAI moderation via `omni-moderation-latest` with a pluggable client seam for other providers       |
| Scope fast-pass      | Expanded English and Thai shopping-intent keywords catch obvious in-scope queries for free                  |
| Scope fallback       | A lightweight `gpt-4o-mini` intent classifier runs only when keywords miss to catch indirect or multilingual retail intent |
| Empty query          | Rejects blank input before processing                                                                       |

The scope guard now uses a tiered strategy rather than a keyword-only gate. Obvious requests such as `Where can I buy shoes?`, `I want a T-shirt`, and `อยากได้ชุดลำลองใส่ไปเที่ยว` pass immediately through the keyword fast-path, while ambiguous queries can still be rescued by the LLM intent classifier. Truly off-topic questions receive a warm concierge redirect instead of a blunt rejection.

#### Layer 2 — Prompt-level guardrails

- The system prompt explicitly instructs the LLM to answer only from provided context.
- It instructs the LLM not to invent any shop names, hours, categories, malls, or floors.
- Retrieved chunks are injected verbatim so the LLM has clear, unambiguous source material.

#### Layer 3 — Output guardrails (`app/guardrails/output_guard.py`)

| Check                  | How it works                                                                                     |
|------------------------|--------------------------------------------------------------------------------------------------|
| Time grounding         | Extracts all `HH:MM` patterns from the LLM answer and verifies each exists in source texts or metadata |
| Floor grounding        | Extracts floor references and confirms they appear in the source data                            |
| Entity grounding       | Checks that shop names and mall names from retrieved sources appear in the combined source text   |
| Confidence scoring     | Based on the best retrieval score: **high** (≥ 0.85), **medium** (≥ 0.65), **low** (< 0.65)     |

If grounding verification fails, the pipeline replaces the generated answer with a warmer concierge fallback: _"I don't have that specific information in my records right now. I can help you find shops, check opening hours, or suggest stores by category — what would you like to know?"_

#### Layer 4 — Source transparency

Every response includes the `sources` array and `guardrails` object, giving the consumer full visibility into:

- Which data chunks were used to answer
- How relevant each chunk was (score)
- Whether the answer passed grounding checks
- The confidence level of the retrieval

---

## Gradio UI

The Gradio interface (`ui/gradio_app.py`) provides two tabs:

### Chat Tab

A web-based chat interface for asking questions about mall shops. It displays:

- The AI-generated answer
- Guardrail results (flagged, in-scope, grounding, confidence)
- Full source chunks with metadata
- A browser-session-backed conversation that reuses the same `session_id` automatically across turns
- A `New Session` button to clear the current chat memory and start a fresh conversation

### Normalization Review Tab (Admin)

An admin panel for reviewing and approving name normalization suggestions:

1. Click "Generate Suggestions" to run the clustering pipeline on the current data.
2. Review the suggested canonical-name → variant mappings.
3. Edit if needed, then approve and apply.
4. Approved mappings persist to `data/name_mappings.json` for future ingestion runs.

Access the UI at **http://localhost:7860** after starting the services.

The Chat tab now exposes a **Retrieval Debug** panel so reviewers can inspect:

- inferred filters extracted from the user query,
- the metadata filter plans tried against Qdrant,
- candidate chunks returned by each plan,
- lexical/vector scoring breakdowns,
- which candidates survived reranking.

---

## Testing

The project currently includes **62 collected tests** across **11 test modules**, all using **pytest** with mocks and stubs (no external services required):

```bash
# Run the full test suite
pytest

# Run with verbose output
pytest -v

# Run a specific test file
pytest tests/test_pipeline.py
```

### Test coverage by module

| Test File              | Tests | What is covered                                                                          |
|------------------------|-------|------------------------------------------------------------------------------------------|
| `test_cleaner.py`      | 3     | Time normalization (10 parametrized format cases), data cleaning, missing column detection |
| `test_normalizer.py`   | 7     | Name clustering, embedding similarity, external reviewer protocol, save/load round-trip, unknown name detection, OpenAI reviewer parsing |
| `test_chunker.py`      | 4     | Single-chunk strategy, hierarchical strategy, metadata preservation, metadata field correctness |
| `test_vector_store.py` | 8     | Embedding batch/query, Qdrant filter construction, collection CRUD, upsert, search results, point count, health check |
| `test_generation.py`   | 4     | Context block building, message assembly, LLM completion, retry exhaustion handling       |
| `test_guardrails.py`   | 12    | Flagged-content blocking, shopping-intent detection, Thai in-scope detection, LLM intent fallback, real moderation client seam, grounded-answer verification, unknown-time detection |
| `test_hybrid_retrieval.py` | 3  | Hybrid filter plans, lexical reranking, minimum-score cutoff behavior                     |
| `test_pipeline.py`     | 5     | Out-of-scope blocking, no-sources fallback, grounded answer flow, ungrounded → fallback replacement, short follow-up query rewriting |
| `test_api.py`          | 4     | Chat endpoint response schema, session-id propagation, session-history reuse, health endpoint, X-Request-ID middleware propagation |
| `test_query_evaluations.py` | 1 | Fixture-driven regression coverage for Thai and product-intent retail queries             |
| `test_session_memory.py` | 2 | In-memory session history retention and truncation behavior                               |

All tests mock external dependencies (OpenAI, Qdrant) and can run offline without any API keys.

---

## Configuration Reference

All settings are managed via environment variables (or a `.env` file). The table below lists every variable with its default:

| Variable                  | Default                    | Description                                                |
|---------------------------|----------------------------|------------------------------------------------------------|
| `OPENAI_API_KEY`          | _(required)_               | OpenAI API key for embeddings and generation               |
| `QDRANT_HOST`             | `localhost`                | Qdrant server hostname                                     |
| `QDRANT_PORT`             | `6333`                     | Qdrant HTTP API port                                       |
| `QDRANT_COLLECTION_NAME`  | `mall_shops`               | Name of the Qdrant collection for shop vectors             |
| `EMBEDDING_MODEL`         | `text-embedding-3-small`   | OpenAI embedding model                                     |
| `EMBEDDING_DIMENSIONS`    | `1536`                     | Embedding vector dimensionality                            |
| `LLM_MODEL`              | `gpt-4o-mini`              | OpenAI chat model for answer generation                    |
| `LLM_TEMPERATURE`        | `0.1`                      | Generation temperature (lower = more deterministic)        |
| `LLM_MAX_TOKENS`         | `1024`                     | Maximum tokens in generated response                       |
| `MODERATION_ENABLED`     | `true`                     | Enable real OpenAI moderation before retrieval             |
| `MODERATION_MODEL`       | `omni-moderation-latest`   | OpenAI moderation model                                    |
| `NORMALIZATION_REVIEW_MODEL` | `gpt-4o-mini`          | OpenAI model used for normalization cluster review         |
| `RETRIEVAL_TOP_K`        | `5`                        | Maximum number of chunks to retrieve per query             |
| `RETRIEVAL_SCORE_THRESHOLD` | `0.5`                   | Minimum cosine similarity score to include a result        |
| `HYBRID_CANDIDATE_MULTIPLIER` | `4`                  | Candidate expansion multiplier before hybrid reranking     |
| `HYBRID_MIN_SCORE`       | `0.2`                      | Minimum hybrid score required to keep a retrieval result   |
| `APP_HOST`               | `0.0.0.0`                  | API server bind address                                    |
| `APP_PORT`               | `8000`                     | API server port                                            |
| `LOG_LEVEL`              | `INFO`                     | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`)        |
| `ENVIRONMENT`            | `development`              | Runtime environment (`development` or `production`)        |
| `DATA_CSV_PATH`          | `data/shops.csv`           | Path to the source CSV file                                |
| `NAME_MAPPINGS_PATH`     | `data/name_mappings.json`  | Path to the persisted normalization mappings                |
| `CHUNK_STRATEGY`         | `single`                   | Chunking mode: `single` (one chunk per shop) or `hierarchical` (summary + detail windows) |
| `CHUNK_MAX_TOKENS`       | `256`                      | Maximum tokens per detail chunk (hierarchical mode)        |
| `CHUNK_OVERLAP_TOKENS`   | `50`                       | Overlap tokens between consecutive detail chunks           |