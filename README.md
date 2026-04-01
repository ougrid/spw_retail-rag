# Retail RAG Chatbot

Production-oriented Retrieval-Augmented Generation chatbot for answering questions about mall shops and tenants from the provided `shops.csv` dataset.

## Features

- FastAPI backend with `POST /chat` and `GET /health`
- Qdrant vector store with metadata-aware retrieval
- OpenAI embeddings and answer generation
- Intelligent mall-name normalization pipeline with persisted mappings
- Multi-layer guardrails for scope, grounding, and confidence
- Source transparency: every answer returns the chunks used to answer
- Gradio UI with chat mode and normalization review mode
- Docker and docker-compose support
- Unit tests for ingestion, retrieval, generation, guardrails, pipeline, and API layers

## Tech Stack

- Python 3.11+
- FastAPI
- Qdrant
- OpenAI API
- sentence-transformers
- Gradio
- pytest

## Why Qdrant

Qdrant was chosen as the vector database because it gives the best balance of production-readiness and operational simplicity for this assignment.

- It is a full database rather than only an indexing library.
- It supports payload-based filtering, which is important for combining semantic search with filters like `mall_name`, `category`, and `floor`.
- It is easy to run locally and in Docker.
- It scales beyond the current toy dataset without changing the architecture.
- Its Python client is straightforward and integrates well with a clean service layer.

## Architecture Overview

```text
shops.csv
	-> loader
	-> cleaner
	-> intelligent normalizer
	-> chunker
	-> OpenAI embeddings
	-> Qdrant

User query
	-> input guardrails
	-> query embedding
	-> Qdrant retrieval
	-> prompt builder
	-> OpenAI generation
	-> output guardrails
	-> answer + sources + guardrail metadata
```

## Project Structure

```text
app/
	api/           FastAPI models, routes, middleware
	generation/    OpenAI chat wrapper and prompt builders
	guardrails/    Input and output validation
	ingestion/     Loader, cleaner, normalizer, chunker
	rag/           End-to-end RAG orchestration
	retrieval/     Embedding client and Qdrant wrapper
data/
	shops.csv
	name_mappings.json
scripts/
	ingest.py
tests/
ui/
	gradio_app.py
```

## Setup

### 1. Create a virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Create `.env` from `.env.example` and fill in the required values:

```env
OPENAI_API_KEY=...
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=mall_shops
EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=gpt-4o-mini
```

### 4. Start Qdrant

If you have Docker:

```bash
docker run -p 6333:6333 qdrant/qdrant:v1.13.6
```

### 5. Run ingestion

```bash
python scripts/ingest.py --auto --recreate
```

### 6. Run the API

```bash
uvicorn app.main:app --reload
```

### 7. Run the Gradio UI

```bash
python ui/gradio_app.py
```

## Docker Usage

### Start API, UI, and Qdrant

```bash
docker compose up --build api ui qdrant
```

### Run the one-off ingestion job

```bash
docker compose --profile ingest run --rm ingest
```

## API Usage

### Chat Request

`POST /chat`

Request:

```json
{
	"query": "Where is Nike in ICONSIAM?"
}
```

Response:

```json
{
	"answer": "Nike is on floor 1 of ICONSIAM and opens at 10:00.",
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

### Health Check

`GET /health`

Response:

```json
{
	"status": "ok",
	"checks": {
		"qdrant": true,
		"openai": true
	}
}
```

## Data Cleaning Strategy

The ingestion cleaner handles the obvious inconsistencies in the supplied CSV.

- Normalizes whitespace across text columns
- Ensures the expected schema is present
- Fills missing descriptions with a safe default message
- Standardizes time formats into `HH:MM` 24-hour format

Supported examples include:

- `10:00 AM`
- `10am`
- `08:00`
- `0900`
- `9.30`
- `10.00`

## Intelligent Normalization Strategy

Mall names are normalized through a production-oriented pipeline:

1. Cluster similar names into candidate groups.
2. Support a review hook for LLM-based semantic validation.
3. Persist approved mappings to `data/name_mappings.json`.
4. Auto-apply known mappings on future ingestion runs.
5. Surface unknown names for review in the Gradio admin tab.

This prevents hardcoded-name logic from becoming the only normalization mechanism.

## Chunking Strategy

### Current dataset

The provided dataset is small, so the default strategy creates one summary chunk per shop.

Example:

```text
Nike is a Sports shop located on floor 1 of ICONSIAM. Athletic footwear and apparel. Open from 10:00 to 22:00.
```

### Production-ready extension path

The chunker already supports a hierarchical mode:

- summary chunk for the shop overview
- detail chunks for longer descriptions
- overlapping token windows
- parent-child chunk relationships
- metadata preserved on every chunk

This means the code can support longer descriptions later without replacing the retrieval design.

## Embedding Strategy

- Document embeddings use OpenAI `text-embedding-3-small`
- Query embeddings use the same model for retrieval consistency
- Local similarity grouping for normalization is kept separate from production retrieval embeddings

## Guardrails Strategy

The system uses multiple guardrail layers instead of relying on a single prompt.

### Input guardrails

- moderation hook for harmful content screening
- topical scope filtering for mall/shop-related requests

### Prompt guardrails

- grounded system prompt
- explicit instruction not to invent missing facts
- retrieved chunks injected directly into the prompt

### Output guardrails

- grounding checks against retrieved metadata and chunk text
- confidence scoring from retrieval scores
- fallback response if the generated answer is not grounded

### Source transparency

Every response includes the source chunks used for answering, including metadata and retrieval scores.

## Testing

Run the full test suite with:

```bash
pytest
```

Current test coverage includes:

- cleaner
- normalizer
- chunker
- vector store
- generation
- guardrails
- pipeline
- API

## Notes

- The current Dockerfile uses a modern pinned Python slim base, but the image scanner may still report upstream OS-package vulnerabilities. That is a base-image supply concern rather than an application-code issue and should be addressed through routine image refreshing in CI.
- The default runtime assumes a valid OpenAI API key and a reachable Qdrant instance.