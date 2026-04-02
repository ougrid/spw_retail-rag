"""FastAPI application entry point."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.middleware import configure_app_middleware
from app.api.routes import router
from app.config import Settings, get_settings
from app.generation.llm import LLMClient
from app.guardrails.input_guard import InputGuard, LLMIntentClassifier
from app.guardrails.openai_moderation import OpenAIModerationClient
from app.guardrails.output_guard import OutputGuard
from app.rag.pipeline import RAGPipeline
from app.rag.query_analyzer import QueryAnalyzer
from app.retrieval.embeddings import EmbeddingClient
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.vector_store import QdrantVectorStore
from app.session_memory import SessionMemoryStore


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not hasattr(app.state, "pipeline"):
        settings = get_settings()
        embedding_client = EmbeddingClient(
            api_key=settings.openai_api_key,
            model=settings.embedding_model,
            dimensions=settings.embedding_dimensions,
        )
        vector_store = QdrantVectorStore(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            collection_name=settings.qdrant_collection_name,
            dimensions=settings.embedding_dimensions,
        )
        llm_client = LLMClient(
            api_key=settings.openai_api_key,
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )
        moderation_client = None
        if settings.moderation_enabled and settings.openai_api_key:
            moderation_client = OpenAIModerationClient(
                api_key=settings.openai_api_key,
                model=settings.moderation_model,
            )
        query_analyzer = QueryAnalyzer.from_paths(
            data_csv_path=settings.data_csv_path,
            name_mappings_path=settings.name_mappings_path,
        )
        retriever = HybridRetriever(
            vector_store=vector_store,
            query_analyzer=query_analyzer,
            top_k=settings.retrieval_top_k,
            candidate_multiplier=settings.hybrid_candidate_multiplier,
            minimum_hybrid_score=settings.hybrid_min_score,
        )
        intent_classifier = LLMIntentClassifier(llm_client)
        app.state.pipeline = RAGPipeline(
            embedding_client=embedding_client,
            vector_store=vector_store,
            llm_client=llm_client,
            input_guard=InputGuard(
                moderation_client=moderation_client,
                intent_classifier=intent_classifier,
            ),
            output_guard=OutputGuard(),
            query_analyzer=query_analyzer,
            retriever=retriever,
            top_k=settings.retrieval_top_k,
            score_threshold=settings.retrieval_score_threshold,
        )
        app.state.session_store = SessionMemoryStore()
        app.state.health_checks = {
            "qdrant": vector_store.health_check,
            "openai": llm_client.health_check,
        }
    yield


def create_app(
    settings: Settings | None = None,
    pipeline: RAGPipeline | None = None,
    health_checks: dict[str, callable] | None = None,
    session_store: SessionMemoryStore | None = None,
) -> FastAPI:
    app = FastAPI(title="Retail RAG API", version="0.1.0", lifespan=lifespan)
    configure_app_middleware(app)
    app.include_router(router)

    if pipeline is not None:
        app.state.pipeline = pipeline
    if health_checks is not None:
        app.state.health_checks = health_checks
    if session_store is not None:
        app.state.session_store = session_store
    if not hasattr(app.state, "session_store"):
        app.state.session_store = SessionMemoryStore()

    return app


app = create_app()
