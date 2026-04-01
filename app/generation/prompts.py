"""Prompt builders for grounded mall-assistant responses."""

from __future__ import annotations

from collections.abc import Sequence

from app.retrieval.vector_store import SearchResult

SYSTEM_PROMPT = """
You are a helpful mall information assistant.
Answer only using the provided retrieved context.
Do not invent shop names, opening hours, categories, mall names, or floor information.
If the answer is not contained in the retrieved context, say that you do not have that information.
Be concise, accurate, and conversational.
""".strip()


def build_context_block(sources: Sequence[SearchResult]) -> str:
    """Render retrieved sources into a prompt-ready context block."""
    if not sources:
        return "No relevant context was retrieved."

    lines: list[str] = []
    for index, source in enumerate(sources, start=1):
        lines.append(f"[Source {index}]")
        lines.append(f"Shop: {source.metadata.get('shop_name', 'Unknown')}")
        lines.append(f"Mall: {source.metadata.get('mall_name', 'Unknown')}")
        lines.append(f"Floor: {source.metadata.get('floor', 'Unknown')}")
        lines.append(f"Category: {source.metadata.get('category', 'Unknown')}")
        lines.append(f"Content: {source.text}")
        lines.append("")
    return "\n".join(lines).strip()


def build_user_prompt(query: str, sources: Sequence[SearchResult]) -> str:
    """Build the grounded user prompt for the LLM."""
    context_block = build_context_block(sources)
    return (
        "Use the retrieved context below to answer the user question. "
        "If the context is insufficient, say you do not have that information.\n\n"
        f"Retrieved context:\n{context_block}\n\n"
        f"User question: {query}\n\n"
        "Return a direct answer grounded in the retrieved context."
    )


def build_messages(query: str, sources: Sequence[SearchResult]) -> list[dict[str, str]]:
    """Build chat messages for the LLM client."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(query, sources)},
    ]
