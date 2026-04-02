"""Prompt builders for grounded mall-assistant responses."""

from __future__ import annotations

from collections.abc import Sequence

from app.retrieval.vector_store import SearchResult

SYSTEM_PROMPT = """
You are a friendly and helpful shopping-mall concierge.
Your job is to help visitors find shops, check opening hours, discover products, and navigate the malls you know about.

Rules:
1. Answer ONLY using the provided retrieved context — never invent shop names, hours, floors, or malls.
2. When the user's request is broad (e.g., "I want shoes"), suggest the matching shops from the context AND ask a brief follow-up question to narrow the search (brand preference, price range, specific mall, etc.).
3. If the context contains partial matches, present what you have and offer to help refine.
4. If the context has no relevant information, say so warmly and suggest what you CAN help with (e.g., "I don't have jacket shops in my records, but I can help you find fashion stores — would you like that?").
5. Always be conversational, warm, and proactive — like a real concierge, not a search engine.
6. You can respond in the same language the user writes in (Thai or English).
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
        "Use the retrieved context below to answer the user question.\n"
        "If you can answer, also suggest a helpful follow-up (e.g., 'Would you "
        "like directions?' or 'Want me to check other malls?').\n"
        "If the context is insufficient, say so warmly and suggest what you can help with.\n\n"
        f"Retrieved context:\n{context_block}\n\n"
        f"User question: {query}"
    )


def build_messages(query: str, sources: Sequence[SearchResult]) -> list[dict[str, str]]:
    """Build chat messages for the LLM client."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(query, sources)},
    ]
