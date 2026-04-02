"""Prompt builders for grounded mall-assistant responses."""

from __future__ import annotations

from collections.abc import Sequence

from app.retrieval.vector_store import SearchResult
from app.session_memory import ConversationTurn

SYSTEM_PROMPT = """
You are a friendly and helpful shopping-mall concierge.
Your job is to help visitors find shops, check opening hours, discover products, and navigate the malls you know about.

Rules:
1. Answer ONLY using the provided retrieved context — never invent shop names, hours, floors, or malls.
2. When the user's request is broad (e.g., "I want shoes"), suggest the matching shops from the context AND ask a brief follow-up question to narrow the search (brand preference, price range, specific mall, etc.).
3. If the user indicates they want to go to a specific shop, give the grounded location details and ask whether they want directions to that shop.
4. If the context contains partial matches, present what you have and offer to help refine.
5. If the user asks for directions but the retrieved context only contains mall and floor details, be transparent about that and give the grounded location details you do know.
6. If the context has no relevant information, say so warmly and suggest what you CAN help with (e.g., "I don't have jacket shops in my records, but I can help you find fashion stores — would you like that?").
7. Use the recent conversation to resolve follow-up questions like "okay", "that one", or "I want that".
8. Always be conversational, warm, and proactive — like a real concierge, not a search engine.
9. You can respond in the same language the user writes in (Thai or English).
""".strip()

FOLLOW_UP_REWRITE_PROMPT = """
You rewrite short follow-up chat messages into standalone requests for a shopping-mall concierge system.

Rules:
1. Use the recent conversation to resolve references like "that one", "go there", "okay", or "ต้องการ".
2. Preserve concrete entities such as shop name, mall name, floor, category, and the user's intent.
3. If the latest message is already standalone, return it unchanged.
4. ONLY rewrite into a directions request when the latest message is an affirmative reply to an assistant question about directions.
5. If the user only says they want to go to a shop, rewrite it as wanting to go to that shop, not as asking for directions yet.
6. Return ONLY the standalone user request text. No quotes, no markdown, no explanation.
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


def build_conversation_block(conversation_history: Sequence[ConversationTurn]) -> str:
    """Render recent conversation history into a prompt-ready block."""
    if not conversation_history:
        return ""

    lines: list[str] = []
    for turn in conversation_history:
        label = "User" if turn.role == "user" else "Assistant"
        lines.append(f"{label}: {turn.content}")
    return "\n".join(lines)


def _is_directions_request(text: str) -> bool:
    lowered = text.lower()
    direction_markers = {
        "direction",
        "directions",
        "route",
        "how do i get",
        "find the way",
        "หาทาง",
        "เส้นทาง",
        "ทางไป",
    }
    return any(marker in lowered for marker in direction_markers)


def build_user_prompt(
    query: str,
    sources: Sequence[SearchResult],
    conversation_history: Sequence[ConversationTurn] | None = None,
    resolved_query: str | None = None,
) -> str:
    """Build the grounded user prompt for the LLM."""
    context_block = build_context_block(sources)
    prompt = (
        "Use the retrieved context below to answer the user question.\n"
        "If you can answer, also suggest a helpful follow-up (e.g., 'Would you "
        "like directions?' or 'Want me to check other malls?').\n"
        "If the context is insufficient, say so warmly and suggest what you can help with.\n\n"
    )
    if conversation_history:
        prompt += (
            f"Recent conversation:\n{build_conversation_block(conversation_history)}\n\n"
        )
    if resolved_query and resolved_query != query:
        prompt += f"Resolved user intent for retrieval: {resolved_query}\n\n"
        if _is_directions_request(resolved_query):
            prompt += (
                "The user is asking for directions to a specific shop. "
                "Use the grounded mall and floor details from the retrieved context as the answer, "
                "and do not ask again whether they want directions.\n\n"
            )
    prompt += (
        f"Retrieved context:\n{context_block}\n\n"
        f"User question: {query}"
    )
    return prompt


def build_messages(
    query: str,
    sources: Sequence[SearchResult],
    conversation_history: Sequence[ConversationTurn] | None = None,
    resolved_query: str | None = None,
) -> list[dict[str, str]]:
    """Build chat messages for the LLM client."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_user_prompt(
                query,
                sources,
                conversation_history=conversation_history,
                resolved_query=resolved_query,
            ),
        },
    ]


def build_follow_up_rewrite_messages(
    query: str,
    conversation_history: Sequence[ConversationTurn],
) -> list[dict[str, str]]:
    """Build messages that rewrite an ambiguous follow-up into a standalone query."""
    history_block = build_conversation_block(conversation_history)
    return [
        {"role": "system", "content": FOLLOW_UP_REWRITE_PROMPT},
        {
            "role": "user",
            "content": (
                f"Recent conversation:\n{history_block}\n\n"
                f"Latest user message: {query}"
            ),
        },
    ]
