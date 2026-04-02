"""Gradio interface for chat and normalization review."""

from __future__ import annotations

import json
import os
from uuid import uuid4

import gradio as gr
import httpx
import pandas as pd

from app.config import get_settings
from app.ingestion.cleaner import clean_shop_data
from app.ingestion.loader import load_csv
from app.ingestion.normalizer import (
    cluster_names,
    detect_unknown_names,
    flatten_mappings,
    load_name_mappings,
    review_clusters,
    save_name_mappings,
)
from app.ingestion.openai_reviewer import OpenAINameReviewer

settings = get_settings()
DEFAULT_API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


def chat_with_api(message, history, api_base_url, session_id):
    response = httpx.post(
        f"{api_base_url.rstrip('/')}/chat",
        json={"query": message, "session_id": session_id or str(uuid4())},
        timeout=30.0,
    )
    response.raise_for_status()
    payload = response.json()
    sources_markdown = (
        "\n\n".join(
            [
                (
                    f"### {source.get('shop_name', 'Unknown')}\n"
                    f"- Mall: {source.get('mall_name', 'Unknown')}\n"
                    f"- Floor: {source.get('floor', 'Unknown')}\n"
                    f"- Category: {source.get('category', 'Unknown')}\n"
                    f"- Score: {source.get('relevance_score', 0):.2f}\n"
                    f"- Chunk: {source.get('chunk_text', '')}"
                )
                for source in payload.get("sources", [])
            ]
        )
        or "No sources returned."
    )

    transcript = (
        history or ""
    ) + f"User: {message}\nAssistant: {payload['answer']}\n\n"
    return (
        transcript,
        payload["answer"],
        json.dumps(payload["guardrails"], indent=2),
        sources_markdown,
        json.dumps(payload.get("retrieval_debug") or {}, indent=2, ensure_ascii=False),
        payload["session_id"],
    )


def reset_chat_session():
    return "", "", "{}", "No sources returned.", "{}", str(uuid4())


def generate_normalization_suggestions():
    raw_df = load_csv(settings.data_csv_path)
    cleaned_df = clean_shop_data(raw_df)
    existing_mappings = load_name_mappings(settings.name_mappings_path)
    unique_names = sorted(cleaned_df["mall_name"].dropna().unique().tolist())
    unknown_names = detect_unknown_names(unique_names, existing_mappings)
    suggestions = cluster_names(unknown_names)
    reviewer = (
        OpenAINameReviewer(
            api_key=settings.openai_api_key,
            model=settings.normalization_review_model,
        )
        if settings.openai_api_key
        else None
    )
    reviewed = review_clusters(suggestions, reviewer=reviewer)

    rows = []
    for canonical_name, variants in reviewed.items():
        rows.append(
            {
                "approved": True,
                "canonical_name": canonical_name,
                "variants": ", ".join(variants),
            }
        )

    return json.dumps(rows, indent=2)


def apply_approved_suggestions(suggestions_text):
    if not suggestions_text or not str(suggestions_text).strip():
        return "No suggestions to apply."

    suggestions = json.loads(suggestions_text)
    if not suggestions:
        return "No suggestions to apply."

    canonical_to_variants: dict[str, list[str]] = {}
    for row in suggestions:
        if not bool(row.get("approved", False)):
            continue
        canonical = str(row.get("canonical_name", "")).strip()
        variants = [
            variant.strip()
            for variant in str(row.get("variants", "")).split(",")
            if variant.strip()
        ]
        if canonical and variants:
            canonical_to_variants[canonical] = variants

    mappings = flatten_mappings(canonical_to_variants)
    existing_mappings = load_name_mappings(settings.name_mappings_path)
    merged = {**existing_mappings, **mappings}
    save_name_mappings(merged, settings.name_mappings_path)
    return f"Saved {len(mappings)} mapping entries to {settings.name_mappings_path}."


def load_current_mappings():
    mappings = load_name_mappings(settings.name_mappings_path)
    return json.dumps(mappings, indent=2)


with gr.Blocks(title="Retail RAG Assistant") as demo:
    gr.Markdown("# Retail RAG Assistant")

    with gr.Tab("Chat"):
        api_base_url = gr.Textbox(label="API Base URL", value=DEFAULT_API_BASE_URL)
        session_id = gr.State(str(uuid4()))
        message = gr.Textbox(label="Your question")
        transcript = gr.Textbox(label="Conversation", lines=12)
        answer_box = gr.Textbox(label="Latest Answer", lines=4)
        guardrails_json = gr.Code(label="Guardrails", language="json")
        sources_markdown = gr.Markdown(label="Sources")
        retrieval_debug_json = gr.Code(label="Retrieval Debug", language="json")
        send_button = gr.Button("Send")
        new_session_button = gr.Button("New Session")

        send_button.click(
            chat_with_api,
            inputs=[message, transcript, api_base_url, session_id],
            outputs=[
                transcript,
                answer_box,
                guardrails_json,
                sources_markdown,
                retrieval_debug_json,
                session_id,
            ],
        )
        new_session_button.click(
            reset_chat_session,
            outputs=[
                transcript,
                answer_box,
                guardrails_json,
                sources_markdown,
                retrieval_debug_json,
                session_id,
            ],
        )

    with gr.Tab("Normalization Review"):
        suggestions_text = gr.Code(label="Generated Suggestions", language="json")
        generate_button = gr.Button("Generate Suggestions")
        apply_button = gr.Button("Apply Approved Suggestions")
        apply_status = gr.Textbox(label="Apply Status")
        current_mappings = gr.Code(label="Current Mappings", language="json")
        refresh_button = gr.Button("Refresh Current Mappings")

        generate_button.click(
            generate_normalization_suggestions, outputs=[suggestions_text]
        )
        apply_button.click(
            apply_approved_suggestions,
            inputs=[suggestions_text],
            outputs=[apply_status],
        )
        refresh_button.click(load_current_mappings, outputs=[current_mappings])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
