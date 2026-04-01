"""Gradio interface for chat and normalization review."""

from __future__ import annotations

import json

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
    save_name_mappings,
)

settings = get_settings()


def chat_with_api(message: str, history: list[dict[str, str]], api_base_url: str):
    response = httpx.post(
        f"{api_base_url.rstrip('/')}/chat",
        json={"query": message},
        timeout=30.0,
    )
    response.raise_for_status()
    payload = response.json()
    sources_markdown = "\n\n".join(
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
    ) or "No sources returned."

    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": payload["answer"]},
    ]
    return history, payload["guardrails"], sources_markdown


def generate_normalization_suggestions() -> pd.DataFrame:
    raw_df = load_csv(settings.data_csv_path)
    cleaned_df = clean_shop_data(raw_df)
    existing_mappings = load_name_mappings(settings.name_mappings_path)
    unique_names = sorted(cleaned_df["mall_name"].dropna().unique().tolist())
    unknown_names = detect_unknown_names(unique_names, existing_mappings)
    suggestions = cluster_names(unknown_names)

    rows = []
    for suggestion in suggestions:
        rows.append(
            {
                "approved": True,
                "canonical_name": suggestion.canonical_name,
                "variants": ", ".join(suggestion.variants),
            }
        )

    return pd.DataFrame(rows, columns=["approved", "canonical_name", "variants"])


def apply_approved_suggestions(suggestions_df: pd.DataFrame) -> str:
    if suggestions_df is None or suggestions_df.empty:
        return "No suggestions to apply."

    canonical_to_variants: dict[str, list[str]] = {}
    for _, row in suggestions_df.iterrows():
        if not bool(row.get("approved", False)):
            continue
        canonical = str(row.get("canonical_name", "")).strip()
        variants = [variant.strip() for variant in str(row.get("variants", "")).split(",") if variant.strip()]
        if canonical and variants:
            canonical_to_variants[canonical] = variants

    mappings = flatten_mappings(canonical_to_variants)
    existing_mappings = load_name_mappings(settings.name_mappings_path)
    merged = {**existing_mappings, **mappings}
    save_name_mappings(merged, settings.name_mappings_path)
    return f"Saved {len(mappings)} mapping entries to {settings.name_mappings_path}."


def load_current_mappings() -> str:
    mappings = load_name_mappings(settings.name_mappings_path)
    return json.dumps(mappings, indent=2)


with gr.Blocks(title="Retail RAG Assistant") as demo:
    gr.Markdown("# Retail RAG Assistant")

    with gr.Tab("Chat"):
        api_base_url = gr.Textbox(label="API Base URL", value="http://localhost:8000")
        chatbot = gr.Chatbot(type="messages", label="Mall Assistant")
        message = gr.Textbox(label="Your question")
        guardrails_json = gr.JSON(label="Guardrails")
        sources_markdown = gr.Markdown(label="Sources")
        send_button = gr.Button("Send")

        send_button.click(
            chat_with_api,
            inputs=[message, chatbot, api_base_url],
            outputs=[chatbot, guardrails_json, sources_markdown],
        )

    with gr.Tab("Normalization Review"):
        suggestions_table = gr.Dataframe(
            headers=["approved", "canonical_name", "variants"],
            datatype=["bool", "str", "str"],
            row_count=(0, "dynamic"),
            col_count=(3, "fixed"),
            label="Generated Suggestions",
            interactive=True,
        )
        generate_button = gr.Button("Generate Suggestions")
        apply_button = gr.Button("Apply Approved Suggestions")
        apply_status = gr.Textbox(label="Apply Status")
        current_mappings = gr.Code(label="Current Mappings", language="json")
        refresh_button = gr.Button("Refresh Current Mappings")

        generate_button.click(generate_normalization_suggestions, outputs=[suggestions_table])
        apply_button.click(apply_approved_suggestions, inputs=[suggestions_table], outputs=[apply_status])
        refresh_button.click(load_current_mappings, outputs=[current_mappings])


if __name__ == "__main__":
    demo.launch()
