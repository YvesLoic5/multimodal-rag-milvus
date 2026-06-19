"""Chainlit chatbot entry point.

Handlers:
  on_chat_start  – session init, welcome message, action buttons
  on_message     – check cache → retrieve → rerank → stream LLM answer
  on_file_upload – ingest PDF / images with progress display (via action)
"""

from __future__ import annotations

import asyncio
import traceback
from pathlib import Path
from typing import Any

import chainlit as cl

from app.cache.redis_cache import get_cache
from app.generation.llm_chain import stream_answer
from app.ingestion.pipeline import ingest_file
from app.retrieval.hybrid_retriever import get_retriever
from app.retrieval.reranker import get_reranker
from app.utils.config import get_settings
from app.utils.logger import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


# ── Session helpers ───────────────────────────────────────────────────────────

def _get_history() -> list[dict[str, str]]:
    return cl.user_session.get("history", [])


def _append_history(role: str, content: str) -> None:
    history = _get_history()
    history.append({"role": role, "content": content})
    cl.user_session.set("history", history)


def _format_sources(hits: list[dict[str, Any]]) -> str:
    lines = ["**Sources retrieved:**\n"]
    for i, h in enumerate(hits, 1):
        meta = h.get("metadata", {})
        source = meta.get("source", "unknown")
        page = meta.get("page", "?")
        score = h.get("rerank_score", h.get("score", 0.0))
        modality = h.get("modality", "text")
        icon = "🖼️" if modality == "image" else "📄"
        lines.append(
            f"{icon} **[{i}]** `{source}` • Page {page} • "
            f"Score: `{score:.3f}` • `{modality}`"
        )
    return "\n".join(lines)


# ── Chat start ────────────────────────────────────────────────────────────────

@cl.on_chat_start
async def on_chat_start() -> None:
    cl.user_session.set("history", [])
    settings = get_settings()

    model_label = "GPT-4o" if settings.use_openai else f"Ollama / {settings.ollama_model}"

    welcome = (
        f"## 👋 Bienvenue dans le Multimodal RAG!\n\n"
        f"Ce système peut répondre à vos questions en s'appuyant sur des **documents PDF** "
        f"et des **images** indexés dans la base vectorielle Milvus.\n\n"
        f"**Modèle actif :** `{model_label}`  \n"
        f"**Collection Milvus :** `{settings.milvus_collection}`\n\n"
        f"Vous pouvez :\n"
        f"- 💬 Poser une question directement\n"
        f"- 📎 Joindre un PDF ou une image via le bouton d'upload\n"
    )
    await cl.Message(content=welcome).send()

    actions = [
        cl.Action(
            name="ingest_action",
            label="📂 Ingérer des documents",
            description="Charger et indexer un PDF ou une image",
            payload={"action": "ingest"},
        ),
        cl.Action(
            name="status_action",
            label="ℹ️ Statut du système",
            description="Vérifier la connexion Milvus et Redis",
            payload={"action": "status"},
        ),
    ]
    await cl.Message(content="Choisissez une action :", actions=actions).send()


# ── Action callbacks ──────────────────────────────────────────────────────────

@cl.action_callback("ingest_action")
async def on_ingest_action(action: cl.Action) -> None:
    await cl.Message(
        content="📎 Veuillez joindre un fichier (PDF ou image) à votre prochain message."
    ).send()


@cl.action_callback("status_action")
async def on_status_action(action: cl.Action) -> None:
    settings = get_settings()
    lines = [
        "### 🔍 Statut du système\n",
        f"- **Milvus** : `{settings.milvus_host}:{settings.milvus_port}`",
        f"- **Collection** : `{settings.milvus_collection}`",
        f"- **Redis** : `{settings.redis_url}`",
        f"- **Modèle LLM** : {'GPT-4o' if settings.use_openai else settings.ollama_model}",
        f"- **Embeddings texte** : `{settings.embedding_model_text}`",
        f"- **Top-K retrieval** : `{settings.top_k_retrieval}`",
        f"- **Top-K rerank** : `{settings.top_k_rerank}`",
    ]
    await cl.Message(content="\n".join(lines)).send()


# ── Main message handler ──────────────────────────────────────────────────────

@cl.on_message
async def on_message(message: cl.Message) -> None:
    # ── Handle file uploads ───────────────────────────────────────────────────
    if message.elements:
        await _handle_file_uploads(message)
        return

    question = message.content.strip()
    if not question:
        return

    # ── Check cache ───────────────────────────────────────────────────────────
    cache = get_cache()
    cached = cache.get(question)
    if cached:
        await cl.Message(
            content=f"⚡ **Cache hit!**\n\n{cached['answer']}"
        ).send()
        sources_text = cached.get("sources_text", "")
        if sources_text:
            await cl.Message(content=sources_text).send()
        _append_history("user", question)
        _append_history("assistant", cached["answer"])
        return

    # ── Retrieval ─────────────────────────────────────────────────────────────
    thinking_msg = cl.Message(content="🔍 Recherche en cours…")
    await thinking_msg.send()

    try:
        retriever = get_retriever()
        reranker = get_reranker()

        loop = asyncio.get_event_loop()
        hits = await loop.run_in_executor(
            None, retriever.retrieve, question
        )
        hits = await loop.run_in_executor(
            None, reranker.rerank, question, hits
        )

        await thinking_msg.remove()

        if not hits:
            await cl.Message(
                content="❌ Aucun document pertinent trouvé pour votre question."
            ).send()
            return

        # ── Stream answer ─────────────────────────────────────────────────────
        history = _get_history()
        answer_msg = cl.Message(content="")
        await answer_msg.send()

        full_answer = ""
        async for token in stream_answer(question, hits, history):
            await answer_msg.stream_token(token)
            full_answer += token

        await answer_msg.update()

        # ── Show sources ──────────────────────────────────────────────────────
        sources_text = _format_sources(hits)
        await cl.Message(content=sources_text).send()

        # ── Cache the result ──────────────────────────────────────────────────
        cache.set(question, {"answer": full_answer, "sources_text": sources_text})

        _append_history("user", question)
        _append_history("assistant", full_answer)

    except Exception as exc:
        await thinking_msg.remove()
        logger.error("Pipeline error", error=str(exc), trace=traceback.format_exc())
        await cl.Message(
            content=f"❌ Une erreur s'est produite : `{exc}`\nConsultez les logs pour plus de détails."
        ).send()


# ── File upload handler ───────────────────────────────────────────────────────

async def _handle_file_uploads(message: cl.Message) -> None:
    supported = {".pdf", ".png", ".jpg", ".jpeg", ".webp"}
    files_processed = 0
    total_text = 0
    total_images = 0

    for element in message.elements:
        if not isinstance(element, cl.File):
            continue
        suffix = Path(element.name).suffix.lower()
        if suffix not in supported:
            await cl.Message(
                content=f"⚠️ Format `{suffix}` non supporté. Utilisez PDF, PNG ou JPG."
            ).send()
            continue

        progress_msg = cl.Message(
            content=f"⏳ Ingestion de `{element.name}` en cours…"
        )
        await progress_msg.send()

        try:
            loop = asyncio.get_event_loop()
            counts = await loop.run_in_executor(
                None,
                ingest_file,
                element.path,
            )
            total_text += counts.get("text_chunks", 0)
            total_images += counts.get("image_chunks", 0)
            files_processed += 1

            await progress_msg.remove()
            await cl.Message(
                content=(
                    f"✅ `{element.name}` indexé avec succès !\n"
                    f"  - 📄 Chunks texte : **{counts['text_chunks']}**\n"
                    f"  - 🖼️ Chunks image : **{counts['image_chunks']}**"
                )
            ).send()
        except Exception as exc:
            await progress_msg.remove()
            logger.error("Ingestion failed", file=element.name, error=str(exc))
            await cl.Message(
                content=f"❌ Échec de l'ingestion de `{element.name}` : `{exc}`"
            ).send()

    if files_processed > 1:
        await cl.Message(
            content=(
                f"📊 **Récapitulatif** : {files_processed} fichiers indexés\n"
                f"  - 📄 Total chunks texte : **{total_text}**\n"
                f"  - 🖼️ Total chunks image : **{total_images}**"
            )
        ).send()
