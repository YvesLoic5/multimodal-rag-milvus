"""LangChain-based answer generation.

Default: ChatOpenAI (GPT-4o) with streaming.
Fallback: LLaVA via Ollama when OPENAI_API_KEY is absent.

The chain receives retrieved + reranked context and the conversation history,
then produces a grounded, source-cited answer.
"""

from __future__ import annotations

from typing import Any, AsyncGenerator

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from app.utils.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

_SYSTEM_PROMPT = """\
You are a helpful AI assistant with access to a knowledge base. Your task is to answer
questions based strictly on the provided context.

Rules:
- Always cite your sources using [Source: <filename>, Page: <page>] notation.
- If the context does not contain enough information, say so clearly.
- When referencing images, describe what you can infer from the image caption.
- Keep answers concise but complete.
- Do NOT invent information beyond what is in the context.
"""


def _build_context_block(hits: list[dict[str, Any]]) -> str:
    """Format retrieved hits into a readable context block for the LLM."""
    lines: list[str] = ["=== RETRIEVED CONTEXT ===\n"]
    for i, hit in enumerate(hits, start=1):
        modality = hit.get("modality", "text")
        source = hit.get("metadata", {}).get("source", "unknown")
        page = hit.get("metadata", {}).get("page", "?")
        score = hit.get("rerank_score", hit.get("score", 0.0))
        content = hit.get("content_text", "")

        lines.append(
            f"[{i}] [{modality.upper()}] Score={score:.3f} | "
            f"Source: {source}, Page: {page}\n{content}\n"
        )
    return "\n".join(lines)


def _build_messages(
    question: str,
    context_block: str,
    history: list[dict[str, str]],
) -> list[BaseMessage]:
    messages: list[BaseMessage] = [SystemMessage(content=_SYSTEM_PROMPT)]

    # Last 5 exchanges from conversation history
    for turn in history[-5:]:
        if turn.get("role") == "user":
            messages.append(HumanMessage(content=turn["content"]))
        elif turn.get("role") == "assistant":
            messages.append(AIMessage(content=turn["content"]))

    messages.append(
        HumanMessage(content=f"{context_block}\n\n=== QUESTION ===\n{question}")
    )
    return messages


def _get_llm(streaming: bool = True) -> Any:
    """Return the appropriate LLM based on configuration."""
    settings = get_settings()
    if settings.use_openai:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model="gpt-4o",
            temperature=0.1,
            streaming=streaming,
            api_key=settings.openai_api_key,
        )
    else:
        from langchain_community.llms import Ollama

        logger.warning("OpenAI key not found — using Ollama fallback", model=settings.ollama_model)
        return Ollama(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            temperature=0.1,
        )


def generate_answer(
    question: str,
    hits: list[dict[str, Any]],
    history: list[dict[str, str]] | None = None,
) -> str:
    """Generate a non-streaming answer (used by evaluation scripts)."""
    history = history or []
    context_block = _build_context_block(hits)
    messages = _build_messages(question, context_block, history)
    llm = _get_llm(streaming=False)
    response = llm.invoke(messages)
    return response.content if hasattr(response, "content") else str(response)


async def stream_answer(
    question: str,
    hits: list[dict[str, Any]],
    history: list[dict[str, str]] | None = None,
) -> AsyncGenerator[str, None]:
    """Stream answer tokens for Chainlit."""
    history = history or []
    context_block = _build_context_block(hits)
    messages = _build_messages(question, context_block, history)
    llm = _get_llm(streaming=True)

    async for chunk in llm.astream(messages):
        token = chunk.content if hasattr(chunk, "content") else str(chunk)
        if token:
            yield token
