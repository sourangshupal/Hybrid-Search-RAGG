"""
Google Gemini integration for LLM generation and embeddings.

Supports:
- Gemini 2.5 Flash, Gemini 2.0 Flash for generation
- text-embedding-004 for embeddings
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

logger = logging.getLogger("hybridrag.gemini")
logger.setLevel(logging.INFO)


@dataclass
class GeminiLLM:
    """
    Google Gemini LLM wrapper.

    Provides async generation for HybridRAG.
    """

    api_key: str
    model: str = "gemini-2.5-flash"
    max_tokens: int = 4096

    def __post_init__(self) -> None:
        from google import genai
        self._client = genai.Client(api_key=self.api_key)

    async def generate_async(
        self,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs,
    ) -> str:
        """
        Generate response from Gemini.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            **kwargs: Additional parameters

        Returns:
            Generated text response
        """
        logger.info(f"[LLM] Starting generation with model={self.model}")
        logger.debug(f"[LLM] Prompt length: {len(prompt)} chars, system_prompt: {'yes' if system_prompt else 'no'}")

        # Combine system prompt with user prompt if provided
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
            logger.debug(f"[LLM] Combined prompt length: {len(full_prompt)} chars")

        try:
            response = await self._client.aio.models.generate_content(
                model=self.model,
                contents=full_prompt,
            )

            result = response.text
            logger.info(f"[LLM] Generation complete: {len(result)} chars response")
            return result
        except Exception as e:
            logger.error(f"[LLM] Generation error: {e}")
            raise


@dataclass
class GeminiEmbedder:
    """
    Google Gemini embedding wrapper.

    Uses text-embedding-004 model (768 dimensions).
    """

    api_key: str
    model: str = "text-embedding-004"
    batch_size: int = 100

    def __post_init__(self) -> None:
        import google.generativeai as genai
        genai.configure(api_key=self.api_key)

    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension (768 for text-embedding-004)."""
        return 768

    async def embed_async(
        self,
        texts: Sequence[str],
    ) -> np.ndarray:
        """
        Embed texts asynchronously.

        Args:
            texts: List of texts to embed

        Returns:
            Numpy array of embeddings
        """
        import google.generativeai as genai
        import asyncio

        if not texts:
            return np.array([], dtype=np.float32)

        all_embeddings: list[list[float]] = []
        loop = asyncio.get_event_loop()

        for i in range(0, len(texts), self.batch_size):
            batch = list(texts[i : i + self.batch_size])

            # Gemini embed_content is sync, wrap for async
            result = await loop.run_in_executor(
                None,
                lambda b=batch: genai.embed_content(
                    model=f"models/{self.model}",
                    content=b,
                    task_type="retrieval_document",
                ),
            )

            # Handle single vs batch response
            if isinstance(result["embedding"][0], list):
                all_embeddings.extend(result["embedding"])
            else:
                all_embeddings.append(result["embedding"])

        return np.array(all_embeddings, dtype=np.float32)


def create_gemini_llm_func(
    api_key: str,
    model: str = "gemini-2.5-flash",
    max_tokens: int = 4096,
) -> Callable[..., str]:
    """
    Create LLM function using Gemini.

    Args:
        api_key: Google AI API key
        model: Model name (gemini-2.5-flash, gemini-2.0-flash)
        max_tokens: Maximum tokens for generation

    Returns:
        Async LLM function
    """
    llm = GeminiLLM(api_key=api_key, model=model, max_tokens=max_tokens)

    async def llm_func(
        prompt: str,
        system_prompt: str | None = None,
        history_messages: list | None = None,
        **kwargs,
    ) -> str:
        return await llm.generate_async(prompt, system_prompt, **kwargs)

    return llm_func


def create_gemini_embedding_func(
    api_key: str,
    model: str = "text-embedding-004",
    batch_size: int = 100,
) -> Callable[[list[str]], np.ndarray]:
    """
    Create embedding function using Gemini.

    Args:
        api_key: Google AI API key
        model: Embedding model name
        batch_size: Batch size for API calls

    Returns:
        Async embedding function
    """
    embedder = GeminiEmbedder(api_key=api_key, model=model, batch_size=batch_size)

    async def embed_func(texts: list[str]) -> np.ndarray:
        return await embedder.embed_async(texts)

    return embed_func
