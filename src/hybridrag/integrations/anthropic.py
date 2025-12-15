"""
Anthropic Claude integration for LLM generation.

Provides async LLM function for HybridRAG.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from anthropic import AsyncAnthropic

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass
class ClaudeLLM:
    """
    Anthropic Claude LLM wrapper.

    Provides async generation for HybridRAG.
    """

    api_key: str
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 4096

    def __post_init__(self) -> None:
        self._client = AsyncAnthropic(api_key=self.api_key)

    async def generate_async(
        self,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs,
    ) -> str:
        """
        Generate response from Claude.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            **kwargs: Additional parameters

        Returns:
            Generated text response
        """
        messages = [{"role": "user", "content": prompt}]

        response = await self._client.messages.create(
            model=self.model,
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            system=system_prompt or "",
            messages=messages,
        )

        return response.content[0].text


def create_llm_func(
    api_key: str,
    model: str = "claude-sonnet-4-20250514",
    max_tokens: int = 4096,
) -> Callable[..., str]:
    """
    Create LLM function for HybridRAG.

    Args:
        api_key: Anthropic API key
        model: Claude model name
        max_tokens: Maximum tokens for generation

    Returns:
        Async LLM function
    """
    llm = ClaudeLLM(api_key=api_key, model=model, max_tokens=max_tokens)

    async def llm_func(
        prompt: str,
        system_prompt: str | None = None,
        history_messages: list | None = None,
        **kwargs,
    ) -> str:
        return await llm.generate_async(prompt, system_prompt, **kwargs)

    return llm_func
