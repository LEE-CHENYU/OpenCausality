"""
LLM Client Abstraction Layer.

Provides a configurable backend for LLM completions.
Default: Anthropic SDK. Fallback: LiteLLM for multi-provider support.

Usage:
    client = get_llm_client()
    response = client.complete("You are a helpful assistant.", "Hello")
    structured = client.complete_structured("System prompt", "User input", schema)
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class LLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    def complete(self, system: str, user: str, max_tokens: int = 1024) -> str:
        """Generate a text completion."""
        ...

    @abstractmethod
    def complete_structured(self, system: str, user: str, schema: dict) -> dict:
        """Generate a structured (JSON) completion matching the given schema."""
        ...


class AnthropicClient(LLMClient):
    """Uses the anthropic Python SDK. Requires ANTHROPIC_API_KEY."""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-5-20250929"):
        import anthropic

        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> str:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return response.content[0].text

    def complete_structured(self, system: str, user: str, schema: dict) -> dict:
        """Use tool_use to extract structured JSON matching the schema."""
        tool_name = "structured_output"
        response = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": user}],
            tools=[
                {
                    "name": tool_name,
                    "description": "Return structured data matching the schema.",
                    "input_schema": schema,
                }
            ],
            tool_choice={"type": "tool", "name": tool_name},
        )
        # Extract the tool use block
        for block in response.content:
            if block.type == "tool_use" and block.name == tool_name:
                return block.input
        # Fallback: try to parse text response
        for block in response.content:
            if block.type == "text":
                try:
                    return json.loads(block.text)
                except json.JSONDecodeError:
                    pass
        return {}


class LiteLLMClient(LLMClient):
    """Uses litellm for multi-provider support."""

    def __init__(self, model: str = "claude-sonnet-4-5-20250929"):
        self._model = model

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> str:
        import litellm

        response = litellm.completion(
            model=self._model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content

    def complete_structured(self, system: str, user: str, schema: dict) -> dict:
        import litellm

        response = litellm.completion(
            model=self._model,
            max_tokens=4096,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "structured_output",
                        "description": "Return structured data matching the schema.",
                        "parameters": schema,
                    },
                }
            ],
            tool_choice={"type": "function", "function": {"name": "structured_output"}},
        )
        msg = response.choices[0].message
        if msg.tool_calls:
            return json.loads(msg.tool_calls[0].function.arguments)
        if msg.content:
            try:
                return json.loads(msg.content)
            except json.JSONDecodeError:
                pass
        return {}


class CodexCLIClient(LLMClient):
    """Shells out to ``codex exec`` or ``claude`` CLI for completions.

    No API key required — uses locally installed CLI tools.
    """

    def __init__(self, provider: str = "codex", model: str = "gpt-5.3-codex"):
        self._provider = provider
        self._model = model

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> str:
        prompt_text = f"System: {system}\n\nUser: {user}"

        if self._provider == "codex":
            cmd = ["codex", "exec", "--full-auto", "-m", self._model, prompt_text]
        elif self._provider == "claude_cli":
            cmd = ["claude", "-p", prompt_text, "--output-format", "text"]
        else:
            raise ValueError(f"Unknown CLI provider: {self._provider}")

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            logger.warning(f"CLI command failed (rc={result.returncode}): {result.stderr[:200]}")
        return result.stdout.strip()

    def complete_structured(self, system: str, user: str, schema: dict) -> dict:
        schema_str = json.dumps(schema, indent=2)
        augmented_user = (
            f"{user}\n\nReturn your answer as a JSON object matching this schema:\n{schema_str}"
        )
        raw = self.complete(system, augmented_user)
        return self._extract_json(raw)

    @staticmethod
    def _extract_json(text: str) -> dict:
        """Extract JSON from text, handling markdown fences."""
        # Try direct parse first
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            pass

        # Try extracting from markdown code fences
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Try finding first { ... } block
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        return {}


def get_llm_client(settings: Any | None = None) -> LLMClient:
    """
    Factory: return LLM client based on settings.

    Provider priority:
    - "anthropic": Anthropic SDK (requires ANTHROPIC_API_KEY)
    - "litellm": LiteLLM multi-provider (works with OPENAI_API_KEY etc.)
    - "codex": Codex CLI (no API key needed)
    - "claude_cli": Claude CLI (no API key needed)

    If provider is "anthropic" but no API key is set, auto-falls back to codex.
    """
    if settings is None:
        from config.settings import get_settings
        settings = get_settings()

    provider = getattr(settings, "llm_provider", "anthropic")
    model = getattr(settings, "llm_model", "claude-sonnet-4-5-20250929")
    api_key = getattr(settings, "anthropic_api_key", "")

    if provider in ("codex", "claude_cli"):
        codex_model = getattr(settings, "codex_model", model)
        logger.info(f"Using {provider} CLI client with model={codex_model}")
        return CodexCLIClient(provider=provider, model=codex_model)

    if provider == "litellm":
        logger.info(f"Using LiteLLM client with model={model}")
        return LiteLLMClient(model=model)

    # Default: anthropic
    if not api_key:
        logger.warning(
            "ANTHROPIC_API_KEY not set. Falling back to codex CLI provider."
        )
        codex_model = getattr(settings, "codex_model", "gpt-5.3-codex")
        return CodexCLIClient(provider="codex", model=codex_model)

    logger.info(f"Using Anthropic client with model={model}")
    return AnthropicClient(api_key=api_key, model=model)
