"""Tests for LLM provider clients and factory."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from shared.llm.client import (
    CodexCLIClient,
    LiteLLMClient,
    get_llm_client,
)


class TestCodexCLIClientBuildsCommand:
    def test_codex_command(self):
        client = CodexCLIClient(provider="codex", model="gpt-5.3-codex")
        with patch("shared.llm.client.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout="Hello world", stderr="",
            )
            result = client.complete("sys", "user")
            assert result == "Hello world"
            cmd = mock_run.call_args[0][0]
            assert cmd[0] == "codex"
            assert "exec" in cmd
            assert "--full-auto" in cmd
            assert "-m" in cmd
            assert "gpt-5.3-codex" in cmd


class TestClaudeCLIClientBuildsCommand:
    def test_claude_cli_command(self):
        client = CodexCLIClient(provider="claude_cli", model="claude-sonnet-4-5-20250929")
        with patch("shared.llm.client.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout="Response text", stderr="",
            )
            result = client.complete("sys", "user")
            assert result == "Response text"
            cmd = mock_run.call_args[0][0]
            assert cmd[0] == "claude"
            assert "-p" in cmd
            assert "--output-format" in cmd


class TestStructuredJsonParsing:
    def test_direct_json(self):
        assert CodexCLIClient._extract_json('{"key": "value"}') == {"key": "value"}

    def test_markdown_fenced_json(self):
        text = '```json\n{"key": "value"}\n```'
        assert CodexCLIClient._extract_json(text) == {"key": "value"}

    def test_json_in_text(self):
        text = 'Here is the result: {"key": "value"} and more text'
        assert CodexCLIClient._extract_json(text) == {"key": "value"}

    def test_empty_on_invalid(self):
        assert CodexCLIClient._extract_json("no json here") == {}

    def test_complete_structured_uses_extract(self):
        client = CodexCLIClient(provider="codex")
        with patch("shared.llm.client.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout='```json\n{"claims": [{"x": 1}]}\n```',
                stderr="",
            )
            result = client.complete_structured("sys", "user", {"type": "object"})
            assert result == {"claims": [{"x": 1}]}


class TestFactoryFallbackNoApiKey:
    def test_falls_back_to_codex(self):
        settings = MagicMock()
        settings.llm_provider = "anthropic"
        settings.llm_model = "claude-sonnet-4-5-20250929"
        settings.anthropic_api_key = ""
        settings.codex_model = "gpt-5.3-codex"

        client = get_llm_client(settings)
        assert isinstance(client, CodexCLIClient)


class TestFactoryLiteLLMSelection:
    def test_selects_litellm(self):
        settings = MagicMock()
        settings.llm_provider = "litellm"
        settings.llm_model = "gpt-4"
        settings.anthropic_api_key = ""

        client = get_llm_client(settings)
        assert isinstance(client, LiteLLMClient)


class TestFactoryCodexSelection:
    def test_selects_codex(self):
        settings = MagicMock()
        settings.llm_provider = "codex"
        settings.llm_model = "gpt-5.3-codex"
        settings.codex_model = "gpt-5.3-codex"

        client = get_llm_client(settings)
        assert isinstance(client, CodexCLIClient)
