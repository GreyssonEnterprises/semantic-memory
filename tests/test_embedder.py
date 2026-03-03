"""Tests for multi-provider embedding abstraction.

Covers ollama and openai backends with mocked HTTP calls.
No external services required.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


class MockResponse:
    """Reusable mock for requests.post responses."""

    def __init__(self, json_data, status_code=200):
        self._json = json_data
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")

    def json(self):
        return self._json


FAKE_768 = [0.1] * 768
FAKE_1536 = [0.2] * 1536


# ── Provider selection ──────────────────────────────────────────────────────


def test_default_provider_is_ollama(monkeypatch):
    """No EMBEDDING_PROVIDER env → ollama backend."""
    monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
    import importlib
    import src.pipeline.embedder as mod

    importlib.reload(mod)
    assert mod.EMBEDDING_PROVIDER == "ollama"
    importlib.reload(mod)


def test_openai_provider_from_env(monkeypatch):
    """EMBEDDING_PROVIDER=openai → openai backend."""
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    import importlib
    import src.pipeline.embedder as mod

    importlib.reload(mod)
    assert mod.EMBEDDING_PROVIDER == "openai"
    monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    importlib.reload(mod)


def test_invalid_provider_raises(monkeypatch):
    """EMBEDDING_PROVIDER=bogus → ValueError on load."""
    monkeypatch.setenv("EMBEDDING_PROVIDER", "bogus")
    import importlib
    import src.pipeline.embedder as mod

    with pytest.raises(ValueError, match="Unknown EMBEDDING_PROVIDER"):
        importlib.reload(mod)

    monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
    importlib.reload(mod)


def test_openai_without_api_key_raises(monkeypatch):
    """EMBEDDING_PROVIDER=openai without OPENAI_API_KEY → ValueError."""
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    import importlib
    import src.pipeline.embedder as mod

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        importlib.reload(mod)

    monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
    importlib.reload(mod)


# ── Ollama backend ──────────────────────────────────────────────────────────


def test_embed_text_ollama(monkeypatch):
    """embed_text calls ollama /api/embed and returns the embedding."""
    monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
    import importlib
    import src.pipeline.embedder as mod

    importlib.reload(mod)

    def mock_post(url, json, timeout=None):
        assert "/api/embed" in url
        assert json["model"] == mod.EMBED_MODEL
        return MockResponse({"embeddings": [FAKE_768]})

    monkeypatch.setattr(mod.requests, "post", mock_post)
    result = mod.embed_text("hello world")
    assert len(result) == 768
    assert result == FAKE_768


def test_embed_batch_ollama(monkeypatch):
    """embed_batch calls ollama with batched input."""
    monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
    import importlib
    import src.pipeline.embedder as mod

    importlib.reload(mod)

    call_count = 0

    def mock_post(url, json, timeout=None):
        nonlocal call_count
        call_count += 1
        n = len(json["input"])
        return MockResponse({"embeddings": [FAKE_768] * n})

    monkeypatch.setattr(mod.requests, "post", mock_post)
    result = mod.embed_batch(["text one", "text two", "text three"])
    assert len(result) == 3
    assert call_count == 1


# ── OpenAI backend ──────────────────────────────────────────────────────────


def test_embed_text_openai(monkeypatch):
    """embed_text with openai provider calls /v1/embeddings."""
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
    import importlib
    import src.pipeline.embedder as mod

    importlib.reload(mod)

    def mock_post(url, json, headers=None, timeout=None):
        assert "/v1/embeddings" in url
        assert "Bearer sk-test-key" in headers.get("Authorization", "")
        return MockResponse({
            "data": [{"embedding": FAKE_1536, "index": 0}],
            "model": "text-embedding-3-small",
            "usage": {"prompt_tokens": 5, "total_tokens": 5},
        })

    monkeypatch.setattr(mod.requests, "post", mock_post)
    result = mod.embed_text("hello world")
    assert len(result) == 1536
    assert result == FAKE_1536

    monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    importlib.reload(mod)


def test_embed_batch_openai(monkeypatch):
    """embed_batch with openai provider batches correctly."""
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
    import importlib
    import src.pipeline.embedder as mod

    importlib.reload(mod)

    def mock_post(url, json, headers=None, timeout=None):
        n = len(json["input"])
        return MockResponse({
            "data": [
                {"embedding": FAKE_1536, "index": i} for i in range(n)
            ],
            "model": "text-embedding-3-small",
            "usage": {"prompt_tokens": 10, "total_tokens": 10},
        })

    monkeypatch.setattr(mod.requests, "post", mock_post)
    result = mod.embed_batch(["one", "two", "three"])
    assert len(result) == 3
    assert all(len(e) == 1536 for e in result)

    monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    importlib.reload(mod)


def test_openai_dimensions_param(monkeypatch):
    """EMBEDDING_DIMENSIONS env is passed in the OpenAI request body."""
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "768")
    import importlib
    import src.pipeline.embedder as mod

    importlib.reload(mod)

    captured_body = {}

    def mock_post(url, json, headers=None, timeout=None):
        captured_body.update(json)
        return MockResponse({
            "data": [{"embedding": FAKE_768, "index": 0}],
            "model": "text-embedding-3-small",
            "usage": {"prompt_tokens": 5, "total_tokens": 5},
        })

    monkeypatch.setattr(mod.requests, "post", mock_post)
    mod.embed_text("test")
    assert captured_body.get("dimensions") == 768

    monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("EMBEDDING_DIMENSIONS", raising=False)
    importlib.reload(mod)


def test_openai_custom_base_url(monkeypatch):
    """OPENAI_BASE_URL overrides the API endpoint (for Azure, vLLM, etc.)."""
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://my-azure.openai.azure.com")
    import importlib
    import src.pipeline.embedder as mod

    importlib.reload(mod)

    captured_url = None

    def mock_post(url, json, headers=None, timeout=None):
        nonlocal captured_url
        captured_url = url
        return MockResponse({
            "data": [{"embedding": FAKE_768, "index": 0}],
            "model": "text-embedding-3-small",
            "usage": {"prompt_tokens": 5, "total_tokens": 5},
        })

    monkeypatch.setattr(mod.requests, "post", mock_post)
    mod.embed_text("test")
    assert captured_url.startswith("https://my-azure.openai.azure.com")

    monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    importlib.reload(mod)


# ── Truncation behavior (both providers) ────────────────────────────────────


def test_embed_batch_truncates_long_text(monkeypatch):
    """Very long texts are truncated before embedding."""
    monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
    import importlib
    import src.pipeline.embedder as mod

    importlib.reload(mod)

    captured_input = None

    def mock_post(url, json, timeout=None):
        nonlocal captured_input
        captured_input = json["input"]
        n = len(captured_input) if isinstance(captured_input, list) else 1
        return MockResponse({"embeddings": [FAKE_768] * n})

    monkeypatch.setattr(mod.requests, "post", mock_post)
    long_text = "x" * 10000
    mod.embed_batch([long_text])
    assert len(captured_input[0]) <= mod.MAX_CHARS


def test_embed_batch_replaces_empty_strings(monkeypatch):
    """Empty strings are replaced with 'empty' to avoid API errors."""
    monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
    import importlib
    import src.pipeline.embedder as mod

    importlib.reload(mod)

    captured_input = None

    def mock_post(url, json, timeout=None):
        nonlocal captured_input
        captured_input = json["input"]
        return MockResponse({"embeddings": [FAKE_768] * len(captured_input)})

    monkeypatch.setattr(mod.requests, "post", mock_post)
    mod.embed_batch(["", "   ", "valid"])
    assert captured_input[0] == "empty"
    assert captured_input[1] == "empty"
    assert captured_input[2] == "valid"


# ── OpenAI response ordering ───────────────────────────────────────────────


def test_openai_batch_respects_index_field(monkeypatch):
    """OpenAI may return embeddings out of order; we sort by index."""
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    import importlib
    import src.pipeline.embedder as mod

    importlib.reload(mod)

    embed_a = [1.0] * 768
    embed_b = [2.0] * 768

    def mock_post(url, json, headers=None, timeout=None):
        return MockResponse({
            "data": [
                {"embedding": embed_b, "index": 1},
                {"embedding": embed_a, "index": 0},
            ],
            "model": "text-embedding-3-small",
            "usage": {"prompt_tokens": 10, "total_tokens": 10},
        })

    monkeypatch.setattr(mod.requests, "post", mock_post)
    result = mod.embed_batch(["first", "second"])
    assert result[0] == embed_a
    assert result[1] == embed_b

    monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    importlib.reload(mod)
