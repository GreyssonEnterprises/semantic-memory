"""Embedding client — supports ollama and OpenAI-compatible APIs.

Provider selection via EMBEDDING_PROVIDER env var ("ollama" or "openai").
Defaults to "ollama" for backward compatibility.

Environment variables:
    EMBEDDING_PROVIDER   - "ollama" (default) or "openai"
    OLLAMA_URL           - Ollama endpoint (default: http://localhost:11434)
    EMBED_MODEL          - Model name. Defaults: nomic-embed-text (ollama),
                           text-embedding-3-small (openai)
    OPENAI_API_KEY       - Required when provider is "openai"
    OPENAI_BASE_URL      - Override API base (default: https://api.openai.com)
                           Useful for Azure OpenAI, vLLM, LiteLLM, etc.
    EMBEDDING_DIMENSIONS - Optional output dimensions (openai only, for models
                           that support the dimensions parameter like
                           text-embedding-3-small). Set to 768 to match
                           nomic-embed-text stores.
"""

import logging
import os
import math
from datetime import datetime

import requests

logger = logging.getLogger(__name__)

# ── Provider configuration ──────────────────────────────────────────────────

EMBEDDING_PROVIDER = os.environ.get("EMBEDDING_PROVIDER", "ollama").lower()

_VALID_PROVIDERS = ("ollama", "openai")
if EMBEDDING_PROVIDER not in _VALID_PROVIDERS:
    raise ValueError(
        f"Unknown EMBEDDING_PROVIDER '{EMBEDDING_PROVIDER}'. "
        f"Must be one of {_VALID_PROVIDERS}"
    )

# Ollama settings
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")

# OpenAI settings
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com").rstrip("/")
EMBEDDING_DIMENSIONS = os.environ.get("EMBEDDING_DIMENSIONS")

if EMBEDDING_PROVIDER == "openai" and not OPENAI_API_KEY:
    raise ValueError(
        "OPENAI_API_KEY is required when EMBEDDING_PROVIDER=openai"
    )

# Model defaults per provider
_DEFAULT_MODELS = {
    "ollama": "nomic-embed-text",
    "openai": "text-embedding-3-small",
}
EMBED_MODEL = os.environ.get("EMBED_MODEL", _DEFAULT_MODELS[EMBEDDING_PROVIDER])

MAX_CHARS = 6000  # ~1500 tokens of headroom for most models

# ── Ollama backend ──────────────────────────────────────────────────────────


def _embed_text_ollama(text: str) -> list[float]:
    resp = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": EMBED_MODEL, "input": text},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["embeddings"][0]


def _embed_batch_ollama(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = requests.post(
            f"{OLLAMA_URL}/api/embed",
            json={"model": EMBED_MODEL, "input": batch},
            timeout=120,
        )
        resp.raise_for_status()
        all_embeddings.extend(resp.json()["embeddings"])
    return all_embeddings


# ── OpenAI backend ──────────────────────────────────────────────────────────


def _openai_headers() -> dict:
    return {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }


def _openai_body(input_data: str | list[str]) -> dict:
    body: dict = {"model": EMBED_MODEL, "input": input_data}
    if EMBEDDING_DIMENSIONS:
        body["dimensions"] = int(EMBEDDING_DIMENSIONS)
    return body


def _embed_text_openai(text: str) -> list[float]:
    resp = requests.post(
        f"{OPENAI_BASE_URL}/v1/embeddings",
        json=_openai_body(text),
        headers=_openai_headers(),
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["data"][0]["embedding"]


def _embed_batch_openai(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = requests.post(
            f"{OPENAI_BASE_URL}/v1/embeddings",
            json=_openai_body(batch),
            headers=_openai_headers(),
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()["data"]
        sorted_data = sorted(data, key=lambda d: d["index"])
        all_embeddings.extend(d["embedding"] for d in sorted_data)
    return all_embeddings


# ── Public API (provider-agnostic) ──────────────────────────────────────────

_TEXT_FN = _embed_text_ollama if EMBEDDING_PROVIDER == "ollama" else _embed_text_openai
_BATCH_FN = _embed_batch_ollama if EMBEDDING_PROVIDER == "ollama" else _embed_batch_openai


def embed_text(text: str) -> list[float]:
    """Embed a single text string. Provider selected by EMBEDDING_PROVIDER env."""
    return _TEXT_FN(text)


def embed_batch(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """Embed multiple texts with truncation and empty-string handling."""
    truncated = [t[:MAX_CHARS] if len(t) > MAX_CHARS else t for t in texts]
    truncated = [t if t.strip() else "empty" for t in truncated]
    return _BATCH_FN(truncated, batch_size=batch_size)


def time_weight(age_days: float) -> float:
    """Logarithmic time decay: recent sessions rank higher, old ones never vanish.

    .. deprecated:: Phase 1 (Cognitive Primitives)
        Replaced by actr_activation() which accounts for both recency AND frequency.
        Retained for backward compatibility. Will be removed in a future version.

    score = 1 / (1 + log(age_days + 1))

    Examples:
        0 days  → 1.0
        1 day   → 0.59
        7 days  → 0.32
        30 days → 0.22
        180 days → 0.16
        365 days → 0.14
    """
    return 1.0 / (1.0 + math.log(age_days + 1))


def actr_activation(
    n: int,
    age_days: float,
    access_timestamps: list[str] | None = None,
    decay: float = 0.5,
    spread_bonus_weight: float = 0.2,
) -> float:
    """ACT-R base-level activation: accounts for both recency AND frequency.

    B = ln(n + 1) - decay * ln(age_days / (n + 1))
    activation_weight = sigmoid(B) * spread_bonus

    Args:
        n: Number of accesses (clamped to min 1).
        age_days: Days since creation (clamped to min 0.01).
        access_timestamps: ISO timestamps of recent accesses (max 20 used).
        decay: Decay parameter (default 0.5).
        spread_bonus_weight: Weight for spaced-repetition bonus (default 0.2).

    Returns:
        Activation weight in (0, 1+]. Always positive.
    """
    import statistics

    # Clamp inputs
    n = max(1, n)
    age_days = max(0.01, age_days)

    # Base-level activation
    B = math.log(n + 1) - decay * math.log(age_days / (n + 1))

    # Sigmoid
    sig = 1.0 / (1.0 + math.exp(-B))

    # Spaced repetition bonus
    spread_bonus = 1.0
    if access_timestamps and len(access_timestamps) >= 2:
        # Use most recent 20
        ts_list = access_timestamps[-20:]
        try:
            parsed = sorted(
                datetime.fromisoformat(t) for t in ts_list
            )
            intervals = [
                (parsed[i + 1] - parsed[i]).total_seconds()
                for i in range(len(parsed) - 1)
            ]
            mean_interval = statistics.mean(intervals)
            if mean_interval > 0:
                temporal_spread = statistics.stdev(intervals) if len(intervals) > 1 else 0.0
                ratio = min(1.0, temporal_spread / mean_interval)
                spread_bonus = 1.0 + spread_bonus_weight * (1.0 - ratio)
        except (ValueError, TypeError):
            spread_bonus = 1.0

    return sig * spread_bonus
