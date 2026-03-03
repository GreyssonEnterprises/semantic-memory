"""Embedding client — wraps local ollama nomic-embed-text."""

import os
import requests
from typing import Union
import math

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = "nomic-embed-text"


def embed_text(text: str) -> list[float]:
    """Embed a single text string via ollama. Returns 768-dim vector."""
    resp = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": EMBED_MODEL, "input": text},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    # ollama /api/embed returns {"embeddings": [[...]]}
    return data["embeddings"][0]


def embed_batch(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """Embed multiple texts. Batches to avoid overloading ollama."""
    all_embeddings = []
    # Truncate very long texts (nomic-embed-text has ~8192 token context, ~4 chars/token)
    max_chars = 6000  # Conservative: ~1500 tokens of headroom
    truncated = [t[:max_chars] if len(t) > max_chars else t for t in texts]
    # Filter empty strings
    truncated = [t if t.strip() else "empty" for t in truncated]
    
    for i in range(0, len(truncated), batch_size):
        batch = truncated[i : i + batch_size]
        resp = requests.post(
            f"{OLLAMA_URL}/api/embed",
            json={"model": EMBED_MODEL, "input": batch},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        all_embeddings.extend(data["embeddings"])
    return all_embeddings


def time_weight(age_days: float) -> float:
    """Logarithmic time decay: recent sessions rank higher, old ones never vanish.
    
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
