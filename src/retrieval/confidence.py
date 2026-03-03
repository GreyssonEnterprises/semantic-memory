"""Bayesian confidence scoring for semantic memories.

Implements confidence updates based on Bayes' theorem, applied when
new memories are stored that may confirm or contradict existing ones.

References:
    Bayes, T. (1763). Philosophical Transactions, 53, 370-418.
    Laplace, P.S. (1812). Theorie analytique des probabilites.
"""

import json
import logging
import os
import re

import requests

logger = logging.getLogger(__name__)

# Classification model config — lightweight local model via ollama
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
CLASSIFY_MODEL = os.environ.get("SEMANTIC_MEMORY_CLASSIFY_MODEL", "qwen2.5:3b")
CLASSIFY_TIMEOUT = int(os.environ.get("SEMANTIC_MEMORY_CLASSIFY_TIMEOUT", "30"))

CONFIDENCE_MIN = 0.1
CONFIDENCE_MAX = 2.0
CONFIDENCE_DEFAULT = 1.0

SIMILARITY_THRESHOLD = 0.85  # Minimum cosine sim to trigger contradiction check

LIKELIHOOD_RATIOS = {
    "strong_confirm": 1.15,
    "weak_confirm": 1.05,
    "neutral": 1.00,
    "weak_contradict": 0.85,
    "strong_contradict": 0.60,
}


def bayesian_update(
    prior_confidence: float,
    evidence_type: str,
) -> float:
    """Update a memory's confidence based on new evidence.

    confidence_new = clamp(prior * likelihood_ratio, CONFIDENCE_MIN, CONFIDENCE_MAX)

    Raises ValueError if evidence_type not in LIKELIHOOD_RATIOS.
    """
    if evidence_type not in LIKELIHOOD_RATIOS:
        raise ValueError(
            f"Unknown evidence_type '{evidence_type}'. "
            f"Must be one of {list(LIKELIHOOD_RATIOS.keys())}"
        )
    ratio = LIKELIHOOD_RATIOS[evidence_type]
    new_confidence = prior_confidence * ratio
    return max(CONFIDENCE_MIN, min(CONFIDENCE_MAX, new_confidence))


def detect_contradictions(
    new_embedding: list[float],
    new_text: str,
    store: "VectorStore",
    agent: str,
    top_k: int = 10,
) -> list[dict]:
    """Find existing memories that may be contradicted by new content.

    1. Search existing memories with new_embedding (top_k results)
    2. For each result with cosine similarity > SIMILARITY_THRESHOLD (0.85):
       - Chroma returns distances. For cosine space: similarity = 1 - distance
       - Call classify_relationship() to determine agree/disagree
       - Compute new confidence via bayesian_update()
    3. Return list of dicts with: memory_id, similarity, evidence_type,
       old_confidence, new_confidence, collection_type ("segment" or "summary")

    Searches BOTH segments and summaries collections.
    """
    contradictions = []

    for collection_type, search_fn in [
        ("segment", store.search_segments),
        ("summary", store.search_summaries),
    ]:
        results = search_fn(new_embedding, n_results=top_k)
        if not results or not results.get("ids") or not results["ids"][0]:
            continue

        ids = results["ids"][0]
        distances = results["distances"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]

        for doc_id, dist, doc, meta in zip(ids, distances, documents, metadatas):
            # Cosine space: similarity = 1 - distance
            similarity = 1.0 - dist

            if similarity < SIMILARITY_THRESHOLD:
                continue

            evidence_type = classify_relationship(new_text, doc)
            old_confidence = float(meta.get("confidence", CONFIDENCE_DEFAULT))
            new_confidence = bayesian_update(old_confidence, evidence_type)

            contradictions.append({
                "memory_id": doc_id,
                "similarity": similarity,
                "evidence_type": evidence_type,
                "old_confidence": old_confidence,
                "new_confidence": new_confidence,
                "collection_type": collection_type,
            })

    return contradictions


def classify_relationship(
    statement_a: str,
    statement_b: str,
) -> str:
    """Classify whether two statements agree, disagree, or are unrelated.

    Uses a lightweight local LLM (Qwen2.5-1.5B via ollama) for classification.
    Truncates statements to 200 chars each to keep the prompt small (~100 tokens).
    Falls back to "neutral" (no confidence change) on any error — never degrades
    confidence due to classification failures.

    The model is configurable via environment variables:
        OLLAMA_URL: ollama endpoint (default: http://localhost:11434)
        SEMANTIC_MEMORY_CLASSIFY_MODEL: model name (default: qwen2.5:1.5b)
        SEMANTIC_MEMORY_CLASSIFY_TIMEOUT: request timeout in seconds (default: 30)

    Args:
        statement_a: Text of the existing memory.
        statement_b: Text of the new memory.

    Returns:
        One of: "strong_confirm", "weak_confirm", "neutral",
        "weak_contradict", "strong_contradict".
    """
    # Truncate to keep prompt small
    a = statement_a[:200].strip()
    b = statement_b[:200].strip()

    if not a or not b:
        return "neutral"

    prompt = (
        "You are a factual consistency classifier. Compare these two statements "
        "and respond with EXACTLY one of these labels:\n"
        "- strong_confirm (they say the same thing)\n"
        "- weak_confirm (they are compatible/consistent)\n"
        "- neutral (they discuss unrelated topics)\n"
        "- weak_contradict (they are somewhat inconsistent)\n"
        "- strong_contradict (they directly contradict each other)\n\n"
        f"Statement A: {a}\n"
        f"Statement B: {b}\n\n"
        "Label:"
    )

    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": CLASSIFY_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    "num_predict": 20,  # Only need a few tokens
                },
            },
            timeout=CLASSIFY_TIMEOUT,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "").strip().lower()

        # Extract the classification label from the response
        # The model might include extra text, so search for known labels
        for label in LIKELIHOOD_RATIOS:
            if label in raw:
                return label

        # If no exact match, try partial matching
        if "contradict" in raw:
            return "strong_contradict" if "strong" in raw else "weak_contradict"
        if "confirm" in raw or "agree" in raw or "same" in raw:
            return "strong_confirm" if "strong" in raw else "weak_confirm"
        if "unrelated" in raw or "neutral" in raw or "different" in raw:
            return "neutral"

        logger.warning(
            "classify_relationship: could not parse LLM response: %r, defaulting to neutral",
            raw,
        )
        return "neutral"

    except requests.exceptions.ConnectionError:
        logger.warning(
            "classify_relationship: ollama not reachable at %s, defaulting to neutral",
            OLLAMA_URL,
        )
        return "neutral"
    except Exception as e:
        logger.warning(
            "classify_relationship: LLM classification failed (%s), defaulting to neutral",
            e,
        )
        return "neutral"
