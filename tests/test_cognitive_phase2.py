"""Tests for Phase 2: Bayesian Confidence Scoring.

Unit tests only — no ollama or external services required.
"""

import json
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.confidence import (
    bayesian_update,
    detect_contradictions,
    classify_relationship,
    CONFIDENCE_MIN,
    CONFIDENCE_MAX,
    LIKELIHOOD_RATIOS,
)


# ── bayesian_update tests ──────────────────────────────────────────────────


def test_bayesian_strong_confirm():
    """prior=1.0, evidence="strong_confirm" → 1.15"""
    assert bayesian_update(1.0, "strong_confirm") == pytest.approx(1.15)


def test_bayesian_strong_contradict():
    """prior=1.0, evidence="strong_contradict" → 0.60"""
    assert bayesian_update(1.0, "strong_contradict") == pytest.approx(0.60)


def test_bayesian_floor():
    """prior=0.1, evidence="strong_contradict" → 0.1 (floor)"""
    result = bayesian_update(0.1, "strong_contradict")
    assert result == pytest.approx(CONFIDENCE_MIN)


def test_bayesian_ceiling():
    """prior=1.8, evidence="strong_confirm" → 2.0 (ceiling)"""
    result = bayesian_update(1.8, "strong_confirm")
    assert result == pytest.approx(CONFIDENCE_MAX)


def test_bayesian_invalid_evidence():
    """Invalid evidence_type → ValueError"""
    with pytest.raises(ValueError):
        bayesian_update(1.0, "invalid_type")


def test_bayesian_neutral_no_change():
    """Neutral evidence doesn't change confidence"""
    assert bayesian_update(0.7, "neutral") == pytest.approx(0.7)


def test_bayesian_weak_confirm():
    """prior=1.0, weak_confirm → 1.05"""
    assert bayesian_update(1.0, "weak_confirm") == pytest.approx(1.05)


def test_bayesian_weak_contradict():
    """prior=1.0, weak_contradict → 0.85"""
    assert bayesian_update(1.0, "weak_contradict") == pytest.approx(0.85)


def test_bayesian_chained_updates():
    """Multiple contradictions erode confidence but hit floor"""
    c = 1.0
    for _ in range(10):
        c = bayesian_update(c, "strong_contradict")
    assert c == pytest.approx(CONFIDENCE_MIN)


# ── classify_relationship tests ────────────────────────────────────────────


def test_classify_relationship_fallback_on_connection_error(monkeypatch):
    """When ollama is unreachable, classify returns 'neutral' (safe fallback)"""
    # Point to an unreachable URL so the real LLM call fails
    monkeypatch.setenv("OLLAMA_URL", "http://localhost:99999")
    # Need to reimport to pick up the new env var
    import importlib
    import src.retrieval.confidence as conf_mod
    importlib.reload(conf_mod)

    result = conf_mod.classify_relationship("Meeting at 3pm", "Meeting moved to 4pm")
    assert result in conf_mod.LIKELIHOOD_RATIOS

    # Restore
    monkeypatch.delenv("OLLAMA_URL", raising=False)
    importlib.reload(conf_mod)


def test_classify_relationship_empty_input():
    """Empty statements return neutral"""
    result = classify_relationship("", "something")
    assert result == "neutral"
    result = classify_relationship("something", "")
    assert result == "neutral"


def test_classify_relationship_returns_valid_label(monkeypatch):
    """Mock ollama to verify response parsing"""
    import src.retrieval.confidence as conf_mod

    class MockResponse:
        status_code = 200
        def raise_for_status(self): pass
        def json(self): return {"response": " strong_contradict\n"}

    def mock_post(*args, **kwargs):
        return MockResponse()

    monkeypatch.setattr(conf_mod.requests, "post", mock_post)
    result = conf_mod.classify_relationship("Meeting at 3pm", "Meeting at 4pm")
    assert result == "strong_contradict"


def test_classify_relationship_parses_fuzzy_response(monkeypatch):
    """Model might say 'The statements contradict each other' instead of exact label"""
    import src.retrieval.confidence as conf_mod

    class MockResponse:
        status_code = 200
        def raise_for_status(self): pass
        def json(self): return {"response": "These statements contradict each other."}

    def mock_post(*args, **kwargs):
        return MockResponse()

    monkeypatch.setattr(conf_mod.requests, "post", mock_post)
    result = conf_mod.classify_relationship("Meeting at 3pm", "Meeting at 4pm")
    assert result in ("weak_contradict", "strong_contradict")


def test_classify_relationship_unparseable_defaults_neutral(monkeypatch):
    """Completely unparseable LLM output → neutral"""
    import src.retrieval.confidence as conf_mod

    class MockResponse:
        status_code = 200
        def raise_for_status(self): pass
        def json(self): return {"response": "I like turtles"}

    def mock_post(*args, **kwargs):
        return MockResponse()

    monkeypatch.setattr(conf_mod.requests, "post", mock_post)
    result = conf_mod.classify_relationship("Meeting at 3pm", "Meeting at 4pm")
    assert result == "neutral"


# ── detect_contradictions tests ────────────────────────────────────────────


def test_detect_contradictions_below_threshold(tmp_path):
    """Unrelated statements with low similarity → no contradictions detected"""
    from src.pipeline.store import VectorStore

    store = VectorStore("bob", base_path=tmp_path)
    # Add a segment with a known embedding (768-dim)
    embedding = [0.0] * 768
    embedding[0] = 1.0  # Simple directional embedding
    store.add_segment(
        segment_id="test_seg_001",
        text="The meeting is at 3pm in conference room B",
        embedding=embedding,
        session_id="session_001",
        date="2026-03-01",
    )

    # Query with orthogonal embedding
    query_embedding = [0.0] * 768
    query_embedding[1] = 1.0  # Orthogonal direction

    results = detect_contradictions(
        new_embedding=query_embedding,
        new_text="I bought groceries at the store",
        store=store,
        agent="bob",
    )
    assert len(results) == 0


# ── compute_final_score with confidence ────────────────────────────────────


def test_compute_final_score_with_confidence():
    """Confidence multiplier affects final score"""
    from src.retrieval.recall import compute_final_score

    score_normal = compute_final_score(0.5, "2026-03-01", confidence=1.0)
    score_low_conf = compute_final_score(0.5, "2026-03-01", confidence=0.5)
    assert score_low_conf < score_normal
    assert score_low_conf == pytest.approx(score_normal * 0.5)


# ── store metadata confidence ─────────────────────────────────────────────


def test_store_segment_has_confidence(tmp_path):
    """add_segment includes confidence field"""
    from src.pipeline.store import VectorStore

    store = VectorStore("bob", base_path=tmp_path)
    embedding = [0.0] * 768
    store.add_segment(
        segment_id="test_001",
        text="test",
        embedding=embedding,
        session_id="sess_001",
        date="2026-03-01",
    )
    result = store.segments.get(ids=["test_001"], include=["metadatas"])
    meta = result["metadatas"][0]
    assert meta["confidence"] == 1.0


def test_store_summary_has_confidence(tmp_path):
    """add_session_summary includes confidence field"""
    from src.pipeline.store import VectorStore

    store = VectorStore("bob", base_path=tmp_path)
    embedding = [0.0] * 768
    store.add_session_summary(
        session_id="sess_conf_001",
        summary_text="summary test",
        embedding=embedding,
        date="2026-03-01",
    )
    result = store.summaries.get(ids=["sess_conf_001"], include=["metadatas"])
    meta = result["metadatas"][0]
    assert meta["confidence"] == 1.0
