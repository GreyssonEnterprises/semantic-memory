"""Tests for Phase 5: Push triggers and backfill script."""

import json
import pytest

from src.retrieval.push import should_push
from src.pipeline.store import VectorStore
from scripts.backfill_cognitive_fields import backfill_collection, COGNITIVE_DEFAULTS


# ── Push trigger tests ──────────────────────────────────────────────


def test_should_push_high_activation():
    """activation=0.95, confidence=0.9, 48h since last push → True"""
    assert should_push(0.95, 0.9, 48.0, False) is True


def test_should_push_low_activation():
    """activation=0.3, confidence=0.9 → False"""
    assert should_push(0.3, 0.9, 48.0, False) is False


def test_should_push_rate_limited():
    """activation=0.95, confidence=0.9, 2h since last push → False"""
    assert should_push(0.95, 0.9, 2.0, False) is False


def test_should_push_unresolved_threads():
    """activation=0.85 (below 0.9 threshold), unresolved=True → True (lower threshold 0.7)"""
    assert should_push(0.85, 0.9, 48.0, True) is True


def test_should_push_low_confidence():
    """activation=0.95, confidence=0.2 → False (confidence < 0.8)"""
    assert should_push(0.95, 0.2, 48.0, False) is False


# ── Backfill tests ──────────────────────────────────────────────────


def _make_store(tmp_path) -> VectorStore:
    """Create a VectorStore with tmp_path for isolation."""
    return VectorStore(agent="bob", base_path=tmp_path)


def _add_bare_segment(store: VectorStore, seg_id: str, source_type: str = "session_segment"):
    """Add a segment WITHOUT cognitive fields (simulates pre-existing data)."""
    metadata = {
        "session_id": "test-session",
        "date": "2026-03-01",
        "domain": "general",
        "topic": "test",
        "tone": "",
        "participants": json.dumps([]),
        "unresolved_threads": json.dumps([]),
        "status": "",
        "source_type": source_type,
        "source_weight": 0.8,
        "agent": "bob",
        "indexed_at": "2026-03-01T00:00:00+00:00",
    }
    store.segments.upsert(
        ids=[seg_id],
        embeddings=[[0.1] * 384],
        documents=["test content"],
        metadatas=[metadata],
    )


def test_backfill_idempotent(tmp_path):
    """Running backfill twice produces no changes on second run."""
    store = _make_store(tmp_path)
    _add_bare_segment(store, "seg-1")

    # First run: should update
    stats1 = backfill_collection(store.segments, "bob", dry_run=False, infer_access=False)
    assert stats1["updated"] == 1
    assert stats1["skipped"] == 0

    # Second run: should skip (already backfilled)
    stats2 = backfill_collection(store.segments, "bob", dry_run=False, infer_access=False)
    assert stats2["updated"] == 0
    assert stats2["skipped"] == 1


def test_backfill_infer_access_vault(tmp_path):
    """Vault source_type → access_count=10 when --infer-access."""
    store = _make_store(tmp_path)
    _add_bare_segment(store, "seg-vault", source_type="vault")

    backfill_collection(store.segments, "bob", dry_run=False, infer_access=True)

    result = store.segments.get(ids=["seg-vault"], include=["metadatas"])
    assert result["metadatas"][0]["access_count"] == 10


def test_backfill_dry_run_no_writes(tmp_path):
    """Dry run mode doesn't modify any data."""
    store = _make_store(tmp_path)
    _add_bare_segment(store, "seg-dry")

    stats = backfill_collection(store.segments, "bob", dry_run=True, infer_access=False)
    assert stats["updated"] == 1  # counted as "would update"

    # Verify no actual changes were written
    result = store.segments.get(ids=["seg-dry"], include=["metadatas"])
    assert "access_count" not in result["metadatas"][0]
    assert "confidence" not in result["metadatas"][0]
    assert "last_pushed_at" not in result["metadatas"][0]
