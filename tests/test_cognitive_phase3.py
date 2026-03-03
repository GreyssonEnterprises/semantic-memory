import pytest
from src.graph.coactivation import CoactivationGraph, DEFAULT_LEARNING_RATE, MAX_WEIGHT, PRUNE_THRESHOLD


@pytest.fixture
def graph(tmp_path):
    return CoactivationGraph("bob", db_path=str(tmp_path / "test_coactivation.db"))


def test_record_co_retrieval_new_edge(graph):
    """Two new memory IDs -> edge created with weight=learning_rate"""
    graph.record_co_retrieval(["mem_a", "mem_b"])
    assocs = graph.get_associations("mem_a", min_weight=0.0)
    assert len(assocs) == 1
    assert assocs[0]["memory_id"] == "mem_b"
    assert assocs[0]["weight"] == pytest.approx(DEFAULT_LEARNING_RATE)


def test_record_co_retrieval_strengthen(graph):
    """Same pair co-retrieved again -> weight increases"""
    graph.record_co_retrieval(["mem_a", "mem_b"])
    graph.record_co_retrieval(["mem_a", "mem_b"])
    assocs = graph.get_associations("mem_a", min_weight=0.0)
    assert assocs[0]["weight"] == pytest.approx(DEFAULT_LEARNING_RATE * 2)
    assert assocs[0]["co_retrieval_count"] == 2


def test_weight_cap(graph):
    """Edge at max weight stays at MAX_WEIGHT"""
    for _ in range(100):
        graph.record_co_retrieval(["mem_a", "mem_b"])
    assocs = graph.get_associations("mem_a", min_weight=0.0)
    assert assocs[0]["weight"] == pytest.approx(MAX_WEIGHT)


def test_decay_reduces_weight(graph):
    """Edge at 1.0, 1 day decay -> weight decreases"""
    for _ in range(10):
        graph.record_co_retrieval(["mem_a", "mem_b"])
    assocs_before = graph.get_associations("mem_a", min_weight=0.0)
    weight_before = assocs_before[0]["weight"]

    graph.apply_decay(days_elapsed=1.0)
    assocs_after = graph.get_associations("mem_a", min_weight=0.0)
    assert assocs_after[0]["weight"] < weight_before


def test_decay_prunes_weak(graph):
    """Edge below PRUNE_THRESHOLD gets deleted"""
    graph.record_co_retrieval(["mem_a", "mem_b"])  # weight = 0.1
    pruned = graph.apply_decay(days_elapsed=20)  # 0.1 * 0.95^20 ~ 0.036 < 0.05
    assert pruned >= 1
    assocs = graph.get_associations("mem_a", min_weight=0.0)
    assert len(assocs) == 0


def test_canonical_ordering(graph):
    """IDs "b" and "a" -> stored as ("a", "b")"""
    graph.record_co_retrieval(["b_mem", "a_mem"])
    assocs_a = graph.get_associations("a_mem", min_weight=0.0)
    assocs_b = graph.get_associations("b_mem", min_weight=0.0)
    assert len(assocs_a) == 1
    assert len(assocs_b) == 1
    assert assocs_a[0]["memory_id"] == "b_mem"
    assert assocs_b[0]["memory_id"] == "a_mem"


def test_hebbian_boost_mutual(graph):
    """Two results with strong edge -> both scores increase"""
    for _ in range(20):
        graph.record_co_retrieval(["mem_a", "mem_b"])

    results = [
        {"id": "mem_a", "score": 0.5, "metadata": {}, "doc": ""},
        {"id": "mem_b", "score": 0.5, "metadata": {}, "doc": ""},
    ]
    boosted = graph.hebbian_boost(results)
    assert boosted[0]["score"] > 0.5
    assert boosted[1]["score"] > 0.5


def test_hebbian_boost_no_edge(graph):
    """Two results with no edge -> scores unchanged"""
    results = [
        {"id": "mem_x", "score": 0.5, "metadata": {}, "doc": ""},
        {"id": "mem_y", "score": 0.5, "metadata": {}, "doc": ""},
    ]
    boosted = graph.hebbian_boost(results)
    assert boosted[0]["score"] == pytest.approx(0.5)
    assert boosted[1]["score"] == pytest.approx(0.5)


def test_single_result_no_coactivation(graph):
    """Single result -> no edge recorded"""
    graph.record_co_retrieval(["mem_a"])
    assocs = graph.get_associations("mem_a", min_weight=0.0)
    assert len(assocs) == 0
