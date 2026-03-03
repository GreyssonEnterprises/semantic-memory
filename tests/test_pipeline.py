"""End-to-end pipeline test: export a session → embed it → store in Chroma → query it."""

import time
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.exporter import list_sessions, extract_conversation
from src.pipeline.embedder import embed_text, embed_batch
from src.pipeline.store import VectorStore


def test_pipeline():
    print("=" * 60)
    print("SEMANTIC MEMORY — Pipeline Integration Test")
    print("=" * 60)

    # Step 1: List and extract a session
    print("\n[1] Listing sessions...")
    sessions = list_sessions(limit=5, min_messages=6)
    print(f"    Found {len(sessions)} sessions")
    if not sessions:
        print("    ERROR: No sessions found!")
        return

    target = sessions[0]
    print(f"    Using: {target['session_id'][:16]}... ({target['date']}, {target['message_count']} msgs)")

    # Step 2: Extract conversation
    print("\n[2] Extracting conversation...")
    conv = extract_conversation(target["path"])
    print(f"    Clean messages: {len(conv['messages'])}")
    print(f"    Raw text: {len(conv['raw_text'])} chars")

    # Step 3: Create basic chunks (simple for now — Patterson's chunker will replace this)
    print("\n[3] Creating chunks (basic sliding window)...")
    raw = conv["raw_text"]
    chunk_size = 500
    overlap = 100
    chunks = []
    i = 0
    while i < len(raw):
        end = min(i + chunk_size, len(raw))
        chunk = raw[i:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        i += chunk_size - overlap
    print(f"    Created {len(chunks)} chunks")

    # Step 4: Generate embeddings via ollama
    print("\n[4] Embedding chunks via nomic-embed-text...")
    t0 = time.time()
    chunk_embeddings = embed_batch([c[:512] for c in chunks])  # Truncate to safe length
    embed_time = time.time() - t0
    print(f"    Embedded {len(chunk_embeddings)} chunks in {embed_time:.2f}s")
    print(f"    Embedding dim: {len(chunk_embeddings[0])}")

    # Step 5: Generate session summary embedding
    print("\n[5] Embedding session summary...")
    # For now, use first 1000 chars as "summary" — real summarizer comes later
    summary_text = f"Session from {conv['date']}. {len(conv['messages'])} messages. Topics discussed: {raw[:500]}"
    summary_embedding = embed_text(summary_text[:512])
    print(f"    Summary embedding dim: {len(summary_embedding)}")

    # Step 6: Store in Chroma
    print("\n[6] Storing in Chroma (bob namespace)...")
    store = VectorStore("bob")
    
    for idx, (chunk, emb) in enumerate(zip(chunks, chunk_embeddings)):
        store.add_segment(
            segment_id=f"{conv['session_id']}_seg_{idx:03d}",
            text=chunk,
            embedding=emb,
            session_id=conv["session_id"],
            date=conv["date"],
            domain="general",
            topic="test",
        )

    store.add_session_summary(
        session_id=conv["session_id"],
        summary_text=summary_text,
        embedding=summary_embedding,
        date=conv["date"],
    )

    stats = store.stats()
    print(f"    Stored! {stats}")

    # Step 7: Test retrieval
    print("\n[7] Testing retrieval...")
    query = "architecture discussion about memory and embeddings"
    print(f"    Query: '{query}'")

    t0 = time.time()
    q_emb = embed_text(query)
    results = store.search_segments(q_emb, n_results=3)
    retrieval_time = time.time() - t0
    print(f"    Retrieval time: {retrieval_time:.3f}s (target: <3s)")

    if results["documents"] and results["documents"][0]:
        for i, (doc, dist) in enumerate(zip(results["documents"][0], results["distances"][0])):
            print(f"\n    Result {i+1} (distance: {dist:.4f}):")
            print(f"    {doc[:150]}...")

    # Step 8: Test summary search
    print("\n[8] Testing summary search...")
    query2 = "productive planning session with multiple participants"
    q_emb2 = embed_text(query2)
    results2 = store.search_summaries(q_emb2, n_results=3)
    if results2["documents"] and results2["documents"][0]:
        for i, (doc, dist) in enumerate(zip(results2["documents"][0], results2["distances"][0])):
            print(f"\n    Result {i+1} (distance: {dist:.4f}):")
            print(f"    {doc[:150]}...")

    print("\n" + "=" * 60)
    print("PIPELINE TEST COMPLETE")
    print(f"  ✓ Session extraction: working")
    print(f"  ✓ Embedding (nomic-embed-text): {len(chunk_embeddings[0])}-dim, {embed_time:.2f}s for {len(chunks)} chunks")
    print(f"  ✓ Chroma storage: {stats['segments_count']} segments, {stats['summaries_count']} summaries")
    print(f"  ✓ Retrieval: {retrieval_time:.3f}s (target: <3s)")
    print("=" * 60)


if __name__ == "__main__":
    test_pipeline()
