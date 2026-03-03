# Semantic Memory

Cross-agent semantic memory infrastructure for Circle Zero. Ingests conversation sessions, chunks them by topic, embeds them via local or cloud models, stores them in ChromaDB, and retrieves them using cognitive scoring drawn from ACT-R activation theory, Bayesian confidence, and Hebbian co-activation.

## Architecture

```
Session JSONL → Exporter → Topic Chunker → Embedder → ChromaDB
                                              ↕
                                     ollama / OpenAI API

Retrieval:  Query → Embed → Vector Search → ACT-R Scoring
                                           → Bayesian Confidence
                                           → Hebbian Co-activation Boost
                                           → Ranked Results
```

**Dual-layer design:**
- **Human Web** — Markdown files (`memory/YYYY-MM-DD.md`), git-tracked, human-readable
- **Agent Web** — ChromaDB vectors + SQLite co-activation graph, protocol-accessible via MCP

Each agent (bob, patterson, dean) gets filesystem-isolated ChromaDB collections and co-activation databases. A `shared` namespace enables cross-agent context.

## Quick Start

```bash
# Install dependencies
uv sync

# Ingest recent sessions (requires ollama with nomic-embed-text)
uv run python -m src.pipeline.ingest --agent bob --limit 30

# Search memories
uv run python -m src.retrieval.recall "what did we decide about the chunking strategy"

# Run as MCP server (stdio transport)
uv run src/mcp_server.py
```

## Embedding Providers

The embedding layer supports **Ollama** (default) and **OpenAI-compatible APIs**, selected at runtime via environment variables. No code changes needed to switch providers.

### Ollama (default)

```bash
export EMBEDDING_PROVIDER=ollama          # optional, this is the default
export OLLAMA_URL=http://localhost:11434   # optional, this is the default
export EMBED_MODEL=nomic-embed-text       # optional, this is the default
```

### OpenAI

```bash
export EMBEDDING_PROVIDER=openai
export OPENAI_API_KEY=sk-...
export EMBED_MODEL=text-embedding-3-small  # optional, this is the default for openai
```

### OpenAI-compatible APIs (Azure, vLLM, LiteLLM, etc.)

```bash
export EMBEDDING_PROVIDER=openai
export OPENAI_API_KEY=your-key
export OPENAI_BASE_URL=https://your-endpoint.openai.azure.com
export EMBED_MODEL=your-deployment-name
```

### Dimension compatibility

ChromaDB collections are locked to the embedding dimension of their first insert. If you have an existing store built with `nomic-embed-text` (768-dim) and want to switch to `text-embedding-3-small` (1536-dim default), request 768 dimensions to stay compatible:

```bash
export EMBEDDING_DIMENSIONS=768
```

Models that support the `dimensions` parameter: `text-embedding-3-small`, `text-embedding-3-large`. Models that don't (`text-embedding-ada-002`) will ignore this setting.

### All embedding environment variables

| Variable | Default | Description |
|---|---|---|
| `EMBEDDING_PROVIDER` | `ollama` | `"ollama"` or `"openai"` |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama API endpoint |
| `EMBED_MODEL` | per provider | `nomic-embed-text` (ollama) or `text-embedding-3-small` (openai) |
| `OPENAI_API_KEY` | — | Required when provider is `openai` |
| `OPENAI_BASE_URL` | `https://api.openai.com` | Override for compatible APIs |
| `EMBEDDING_DIMENSIONS` | unset | Output dimensions (openai only) |

## Cognitive Primitives

Five scoring primitives layered onto vector similarity, each grounded in published cognitive science:

| Primitive | Source | Effect |
|---|---|---|
| **ACT-R Activation** | Anderson (1993) | Frequently accessed, recently used memories rank higher |
| **Bayesian Confidence** | Bayes (1763) | Contradicted memories score lower; confirmed memories score higher |
| **Hebbian Co-activation** | Hebb (1949) | Memories retrieved together strengthen their mutual association |
| **Sequential Patterns** | Temporal learning | Common retrieval sequences enable predictive recall |
| **Push Triggers** | Proactive retrieval | High-activation memories surface without explicit query |

Scoring formula:

```
score = cosine_similarity × actr_activation(n, age) × source_weight × confidence + hebbian_boost
```

See [docs/COGNITIVE-PRIMITIVES-SPEC.md](docs/COGNITIVE-PRIMITIVES-SPEC.md) for the full specification with formulas, edge cases, and test plans.

## MCP Server

The MCP server exposes semantic memory to any MCP-compatible client (Claude Code, Cursor, Claude Desktop, etc.).

**Tools:**
- `search_memory` — Semantic search with domain filtering and cognitive scoring
- `store_memory` — Persist facts/decisions with automatic embedding and classification
- `list_recent` — Browse recent daily memory logs
- `get_procedures` — Search procedural memory for workflows

**Claude Code configuration:**

```json
{
  "mcpServers": {
    "semantic-memory": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/semantic-memory", "src/mcp_server.py"],
      "env": {
        "EMBEDDING_PROVIDER": "openai",
        "OPENAI_API_KEY": "sk-...",
        "SEMANTIC_MEMORY_STORE": "/path/to/vectors",
        "SEMANTIC_MEMORY_AGENT": "bob"
      }
    }
  }
}
```

See [docs/MCP-MEMORY-SERVER-SPEC.md](docs/MCP-MEMORY-SERVER-SPEC.md) for the full server specification.

## Project Structure

```
src/
├── pipeline/
│   ├── embedder.py        # Multi-provider embedding (ollama / openai)
│   ├── exporter.py        # OpenClaw session JSONL → clean conversation
│   ├── ingest.py          # Bulk ingest pipeline
│   └── store.py           # ChromaDB vector store with agent isolation
├── chunker/
│   └── topic_chunker.py   # Topic segmentation + sliding window chunking
├── retrieval/
│   ├── recall.py          # Search with cognitive scoring
│   ├── confidence.py      # Bayesian confidence updates
│   ├── context_loader.py  # Agent-friendly context injection
│   └── push.py            # Proactive memory surfacing triggers
├── graph/
│   ├── coactivation.py    # Hebbian co-activation graph (SQLite)
│   └── sequences.py       # Sequential pattern detection
└── mcp_server.py          # FastMCP server (stdio transport)

scripts/
├── backfill_classify.py       # LLM classification backfill for existing segments
└── backfill_cognitive_fields.py  # Add cognitive metadata to pre-existing memories

docs/
├── COGNITIVE-PRIMITIVES-SPEC.md  # Full cognitive architecture specification
├── MCP-MEMORY-SERVER-SPEC.md     # MCP server API specification
└── PHILOSOPHY.md                 # Circle Zero cognitive extension philosophy
```

## Testing

```bash
uv run python -m pytest tests/ -v
```

86 tests covering ACT-R activation, Bayesian confidence, Hebbian co-activation, sequential patterns, push triggers, embedding providers, and the ingest pipeline. The test suite requires no external services — all API calls are mocked.

## Dependencies

- Python ≥ 3.12
- [ChromaDB](https://www.trychroma.com/) — Vector storage
- [requests](https://docs.python-requests.org/) — HTTP client for embedding APIs
- [MCP](https://modelcontextprotocol.io/) — Model Context Protocol server SDK

Embedding backends (one required):
- [Ollama](https://ollama.com/) with `nomic-embed-text` — Local, no API key needed
- OpenAI API or any compatible endpoint — Requires API key
