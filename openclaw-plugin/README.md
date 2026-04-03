# greysson-memory (OpenClaw Plugin)

OpenClaw plugin that provides semantic memory search and topic document scanning on every conversational turn. Part of the [semantic-memory](https://github.com/GreyssonEnterprises/semantic-memory) system.

## Three Memory Layers

```
┌──────────────────────────────────────────────────────────────┐
│                    Agent Conversation                         │
│                                                              │
│  ┌─────────────┐  ┌─────────────────┐  ┌────────────────┐   │
│  │  Layer 1     │  │  Layer 2         │  │  Layer 3        │  │
│  │  LCM         │  │  Semantic Memory │  │  Crystallized   │  │
│  │  (lossless-  │  │  (ChromaDB +     │  │  Topics         │  │
│  │   claw)      │  │   embeddings)    │  │  (markdown)     │  │
│  │              │  │                  │  │                 │  │
│  │  DAG-based   │  │  Vector search   │  │  Living topic   │  │
│  │  conversation│  │  across all      │  │  documents in   │  │
│  │  compaction  │  │  ingested        │  │  memory/topics/ │  │
│  │              │  │  memory files    │  │                 │  │
│  │  Separate    │  │  This plugin     │  │  This plugin    │  │
│  │  plugin      │  │                  │  │                 │  │
│  └─────────────┘  └─────────────────┘  └────────────────┘   │
└──────────────────────────────────────────────────────────────┘
```

## How It Works

On every conversational turn (not heartbeat/cron), the plugin:

1. **Topic Scanner** — scans `memory/topics/*.md` for filenames matching the prompt, extracts Key Conclusions and Open Questions
2. **Semantic Search** — queries ChromaDB via `pai-memory-recall` CLI (primary) or direct embeddings (fallback)
3. **Context Assembly** — combines results into `prependContext` (capped at `maxPrependChars`)
4. **Crystallization Reminder** — appends a checkpoint reminder to `appendSystemContext`

On heartbeat triggers, the plugin runs incremental indexing instead — processing new session JSONL files into ChromaDB embeddings so they become searchable.

## Configuration

All settings are in `openclaw.plugin.json` with sensible defaults:

| Setting | Default | Description |
|---------|---------|-------------|
| `enabled` | `true` | Master kill switch |
| `topicsDir` | `memory/topics` | Workspace-relative path to topic docs |
| `chromaDbPath` | `semantic-memory/vectors` | ChromaDB vector store path |
| `ollamaBaseUrl` | `http://localhost:11434` | Ollama API endpoint |
| `embeddingModel` | `nomic-embed-text` | Embedding model name |
| `maxTopicDocs` | `3` | Max topic documents to inject |
| `maxMemoryResults` | `5` | Max semantic memory results |
| `minMemoryScore` | `0.5` | Minimum similarity threshold |
| `maxPrependChars` | `2000` | Hard cap on injected context size |
| `crystallizationReminder` | `true` | Include the crystallization checkpoint |
| `skipTriggers` | `["cron"]` | Triggers that skip memory injection |

## Installation

```bash
cd openclaw-plugin
npm install
npm run build
```

Then configure in your OpenClaw `openclaw.json` plugins section.

## Design Decisions

1. **No chromadb npm dependency** — queries ChromaDB via HTTP API or `pai-memory-recall` CLI
2. **Parallel execution** — topic scanning and semantic search run via `Promise.all`
3. **Budget-capped output** — total injected context hard-capped at `maxPrependChars`
4. **Per-agent memory** — agent ID from hook context routes to the correct ChromaDB collection
5. **Heartbeat indexing** — runs `incremental-ingest` on heartbeat to keep memories fresh
6. **Graceful degradation** — every external call wrapped in try/catch, never crashes the hook pipeline
