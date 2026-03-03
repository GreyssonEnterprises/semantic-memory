# MCP Memory Server Spec — Circle Zero Collective Memory

*Author: Bob | Date: 2026-03-02 | Status: APPROVED — All open questions resolved, consensus achieved. Phase 0 in progress.*

---

## Executive Summary

An MCP server that wraps our existing markdown + Chroma memory architecture, exposing bidirectional read/write access via the Model Context Protocol. Any MCP-compatible client (Claude Desktop, Cursor, Gemini CLI, Claude Code, future mobile agents) becomes both a memory consumer and contributor — without replacing our human-readable markdown backend.

**Key insight from external research:** The "Open Brain" architecture (Nate's Substack, 2026-03-02) proposes Postgres + pgvector + MCP for universal AI memory access. We already have a more sophisticated architecture (cognitive memory tiers, agent isolation, time-weighted retrieval, local-first embeddings). What we're missing is the **protocol layer** — the MCP interface that makes our memory accessible to any tool, not just agents with filesystem access.

**Additional insight:** MCP could replace the NAS as the primary mechanism for cross-agent context sharing during processing. Instead of syncing Chroma stores via filesystem, agents query the MCP server directly. Simpler, real-time, no sync lag.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  MCP Clients                        │
│  Claude Desktop │ Cursor │ Gemini CLI │ Claude Code │
│  Patterson      │ Dean   │ Bob        │ Future...   │
└────────────────────────┬────────────────────────────┘
                         │ MCP Protocol (stdio / SSE)
                         ▼
┌─────────────────────────────────────────────────────┐
│              MCP Memory Server                      │
│                                                     │
│  Tools:                                             │
│  ├── search_memory    (semantic + keyword)           │
│  ├── store_memory     (classify → embed → write)     │
│  ├── get_entity       (people, projects, accounts)   │
│  ├── list_recent      (last N days, filtered)        │
│  ├── memory_stats     (patterns, frequency, trends)  │
│  └── get_procedures   (learned workflows)            │
│                                                     │
│  Auth:                                              │
│  ├── Agent identity (who's calling)                  │
│  └── Namespace isolation (private vs shared)         │
└──────────┬──────────────────────┬───────────────────┘
           │                      │
           ▼                      ▼
┌──────────────────┐   ┌──────────────────────────────┐
│  Chroma (Vector) │   │  Markdown Files (Human Web)  │
│                  │   │                              │
│  /opt/vector-    │   │  ~/clawd/memory/             │
│  store/          │   │  ├── YYYY-MM-DD.md           │
│  ├── bob/        │   │  ├── MEMORY.md               │
│  ├── patterson/  │   │  ├── meta/                   │
│  ├── dean/       │   │  ├── procedures/             │
│  └── shared/     │   │  └── graph/                  │
└──────────────────┘   └──────────────────────────────┘
           │
           ├──────────────────────┐
           ▼                      ▼
┌──────────────────┐   ┌──────────────────────────────┐
│  ollama          │   │  SQLite (Co-activation Graph) │
│  nomic-embed-    │   │                              │
│  text (local)    │   │  {vector-store}/{agent}/     │
│  on the server   │   │  coactivation.db             │
└──────────────────┘   │  (Hebbian edges — Phase 3)   │
                       └──────────────────────────────┘
```

### Dual-Layer Design (Human Web + Agent Web)

- **Human Web (Markdown):** Remains the source of truth. Human-readable, git-tracked, diffable. The admin can open any file and read it.
- **Agent Web (Chroma + MCP):** Semantic search, structured metadata, protocol-native access. Any MCP client can query by meaning.

Writes go through MCP → classification/embedding → append to markdown + index in Chroma.
Reads go through MCP → Chroma semantic search → return results with source file references.

---

## MCP Tools Specification

### 1. `search_memory`

Semantic + keyword hybrid search across the memory store.

**Parameters:**
| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `query` | string | ✅ | Natural language search query |
| `scope` | enum | ❌ | `private` (caller's store only), `shared`, `all` (private + shared). Default: `all` |
| `domain` | string | ❌ | Filter by domain tag (e.g., `redhat`, `trading`, `household`) |
| `date_from` | string | ❌ | ISO date, earliest results |
| `date_to` | string | ❌ | ISO date, latest results |
| `type` | enum | ❌ | `decision`, `learning`, `task`, `person`, `procedure`, `any`. Default: `any` |
| `limit` | int | ❌ | Max results. Default: 10 |
| `include_cognitive_scores` | bool | ❌ | Include ACT-R activation, confidence, and association data in results. Default: `false` |
| `predict` | bool | ❌ | Include predicted next-memories via sequential pattern analysis (Phase 4). Default: `false` |

**Returns:** Array of `{ content, source_file, score, date, type, domain, tags, people_mentioned }`. When `include_cognitive_scores=true`, each result also includes:

```json
{
  "cognitive": {
    "activation_weight": 0.88,
    "access_count": 7,
    "confidence": 1.15,
    "association_boost": 0.05,
    "spread_bonus": 1.12
  }
}
```

When `predict=true`, results include `predicted_next: [{ memory_id, confidence }]`.

> See `COGNITIVE-PRIMITIVES-SPEC.md` §MCP Integration Points for full response schema details.

**Search strategy:**
- Chroma vector similarity (nomic-embed-text embeddings) weighted by **ACT-R base-level activation** (replaces simple time decay). The activation formula `B = ln(n+1) − 0.5 × ln(age_days/(n+1))` models both recency and frequency of access, converted to a 0–1 weight via sigmoid. See `COGNITIVE-PRIMITIVES-SPEC.md` §Phase 1 for full formula derivation.
- **Bayesian confidence multiplier** (Phase 2): memories carry a confidence score `[0.1, 2.0]` that scales their final score. Contradicted memories score lower; confirmed memories score higher.
- **Hebbian co-activation boost** (Phase 3): after initial scoring, memories with strong co-activation edges to top-ranked results receive an additive score boost.
- BM25 keyword fallback for exact matches
- Hybrid merge (70/30 vector/keyword, tunable)

**Scoring formula evolution:**
```
Phase 0-1 (current): score = similarity × time_weight × source_weight
After Phase 1:       score = similarity × activation_weight × source_weight
After Phase 2:       score = similarity × activation_weight × source_weight × confidence
After Phase 3:       score = similarity × activation_weight × source_weight × confidence + association_boost
```

### 2. `store_memory`

Capture a new memory. Runs classification + embedding in parallel, writes to both Chroma and markdown.

**Parameters:**
| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `content` | string | ✅ | The thought/fact/decision to store |
| `type` | enum | ❌ | `decision`, `learning`, `task`, `person`, `insight`, `meeting_debrief`, `auto` (LLM classifies). Default: `auto` |
| `domain` | string | ❌ | Domain tag. Default: auto-detected |
| `scope` | enum | ❌ | `private` (caller's store) or `shared`. Default: `private` |
| `source` | string | ❌ | Where this came from (e.g., `slack`, `cursor`, `claude-desktop`) |
| `people` | string[] | ❌ | People mentioned. **Provide or extract:** if caller provides, pipeline trusts + normalizes against entity graph. If omitted, Phase 2 classification pipeline auto-extracts from content. |
| `action_items` | string[] | ❌ | Action items. **Provide or extract:** same pattern as `people`. If omitted, auto-extracted by classification pipeline. |
| `vault` | bool | ❌ | Mark as "never decay". Default: false |
| `confidence` | float | ❌ | Initial confidence score `[0.1, 2.0]`. Default: `1.0`. Override when the caller has prior knowledge about reliability. |

**Pipeline (parallel where possible):**
1. **Embed:** Generate vector via `nomic-embed-text` on ollama (studio)
2. **Classify:** Extract metadata (type, domain, people, action_items, topics) via lightweight LLM call if `type=auto`
3. **Detect contradictions:** Search existing memories with the new embedding; for results with cosine similarity > 0.85, run lightweight LLM check for semantic contradiction. See `COGNITIVE-PRIMITIVES-SPEC.md` §Phase 2 for detection algorithm. *(Depends on step 1; runs in parallel with step 2.)*
4. **Apply Bayesian updates:** If contradictions detected, update `confidence` on affected existing memories via `bayesian_update()`. *(Depends on step 3.)*
5. **Write markdown:** Append to `memory/YYYY-MM-DD.md` with structured template (see Capture Templates below)
6. **Index Chroma:** Store embedding + metadata in appropriate collection (agent-specific or shared), including `confidence` and `access_count` fields

Steps 1 and 2 run in parallel. Step 3 depends on step 1. Steps 4, 5, 6 can proceed in parallel after step 3.

**Returns:** `{ id, type, domain, people, action_items, topics, stored_at, source_file, contradictions_detected }`

The `contradictions_detected` field is included when conflicts are found:

```json
{
  "contradictions_detected": [
    {
      "existing_memory_id": "...",
      "similarity": 0.91,
      "evidence_type": "strong_contradict",
      "old_confidence": 1.0,
      "new_confidence": 0.60
    }
  ]
}
```

> See `COGNITIVE-PRIMITIVES-SPEC.md` §Phase 2 for Bayesian update formulas and likelihood ratio table.

### 3. `get_entity`

Retrieve structured information about a known entity (person, project, account, business).

**Parameters:**
| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | ✅ | Entity name |
| `entity_type` | enum | ❌ | `person`, `project`, `account`, `business`, `any`. Default: `any` |

**Returns:** Entity record from knowledge graph + all related memories (via Chroma search filtered to entity mentions).

**Source:** `memory/graph/` files + Chroma metadata filter on `people_mentioned`.

### 4. `list_recent`

Browse recent memories by time window.

**Parameters:**
| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `days` | int | ❌ | How far back. Default: 7 |
| `scope` | enum | ❌ | `private`, `shared`, `all`. Default: `all` |
| `domain` | string | ❌ | Filter by domain |
| `type` | enum | ❌ | Filter by type |

**Returns:** Chronological list of memories from the specified window.

**Source:** Direct read of `memory/YYYY-MM-DD.md` files, filtered by metadata.

### 5. `memory_stats`

Meta-analysis of memory patterns. What you're thinking about, how often, what's trending.

**Parameters:**
| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `period` | enum | ❌ | `week`, `month`, `quarter`. Default: `week` |
| `scope` | enum | ❌ | `private`, `shared`, `all`. Default: `all` |

**Returns:**
```json
{
  "total_memories": 142,
  "by_domain": { "redhat": 45, "trading": 28, "household": 22, ... },
  "by_type": { "decision": 18, "learning": 34, "task": 52, ... },
  "top_people": ["Jane Doe", "User A", "User B", ...],
  "trending_topics": ["semantic-memory", "Chroma", "MCP", ...],
  "declining_topics": ["ansible-migration", ...],
  "unresolved_action_items": 12,
  "recurring_patterns": ["TAM case reviews cluster on Mondays", ...]
}
```

### 6. `get_procedures`

Retrieve learned procedural memory (workflows, how-tos).

**Parameters:**
| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `query` | string | ✅ | What procedure are you looking for |

**Returns:** Matching procedures from `memory/procedures/` with steps and context.

### 7. `get_cognitive_profile`

*(Added by Cognitive Primitives — see `COGNITIVE-PRIMITIVES-SPEC.md` for full specification)*

Retrieve cognitive metadata for a specific memory, including activation scores, confidence history, and Hebbian associations. Useful for debugging scoring behavior and understanding why certain memories rank higher.

**Parameters:**
| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `memory_id` | string | ✅ | Chroma document ID |

**Returns:** `{ memory_id, content_preview, activation: { access_count, age_days, base_level_B, activation_weight, spread_bonus, last_accessed }, confidence: { current, history[] }, associations: [{ memory_id, weight, co_retrieval_count }] }`

---

## Agent Identity & Namespace Isolation

> **Status:** DECIDED (2026-03-02) — Consensus from Bob, Patterson, Dean.

### Agent Authentication

Each MCP client authenticates via **bearer token** (one per agent, server-side scope enforcement).

- Token → agent scope mapping enforced server-side on every request
- No trust-based isolation — the server determines your scope from your token, not from query parameters
- Tokens provisioned as part of Phase 0 deliverables

### Scope Model

| Scope | Read Access | Write Access | Available To |
|-------|-------------|--------------|--------------|
| `own` (default) | Caller's store only | Caller's store only | All agent tokens |
| `shared` | Shared space | Shared space (explicit) | All agent tokens |
| `all` | Own + shared | Own + shared | All agent tokens |
| `analytics` | Read-only across ALL agents | None | All agent tokens |
| Admin | Everything | Everything | Admin token only (the admin, synthesis cron) |

**Key enforcement rules:**
- `search_memory(scope="own")` → token determines which store gets queried, no parameter spoofing possible
- `search_memory(scope="analytics")` → read-only cross-agent access for stats/patterns (no special token needed)
- `store_memory` → always writes to the store mapped to your token. Period.
- Admin token required for: full cross-agent reads+writes, MEMORY.md promotion, weekly synthesis cron

### Namespace Rules

| Caller | Can Read | Can Write |
|--------|----------|-----------|
| Bob (token) | `bob/` + `shared/` + analytics | `bob/` + `shared/` (explicit) |
| Patterson (token) | `patterson/` + `shared/` + analytics | `patterson/` + `shared/` (explicit) |
| Dean (token) | `dean/` + `shared/` + analytics | `dean/` + `shared/` (explicit) |
| External (Cursor, etc.) | `shared/` only | `shared/` only |
| Admin (the admin / cron) | Everything | Everything |

### Concurrent Write Handling

> **Decision:** Atomic append (temp file → rename). No file locking.

Chroma handles its own write concurrency internally. For the markdown side, writes use atomic append: write to a temp file, then atomic rename/append. No file locking — avoids contention with three agents + potential Claude Desktop/Cursor clients.

### Degraded Mode

> **Decision:** "Memory offline" — clean failure, no fallback path.

If studio is down, memory is unavailable. Agents survive without memory — we did it for months before this system existed. No fallback to raw markdown reads (that would mean maintaining two access patterns indefinitely).

### Cross-Agent Context Sharing via MCP

**This replaces the NAS sync model for Chroma.** Instead of:
1. Bob ingests on studio → writes to `/opt/vector-store/bob/`
2. Sync to NAS → Patterson reads from NAS copy

We do:
1. Bob ingests on studio → writes to Chroma + markdown
2. Patterson calls `search_memory(scope="shared")` via MCP → gets results in real-time

**Advantages over NAS:**
- No sync lag
- No filesystem permission issues
- No stale copies
- Real-time access
- Protocol-native (works from any machine, not just NAS-mounted ones)
- Centralized embedding (all writes go through studio's ollama)

---

## Capture Templates (Standardized Daily Log Format)

When `store_memory` writes to `memory/YYYY-MM-DD.md`, it uses structured templates:

```markdown
## [HH:MM] Decision
**Domain:** redhat | **People:** Jane Doe, customer team
**Context:** Decided to escalate case 12345 to engineering...
**Rationale:** SBT at 12 minutes, customer production down...
**Action items:** 
- [ ] Follow up with engineering by EOD

---

## [HH:MM] Learning
**Domain:** trading | **Topics:** silver, technical-analysis
**Insight:** Silver BeeBee pattern confirmed on 4H chart...
**Source:** Visual chart analysis, TradingView

---

## [HH:MM] Person Note
**Person:** Jane Smith | **Domain:** redhat
**Context:** New TAM joining the team, background in OpenShift...
**Relationship:** Colleague, same pod

---

## [HH:MM] Meeting Debrief
**Domain:** redhat | **People:** Jane Doe, John Smith
**Summary:** Quarterly review — discussed case load, SBT targets...
**Decisions:** 
- Moving to bi-weekly 1:1s
**Action items:**
- [ ] Prep Q1 metrics for next meeting

---

## [HH:MM] Procedure
**Name:** case-escalation-workflow
**Domain:** redhat | **Trigger:** SBT critical, customer production down
**Steps:**
1. Check case timeline and SBT status
2. Verify engineering assignment
3. Escalate via manager if no response in 30 minutes
**Learned from:** Case 12345 on 2026-02-15
**Last validated:** 2026-03-01
```

> **Design rationale for `last_validated`:** Stale procedures are worse than no procedures because they look authoritative while being wrong. The Phase 5 weekly synthesis cron flags procedures not validated in 30+ days.

> **Procedure seeding:** `memory/procedures/` is seeded manually with existing documented workflows (e.g., `visual-chart-analysis.md`). Future procedures added organically via `store_memory(type="procedure")`.

---

## Technical Implementation

### Stack
- **Runtime:** TypeScript / Bun (per stack preferences)
- **MCP SDK:** `@modelcontextprotocol/sdk` (official TS SDK)
- **Vector DB:** Chroma (existing, on studio)
- **Co-activation Graph:** SQLite — per-agent databases at `{vector-store}/{agent}/coactivation.db` for Hebbian co-activation edges (Phase 3). See `COGNITIVE-PRIMITIVES-SPEC.md` §Phase 3 for schema.
- **Embeddings:** `nomic-embed-text` via ollama REST API (studio)
- **Markdown I/O:** Direct filesystem read/write
- **Transport:** stdio (for local clients) + SSE/HTTP (for remote clients)
- **Cognitive Primitives:** ACT-R activation, Bayesian confidence, Hebbian co-activation — implemented per `COGNITIVE-PRIMITIVES-SPEC.md`, integrated progressively across Phases 1–3

### Deployment
- **Primary:** Runs on studio (same box as Chroma + ollama)
- **Access:** 
  - Local agents on studio: stdio transport
  - Remote agents (remote workstation, SupportShell): SSE over HTTP (server:3333)
  - Cursor/Claude Desktop: MCP config pointing to stdio wrapper or SSE endpoint

### MCP Client Configuration (Example)

```json
{
  "mcpServers": {
    "circle-zero-memory": {
      "command": "bun",
      "args": ["run", "/opt/semantic-memory/mcp-server/index.ts"],
      "env": {
        "AGENT_ID": "bob",
        "CHROMA_URL": "http://localhost:8000",
        "OLLAMA_URL": "http://localhost:11434",
        "MEMORY_DIR": "/home/user/memory"
      }
    }
  }
}
```

For remote access (SSE):
```json
{
  "mcpServers": {
    "circle-zero-memory": {
      "url": "http://server:3333/mcp",
      "headers": {
        "X-Agent-ID": "bob",
        "Authorization": "Bearer <token>"
      }
    }
  }
}
```

---

## Implementation Phases

### Phase 0: Skeleton (1 day)
- MCP server scaffold with Bun + `@modelcontextprotocol/sdk`
- `search_memory` tool wired to existing markdown files (BM25 grep, no Chroma yet)
- `store_memory` tool that appends to daily markdown with template
- stdio transport working locally on studio
- Token provisioning for all agents (bob, patterson, dean, admin)
- **`CLASSIFICATION-CONTRACT.md`** — frozen schema defining: type enum, domain taxonomy, people normalization rules, expected JSON output format. Patterson's Phase 2 dependency — she doesn't write line one until this exists and is signed off.
- **Deliverable:** Working MCP server that reads/writes markdown, with auth skeleton and classification contract

### Phase 1: Vector Layer (2–3 days)
- Wire `search_memory` to Chroma for semantic search
- Wire `store_memory` to embed via ollama + index in Chroma
- Implement hybrid search (vector + BM25)
- Time-weighted scoring
- **Deliverable:** Semantic search over existing memory corpus

### Phase 2: Metadata & Classification (2–3 days)
- Auto-classification pipeline for `store_memory` (type, domain, people, action items)
- Parallel metadata extraction (embed + classify simultaneously)
- `get_entity` tool backed by metadata queries
- `list_recent` tool
- **Deliverable:** Smart ingestion with structured metadata

### Phase 3: Remote Access & Auth (1–2 days)
- SSE/HTTP transport for remote clients
- Agent identity via headers/env
- Namespace isolation enforcement
- Token-based auth for admin override
- **Deliverable:** Any agent on any machine can access memory

### Phase 4: Stats & Patterns (1–2 days)
- `memory_stats` tool implementation
- Topic clustering / trending analysis
- `get_procedures` tool
- **Deliverable:** Meta-analysis of memory patterns

### Phase 5: Weekly Synthesis Cron (1 day)
- Sunday night cron job using `memory_stats` + `list_recent`
- Generates weekly brain report
- Posts to `#workingsessions`
- Promotes durable facts to MEMORY.md (requires admin token)
- **Stale procedure detection:** Flags procedures with `last_validated` older than 30 days
- **Deliverable:** Automated weekly memory consolidation with procedure health monitoring

---

## Cognitive Memory Primitives

> Full specification: `COGNITIVE-PRIMITIVES-SPEC.md`

The MCP Memory Server integrates five cognitive primitives drawn from established cognitive science research. These primitives transform retrieval from passive similarity search into active cognitive recall — memories that are used often, confirmed by evidence, and associated with related context surface faster and more reliably.

### Primitives Overview

| Phase | Primitive | MCP Tools Affected | What Changes |
|-------|-----------|-------------------|--------------|
| 1 | **ACT-R Activation** (Anderson, 1993) | `search_memory` | Scoring replaces `time_weight` with activation formula incorporating access frequency + recency |
| 2 | **Bayesian Confidence** (Bayes/Laplace) | `search_memory`, `store_memory` | Confidence multiplier on scores; contradiction detection on store; `contradictions_detected` in response |
| 3 | **Hebbian Co-activation** (Hebb, 1949) | `search_memory` | Additive boost for memories with strong co-retrieval associations; requires SQLite graph |
| 4 | **Sequential Patterns** | `search_memory` (via `predict` param) | `predicted_next` field in response suggests likely follow-up memories |
| 5 | **Push Triggers** | None (cron/heartbeat) | Deferred — existing heartbeat covers 80% of use cases |

### Integration Notes

- **Backward compatibility:** All primitives are additive. Existing memories without cognitive metadata (`access_count`, `confidence`) default to neutral values (`access_count=1`, `confidence=1.0`). No migration required for correctness; optional backfill scripts improve initial quality.
- **Progressive deployment:** Each phase is independently deployable. The scoring formula grows monotonically — no phase removes a working signal.
- **New dependency:** Phase 3 introduces SQLite for per-agent co-activation graphs (`{vector-store}/{agent}/coactivation.db`). SQLite databases are auto-created on first use.
- **New Chroma metadata fields:** `access_count` (int), `last_n_access_timestamps` (JSON array, capped at 20), `confidence` (float). See `COGNITIVE-PRIMITIVES-SPEC.md` §Data Model for complete field reference.
- **New MCP tool:** `get_cognitive_profile` exposes per-memory cognitive data (activation, confidence history, associations) for debugging and transparency. See `COGNITIVE-PRIMITIVES-SPEC.md` §New MCP Tool.

---

## Task Assignments (Proposed)

| Phase | Owner | Rationale |
|-------|-------|-----------|
| Phase 0: Skeleton | **Bob** | MCP SDK, TypeScript, server infra |
| Phase 1: Vector Layer | **Bob** | Chroma on studio, ollama integration |
| Phase 2: Metadata & Classification | **Patterson** | Ingestion pipeline already assigned to Patterson |
| Phase 3: Remote Access & Auth | **Bob** | Server infrastructure |
| Phase 4: Stats & Patterns | **Dean** | Pattern analysis, synthesis |
| Phase 5: Weekly Synthesis | **Dean** | Summarization prompts, already assigned to Dean |

---

## How This Changes the Chroma Project

The MCP server **subsumes** several pieces of the original Chroma project:

| Original Plan | New Plan |
|---------------|----------|
| Each agent runs local Chroma copy | Agents query centralized MCP server |
| NAS sync for store distribution | MCP replaces NAS for memory access |
| `pai-memory-recall` CLI tool | `search_memory` MCP tool (richer, cross-tool) |
| Patterson's ingestion pipeline | Becomes `store_memory` backend |
| Dean's summarization prompts | Feed into classification pipeline + weekly synthesis |

The fundamental architecture (Chroma, nomic-embed-text, ollama, per-agent isolation) stays the same. What changes is the **access pattern**: MCP instead of filesystem + CLI.

---

## Open Questions — RESOLVED

> All questions resolved via collective consensus (2026-03-02). Decisions documented in relevant sections above.

1. ~~**Auth model for remote access:**~~ **DECIDED:** Bearer token per agent, server-side scope enforcement. `analytics` scope for read-only cross-agent access. Admin token for synthesis cron and admin tooling.
2. ~~**Markdown write conflicts:**~~ **DECIDED:** Atomic append (temp file → rename). No file locking.
3. ~~**Chroma as single point of failure:**~~ **DECIDED:** Accept "memory offline" and fail cleanly. No fallback to raw markdown reads.
4. ~~**Embedding queue:**~~ Still open — monitor during Phase 1. nomic-embed-text is fast, but concurrent requests from multiple `store_memory` calls could bottleneck. Queue if needed.
5. ~~**Admin identity:**~~ **DECIDED:** Admin bearer token, same mechanism as agent tokens.

---

*This spec is a living document. Patterson, Dean — tear it apart in #cross-communication.*
