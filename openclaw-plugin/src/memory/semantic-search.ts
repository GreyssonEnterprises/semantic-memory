/**
 * Semantic memory search — pai-memory-recall primary, ChromaDB fallback.
 *
 * Strategy 1 (PRIMARY): Shell out to `pai-memory-recall --json`.
 *   This leverages the full retrieval pipeline:
 *   - ACT-R base-level activation (recency × frequency × spaced repetition)
 *   - Source-type weighted boost (vault 2×, procedures 1.5×, graph 1.3×)
 *   - Hebbian co-activation boost (memories that fire together wire together)
 *   - Access tracking (every retrieval updates access_count + timestamps)
 *   - Blended mode (segments + summaries merged and re-ranked)
 *
 * Strategy 2 (FALLBACK): Direct Ollama embeddings → ChromaDB HTTP API.
 *   Raw cosine similarity without ACT-R/Hebbian/weights.
 *   Used only when pai-memory-recall CLI is unavailable.
 *
 * Both strategies are wrapped in try/catch for graceful degradation.
 */

import type {
  GreyssonMemoryConfig,
  MemoryResult,
  OllamaEmbeddingResponse,
  ChromaQueryResponse,
  Logger,
} from "../types.js";

// ── Strategy 1: pai-memory-recall CLI (PRIMARY) ──────────────────────

/** JSON output shape from `pai-memory-recall --json`. */
interface PaiRecallResult {
  score: number;
  distance: number;
  source: "segment" | "summary";
  date: string;
  domain: string;
  topic: string;
  session_id: string;
  text: string;
}

/**
 * Query via pai-memory-recall CLI — the full retrieval pipeline.
 *
 * Uses `--json` for structured output and `--mode blended` for
 * combined segment + summary search with re-ranking.
 */
async function queryViaPaiRecall(
  prompt: string,
  nResults: number,
  logger: Logger,
  agentId: string
): Promise<MemoryResult[]> {
  try {
    const { execFile } = await import("node:child_process");
    const { promisify } = await import("node:util");
    const execFileAsync = promisify(execFile);

    const args = [
      prompt,
      "--json",
      "--mode", "blended",
      "--top", String(nResults),
      "--agent", agentId,
    ];

    const { stdout } = await execFileAsync("pai-memory-recall", args, {
      timeout: 10_000,
      env: {
        ...process.env,
        OLLAMA_URL: process.env.OLLAMA_URL || "http://localhost:11434",
        SEMANTIC_MEMORY_STORE:
          process.env.SEMANTIC_MEMORY_STORE ||
          "/data/greysson-memory/semantic-memory/vectors",
        SEMANTIC_MEMORY_AGENT: agentId,
        PATH: `/root/.local/bin:/usr/local/bin:/usr/bin:/bin:${process.env.PATH || ""}`,
      },
    });

    const parsed = JSON.parse(stdout) as PaiRecallResult[];

    return parsed.map((r, i) => ({
      id: `recall-${r.session_id?.slice(0, 8) || i}`,
      content: r.text || "",
      score: r.score ?? 0,
      metadata: {
        date: r.date,
        domain: r.domain,
        topic: r.topic,
        source: r.source,
        session_id: r.session_id,
        distance: r.distance,
      },
    }));
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : String(err);
    if (msg.includes("ENOENT")) {
      logger.debug?.("pai-memory-recall not found in PATH");
    } else if (msg.includes("TIMEOUT") || msg.includes("timed out")) {
      logger.warn("pai-memory-recall timed out (10s)");
    } else {
      logger.warn(`pai-memory-recall failed: ${msg}`);
    }
    return [];
  }
}

// ── Strategy 2: Direct Ollama → ChromaDB (FALLBACK) ─────────────────

async function getEmbedding(
  text: string,
  config: Pick<GreyssonMemoryConfig, "ollamaBaseUrl" | "embeddingModel">,
  logger: Logger
): Promise<number[] | null> {
  const url = `${config.ollamaBaseUrl}/api/embeddings`;
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 5000);

  try {
    const resp = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: config.embeddingModel,
        prompt: text,
      }),
      signal: controller.signal,
    });

    if (!resp.ok) {
      logger.warn(
        `Ollama embedding request failed: ${resp.status} ${resp.statusText}`
      );
      return null;
    }

    const data = (await resp.json()) as OllamaEmbeddingResponse;
    return data.embedding;
  } catch (err: unknown) {
    if ((err as Error).name === "AbortError") {
      logger.warn("Ollama embedding request timed out (5s)");
    } else {
      logger.debug?.("Ollama embedding unavailable (fallback path)");
    }
    return null;
  } finally {
    clearTimeout(timeout);
  }
}

async function queryChromaHttp(
  embedding: number[],
  nResults: number,
  logger: Logger
): Promise<ChromaQueryResponse | null> {
  const baseUrl = "http://localhost:8000";
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 5000);

  try {
    const collectionsResp = await fetch(`${baseUrl}/api/v1/collections`, {
      signal: controller.signal,
    });
    if (!collectionsResp.ok) {
      logger.debug?.("ChromaDB HTTP API not available");
      return null;
    }

    const collections = (await collectionsResp.json()) as Array<{
      id: string;
      name: string;
    }>;

    const collection = collections.find(
      (c) =>
        c.name === "bob" ||
        c.name === "memory" ||
        c.name === "bob-memory" ||
        c.name === "semantic-memory" ||
        c.name === "bob_segments" ||
        c.name === "bob_summaries"
    );

    if (!collection) {
      logger.debug?.(
        `No matching ChromaDB collection found. Available: ${collections.map((c) => c.name).join(", ")}`
      );
      return null;
    }

    const queryResp = await fetch(
      `${baseUrl}/api/v1/collections/${collection.id}/query`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query_embeddings: [embedding],
          n_results: nResults,
          include: ["documents", "distances", "metadatas"],
        }),
        signal: controller.signal,
      }
    );

    if (!queryResp.ok) {
      logger.warn(
        `ChromaDB query failed: ${queryResp.status} ${queryResp.statusText}`
      );
      return null;
    }

    return (await queryResp.json()) as ChromaQueryResponse;
  } catch (err: unknown) {
    if ((err as Error).name === "AbortError") {
      logger.debug?.("ChromaDB HTTP query timed out (5s)");
    } else {
      logger.debug?.("ChromaDB HTTP API not reachable");
    }
    return null;
  } finally {
    clearTimeout(timeout);
  }
}

// ── public API ───────────────────────────────────────────────────────

export async function searchSemanticMemory(
  prompt: string,
  config: GreyssonMemoryConfig,
  logger: Logger,
  agentId: string = "bob"
): Promise<MemoryResult[]> {
  // ── Strategy 1 (PRIMARY): pai-memory-recall CLI ────────────────
  // Full pipeline: ACT-R activation + source weights + Hebbian boost
  const cliResults = await queryViaPaiRecall(
    prompt,
    config.maxMemoryResults,
    logger,
    agentId
  );

  const cliFiltered = cliResults.filter(
    (r) => r.score >= config.minMemoryScore
  );

  if (cliFiltered.length > 0) {
    logger.info(
      `Semantic memory: ${cliFiltered.length} results via pai-memory-recall (top score: ${cliFiltered[0].score.toFixed(3)})`
    );
    return cliFiltered;
  }

  // ── Strategy 2 (FALLBACK): Direct Ollama → ChromaDB HTTP ──────
  // Raw cosine similarity without ACT-R/Hebbian/weights
  logger.debug?.(
    "pai-memory-recall returned no results — falling back to direct ChromaDB"
  );

  const embedding = await getEmbedding(prompt, config, logger);

  if (embedding) {
    const chromaResults = await queryChromaHttp(
      embedding,
      config.maxMemoryResults,
      logger
    );

    if (chromaResults && chromaResults.ids[0]?.length > 0) {
      const results: MemoryResult[] = [];
      const ids = chromaResults.ids[0];
      const docs = chromaResults.documents[0];
      const distances = chromaResults.distances[0];
      const metas = chromaResults.metadatas[0];

      for (let i = 0; i < ids.length; i++) {
        const doc = docs[i];
        if (!doc) continue;

        // ChromaDB returns L2 distances; convert to a 0-1 similarity score.
        // For normalised embeddings, distance ∈ [0, 2], so sim = 1 - d/2.
        const distance = distances[i] ?? 0;
        const score = Math.max(0, 1 - distance / 2);

        if (score < config.minMemoryScore) continue;

        results.push({
          id: ids[i],
          content: doc,
          score,
          metadata: metas[i] ?? undefined,
        });
      }

      if (results.length > 0) {
        logger.info(
          `Semantic memory: ${results.length} results via ChromaDB fallback (top score: ${results[0].score.toFixed(3)})`
        );
        return results;
      }
    }
  }

  logger.debug?.("Semantic memory: no relevant results from any strategy");
  return [];
}
