/**
 * before_prompt_build hook — the unified memory injector.
 *
 * Runs on every turn (except heartbeat/cron), performing:
 *   1. Topic document scanning (filename fuzzy-match)
 *   2. Semantic memory search (ChromaDB via Ollama embeddings)
 *   3. Context assembly (prependContext + appendSystemContext)
 *
 * Layer 1 (LCM) is handled by @martian-engineering/lossless-claw and
 * is deliberately NOT duplicated here.
 */

import { join } from "node:path";
import { scanTopicDocuments } from "../memory/topic-scanner.js";
import { searchSemanticMemory } from "../memory/semantic-search.js";
import type { GreyssonMemoryConfig, TopicMatch, MemoryResult, Logger } from "../types.js";

// ── heartbeat indexing ────────────────────────────────────────────────

async function runIncrementalIngest(agentId: string, logger: Logger): Promise<void> {
  try {
    const { execFile } = await import("node:child_process");
    const { promisify } = await import("node:util");
    const { existsSync } = await import("node:fs");
    const execFileAsync = promisify(execFile);

    const home = process.env.HOME || "/home/openclaw";
    const sessionDir = `${home}/.openclaw/agents/${agentId}/sessions`;

    // Use writable copy on state PVC if available (OCP), else image path (LXC)
    const smWritable = `${home}/.openclaw/semantic-memory`;
    const smDir = existsSync(`${smWritable}/.venv`) ? smWritable : "/opt/semantic-memory";

    const { stdout, stderr } = await execFileAsync(
      "uv",
      [
        "run", "python", "-m", "src.pipeline.ingest",
        "--agent", agentId,
        "--session-dir", sessionDir,
        "--mode", "topic",
        "--min-messages", "4",
      ],
      {
        cwd: smDir,
        timeout: 60_000,
        env: {
          ...process.env,
          PATH: `${home}/.local/bin:/usr/local/bin:/usr/bin:/bin:${process.env.PATH || ""}`,
          SEMANTIC_MEMORY_STORE:
            process.env.SEMANTIC_MEMORY_STORE ||
            "/data/greysson-memory/semantic-memory/vectors",
          EMBEDDING_PROVIDER: process.env.EMBEDDING_PROVIDER || "openai",
          OPENAI_API_KEY:
            process.env.OPENAI_API_KEY ||
            process.env.OPENAI_API_KEY_EMBEDDINGS || "",
          UV_PYTHON_INSTALL_DIR:
            process.env.UV_PYTHON_INSTALL_DIR ||
            `${home}/.local/share/uv/python`,
        },
      }
    );

    if (stdout.trim()) {
      logger.info(`[memory-ingest] ${agentId}: ${stdout.trim()}`);
    }
    if (stderr.trim()) {
      logger.debug?.(`[memory-ingest] ${agentId} stderr: ${stderr.trim()}`);
    }
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : String(err);
    if (msg.includes("ENOENT")) {
      logger.debug?.("[memory-ingest] uv or /opt/semantic-memory not found — skipping");
    } else if (msg.includes("TIMEOUT") || msg.includes("timed out")) {
      logger.warn("[memory-ingest] indexing timed out (60s)");
    } else {
      logger.warn(`[memory-ingest] failed: ${msg}`);
    }
  }
}

// ── static system context (cacheable) ────────────────────────────────

const CRYSTALLIZATION_REMINDER = `## Semantic Crystallization Checkpoint
After substantive exchanges: search memory/topics/ for existing docs on discussed topics.
Create or update topic documents at natural inflection points (conclusions, decisions, understanding shifts).
Template: memory/templates/topic-document.md`;

// ── context formatting ───────────────────────────────────────────────

function formatTopicMatches(matches: TopicMatch[]): string {
  if (matches.length === 0) return "";

  const lines = ["## 📂 Relevant Topic Documents"];

  for (const m of matches) {
    lines.push(`### ${m.title} → \`${m.relativePath}\``);
    if (m.keyConclusions) {
      lines.push(`**Conclusions:** ${m.keyConclusions}`);
    }
    if (m.openQuestions) {
      lines.push(`**Open Qs:** ${m.openQuestions}`);
    }
  }

  return lines.join("\n");
}

function formatMemoryResults(results: MemoryResult[]): string {
  if (results.length === 0) return "";

  const lines = ["## 🧠 Recalled Memory"];

  for (const r of results) {
    // Truncate individual chunks to keep total size reasonable
    const snippet = r.content.length > 300
      ? r.content.slice(0, 297) + "..."
      : r.content;
    lines.push(`- (${r.score.toFixed(2)}) ${snippet}`);
  }

  return lines.join("\n");
}

function assembleContext(
  topicSection: string,
  memorySection: string,
  maxChars: number
): string {
  const parts: string[] = [];

  if (topicSection) parts.push(topicSection);
  if (memorySection) parts.push(memorySection);

  if (parts.length === 0) return "";

  let combined = parts.join("\n\n");

  // Hard truncation to stay within budget
  if (combined.length > maxChars) {
    combined = combined.slice(0, maxChars - 3) + "...";
  }

  return combined;
}

// ── hook types (matching OpenClaw plugin SDK) ────────────────────────

interface BeforePromptBuildEvent {
  prompt: string;
  messages: unknown[];
}

interface AgentContext {
  agentId?: string;
  sessionKey?: string;
  sessionId?: string;
  workspaceDir?: string;
  trigger?: string;
  channelId?: string;
}

interface BeforePromptBuildResult {
  systemPrompt?: string;
  prependContext?: string;
  prependSystemContext?: string;
  appendSystemContext?: string;
}

// ── exported hook factory ────────────────────────────────────────────

export function createBeforePromptBuildHook(
  getConfig: () => GreyssonMemoryConfig,
  logger: Logger,
  resolvePath: (rel: string) => string
) {
  return async function beforePromptBuild(
    event: BeforePromptBuildEvent,
    ctx: AgentContext
  ): Promise<BeforePromptBuildResult> {
    const config = getConfig();

    // ── bail-out checks ──────────────────────────────────────────

    if (!config.enabled) {
      return {};
    }

    // On heartbeat: run indexing instead of search (no memory injection needed)
    if (ctx.trigger === "heartbeat") {
      const agentId = ctx.agentId || "bob";
      logger.debug?.(`Heartbeat trigger — running incremental ingest for ${agentId}`);
      await runIncrementalIngest(agentId, logger);
      return {};
    }

    // Skip other non-interactive triggers (cron, dreaming, etc.)
    if (ctx.trigger && config.skipTriggers.includes(ctx.trigger)) {
      logger.debug?.(`Skipping memory injection for trigger: ${ctx.trigger}`);
      return {};
    }

    const prompt = event.prompt;
    if (!prompt || prompt.trim().length < 5) {
      return {};
    }

    // ── resolve paths ────────────────────────────────────────────

    const workspaceDir = ctx.workspaceDir || resolvePath(".");
    const topicsDir = join(workspaceDir, config.topicsDir);

    // ── parallel search ──────────────────────────────────────────

    const agentId = ctx.agentId || "bob";

    const [topicMatches, memoryResults] = await Promise.all([
      scanTopicDocuments(prompt, topicsDir, config.maxTopicDocs, logger)
        .catch((err) => {
          logger.warn("Topic scan failed", err);
          return [] as TopicMatch[];
        }),
      searchSemanticMemory(prompt, config, logger, agentId)
        .catch((err) => {
          logger.warn("Semantic search failed", err);
          return [] as MemoryResult[];
        }),
    ]);

    // ── assemble result ──────────────────────────────────────────

    const topicSection = formatTopicMatches(topicMatches);
    const memorySection = formatMemoryResults(memoryResults);
    const prependContext = assembleContext(
      topicSection,
      memorySection,
      config.maxPrependChars
    );

    const result: BeforePromptBuildResult = {};

    if (prependContext) {
      result.prependContext = prependContext;
      logger.info(
        `Memory injected: ${topicMatches.length} topics, ${memoryResults.length} memories (${prependContext.length} chars)`
      );
    }

    if (config.crystallizationReminder) {
      result.appendSystemContext = CRYSTALLIZATION_REMINDER;
    }

    return result;
  };
}
