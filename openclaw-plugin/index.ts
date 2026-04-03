/**
 * greysson-memory — OpenClaw plugin entry point.
 *
 * Unifies semantic memory (ChromaDB/Ollama) and topic crystallization
 * into a single before_prompt_build hook. Does NOT duplicate LCM
 * conversation compaction (handled by @martian-engineering/lossless-claw).
 */

import { createBeforePromptBuildHook } from "./src/hooks/before-prompt-build.js";
import { DEFAULT_CONFIG } from "./src/types.js";
import type { GreyssonMemoryConfig } from "./src/types.js";
import type { OpenClawPluginApi } from "openclaw/plugin-sdk";

export default {
  id: "greysson-memory",

  configSchema: {
    type: "object" as const,
    properties: {
      enabled: { type: "boolean" as const, default: true },
      topicsDir: { type: "string" as const, default: "memory/topics" },
      chromaDbPath: {
        type: "string" as const,
        default: "semantic-memory/vectors",
      },
      ollamaBaseUrl: {
        type: "string" as const,
        default: "http://localhost:11434",
      },
      embeddingModel: { type: "string" as const, default: "nomic-embed-text" },
      maxTopicDocs: { type: "number" as const, default: 3 },
      maxMemoryResults: { type: "number" as const, default: 5 },
      minMemoryScore: { type: "number" as const, default: 0.5 },
      maxPrependChars: { type: "number" as const, default: 2000 },
      crystallizationReminder: { type: "boolean" as const, default: true },
      skipTriggers: {
        type: "array" as const,
        items: { type: "string" as const },
        default: ["cron"],
      },
    },
  },

  register(api: OpenClawPluginApi) {
    const logger = api.logger;

    /** Merge plugin config with defaults. */
    function getConfig(): GreyssonMemoryConfig {
      const raw = (api.pluginConfig ?? {}) as Partial<GreyssonMemoryConfig>;
      return { ...DEFAULT_CONFIG, ...raw };
    }

    const hook = createBeforePromptBuildHook(
      getConfig,
      logger,
      api.resolvePath.bind(api)
    );

    // Note: registerHook's TS type uses InternalHookHandler (single-arg), but
    // the runtime dispatches before_prompt_build hooks with (event, ctx).
    // This is a known SDK type gap — the cast is safe and matches runtime behavior.
    api.registerHook(
      "before_prompt_build",
      hook as unknown as Parameters<typeof api.registerHook>[1]
    );

    logger.info("greysson-memory plugin registered (before_prompt_build hook)");
  },
};
