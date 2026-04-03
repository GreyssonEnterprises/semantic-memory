/** Greysson Memory plugin configuration. */
export interface GreyssonMemoryConfig {
  enabled: boolean;
  topicsDir: string;
  chromaDbPath: string;
  ollamaBaseUrl: string;
  embeddingModel: string;
  maxTopicDocs: number;
  maxMemoryResults: number;
  minMemoryScore: number;
  maxPrependChars: number;
  crystallizationReminder: boolean;
  skipTriggers: string[];
}

/** Defaults — mirrors openclaw.plugin.json configSchema defaults. */
export const DEFAULT_CONFIG: GreyssonMemoryConfig = {
  enabled: true,
  topicsDir: "memory/topics",
  chromaDbPath: "semantic-memory/vectors",
  ollamaBaseUrl: "http://localhost:11434",
  embeddingModel: "nomic-embed-text",
  maxTopicDocs: 3,
  maxMemoryResults: 5,
  minMemoryScore: 0.5,
  maxPrependChars: 2000,
  crystallizationReminder: true,
  skipTriggers: ["cron"],
};

/** A matched topic document with extracted sections. */
export interface TopicMatch {
  filename: string;
  relativePath: string;
  title: string;
  keyConclusions: string;
  openQuestions: string;
  score: number; // 0-1, higher = better match
}

/** A semantic memory search result. */
export interface MemoryResult {
  id: string;
  content: string;
  score: number; // cosine similarity, higher = more similar
  metadata?: Record<string, unknown>;
}

/** Ollama embedding API response. */
export interface OllamaEmbeddingResponse {
  embedding: number[];
}

/** ChromaDB query response shape (simplified). */
export interface ChromaQueryResponse {
  ids: string[][];
  documents: (string | null)[][];
  distances: number[][];
  metadatas: (Record<string, unknown> | null)[][];
}

/** Minimal logger interface matching OpenClaw's api.logger. */
export interface Logger {
  info(msg: string, ...args: unknown[]): void;
  warn(msg: string, ...args: unknown[]): void;
  error(msg: string, ...args: unknown[]): void;
  debug?(msg: string, ...args: unknown[]): void;
}
