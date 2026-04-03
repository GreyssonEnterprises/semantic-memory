/**
 * Topic document scanner.
 *
 * Scans workspace memory/topics/ for markdown docs whose filenames
 * fuzzy-match the user's prompt, extracts Key Conclusions and Open
 * Questions sections, and returns scored matches.
 */

import { readdir, readFile } from "node:fs/promises";
import { join, basename } from "node:path";
import type { TopicMatch, Logger } from "../types.js";

// ── helpers ──────────────────────────────────────────────────────────

/** Cheap tokeniser: lowercase, strip punctuation, split on whitespace. */
function extractTokens(text: string): Set<string> {
  return new Set(
    text
      .toLowerCase()
      .replace(/[^a-z0-9\s-]/g, " ")
      .split(/\s+/)
      .filter((t) => t.length > 2) // skip tiny words
  );
}

/**
 * Score a filename against a set of prompt tokens.
 * Returns 0-1 where 1 = every filename token appeared in the prompt.
 */
function scoreFilename(filename: string, promptTokens: Set<string>): number {
  // strip extension, split on dashes / underscores
  const slug = filename.replace(/\.md$/i, "");
  const fileTokens = slug
    .toLowerCase()
    .split(/[-_\s]+/)
    .filter((t) => t.length > 2);
  if (fileTokens.length === 0) return 0;

  let hits = 0;
  for (const ft of fileTokens) {
    for (const pt of promptTokens) {
      // substring match — "memory" matches "semantic-memory"
      if (pt.includes(ft) || ft.includes(pt)) {
        hits++;
        break;
      }
    }
  }
  return hits / fileTokens.length;
}

// ── section extraction ───────────────────────────────────────────────

const SECTION_RE =
  /^##\s+(Key Conclusions|Open Questions|Current Understanding)\s*$/im;

/**
 * Pull the content under a specific H2 heading, stopping at the next H2
 * or end-of-file.  Returns empty string if section not found.
 */
function extractSection(markdown: string, heading: string): string {
  const re = new RegExp(
    `^##\\s+${heading}\\s*$([\\s\\S]*?)(?=^##\\s|$(?!\\s))`,
    "im"
  );
  const m = markdown.match(re);
  if (!m) return "";
  return m[1].trim().slice(0, 500); // hard cap per section
}

/** Pull the first H1 title from the doc. */
function extractTitle(markdown: string): string {
  const m = markdown.match(/^#\s+(.+)$/m);
  return m ? m[1].trim() : "";
}

// ── public API ───────────────────────────────────────────────────────

export async function scanTopicDocuments(
  prompt: string,
  topicsDir: string,
  maxResults: number,
  logger: Logger
): Promise<TopicMatch[]> {
  let files: string[];
  try {
    files = await readdir(topicsDir);
  } catch (err: unknown) {
    const code = (err as NodeJS.ErrnoException).code;
    if (code === "ENOENT") {
      logger.debug?.(`Topics dir does not exist yet: ${topicsDir}`);
    } else {
      logger.warn(`Failed to read topics dir: ${topicsDir}`, err);
    }
    return [];
  }

  const mdFiles = files.filter((f) => f.endsWith(".md"));
  if (mdFiles.length === 0) return [];

  const promptTokens = extractTokens(prompt);
  if (promptTokens.size === 0) return [];

  // Score all files, keep those above threshold
  const scored: { file: string; score: number }[] = [];
  for (const file of mdFiles) {
    const score = scoreFilename(file, promptTokens);
    if (score >= 0.3) {
      scored.push({ file, score });
    }
  }

  // Sort by score descending, take top N
  scored.sort((a, b) => b.score - a.score);
  const top = scored.slice(0, maxResults);

  if (top.length === 0) return [];

  // Read & extract sections in parallel
  const results = await Promise.all(
    top.map(async ({ file, score }): Promise<TopicMatch | null> => {
      try {
        const fullPath = join(topicsDir, file);
        const content = await readFile(fullPath, "utf-8");
        const title = extractTitle(content) || file.replace(/\.md$/, "");
        const keyConclusions =
          extractSection(content, "Key Conclusions") ||
          extractSection(content, "Current Understanding");
        const openQuestions = extractSection(content, "Open Questions");

        // Skip if we couldn't extract anything useful
        if (!keyConclusions && !openQuestions) return null;

        return {
          filename: file,
          relativePath: `memory/topics/${file}`,
          title,
          keyConclusions,
          openQuestions,
          score,
        };
      } catch (err) {
        logger.warn(`Failed to read topic doc: ${file}`, err);
        return null;
      }
    })
  );

  return results.filter((r): r is TopicMatch => r !== null);
}
