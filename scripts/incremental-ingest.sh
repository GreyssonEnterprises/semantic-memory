#!/usr/bin/env bash
# incremental-ingest.sh — Ingest new sessions for local agent
# Run via OpenClaw cron. Designed for studio (grimm's machine).
#
# Flow:
#   1. Ingest new sessions for Bob
#   2. Backup all stores to NAS

set -euo pipefail

PROJECT_DIR="$HOME/clawd/projects/semantic-memory"
cd "$PROJECT_DIR"

LOG_FILE="data/logs/ingest-$(date +%Y-%m-%d).log"
mkdir -p data/logs

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG_FILE"; }

log "=== Incremental ingest starting ==="

# Step 1: Ingest new sessions for agent
# The ingest script skips already-embedded sessions (checks by session_id in Chroma)

log "Ingesting Bob sessions..."
BOB_SESSION_DIR="$HOME/.openclaw/agents/main/sessions"
uv run python -m src.pipeline.ingest \
  --agent bob \
  --session-dir "$BOB_SESSION_DIR" \
  --mode topic \
  --min-messages 4 2>>"$LOG_FILE" || log "WARNING: Bob ingest failed"

# Step 2: Backup to NAS
NAS_BACKUP="/Volumes/public/projects/business/warehouse/semantic-memory-cross-agent/data/vectors-backup"
if [ -d "/Volumes/public/" ]; then
  log "Backing up stores to NAS..."
  mkdir -p "$NAS_BACKUP"
  rsync -az --no-group --no-owner data/vectors/ "$NAS_BACKUP/" 2>>"$LOG_FILE" \
    || log "WARNING: NAS backup failed"
else
  log "NAS not mounted, skipping backup"
fi

# Summary
BOB_SEGS=$(uv run python -c "
from src.pipeline.store import VectorStore
s = VectorStore('bob')
print(s.segments.count())
" 2>/dev/null || echo "?")

log "=== Ingest complete. Segments: bob=$BOB_SEGS ==="
echo "Incremental ingest complete. bob=$BOB_SEGS segments."
