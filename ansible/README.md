# ansible/

Idempotent OpenClaw config management across Circle Zero PAI nodes (shale, studio, caroline).

## Quick Start

```bash
# Install deps
uv pip install json5 ansible

# Dry run (check mode)
cd ansible/
ansible-playbook -i inventory.yml openclaw-memory.yml --check

# Apply to all hosts
ansible-playbook -i inventory.yml openclaw-memory.yml

# Apply to single host
ansible-playbook -i inventory.yml openclaw-memory.yml --limit shale
```

## What It Does

**`openclaw-memory.yml`** — Patches `memorySearch` config on each PAI:
- Switches `provider` from `"openai"` (shim) to native `"ollama"`
- Removes obsolete `remote` block (baseUrl/apiKey shim pointing at Ollama)
- Validates config with `openclaw config validate` (v2026.3.2+)
- Restarts gateway only if changes were made
- Idempotent: safe to run multiple times

## Architecture

```
shale (192.168.1.25)    → Patterson  → ~/.openclaw/openclaw.json
studio (192.168.1.22)   → Bob        → ~/.openclaw/openclaw.json
caroline                → Dean       → ~/.openclaw/openclaw.json
                                        ↓
                              All share Ollama @ 192.168.1.21:11434
                              (nomic-embed-text for embeddings)
```

Three independent OpenClaw gateways, one shared Ollama backend.
Each playbook run patches all three configs identically.

## Adding New Config Changes

1. Add a new vars file in `vars/`
2. Create a new playbook (or add tasks to existing)
3. Use `scripts/patch-openclaw-config.py` for JSON5-safe config patching
4. Always validate + conditional restart

## Config Patcher

`scripts/patch-openclaw-config.py` handles OpenClaw's JSON5 configs:
- Parses JSON5 (unquoted keys, trailing commas, comments)
- Deep-merges patches at any dotted path
- Removes specified keys
- Creates `.bak` backup before writing
- Writes back as standard JSON (OpenClaw accepts both)

**Note:** Comments in the original JSON5 config are not preserved after patching.
The output is valid JSON which OpenClaw handles identically.
