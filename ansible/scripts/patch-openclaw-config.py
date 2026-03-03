#!/usr/bin/env python3
"""Patch OpenClaw JSON5 config files idempotently.

OpenClaw configs use JSON5-ish syntax (unquoted keys, single-quoted strings,
trailing commas). This script parses with json5, applies a deep-merge patch
to a specific config path, optionally removes keys, and writes back valid JSON
(which OpenClaw also accepts).

Usage:
    patch-openclaw-config.py <config_path> <json_patch> [--remove-keys key1,key2] [--config-path dotted.path]
    
Example:
    patch-openclaw-config.py ~/.openclaw/openclaw.json \
        '{"provider":"ollama","enabled":true}' \
        --config-path agents.defaults.memorySearch \
        --remove-keys remote

Exit codes:
    0 = patched successfully (or already at desired state)
    1 = error
    2 = config file not found
"""

import argparse
import copy
import json
import sys
from pathlib import Path

try:
    import json5
except ImportError:
    print("ERROR: json5 package required. Install with: uv pip install json5", file=sys.stderr)
    sys.exit(1)


def deep_merge(base: dict, patch: dict) -> dict:
    """Deep merge patch into base. Patch values win on conflict."""
    result = copy.deepcopy(base)
    for key, value in patch.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def get_nested(d: dict, path: str) -> dict:
    """Get a nested dict value by dotted path."""
    parts = path.split(".")
    current = d
    for part in parts:
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def set_nested(d: dict, path: str, value) -> None:
    """Set a nested dict value by dotted path, creating intermediates."""
    parts = path.split(".")
    current = d
    for part in parts[:-1]:
        if part not in current or not isinstance(current[part], dict):
            current[part] = {}
        current = current[part]
    current[parts[-1]] = value


def main():
    parser = argparse.ArgumentParser(description="Patch OpenClaw JSON5 config files")
    parser.add_argument("config_path", help="Path to openclaw.json")
    parser.add_argument("patch_json", help="JSON patch to apply")
    parser.add_argument("--config-path", dest="config_dotpath", default=None,
                        help="Dotted path within config to patch (e.g. agents.defaults.memorySearch)")
    parser.add_argument("--remove-keys", dest="remove_keys", default=None,
                        help="Comma-separated keys to remove from the target path")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would change without writing")
    parser.add_argument("--backup", action="store_true", default=True,
                        help="Create .bak backup before writing (default: true)")
    args = parser.parse_args()

    config_file = Path(args.config_path).expanduser()
    if not config_file.exists():
        print(f"ERROR: Config file not found: {config_file}", file=sys.stderr)
        sys.exit(2)

    # Parse existing config (JSON5)
    try:
        with open(config_file, "r") as f:
            config = json5.load(f)
    except Exception as e:
        print(f"ERROR: Failed to parse config: {e}", file=sys.stderr)
        sys.exit(1)

    # Parse the patch
    try:
        patch = json.loads(args.patch_json)
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid patch JSON: {e}", file=sys.stderr)
        sys.exit(1)

    # Apply patch at the specified path (or root)
    if args.config_dotpath:
        existing = get_nested(config, args.config_dotpath)
        if existing is None:
            existing = {}
        merged = deep_merge(existing, patch)
        
        # Remove specified keys
        if args.remove_keys:
            for key in args.remove_keys.split(","):
                key = key.strip()
                if key in merged:
                    del merged[key]
                    print(f"  Removed key: {args.config_dotpath}.{key}")
        
        set_nested(config, args.config_dotpath, merged)
    else:
        config = deep_merge(config, patch)
        if args.remove_keys:
            for key in args.remove_keys.split(","):
                key = key.strip()
                if key in config:
                    del config[key]
                    print(f"  Removed key: {key}")

    if args.dry_run:
        print("DRY RUN — would write:")
        target = get_nested(config, args.config_dotpath) if args.config_dotpath else config
        print(json.dumps(target, indent=2))
        return

    # Backup
    if args.backup:
        backup_path = config_file.with_suffix(".json.bak")
        import shutil
        shutil.copy2(config_file, backup_path)
        print(f"  Backup: {backup_path}")

    # Write back as formatted JSON (OpenClaw accepts both JSON and JSON5)
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")
    
    print(f"  Patched: {config_file}")
    if args.config_dotpath:
        result = get_nested(config, args.config_dotpath)
        print(f"  {args.config_dotpath} =")
        print(json.dumps(result, indent=4))


if __name__ == "__main__":
    main()
