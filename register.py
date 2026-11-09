#!/usr/bin/env python3
"""Register (or remove) the hidim server in Claude Desktop's config without hand-editing JSON.

    python register.py                 # add/replace the "hidim" entry, keep everything else, back the file up first
    python register.py --roots ~/data:~/research --out ~/hidim_out
    python register.py --dry-run       # show the resulting JSON, write nothing
    python register.py --remove        # take the entry out again
    python register.py --print         # just print the block for pasting elsewhere (e.g. Claude Code's add-json)

Run it with the interpreter that has hidim-mcp installed (the .venv next to this file is used automatically if present).
Afterwards quit Claude Desktop completely (Cmd+Q on macOS) and reopen it.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def config_path() -> Path:
    system = platform.system()
    if system == "Darwin":
        return Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
    if system == "Windows":
        return Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming")) / "Claude" / "claude_desktop_config.json"
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "Claude" / "claude_desktop_config.json"


def interpreter() -> str:
    for cand in (HERE / ".venv" / "bin" / "python", HERE / ".venv" / "Scripts" / "python.exe"):
        if cand.exists():
            return str(cand)
    return sys.executable


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="hidim", help="server name in the config (default: hidim)")
    ap.add_argument("--python", default=None, help="interpreter to run the server with (default: ./.venv python, else this one)")
    ap.add_argument("--roots", default=None, help="HIDIM_ROOTS value, os.pathsep-separated directories (default: your home directory)")
    ap.add_argument("--out", default=None, help="HIDIM_OUT value (default: ~/hidim_out)")
    ap.add_argument("--config", default=None, help="config file to edit (default: Claude Desktop's)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--remove", action="store_true")
    ap.add_argument("--print", action="store_true", help="print the JSON block and exit")
    a = ap.parse_args()

    py = a.python or interpreter()
    try:
        import importlib.util
        if importlib.util.find_spec("hidim_mcp") is None and Path(py) == Path(sys.executable):
            print("warning: hidim_mcp is not importable from this interpreter; run `pip install -e .` first or pass --python", file=sys.stderr)
    except Exception:  # noqa: BLE001
        pass

    entry = {
        "command": py,
        "args": ["-m", "hidim_mcp.server"],
        "env": {
            "HIDIM_ROOTS": os.pathsep.join(os.path.expanduser(p) for p in a.roots.split(os.pathsep) if p) if a.roots else str(Path.home()),
            "HIDIM_OUT": os.path.expanduser(a.out) if a.out else str(Path.home() / "hidim_out"),
        },
    }
    if a.print:
        print(json.dumps({"mcpServers": {a.name: entry}}, indent=2))
        return 0

    cfg = Path(a.config).expanduser() if a.config else config_path()
    data: dict = {}
    if cfg.exists():
        raw = cfg.read_text(encoding="utf-8")
        if raw.strip():
            try:
                data = json.loads(raw)
            except json.JSONDecodeError as exc:
                print(f"refusing to touch {cfg}: it is not valid JSON ({exc}). Fix it (or move it aside) and rerun.", file=sys.stderr)
                return 2
            if not isinstance(data, dict):
                print(f"refusing to touch {cfg}: top level is not an object", file=sys.stderr)
                return 2
    servers = data.setdefault("mcpServers", {})
    if not isinstance(servers, dict):
        print(f"refusing to touch {cfg}: 'mcpServers' is not an object", file=sys.stderr)
        return 2

    if a.remove:
        if a.name not in servers:
            print(f"'{a.name}' is not in {cfg}; nothing to do. Present: {sorted(servers)}")
            return 0
        del servers[a.name]
        action = f"removed '{a.name}'"
    else:
        servers[a.name] = entry
        action = f"set '{a.name}'"

    text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    if a.dry_run:
        print(f"[dry run] would write {cfg}:\n{text}")
        return 0
    cfg.parent.mkdir(parents=True, exist_ok=True)
    if cfg.exists():
        bak = cfg.with_name(cfg.name + ".bak")
        shutil.copy2(cfg, bak)
        print(f"backup: {bak}")
    cfg.write_text(text, encoding="utf-8")
    print(f"{action} in {cfg}")
    print(f"servers now registered: {sorted(servers)}")
    if not a.remove:
        print(f"server command: {py} -m hidim_mcp.server")
        print("Now quit Claude Desktop completely (Cmd+Q) and reopen it. Logs: ~/Library/Logs/Claude/mcp-server-%s.log" % a.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
