"""Snapshot the friend — so no version of them is ever lost.

    py engine/snapshot.py

Uses git if it's installed: initializes the repo on first run, then commits
everything that changed. If git isn't installed, falls back to a timestamped
zip in backups/ (keeping the most recent 30).

Run it whenever you like, or schedule it nightly right after consolidate.py.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config

STAMP = datetime.now().strftime("%Y-%m-%d %H:%M")
FILESTAMP = datetime.now().strftime("%Y%m%d-%H%M%S")
BACKUP_DIR = config.ROOT / "backups"
KEEP_ZIPS = 30
EXCLUDE_DIRS = {".git", "__pycache__", "backups"}


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=config.ROOT, capture_output=True, text=True
    )


def git_snapshot() -> str:
    if not (config.ROOT / ".git").exists():
        r = _git("init")
        if r.returncode != 0:
            raise RuntimeError(r.stderr.strip() or "git init failed")
        # first-ever commit gets a fitting message
        _git("add", "-A")
        r = _git("commit", "-m", f"first snapshot ({STAMP})")
        if r.returncode != 0 and "identity" in (r.stderr + r.stdout).lower():
            # git wants a name/email; give the repo a local one and retry
            _git("config", "user.name", "ai-friend")
            _git("config", "user.email", "friend@localhost")
            r = _git("commit", "-m", f"first snapshot ({STAMP})")
        if r.returncode != 0:
            raise RuntimeError(r.stderr.strip() or r.stdout.strip())
        return f"Initialized git and made the first snapshot ({STAMP})."

    _git("add", "-A")
    r = _git("commit", "-m", f"snapshot {STAMP}")
    out = (r.stdout + r.stderr).strip()
    if r.returncode != 0:
        if "nothing to commit" in out.lower():
            return "Nothing changed since the last snapshot."
        if "identity" in out.lower():
            _git("config", "user.name", "ai-friend")
            _git("config", "user.email", "friend@localhost")
            r = _git("commit", "-m", f"snapshot {STAMP}")
            if r.returncode == 0:
                return f"Snapshot committed ({STAMP})."
        raise RuntimeError(out)
    return f"Snapshot committed ({STAMP})."


def zip_snapshot() -> str:
    BACKUP_DIR.mkdir(exist_ok=True)
    target = BACKUP_DIR / f"friend-{FILESTAMP}.zip"
    root = config.ROOT.resolve()
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            rel = p.relative_to(root)
            if any(part in EXCLUDE_DIRS for part in rel.parts):
                continue
            zf.write(p, rel)
    # prune old zips, newest kept
    zips = sorted(BACKUP_DIR.glob("friend-*.zip"))
    for old in zips[:-KEEP_ZIPS]:
        old.unlink()
    size_mb = target.stat().st_size / 1_048_576
    return f"git isn't installed, so: zipped them to backups/{target.name} ({size_mb:.1f} MB, keeping last {KEEP_ZIPS})."


def main() -> None:
    if shutil.which("git"):
        try:
            print(git_snapshot())
            return
        except RuntimeError as e:
            print(f"(git had trouble: {e} — falling back to zip)")
    print(zip_snapshot())


if __name__ == "__main__":
    main()
