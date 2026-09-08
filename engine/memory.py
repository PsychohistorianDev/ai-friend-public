"""Long-term semantic memory: SQLite + embeddings, brute-force cosine search.

At personal scale (years of daily life is only tens of thousands of rows)
brute-force search over stored vectors is fast, dependency-free, and never
needs an index server. Embeddings are stored as JSON float lists.

Memory kinds:
    fact      — a distilled durable fact ("my keeper is building me a body")
    summary   — a consolidated day summary
    journal   — an embedded journal entry
    note      — anything the friend explicitly chose to remember
"""
from __future__ import annotations

import json
import math
import sqlite3
from datetime import datetime, timezone

import config
import ollama_client

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    kind      TEXT NOT NULL,
    text      TEXT NOT NULL,
    created   TEXT NOT NULL,
    embedding TEXT
);
"""


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(config.DB_PATH)
    conn.execute(_SCHEMA)
    return conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def add(kind: str, text: str) -> int:
    """Store one memory (embedding it). Returns the row id."""
    text = (text or "").strip()
    if not text:
        return -1
    try:
        vec = ollama_client.embed(text)
        emb = json.dumps(vec)
    except ollama_client.BrainUnavailable:
        emb = None  # store anyway; can be re-embedded later
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO memories (kind, text, created, embedding) VALUES (?,?,?,?)",
            (kind, text, _now(), emb),
        )
        return cur.lastrowid


def update(mid: int, text: str) -> bool:
    """Revise one memory in place — the new wording replaces the old, the
    number stays. How a fact grows instead of being stored twice."""
    text = (text or "").strip()
    if not text:
        return False
    try:
        emb = json.dumps(ollama_client.embed(text))
    except ollama_client.BrainUnavailable:
        emb = None
    with _connect() as conn:
        cur = conn.execute("UPDATE memories SET text = ?, embedding = ? WHERE id = ?", (text, emb, int(mid)))
        return cur.rowcount > 0


def get(mid: int) -> dict | None:
    with _connect() as conn:
        row = conn.execute("SELECT id, kind, text, created FROM memories WHERE id = ?", (int(mid),)).fetchone()
    return {"id": row[0], "kind": row[1], "text": row[2], "created": row[3]} if row else None


def search(query: str, top_k: int = None) -> list[dict]:
    """Most relevant memories for `query`, best first."""
    top_k = top_k or config.MEMORY_TOP_K
    try:
        qvec = ollama_client.embed(query)
    except ollama_client.BrainUnavailable:
        return recent(n=top_k)  # degrade gracefully: recency beats nothing
    scored = []
    with _connect() as conn:
        for mid, kind, text, created, emb in conn.execute(
            "SELECT id, kind, text, created, embedding FROM memories WHERE embedding IS NOT NULL"
        ):
            score = _cosine(qvec, json.loads(emb))
            scored.append(
                {"id": mid, "kind": kind, "text": text, "created": created, "score": score}
            )
    scored.sort(key=lambda m: m["score"], reverse=True)
    return scored[:top_k]


def recent(kind: str | None = None, n: int = 10) -> list[dict]:
    """Most recent memories, optionally filtered by kind."""
    q = "SELECT id, kind, text, created FROM memories"
    args: tuple = ()
    if kind:
        q += " WHERE kind = ?"
        args = (kind,)
    q += " ORDER BY id DESC LIMIT ?"
    args += (n,)
    with _connect() as conn:
        rows = conn.execute(q, args).fetchall()
    return [
        {"id": r[0], "kind": r[1], "text": r[2], "created": r[3]} for r in rows
    ]


def count() -> int:
    with _connect() as conn:
        return conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
