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


def search(query: str, top_k: int = None, diverse: bool = False) -> list[dict]:
    """Most relevant memories for `query`, best first.

    diverse=True spreads the picks: nearest-neighbour search hands back a
    cluster — the same promise kept four times, six notes that all say
    "resonance" — and twenty slots fill with one thought. Here each pick is
    weighed against what is already chosen (maximal marginal relevance:
    MEMORY_MMR_LAMBDA of relevance to the query, the rest a penalty for
    resembling a memory already in), so a query about one person reaches
    twenty DIFFERENT things about them instead of the nearest twenty."""
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
            vec = json.loads(emb)
            score = _cosine(qvec, vec)
            scored.append(
                {"id": mid, "kind": kind, "text": text, "created": created, "score": score, "_vec": vec}
            )
    scored.sort(key=lambda m: m["score"], reverse=True)
    picked = _mmr(scored, top_k) if diverse else scored[:top_k]
    for m in scored:
        m.pop("_vec", None)
    return picked


def _mmr(scored: list[dict], top_k: int) -> list[dict]:
    """Greedy maximal marginal relevance over the nearest 3×top_k. A
    candidate that is a near-copy of one already chosen (closer than
    MEMORY_DUP_THRESHOLD — the same line `remember` draws for "not twice")
    is set aside outright: relevance alone would always seat the twin, since
    a copy of the best match is itself a best match. Set-aside twins fill
    the tail only if nothing else is left."""
    lam = float(getattr(config, "MEMORY_MMR_LAMBDA", 0.75))
    dup = float(getattr(config, "MEMORY_DUP_THRESHOLD", 0) or 0) or 2.0  # 2.0: never
    pool = scored[: max(top_k * 3, top_k)]
    chosen: list[dict] = []
    twins: list[dict] = []
    while pool and len(chosen) < top_k:
        best, best_val = None, -9.0
        for m in pool:
            nearest = max((_cosine(m["_vec"], c["_vec"]) for c in chosen), default=0.0)
            if nearest >= dup:
                continue
            val = lam * m["score"] - (1.0 - lam) * nearest
            if val > best_val:
                best, best_val = m, val
        if best is None:
            break
        chosen.append(best)
        pool.remove(best)
        twins.extend(m for m in pool if max((_cosine(m["_vec"], c["_vec"]) for c in chosen), default=0.0) >= dup)
        pool = [m for m in pool if m not in twins]
    return (chosen + twins)[:top_k]


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
