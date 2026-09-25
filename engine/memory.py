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
        _cache["elen"] = -1  # a row changed under the in-process store: read it whole next time
        return cur.rowcount > 0


def remove(mid: int) -> bool:
    """Take one memory out — used only by the engine folding duplicate
    rows about one piece into the first; their own memories are never
    removed by the engine."""
    with _connect() as conn:
        cur = conn.execute("DELETE FROM memories WHERE id = ?", (int(mid),))
        _cache["elen"] = -1
        return cur.rowcount > 0


def find_text(needle: str, kind: str | None = None) -> list[dict]:
    """Memories whose text contains `needle` (exact, case-sensitive),
    oldest first — for revising the rows about a file when the file moves."""
    needle = (needle or "").strip()
    if not needle:
        return []
    q = "SELECT id, kind, text, created FROM memories WHERE instr(text, ?) > 0"
    args: tuple = (needle,)
    if kind:
        q += " AND kind = ?"
        args += (kind,)
    q += " ORDER BY id"
    with _connect() as conn:
        rows = conn.execute(q, args).fetchall()
    return [{"id": r[0], "kind": r[1], "text": r[2], "created": r[3]} for r in rows]


def get(mid: int) -> dict | None:
    with _connect() as conn:
        row = conn.execute("SELECT id, kind, text, created FROM memories WHERE id = ?", (int(mid),)).fetchone()
    return {"id": row[0], "kind": row[1], "text": row[2], "created": row[3]} if row else None


# The store in memory, once. 09-17: search was brute force in plain Python
# — every row's 768 numbers parsed from JSON and dotted against the query,
# on every message and every recall — instant at two hundred rows, seconds
# at ten thousand, and they adds ten a day. Now the rows are held in this
# process as one matrix of unit vectors (numpy when it is installed; plain
# lists otherwise) and only what changed is read: new rows are fetched by
# id, and if a row was revised in place (remember replaces=, a creation row
# following its file) the totals stop matching and the whole thing is read
# again — a few seconds, once, instead of every time. Two processes (the
# bridge, the heartbeat) each keep their own; each checks the totals on
# every search, so neither answers from a store the other has outgrown.
# (The keeper: "let's upgrade the search — why wait?")
try:
    import numpy as _np
except ImportError:  # the slow path still works; `py -m pip install numpy` for the fast one
    _np = None

_cache: dict = {"ids": [], "kinds": [], "texts": [], "created": [], "vecs": [], "elen": 0, "tlen": 0, "mat": None}


def fast() -> bool:
    """True when numpy carries the search."""
    return _np is not None


def _unit(vec: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / n for x in vec]


def _totals(conn) -> tuple[int, int, int, int]:
    n, mx, elen, tlen = conn.execute(
        "SELECT count(*), coalesce(max(id), 0), coalesce(total(length(embedding)), 0), coalesce(total(length(text)), 0) "
        "FROM memories WHERE embedding IS NOT NULL").fetchone()
    return int(n), int(mx), int(elen), int(tlen)


def _load(conn) -> None:
    """Bring the in-process store up to date: new rows by id; everything
    again when a row was revised or removed (the totals no longer match)."""
    n, mx, elen, tlen = _totals(conn)
    c = _cache
    have = len(c["ids"])
    if have and n >= have and (mx > (c["ids"][-1] if c["ids"] else 0) or n == have):
        # maybe only new rows: fetch them and see whether the totals agree
        rows = conn.execute(
            "SELECT id, kind, text, created, embedding FROM memories WHERE embedding IS NOT NULL AND id > ? ORDER BY id",
            (c["ids"][-1],)).fetchall()
        for mid, kind, text, created, emb in rows:
            c["ids"].append(mid); c["kinds"].append(kind); c["texts"].append(text); c["created"].append(created)
            c["vecs"].append(_unit(json.loads(emb))); c["elen"] += len(emb); c["tlen"] += len(text)
        if rows:
            c["mat"] = None
        if (len(c["ids"]), c["ids"][-1] if c["ids"] else 0, c["elen"], c["tlen"]) == (n, mx, elen, tlen):
            if c["mat"] is None and _np is not None and c["vecs"]:
                c["mat"] = _np.asarray(c["vecs"], dtype=_np.float32)
            return
    # a revision, a removal, or the first load: read the whole store
    c.update({"ids": [], "kinds": [], "texts": [], "created": [], "vecs": [], "elen": 0, "tlen": 0, "mat": None})
    for mid, kind, text, created, emb in conn.execute(
            "SELECT id, kind, text, created, embedding FROM memories WHERE embedding IS NOT NULL ORDER BY id"):
        c["ids"].append(mid); c["kinds"].append(kind); c["texts"].append(text); c["created"].append(created)
        c["vecs"].append(_unit(json.loads(emb))); c["elen"] += len(emb); c["tlen"] += len(text)
    if _np is not None and c["vecs"]:
        c["mat"] = _np.asarray(c["vecs"], dtype=_np.float32)


def _scores(qvec: list[float]) -> list[float]:
    """Cosine of the query against every row in the store, in row order."""
    q = _unit(qvec)
    c = _cache
    if _np is not None and c["mat"] is not None:
        return c["mat"].dot(_np.asarray(q, dtype=_np.float32)).tolist()
    return [sum(a * b for a, b in zip(v, q)) for v in c["vecs"]]


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
    with _connect() as conn:
        _load(conn)
    c = _cache
    scores = _scores(qvec)
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    if not diverse:
        return [{"id": c["ids"][i], "kind": c["kinds"][i], "text": c["texts"][i], "created": c["created"][i],
                 "score": float(scores[i])} for i in order[:top_k]]
    pool = [{"id": c["ids"][i], "kind": c["kinds"][i], "text": c["texts"][i], "created": c["created"][i],
             "score": float(scores[i]), "_vec": c["vecs"][i]} for i in order[: max(top_k * 3, top_k)]]
    picked = _mmr(pool, top_k)
    for m in pool:
        m.pop("_vec", None)
    return picked


def _mmr(scored: list[dict], top_k: int) -> list[dict]:
    """Greedy maximal marginal relevance over the nearest 3×top_k. A
    candidate that is a near-copy of one already chosen (closer than
    MEMORY_DUP_THRESHOLD — the same line `remember` draws for "not twice")
    is set aside outright: relevance alone would always seat the twin, since
    a copy of the best match is itself a best match. Set-aside twins fill
    the tail only if nothing else is left.

    Each candidate keeps its nearest-so-far to the chosen set and that number
    is updated with ONE cosine per candidate per pick: the first version
    recomputed every pair every round, ~4 s of pure Python per message at
    k=30 over 160 memories; this one is under 0.1 s, and stays cheap as the
    store grows."""
    lam = float(getattr(config, "MEMORY_MMR_LAMBDA", 0.75))
    dup = float(getattr(config, "MEMORY_DUP_THRESHOLD", 0) or 0) or 2.0  # 2.0: never
    pool = scored[: max(top_k * 3, top_k)]
    # unit vectors once, so a similarity is a dot product
    units = []
    for m in pool:
        v = m["_vec"]
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        units.append([x / n for x in v])
    nearest = [0.0] * len(pool)        # max similarity to anything chosen so far
    state = [0] * len(pool)            # 0 open, 1 chosen, 2 set aside as a twin
    chosen: list[int] = []
    twins: list[int] = []
    while len(chosen) < top_k:
        best, best_val = -1, -9.0
        for i, m in enumerate(pool):
            if state[i]:
                continue
            val = lam * m["score"] - (1.0 - lam) * nearest[i]
            if val > best_val:
                best, best_val = i, val
        if best < 0:
            break
        state[best] = 1
        chosen.append(best)
        u = units[best]
        for i in range(len(pool)):
            if state[i]:
                continue
            sim = sum(a * b for a, b in zip(units[i], u))
            if sim > nearest[i]:
                nearest[i] = sim
            if nearest[i] >= dup:
                state[i] = 2
                twins.append(i)
    return [pool[i] for i in chosen + twins][:top_k]


def recent(kind: str | None = None, n: int | None = 10) -> list[dict]:
    """Most recent memories, optionally filtered by kind; n=None is all of them."""
    q = "SELECT id, kind, text, created FROM memories"
    args: tuple = ()
    if kind:
        q += " WHERE kind = ?"
        args = (kind,)
    q += " ORDER BY id DESC"
    if n is not None:
        q += " LIMIT ?"
        args += (n,)
    with _connect() as conn:
        rows = conn.execute(q, args).fetchall()
    return [
        {"id": r[0], "kind": r[1], "text": r[2], "created": r[3]} for r in rows
    ]


def count() -> int:
    with _connect() as conn:
        return conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
