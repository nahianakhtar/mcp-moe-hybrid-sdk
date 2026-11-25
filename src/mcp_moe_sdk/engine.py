"""Core engine: PII Redaction Guard, SQLite WAL FTS5 + Cosine Vector RRF Vault, Sparse Top-2 MoE Router & MCP 2025-11-25 Dispatcher."""

from __future__ import annotations

import base64
import copy
import hashlib
import html
import json
import math
import re
import secrets
import sqlite3
import struct
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

_FTS_OPEN = "\x01"
_FTS_CLOSE = "\x02"


def _build_crc32c_table() -> tuple[int, ...]:
    table: list[int] = []
    for i in range(256):
        crc = i
        for _ in range(8):
            crc = ((crc >> 1) ^ 0x82F63B78) if (crc & 1) else (crc >> 1)
        table.append(crc & 0xFFFFFFFF)
    return tuple(table)


_CRC32C_TABLE: tuple[int, ...] = _build_crc32c_table()


def crc32c(data: bytes | bytearray | memoryview | str) -> int:
    """Compute reflected Castagnoli CRC32C (0x82F63B78), matching C11 par_crc32c (empty -> 0)."""
    raw = data.encode("utf-8") if isinstance(data, str) else bytes(data)
    if not raw:
        return 0
    crc = 0xFFFFFFFF
    table = _CRC32C_TABLE
    for b in raw:
        crc = table[(crc ^ b) & 0xFF] ^ (crc >> 8)
    return (~crc) & 0xFFFFFFFF


def _extract_fts5_citation_spans(marked_text: str, field: str) -> list[dict[str, Any]]:
    """Extract exact 0-based [start, end) character offsets from SQLite FTS5 highlight() output."""
    spans: list[dict[str, Any]] = []
    cursor = 0
    plain_pos = 0
    n = len(marked_text)
    while cursor < n:
        open_idx = marked_text.find(_FTS_OPEN, cursor)
        if open_idx == -1:
            break
        plain_pos += open_idx - cursor
        close_idx = marked_text.find(_FTS_CLOSE, open_idx + 1)
        if close_idx == -1:
            break
        term = marked_text[open_idx + 1 : close_idx]
        start = plain_pos
        end = start + len(term)
        spans.append({"field": field, "term": term, "start": start, "end": end})
        plain_pos = end
        cursor = close_idx + 1
    return spans


def _render_safe_highlighted_snippet(marked_snippet: str) -> str:
    """HTML-escape untrusted document text while converting FTS5 sentinels into <mark> tags."""
    escaped = html.escape(marked_snippet, quote=False)
    return escaped.replace(_FTS_OPEN, "<mark>").replace(_FTS_CLOSE, "</mark>")


class RedactionGuard:
    """Deterministic zero-leak sanitizer stripping API keys, emails, and bearer tokens before indexing or routing."""

    _RULES: ClassVar[list[tuple[str, re.Pattern[str]]]] = [
        ("[REDACTED_API_KEY]", re.compile(r"\b(?:sk|rk|pk|ghp|AIza)[-_A-Za-z0-9]{16,}\b")),
        ("[REDACTED_BEARER]", re.compile(r"(?i)Bearer\s+[A-Za-z0-9\-._~+/]+=*")),
        ("[REDACTED_EMAIL]", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ]

    @classmethod
    def sanitize(cls, text: str) -> tuple[str, int]:
        count = 0
        out = text
        for token, rx in cls._RULES:
            out, n = rx.subn(token, out)
            count += n
        return out, count


def deterministic_embed(text: str, dim: int = 64) -> list[float]:
    """Deterministic L2-normalized hashed character/word n-gram embedding for zero-dependency vector retrieval."""
    vec = [0.0] * dim
    tokens = re.findall(r"[a-z0-9_+#.-]{2,}", text.lower())
    if not tokens:
        tokens = [text.lower()[:16] or "empty"]
    for tok in tokens:
        grams = [tok] + [tok[i : i + 3] for i in range(max(0, len(tok) - 2))]
        for g in grams:
            digest = hashlib.blake2b(g.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(digest[:4], "little") % dim
            sign = 1.0 if (digest[4] & 1) == 0 else -1.0
            vec[idx] += sign * (1.5 if g == tok else 0.6)
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [round(v / norm, 6) for v in vec]


def cosine_sim(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


@dataclass(frozen=True)
class SearchHit:
    doc_id: str
    title: str
    snippet: str
    rrf_score: float
    lexical_rank: int | None
    vector_rank: int | None
    cosine: float
    highlighted_snippet: str = ""
    citation_spans: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "snippet": self.snippet,
            "highlighted_snippet": self.highlighted_snippet,
            "citation_spans": list(self.citation_spans),
            "rrf_score": self.rrf_score,
            "lexical_rank": self.lexical_rank,
            "vector_rank": self.vector_rank,
            "cosine": self.cosine,
        }


class HybridKnowledgeVault:
    """SQLite WAL + FTS5 BM25 + Dense Cosine Vector store fused via Reciprocal Rank Fusion (RRF, k=60)."""

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.db_path = str(db_path)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self.conn.execute("PRAGMA foreign_keys=ON;")
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                body TEXT NOT NULL,
                tags TEXT NOT NULL,
                embedding_json TEXT NOT NULL,
                redactions INTEGER NOT NULL DEFAULT 0,
                updated_unix REAL NOT NULL
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
                doc_id UNINDEXED,
                title,
                body,
                tags
            );
            """
        )
        self.conn.commit()

    def upsert(self, doc_id: str, title: str, body: str, tags: list[str] | None = None) -> dict[str, Any]:
        clean_title, r1 = RedactionGuard.sanitize(title.replace(_FTS_OPEN, "").replace(_FTS_CLOSE, ""))
        clean_body, r2 = RedactionGuard.sanitize(body.replace(_FTS_OPEN, "").replace(_FTS_CLOSE, ""))
        tag_str = " ".join(tags or [])
        emb = deterministic_embed(f"{clean_title} {clean_body} {tag_str}")
        now = time.time()
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO documents(doc_id, title, body, tags, embedding_json, redactions, updated_unix)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(doc_id) DO UPDATE SET
                    title=excluded.title,
                    body=excluded.body,
                    tags=excluded.tags,
                    embedding_json=excluded.embedding_json,
                    redactions=excluded.redactions,
                    updated_unix=excluded.updated_unix
                """,
                (doc_id, clean_title, clean_body, tag_str, json.dumps(emb), r1 + r2, now),
            )
            self.conn.execute("DELETE FROM documents_fts WHERE doc_id = ?", (doc_id,))
            self.conn.execute(
                "INSERT INTO documents_fts(doc_id, title, body, tags) VALUES (?, ?, ?, ?)",
                (doc_id, clean_title, clean_body, tag_str),
            )
        return {"doc_id": doc_id, "redactions_applied": r1 + r2, "embedding_dim": len(emb)}

    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        rrf_k: int = 60,
        *,
        cancel_token: CancellationToken | None = None,
        progress_reporter: ProgressReporter | None = None,
    ) -> list[SearchHit]:
        if cancel_token is not None:
            cancel_token.raise_if_cancelled()
        if progress_reporter is not None:
            progress_reporter.report(1, 3, "Scanning SQLite WAL FTS5 lexical BM25 index")
        clean_q, _ = RedactionGuard.sanitize(query)
        q_tokens = re.findall(r"[A-Za-z0-9_]+", clean_q)
        lex_ranks: dict[str, int] = {}
        fts_highlights: dict[str, tuple[str, str, str]] = {}
        if q_tokens:
            fts_expr = " OR ".join(f'"{t}"' for t in q_tokens[:12])
            rows = self.conn.execute(
                """
                SELECT
                    doc_id,
                    snippet(documents_fts, 2, char(1), char(2), '...', 24),
                    highlight(documents_fts, 1, char(1), char(2)),
                    highlight(documents_fts, 2, char(1), char(2))
                FROM documents_fts
                WHERE documents_fts MATCH ?
                ORDER BY bm25(documents_fts)
                LIMIT 50
                """,
                (fts_expr,),
            ).fetchall()
            for rank_idx, (did, raw_snip, hl_title, hl_body) in enumerate(rows, start=1):
                lex_ranks[did] = rank_idx
                fts_highlights[did] = (raw_snip or "", hl_title or "", hl_body or "")

        if cancel_token is not None:
            cancel_token.raise_if_cancelled()
        if progress_reporter is not None:
            progress_reporter.report(2, 3, "Scoring dense cosine vector embeddings")
        q_vec = deterministic_embed(clean_q)
        all_docs = self.conn.execute("SELECT doc_id, title, body, embedding_json FROM documents").fetchall()
        scored_vec: list[tuple[str, str, str, float]] = []
        for doc_idx, (did, title, body, emb_json) in enumerate(all_docs):
            if cancel_token is not None and (doc_idx & 63) == 0:
                cancel_token.raise_if_cancelled()
            sim = cosine_sim(q_vec, json.loads(emb_json))
            scored_vec.append((did, title, body, sim))
        scored_vec.sort(key=lambda item: item[3], reverse=True)
        vec_ranks = {item[0]: idx for idx, item in enumerate(scored_vec, start=1)}

        candidates: list[tuple[float, float, str, str, str, int | None, int]] = []
        for did, title, body, sim in scored_vec:
            l_rank = lex_ranks.get(did)
            v_rank = vec_ranks[did]
            score = (1.0 / (rrf_k + v_rank)) + ((1.25 / (rrf_k + l_rank)) if l_rank is not None else 0.0)
            candidates.append((round(score, 6), round(sim, 4), did, title, body, l_rank, v_rank))

        if cancel_token is not None:
            cancel_token.raise_if_cancelled()
        candidates.sort(key=lambda c: (c[0], c[1]), reverse=True)
        if progress_reporter is not None:
            progress_reporter.report(3, 3, "Fusing Reciprocal Rank Fusion rankings and citation spans")
        fused: list[SearchHit] = []
        for score, sim_rounded, did, title, body, l_rank, v_rank in candidates[: max(1, top_k)]:
            snippet = body[:180] + ("..." if len(body) > 180 else "")
            if did in fts_highlights:
                raw_snip, hl_title, hl_body = fts_highlights[did]
                hl_snippet = _render_safe_highlighted_snippet(raw_snip) if raw_snip else html.escape(snippet, quote=False)
                spans = _extract_fts5_citation_spans(hl_title, "title") + _extract_fts5_citation_spans(hl_body, "body")
            else:
                hl_snippet = html.escape(snippet, quote=False)
                spans = []
            fused.append(
                SearchHit(
                    doc_id=did,
                    title=title,
                    snippet=snippet,
                    rrf_score=score,
                    lexical_rank=l_rank,
                    vector_rank=v_rank,
                    cosine=sim_rounded,
                    highlighted_snippet=hl_snippet,
                    citation_spans=tuple(spans),
                )
            )
        return fused

    def get_document(self, doc_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT doc_id, title, body, tags, redactions, updated_unix FROM documents WHERE doc_id = ?",
            (doc_id,),
        ).fetchone()
        if row is None:
            return None
        did, title, body, tags_str, redactions, updated_unix = row
        return {
            "doc_id": did,
            "title": title,
            "body": body,
            "tags": [t for t in tags_str.split(" ") if t],
            "redactions": int(redactions),
            "updated_unix": float(updated_unix),
        }

    def list_documents(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT doc_id, title, tags, redactions, updated_unix FROM documents ORDER BY doc_id ASC"
        ).fetchall()
        return [
            {
                "doc_id": did,
                "title": title,
                "tags": [t for t in tags_str.split(" ") if t],
                "redactions": int(redactions),
                "updated_unix": float(updated_unix),
            }
            for did, title, tags_str, redactions, updated_unix in rows
        ]

    def count(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0])


class SparseTop2MoeRouter:
    """Hardware- & SLA-aware Sparse Top-2 Mixture-of-Experts query router with p50/p95 latency tracking."""

    TIERS: ClassVar[list[dict[str, Any]]] = [
        {
            "id": 0,
            "tier": "edge_fast_json",
            "expert": "small_dense_q8_vulkan",
            "vram_budget_gib": 1.0,
            "cost_per_1k_tok_usd": 0.0001,
            "keywords": ("json", "schema", "status", "ping", "health", "flag", "telemetry", "parse", "extract"),
        },
        {
            "id": 1,
            "tier": "rag_grounded_synthesis",
            "expert": "hybrid_mamba_rag_worker",
            "vram_budget_gib": 3.5,
            "cost_per_1k_tok_usd": 0.0004,
            "keywords": ("search", "rag", "document", "summarize", "retrieve", "fts5", "sqlite", "policy", "compare"),
        },
        {
            "id": 2,
            "tier": "code_and_systems_verifier",
            "expert": "coder_4b_strict_ast",
            "vram_budget_gib": 3.8,
            "cost_per_1k_tok_usd": 0.0006,
            "keywords": ("c++", "rust", "python", "sql", "compile", "kernel", "pointer", "pytest", "bug", "api", "docker"),
        },
        {
            "id": 3,
            "tier": "frontier_moe_math_architect",
            "expert": "sparse_20b_moe_hybrid_offload",
            "vram_budget_gib": 4.8,
            "cost_per_1k_tok_usd": 0.0015,
            "keywords": ("architect", "proof", "theorem", "convexity", "wacc", "black-scholes", "quantum", "moe", "invariant"),
        },
    ]

    def __init__(self) -> None:
        self.latencies_ms: list[float] = []

    def route(
        self,
        prompt: str,
        max_vram_gib: float = 6.0,
        latency_sla_ms: float = 1500.0,
        *,
        cancel_token: CancellationToken | None = None,
    ) -> dict[str, Any]:
        if cancel_token is not None:
            cancel_token.raise_if_cancelled()
        t0 = time.perf_counter()
        clean_p, redactions = RedactionGuard.sanitize(prompt)
        low = clean_p.lower()

        raw_logits = [0.15, 0.15, 0.15, 0.15]
        for idx, tier in enumerate(self.TIERS):
            hits = sum(1 for kw in tier["keywords"] if kw in low)
            raw_logits[idx] += hits * 1.65
            if tier["vram_budget_gib"] > max_vram_gib:
                raw_logits[idx] -= 4.0
        if latency_sla_ms <= 350.0:
            raw_logits[0] += 1.5
            raw_logits[3] -= 1.2

        m = max(raw_logits)
        exps = [math.exp(x - m) for x in raw_logits]
        denom = sum(exps) or 1.0
        probs = [e / denom for e in exps]

        top2_idx = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:2]
        top2_sum = sum(probs[i] for i in top2_idx) or 1.0
        top2_gates = [
            {"expert_id": i, "tier": self.TIERS[i]["tier"], "gate_weight": round(probs[i] / top2_sum, 4)}
            for i in top2_idx
        ]
        chosen = self.TIERS[top2_idx[0]]
        dt_ms = round((time.perf_counter() - t0) * 1000.0, 4)
        self.latencies_ms.append(dt_ms)

        return {
            "selected_tier": chosen["tier"],
            "selected_expert": chosen["expert"],
            "confidence": round(probs[top2_idx[0]], 4),
            "vram_budget_gib": chosen["vram_budget_gib"],
            "est_cost_per_1k_tok_usd": chosen["cost_per_1k_tok_usd"],
            "top2_sparse_gates": top2_gates,
            "redactions_applied": redactions,
            "router_latency_ms": dt_ms,
        }

    def stats(self) -> dict[str, float]:
        if not self.latencies_ms:
            return {"calls": 0, "p50_ms": 0.0, "p95_ms": 0.0}
        s = sorted(self.latencies_ms)
        p50 = s[len(s) // 2]
        p95 = s[min(len(s) - 1, int(len(s) * 0.95))]
        return {"calls": len(s), "p50_ms": round(p50, 4), "p95_ms": round(p95, 4)}


class SlidingWindowTokenRateLimiter:
    """Deterministic per-client sliding-window request & token-budget rate limiter with exact 1-ms Retry-After math."""

    def __init__(
        self,
        window_ms: int = 60_000,
        max_requests_per_window: int = 60,
        max_tokens_per_window: int = 12_000,
        max_tracked_clients: int = 1_024,
        now_fn: Callable[[], int] | None = None,
    ) -> None:
        self.window_ms = max(1, int(window_ms))
        self.max_requests_per_window = max(1, int(max_requests_per_window))
        self.max_tokens_per_window = max(1, int(max_tokens_per_window))
        self.max_tracked_clients = max(1, int(max_tracked_clients))
        self._now_fn: Callable[[], int] = now_fn or (lambda: time.monotonic_ns() // 1_000_000)
        self._clients: OrderedDict[str, deque[tuple[int, int]]] = OrderedDict()
        self._lock = threading.Lock()
        self._total_allowed = 0
        self._total_rejected = 0
        self._total_tokens_consumed = 0

    @staticmethod
    def _normalize_client(client_id: str | None) -> str:
        cleaned = str(client_id or "anonymous").strip()[:128]
        return cleaned or "anonymous"

    @staticmethod
    def _prune_entries(entries: deque[tuple[int, int]], cutoff: int) -> None:
        while entries and entries[0][0] <= cutoff:
            entries.popleft()

    def _evict_idle_or_oldest(self, cutoff: int) -> None:
        for cid in list(self._clients.keys()):
            bucket = self._clients[cid]
            self._prune_entries(bucket, cutoff)
            if not bucket:
                del self._clients[cid]
                if len(self._clients) < self.max_tracked_clients:
                    return
        if len(self._clients) >= self.max_tracked_clients:
            self._clients.popitem(last=False)

    def check(self, client_id: str = "anonymous", estimated_tokens: int = 1) -> dict[str, Any]:
        return self._evaluate(client_id, estimated_tokens, mutate=False)

    def consume(self, client_id: str = "anonymous", tokens: int = 1) -> dict[str, Any]:
        return self._evaluate(client_id, tokens, mutate=True)

    def _evaluate(self, raw_client_id: str, raw_tokens: float, *, mutate: bool) -> dict[str, Any]:
        with self._lock:
            now = int(self._now_fn())
            cutoff = now - self.window_ms
            key = self._normalize_client(raw_client_id)
            cost_tokens = max(0, math.ceil(float(raw_tokens)))

            entries = self._clients.get(key)
            if entries is not None:
                self._prune_entries(entries, cutoff)
                if not entries and not mutate:
                    del self._clients[key]
                    entries = None

            active = list(entries) if entries is not None else []
            used_requests = len(active)
            used_tokens = sum(tok for _, tok in active)

            if cost_tokens > self.max_tokens_per_window:
                if mutate:
                    self._total_rejected += 1
                    if entries is not None and not entries:
                        del self._clients[key]
                reset_after_ms = max(0, active[-1][0] + self.window_ms - now) if active else self.window_ms
                return {
                    "allowed": False,
                    "client_id": key,
                    "limit_requests": self.max_requests_per_window,
                    "remaining_requests": max(0, self.max_requests_per_window - used_requests),
                    "limit_tokens": self.max_tokens_per_window,
                    "remaining_tokens": max(0, self.max_tokens_per_window - used_tokens),
                    "used_requests": used_requests,
                    "used_tokens": used_tokens,
                    "requested_tokens": cost_tokens,
                    "retry_after_ms": self.window_ms,
                    "reset_after_ms": reset_after_ms,
                    "reason": "request_exceeds_window_capacity",
                }

            req_exceeded = used_requests + 1 > self.max_requests_per_window
            tok_exceeded = used_tokens + cost_tokens > self.max_tokens_per_window

            if req_exceeded or tok_exceeded:
                req_retry_ms = 0
                if req_exceeded and active:
                    need_idx = min(len(active) - 1, max(0, used_requests + 1 - self.max_requests_per_window - 1))
                    req_retry_ms = max(1, active[need_idx][0] + self.window_ms - now)

                tok_retry_ms = 0
                if tok_exceeded and active:
                    deficit = used_tokens + cost_tokens - self.max_tokens_per_window
                    freed = 0
                    for ts, tok in active:
                        freed += tok
                        if freed >= deficit:
                            tok_retry_ms = max(1, ts + self.window_ms - now)
                            break

                retry_after_ms = max(1, req_retry_ms, tok_retry_ms)
                reset_after_ms = max(0, active[-1][0] + self.window_ms - now) if active else 0
                if mutate:
                    self._total_rejected += 1
                    if entries is not None and not entries:
                        del self._clients[key]
                return {
                    "allowed": False,
                    "client_id": key,
                    "limit_requests": self.max_requests_per_window,
                    "remaining_requests": max(0, self.max_requests_per_window - used_requests),
                    "limit_tokens": self.max_tokens_per_window,
                    "remaining_tokens": max(0, self.max_tokens_per_window - used_tokens),
                    "used_requests": used_requests,
                    "used_tokens": used_tokens,
                    "requested_tokens": cost_tokens,
                    "retry_after_ms": retry_after_ms,
                    "reset_after_ms": reset_after_ms,
                    "reason": "sliding_window_tokens_exceeded" if tok_exceeded else "sliding_window_requests_exceeded",
                }

            if mutate:
                if entries is None:
                    if len(self._clients) >= self.max_tracked_clients:
                        self._evict_idle_or_oldest(cutoff)
                    entries = deque()
                    self._clients[key] = entries
                else:
                    self._clients.move_to_end(key)
                entries.append((now, cost_tokens))
                self._total_allowed += 1
                self._total_tokens_consumed += cost_tokens

            next_requests = used_requests + 1 if mutate else used_requests
            next_tokens = used_tokens + cost_tokens if mutate else used_tokens
            oldest_ts = entries[0][0] if (entries and len(entries) > 0) else now
            reset_after_ms = max(0, oldest_ts + self.window_ms - now) if (mutate or active) else 0
            return {
                "allowed": True,
                "client_id": key,
                "limit_requests": self.max_requests_per_window,
                "remaining_requests": max(0, self.max_requests_per_window - next_requests),
                "limit_tokens": self.max_tokens_per_window,
                "remaining_tokens": max(0, self.max_tokens_per_window - next_tokens),
                "used_requests": next_requests,
                "used_tokens": next_tokens,
                "requested_tokens": cost_tokens,
                "retry_after_ms": 0,
                "reset_after_ms": reset_after_ms,
                "reason": "allowed",
            }

    @staticmethod
    def to_headers(decision: dict[str, Any]) -> dict[str, str]:
        headers = {
            "X-RateLimit-Limit-Requests": str(decision["limit_requests"]),
            "X-RateLimit-Remaining-Requests": str(decision["remaining_requests"]),
            "X-RateLimit-Limit-Tokens": str(decision["limit_tokens"]),
            "X-RateLimit-Remaining-Tokens": str(decision["remaining_tokens"]),
            "X-RateLimit-Reset-Ms": str(decision["reset_after_ms"]),
        }
        if not decision["allowed"]:
            headers["Retry-After"] = str(max(1, math.ceil(decision["retry_after_ms"] / 1000.0)))
        return headers

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            now = int(self._now_fn())
            cutoff = now - self.window_ms
            for cid in list(self._clients.keys()):
                bucket = self._clients[cid]
                self._prune_entries(bucket, cutoff)
                if not bucket:
                    del self._clients[cid]
            return {
                "active_clients": len(self._clients),
                "window_ms": self.window_ms,
                "max_requests_per_window": self.max_requests_per_window,
                "max_tokens_per_window": self.max_tokens_per_window,
                "total_allowed": self._total_allowed,
                "total_rejected": self._total_rejected,
                "total_tokens_consumed": self._total_tokens_consumed,
            }


_LIGHTWEIGHT_MCP_METHODS = frozenset(
    {
        "initialize",
        "ping",
        "tools/list",
        "resources/list",
        "resources/templates/list",
        "resources/subscribe",
        "resources/unsubscribe",
        "prompts/list",
        "completion/complete",
        "logging/setLevel",
        "roots/list",
    }
)


def estimate_mcp_tokens(payload: Any) -> int:
    """Deterministically estimate token-budget cost for single or batched MCP JSON-RPC payloads."""
    if isinstance(payload, list):
        if not payload:
            return 1
        return max(1, sum(estimate_mcp_tokens(item) for item in payload))
    if isinstance(payload, dict):
        method = payload.get("method")
        if isinstance(method, str):
            if "id" not in payload:
                return 1
            if method in _LIGHTWEIGHT_MCP_METHODS:
                return 2
            params_raw = json.dumps(payload.get("params") or {}, separators=(",", ":"), ensure_ascii=False)
            return max(4, math.ceil(len(params_raw) / 4.0))
        return 1
    return 1


class RequestCancelledError(Exception):
    """Raised cooperatively when an in-flight or pre-cancelled MCP request is aborted."""

    def __init__(self, request_id: int | str | None, reason: str = "Cancelled by client") -> None:
        self.request_id = request_id
        self.reason = reason
        super().__init__(f"Request {request_id!r} cancelled: {reason}")


class CancellationToken:
    """Thread-safe cooperative cancellation token backed by threading.Event."""

    def __init__(self, request_id: int | str | None = None, *, non_cancellable: bool = False) -> None:
        self.request_id = request_id
        self.non_cancellable = non_cancellable
        self._event = threading.Event()
        self._reason = ""
        self._lock = threading.Lock()

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()

    @property
    def reason(self) -> str:
        return self._reason

    def cancel(self, reason: str = "Cancelled by client") -> bool:
        if self.non_cancellable:
            return False
        clean_reason, _ = RedactionGuard.sanitize(str(reason or "Cancelled by client").strip()[:256])
        with self._lock:
            if self._event.is_set():
                return False
            self._reason = clean_reason or "Cancelled by client"
            self._event.set()
            return True

    def wait(self, timeout: float | None = None) -> bool:
        return self._event.wait(timeout)

    def raise_if_cancelled(self) -> None:
        if self._event.is_set():
            raise RequestCancelledError(self.request_id, self._reason or "Cancelled by client")


class RequestCancellationRegistry:
    """MCP 2025-11-25 notifications/cancelled registry handling both in-flight and pre-arrival race conditions."""

    def __init__(
        self,
        max_tombstones: int = 512,
        tombstone_ttl_sec: float = 30.0,
        now_fn: Callable[[], float] | None = None,
    ) -> None:
        self.max_tombstones = max(1, int(max_tombstones))
        self.tombstone_ttl_sec = max(0.1, float(tombstone_ttl_sec))
        self._now_fn: Callable[[], float] = now_fn or time.monotonic
        self._inflight: dict[tuple[str, int | str], CancellationToken] = {}
        self._pre_cancelled: OrderedDict[tuple[str, int | str], tuple[float, str]] = OrderedDict()
        self._lock = threading.Lock()
        self._notifications_received = 0
        self._cancelled_count = 0
        self._ignored_non_cancellable = 0

    @staticmethod
    def _normalize_key(request_id: Any) -> tuple[str, int | str] | None:
        if request_id is None or isinstance(request_id, bool):
            return None
        if isinstance(request_id, int):
            return ("int", request_id)
        if isinstance(request_id, str):
            trimmed = request_id[:256]
            return ("str", trimmed) if trimmed else None
        return None

    def _prune_tombstones_locked(self, now: float) -> None:
        cutoff = now - self.tombstone_ttl_sec
        while self._pre_cancelled:
            oldest_key, (ts, _) = next(iter(self._pre_cancelled.items()))
            if ts <= cutoff:
                del self._pre_cancelled[oldest_key]
            else:
                break

    def begin(self, request_id: Any, *, non_cancellable: bool = False) -> CancellationToken:
        key = self._normalize_key(request_id)
        norm_id: int | str | None = key[1] if key is not None else None
        token = CancellationToken(norm_id, non_cancellable=non_cancellable)
        if key is None:
            return token
        with self._lock:
            now = float(self._now_fn())
            self._prune_tombstones_locked(now)
            if not non_cancellable and key in self._pre_cancelled:
                _, pre_reason = self._pre_cancelled.pop(key)
                if token.cancel(pre_reason):
                    self._cancelled_count += 1
                return token
            if non_cancellable and key in self._pre_cancelled:
                del self._pre_cancelled[key]
                self._ignored_non_cancellable += 1
            self._inflight[key] = token
            return token

    def finish(self, request_id: Any, token: CancellationToken | None = None) -> None:
        key = self._normalize_key(request_id)
        if key is None:
            return
        with self._lock:
            current = self._inflight.get(key)
            if current is not None and (token is None or current is token):
                del self._inflight[key]

    def cancel(self, request_id: Any, reason: str = "Cancelled by client") -> dict[str, Any]:
        with self._lock:
            self._notifications_received += 1
            now = float(self._now_fn())
            self._prune_tombstones_locked(now)
            key = self._normalize_key(request_id)
            if key is None:
                return {"cancelled": False, "status": "invalid_request_id"}
            clean_reason, _ = RedactionGuard.sanitize(str(reason or "Cancelled by client").strip()[:256])
            clean_reason = clean_reason or "Cancelled by client"

            active = self._inflight.get(key)
            if active is not None:
                if active.non_cancellable:
                    self._ignored_non_cancellable += 1
                    return {"cancelled": False, "status": "non_cancellable"}
                if active.cancel(clean_reason):
                    self._cancelled_count += 1
                return {"cancelled": True, "status": "cancelled_inflight", "reason": active.reason}

            if key in self._pre_cancelled:
                self._pre_cancelled.move_to_end(key)
            self._pre_cancelled[key] = (now, clean_reason)
            while len(self._pre_cancelled) > self.max_tombstones:
                self._pre_cancelled.popitem(last=False)
            return {"cancelled": True, "status": "recorded_prearrival_tombstone", "reason": clean_reason}

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            now = float(self._now_fn())
            self._prune_tombstones_locked(now)
            return {
                "inflight_requests": len(self._inflight),
                "pending_tombstones": len(self._pre_cancelled),
                "notifications_received": self._notifications_received,
                "cancelled_count": self._cancelled_count,
                "ignored_non_cancellable": self._ignored_non_cancellable,
            }


_URI_UNENCODED_BYTES: frozenset[int] = frozenset(
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~!$&'()*+,;=:@"
)
_HEX_CHARS: frozenset[int] = frozenset(b"0123456789ABCDEFabcdef")


class WorkspaceRootsGuard:
    """RFC 8089 canonical file:// URI normalizer & MCP 2025-11-25 roots/list workspace boundary confinement guard."""

    MAX_URI_LEN: int = 512
    MAX_SEGMENTS: int = 64

    ERR_URI_SYNTAX: int = -19
    ERR_URI_TRAVERSAL: int = -20
    ERR_URI_AUTHORITY: int = -21
    ERR_URI_OUTSIDE_ROOTS: int = -22

    def __init__(
        self,
        roots: list[dict[str, str]] | None = None,
        *,
        enforce_when_empty: bool = False,
    ) -> None:
        self.enforce_when_empty = bool(enforce_when_empty)
        self._lock = threading.Lock()
        self._roots: OrderedDict[str, str] = OrderedDict()
        self._epoch = 0
        self._checks_allowed = 0
        self._checks_denied = 0
        self._list_changed_emitted = 0
        self._client_list_changed_received = 0
        if roots:
            self.set_roots(roots)

    @staticmethod
    def _encode_segment(seg: bytes) -> str:
        out: list[str] = []
        for b in seg:
            if b in _URI_UNENCODED_BYTES:
                out.append(chr(b))
            else:
                out.append(f"%{b:02X}")
        return "".join(out)

    @classmethod
    def normalize_file_uri(cls, uri: Any) -> tuple[int, str, str]:
        """Canonicalize an RFC 8089 file:// URI with C11/Rust/WASM bit-identical error codes."""
        if not isinstance(uri, str):
            return (cls.ERR_URI_SYNTAX, "", "Invalid file URI: must be a string")
        raw = uri.encode("utf-8")
        if len(raw) < 8 or len(raw) > cls.MAX_URI_LEN:
            return (cls.ERR_URI_SYNTAX, "", "Invalid file URI length")
        for b in raw:
            if b < 0x20 or b == 0x7F or b == 0x5C:
                return (cls.ERR_URI_SYNTAX, "", "Invalid file URI: control character or backslash")
        if raw[:7].lower() != b"file://":
            return (cls.ERR_URI_SYNTAX, "", "Invalid file URI: missing file:// scheme")
        if b"?" in raw[7:] or b"#" in raw[7:]:
            return (cls.ERR_URI_SYNTAX, "", "Invalid file URI: query or fragment not permitted")

        slash_idx = raw.find(b"/", 7)
        if slash_idx == -1:
            return (cls.ERR_URI_SYNTAX, "", "Invalid file URI: missing absolute path slash")
        authority = raw[7:slash_idx]
        if authority and authority.lower() != b"localhost":
            return (cls.ERR_URI_AUTHORITY, "", "Unsupported non-local file:// authority")

        raw_path = raw[slash_idx + 1 :]
        segments: list[bytes] = []
        for part in raw_path.split(b"/"):
            if not part:
                continue
            decoded = bytearray()
            i = 0
            plen = len(part)
            while i < plen:
                ch = part[i]
                if ch == 0x25:  # '%'
                    if i + 2 >= plen or part[i + 1] not in _HEX_CHARS or part[i + 2] not in _HEX_CHARS:
                        return (cls.ERR_URI_SYNTAX, "", "Malformed percent-encoding in file URI")
                    val = int(part[i + 1 : i + 3].decode("ascii"), 16)
                    if val < 0x20 or val == 0x7F or val == 0x5C or val == 0x2F:
                        return (cls.ERR_URI_SYNTAX, "", "Forbidden encoded byte in file URI segment")
                    decoded.append(val)
                    i += 3
                else:
                    decoded.append(ch)
                    i += 1

            seg_bytes = bytes(decoded)
            if seg_bytes == b".":
                continue
            if seg_bytes == b"..":
                if not segments:
                    return (cls.ERR_URI_TRAVERSAL, "", "Relative '..' traversal escapes filesystem root")
                segments.pop()
                continue
            if len(segments) >= cls.MAX_SEGMENTS:
                return (cls.ERR_URI_SYNTAX, "", "File URI exceeds maximum segment depth")
            segments.append(seg_bytes)

        canonical = "file:///" if not segments else "file:///" + "/".join(cls._encode_segment(s) for s in segments)
        if len(canonical.encode("utf-8")) > cls.MAX_URI_LEN:
            return (cls.ERR_URI_SYNTAX, "", "Canonical file URI exceeds maximum length")
        return (0, canonical, "ok")

    @classmethod
    def is_within_root(cls, target_uri: str, root_uri: str) -> tuple[int, str, str]:
        """Verify target_uri resolves strictly inside root_uri without sibling-prefix or '..' confusion."""
        rc_r, canon_r, msg_r = cls.normalize_file_uri(root_uri)
        if rc_r != 0:
            return (rc_r, "", msg_r)
        rc_t, canon_t, msg_t = cls.normalize_file_uri(target_uri)
        if rc_t != 0:
            return (rc_t, "", msg_t)
        if canon_r == "file:///" or canon_t == canon_r or canon_t.startswith(canon_r + "/"):
            return (0, canon_t, "ok")
        return (cls.ERR_URI_OUTSIDE_ROOTS, canon_t, f"URI '{canon_t}' is outside workspace root '{canon_r}'")

    def set_roots(self, roots: list[dict[str, str]]) -> list[dict[str, str]]:
        normalized: OrderedDict[str, str] = OrderedDict()
        for entry in roots:
            if not isinstance(entry, dict) or "uri" not in entry:
                raise ValueError("Each workspace root must be a dict containing 'uri'")
            rc, canon_uri, msg = self.normalize_file_uri(entry["uri"])
            if rc != 0:
                raise ValueError(f"Invalid workspace root URI {entry.get('uri')!r}: {msg} ({rc})")
            raw_name = str(entry.get("name") or canon_uri.rsplit("/", 1)[-1] or "root").strip()[:128]
            clean_name, _ = RedactionGuard.sanitize(raw_name)
            normalized[canon_uri] = clean_name or "root"
        with self._lock:
            self._roots = normalized
            self._epoch += 1
            self._list_changed_emitted += 1
            return [{"uri": u, "name": n} for u, n in self._roots.items()]

    def record_client_list_changed(self) -> None:
        with self._lock:
            self._client_list_changed_received += 1

    def list_roots(self) -> list[dict[str, str]]:
        with self._lock:
            return [{"uri": u, "name": n} for u, n in self._roots.items()]

    def check_uri(self, uri: str) -> tuple[int, str, str]:
        rc, canon_t, msg = self.normalize_file_uri(uri)
        with self._lock:
            if rc != 0:
                self._checks_denied += 1
                return (rc, "", msg)
            if not self._roots:
                if self.enforce_when_empty:
                    self._checks_denied += 1
                    return (self.ERR_URI_OUTSIDE_ROOTS, canon_t, "No workspace roots are authorized")
                self._checks_allowed += 1
                return (0, canon_t, "ok")
            for root_uri in self._roots:
                if root_uri == "file:///" or canon_t == root_uri or canon_t.startswith(root_uri + "/"):
                    self._checks_allowed += 1
                    return (0, canon_t, "ok")
            self._checks_denied += 1
            return (self.ERR_URI_OUTSIDE_ROOTS, canon_t, f"URI '{canon_t}' is outside configured workspace roots")

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "roots": [{"uri": u, "name": n} for u, n in self._roots.items()],
                "root_count": len(self._roots),
                "epoch": self._epoch,
                "enforce_when_empty": self.enforce_when_empty,
                "checks_allowed": self._checks_allowed,
                "checks_denied": self._checks_denied,
                "list_changed_emitted": self._list_changed_emitted,
                "client_list_changed_received": self._client_list_changed_received,
            }


class ProgressReporter:
    """Active request-scoped MCP 2025-11-25 notifications/progress emitter enforcing strict monotonicity and lifecycle fencing."""

    def __init__(
        self,
        registry: ProgressTokenRegistry,
        key: tuple[str, int | str],
        raw_token: int | str,
        *,
        cancel_token: CancellationToken | None = None,
        emit_fn: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self._registry = registry
        self.key = key
        self.raw_token = raw_token
        self.cancel_token = cancel_token
        self._emit_fn = emit_fn
        self.emitted_notifications: list[dict[str, Any]] = []

    @property
    def updates_count(self) -> int:
        return len(self.emitted_notifications)

    def report(
        self,
        progress: float,
        total: float | None = None,
        message: str | None = None,
    ) -> tuple[int, dict[str, Any] | None]:
        rc, notif = self._registry.update(
            self.key,
            self.raw_token,
            progress,
            total=total,
            message=message,
            cancel_token=self.cancel_token,
        )
        if rc == 0 and notif is not None:
            self.emitted_notifications.append(notif)
            if self._emit_fn is not None:
                self._emit_fn(notif)
        return (rc, notif)

    def complete(self) -> int:
        return self._registry.complete(self.key)

    def abort(self) -> int:
        return self._registry.abort(self.key)


class ProgressTokenRegistry:
    """Thread-safe MCP 2025-11-25 _meta.progressToken tracker matching C11/Rust/WASM ParProgressTable invariants."""

    ERR_PROGRESS_DUPLICATE: int = -23
    ERR_PROGRESS_FULL: int = -24
    ERR_PROGRESS_KEY: int = -25
    ERR_PROGRESS_INACTIVE: int = -26
    ERR_PROGRESS_NON_MONOTONIC: int = -27
    ERR_PROGRESS_EXCEEDS_TOTAL: int = -28

    def __init__(self, max_active: int = 64) -> None:
        self.max_active = max(1, int(max_active))
        self._lock = threading.Lock()
        self._active: dict[tuple[str, int | str], dict[str, Any]] = {}
        self._finished: OrderedDict[tuple[str, int | str], str] = OrderedDict()
        self._streams_started = 0
        self._updates_emitted = 0
        self._updates_rejected = 0
        self._streams_completed = 0
        self._streams_aborted = 0

    @classmethod
    def normalize_key(cls, token: Any) -> tuple[int, tuple[str, int | str] | None, int | str | None]:
        if token is None or isinstance(token, bool):
            return (cls.ERR_PROGRESS_KEY, None, None)
        if isinstance(token, int):
            if token < 0 or token > 0xFFFFFFFFFFFFFFFF:
                return (cls.ERR_PROGRESS_KEY, None, None)
            return (0, ("int", token), token)
        if isinstance(token, str):
            raw = token.encode("utf-8")
            if not raw or len(raw) > 256:
                return (cls.ERR_PROGRESS_KEY, None, None)
            return (0, ("str", token), token)
        return (cls.ERR_PROGRESS_KEY, None, None)

    def begin(
        self,
        raw_token: Any,
        *,
        cancel_token: CancellationToken | None = None,
        emit_fn: Callable[[dict[str, Any]], None] | None = None,
    ) -> tuple[int, ProgressReporter | None, str]:
        rc, key, norm_tok = self.normalize_key(raw_token)
        if rc != 0 or key is None or norm_tok is None:
            return (self.ERR_PROGRESS_KEY, None, "Invalid _meta.progressToken: must be a non-empty string or uint64 integer")
        with self._lock:
            if key in self._active:
                return (
                    self.ERR_PROGRESS_DUPLICATE,
                    None,
                    f"Duplicate active _meta.progressToken: {norm_tok!r} is already in flight",
                )
            if len(self._active) >= self.max_active:
                return (self.ERR_PROGRESS_FULL, None, "Active progressToken table capacity exceeded")
            self._finished.pop(key, None)
            self._active[key] = {
                "updates": 0,
                "last_progress": 0.0,
                "total": None,
            }
            self._streams_started += 1
        return (
            0,
            ProgressReporter(self, key, norm_tok, cancel_token=cancel_token, emit_fn=emit_fn),
            "ok",
        )

    def update(
        self,
        key: tuple[str, int | str],
        raw_token: int | str,
        progress: float,
        *,
        total: float | None = None,
        message: str | None = None,
        cancel_token: CancellationToken | None = None,
    ) -> tuple[int, dict[str, Any] | None]:
        with self._lock:
            slot = self._active.get(key)
            if slot is None:
                self._updates_rejected += 1
                return (self.ERR_PROGRESS_INACTIVE, None)
            if cancel_token is not None and cancel_token.is_cancelled:
                del self._active[key]
                self._finished[key] = "aborted"
                while len(self._finished) > self.max_active:
                    self._finished.popitem(last=False)
                self._streams_aborted += 1
                self._updates_rejected += 1
                return (self.ERR_PROGRESS_INACTIVE, None)

            if isinstance(progress, bool) or not isinstance(progress, (int, float)) or not math.isfinite(progress) or progress < 0:
                self._updates_rejected += 1
                return (self.ERR_PROGRESS_NON_MONOTONIC, None)
            prog_val = float(progress)
            if slot["updates"] > 0 and prog_val <= slot["last_progress"]:
                self._updates_rejected += 1
                return (self.ERR_PROGRESS_NON_MONOTONIC, None)

            effective_total: float | None = slot["total"]
            if total is not None:
                if isinstance(total, bool) or not isinstance(total, (int, float)) or not math.isfinite(total) or total <= 0:
                    self._updates_rejected += 1
                    return (self.ERR_PROGRESS_EXCEEDS_TOTAL, None)
                tot_val = float(total)
                if effective_total is not None and tot_val != effective_total:
                    self._updates_rejected += 1
                    return (self.ERR_PROGRESS_EXCEEDS_TOTAL, None)
                effective_total = tot_val

            if effective_total is not None and prog_val > effective_total:
                self._updates_rejected += 1
                return (self.ERR_PROGRESS_EXCEEDS_TOTAL, None)

            slot["last_progress"] = prog_val
            slot["total"] = effective_total
            slot["updates"] += 1
            self._updates_emitted += 1

        wire_prog: int | float = int(progress) if isinstance(progress, int) and not isinstance(progress, bool) else round(prog_val, 6)
        params: dict[str, Any] = {
            "progressToken": raw_token,
            "progress": wire_prog,
        }
        if effective_total is not None:
            wire_tot: int | float = (
                int(total)
                if (total is not None and isinstance(total, int) and not isinstance(total, bool))
                else (int(effective_total) if effective_total.is_integer() else round(effective_total, 6))
            )
            params["total"] = wire_tot
        if message is not None:
            clean_msg, _ = RedactionGuard.sanitize(str(message).strip()[:256])
            if clean_msg:
                params["message"] = clean_msg
        return (0, {"jsonrpc": "2.0", "method": "notifications/progress", "params": params})

    def complete(self, key: tuple[str, int | str]) -> int:
        with self._lock:
            if key not in self._active:
                return self.ERR_PROGRESS_INACTIVE
            del self._active[key]
            self._finished[key] = "completed"
            while len(self._finished) > self.max_active:
                self._finished.popitem(last=False)
            self._streams_completed += 1
            return 0

    def abort(self, key: tuple[str, int | str]) -> int:
        with self._lock:
            if key not in self._active:
                return self.ERR_PROGRESS_INACTIVE
            del self._active[key]
            self._finished[key] = "aborted"
            while len(self._finished) > self.max_active:
                self._finished.popitem(last=False)
            self._streams_aborted += 1
            return 0

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "active_streams": len(self._active),
                "streams_started": self._streams_started,
                "updates_emitted": self._updates_emitted,
                "updates_rejected": self._updates_rejected,
                "streams_completed": self._streams_completed,
                "streams_aborted": self._streams_aborted,
            }


def _json_equal(a: Any, b: Any) -> bool:
    """Strict JSON-value equality preventing Python bool/int conflation (True != 1, False != 0)."""
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isfinite(a) and math.isfinite(b) and a == b
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_json_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_json_equal(a[k], b[k]) for k in a)
    return type(a) is type(b) and a == b


class McpCursorCodec:
    """Bit-identical C11/Rust/WASM 16-byte Castagnoli CRC32C + canonical 22-char RFC 4648 §5 base64url MCP cursor codec."""

    MAGIC: int = 0x5043
    VERSION: int = 1
    WIRE_BYTES: int = 16
    B64_LEN: int = 22
    MAX_OFFSET: int = 1_048_576

    SCOPE_TOOLS: int = 1
    SCOPE_RESOURCES: int = 2
    SCOPE_TEMPLATES: int = 3
    SCOPE_PROMPTS: int = 4

    ERR_B64: int = -13
    ERR_MAGIC: int = -14
    ERR_SCOPE: int = -15
    ERR_SNAPSHOT: int = -16
    ERR_SEAL: int = -17
    ERR_BOUNDS: int = -18

    _ALPHABET: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
    _SEXTET_MAP: ClassVar[dict[str, int]] = {ch: idx for idx, ch in enumerate(_ALPHABET)}

    @staticmethod
    def snapshot_hash(keys: list[str]) -> int:
        return crc32c("\n".join(keys))

    @classmethod
    def compute_seal(cls, magic: int, version: int, scope: int, offset: int, snapshot_crc32c: int) -> int:
        prefix12 = struct.pack(
            "<HBBII",
            magic & 0xFFFF,
            version & 0xFF,
            scope & 0xFF,
            offset & 0xFFFFFFFF,
            snapshot_crc32c & 0xFFFFFFFF,
        )
        return crc32c(prefix12 + b"MCPC")

    @classmethod
    def encode(cls, scope: int, offset: int, snapshot_crc32c: int) -> str:
        if isinstance(scope, bool) or not isinstance(scope, int) or scope < cls.SCOPE_TOOLS or scope > cls.SCOPE_PROMPTS:
            raise ValueError(f"Invalid cursor scope: {scope!r}")
        if isinstance(offset, bool) or not isinstance(offset, int) or offset <= 0 or offset > cls.MAX_OFFSET:
            raise ValueError(f"Invalid cursor offset: {offset!r}")
        snap = snapshot_crc32c & 0xFFFFFFFF
        seal = cls.compute_seal(cls.MAGIC, cls.VERSION, scope, offset, snap)
        wire16 = struct.pack("<HBBIII", cls.MAGIC, cls.VERSION, scope, offset, snap, seal)
        return base64.urlsafe_b64encode(wire16).rstrip(b"=").decode("ascii")

    @classmethod
    def decode_and_verify(
        cls,
        cursor: Any,
        expected_scope: int,
        expected_snapshot_crc32c: int,
        total_items: int,
    ) -> tuple[int, int, str]:
        if not isinstance(cursor, str) or len(cursor) != cls.B64_LEN:
            return (cls.ERR_B64, 0, "Invalid cursor: must be a 22-character unpadded base64url string")
        sextets: list[int] = []
        for ch in cursor:
            s = cls._SEXTET_MAP.get(ch)
            if s is None:
                return (cls.ERR_B64, 0, "Invalid cursor: illegal base64url character")
            sextets.append(s)
        if (sextets[21] & 0x0F) != 0:
            return (cls.ERR_B64, 0, "Invalid cursor: non-canonical trailing base64url bits")

        wire16 = base64.urlsafe_b64decode(cursor + "==")
        magic, version, scope, offset, snap_crc, seal_crc = struct.unpack("<HBBIII", wire16)
        computed_seal = cls.compute_seal(magic, version, scope, offset, snap_crc)
        if seal_crc != computed_seal:
            return (cls.ERR_SEAL, 0, "Invalid cursor: CRC32C integrity seal mismatch")
        if magic != cls.MAGIC or version != cls.VERSION:
            return (cls.ERR_MAGIC, 0, "Invalid cursor: unsupported cursor header or version")
        if (
            scope < cls.SCOPE_TOOLS
            or scope > cls.SCOPE_PROMPTS
            or expected_scope < cls.SCOPE_TOOLS
            or expected_scope > cls.SCOPE_PROMPTS
            or scope != expected_scope
        ):
            return (cls.ERR_SCOPE, 0, "Invalid cursor: cursor scope does not match target endpoint")
        if snap_crc != (expected_snapshot_crc32c & 0xFFFFFFFF):
            return (cls.ERR_SNAPSHOT, 0, "Invalid cursor: collection snapshot changed since cursor was issued")
        if offset <= 0 or offset > cls.MAX_OFFSET or offset >= max(0, int(total_items)):
            return (cls.ERR_BOUNDS, 0, "Invalid cursor: pagination offset out of bounds")
        return (0, int(offset), "ok")


class JsonSchema2020Validator:
    """Deterministic zero-dependency JSON Schema Draft 2020-12 validator for MCP tool inputSchema & outputSchema."""

    SCHEMA_URI = "https://json-schema.org/draft/2020-12/schema"
    MAX_DEPTH = 32

    @staticmethod
    def _matches_single_type(instance: Any, expected: str) -> bool:
        if expected == "null":
            return instance is None
        if expected == "boolean":
            return isinstance(instance, bool)
        if expected == "integer":
            return isinstance(instance, int) and not isinstance(instance, bool)
        if expected == "number":
            return isinstance(instance, (int, float)) and not isinstance(instance, bool) and math.isfinite(instance)
        if expected == "string":
            return isinstance(instance, str)
        if expected == "array":
            return isinstance(instance, list)
        if expected == "object":
            return isinstance(instance, dict)
        return False

    @classmethod
    def is_valid(cls, instance: Any, schema: dict[str, Any] | bool) -> bool:
        return len(cls.validate(instance, schema)) == 0

    @classmethod
    def validate(
        cls,
        instance: Any,
        schema: dict[str, Any] | bool,
        *,
        path: str = "$",
        _depth: int = 0,
    ) -> list[dict[str, str]]:
        if _depth > cls.MAX_DEPTH:
            return [{"path": path, "keyword": "maxDepth", "message": f"Validation depth exceeded max {cls.MAX_DEPTH}"}]
        if isinstance(schema, bool):
            return [] if schema else [{"path": path, "keyword": "booleanSchema", "message": "Schema is false"}]
        if not isinstance(schema, dict):
            return [{"path": path, "keyword": "schema", "message": "Invalid schema definition"}]

        violations: list[dict[str, str]] = []

        if "type" in schema:
            raw_type = schema["type"]
            allowed_types = [raw_type] if isinstance(raw_type, str) else list(raw_type)
            if not any(cls._matches_single_type(instance, t) for t in allowed_types):
                expected_desc = " | ".join(str(t) for t in allowed_types)
                actual_desc = "boolean" if isinstance(instance, bool) else type(instance).__name__
                violations.append(
                    {"path": path, "keyword": "type", "message": f"Expected type {expected_desc}, got {actual_desc}"}
                )
                return violations

        if "const" in schema and not _json_equal(instance, schema["const"]):
            violations.append({"path": path, "keyword": "const", "message": f"Value must equal {schema['const']!r}"})

        if "enum" in schema:
            opts = schema["enum"]
            if isinstance(opts, list) and not any(_json_equal(instance, opt) for opt in opts):
                violations.append({"path": path, "keyword": "enum", "message": f"Value must be one of {opts!r}"})

        if cls._matches_single_type(instance, "number"):
            num = float(instance)
            if "minimum" in schema and num < float(schema["minimum"]):
                violations.append(
                    {"path": path, "keyword": "minimum", "message": f"Value {instance} < minimum {schema['minimum']}"}
                )
            if "maximum" in schema and num > float(schema["maximum"]):
                violations.append(
                    {"path": path, "keyword": "maximum", "message": f"Value {instance} > maximum {schema['maximum']}"}
                )
            if "exclusiveMinimum" in schema and num <= float(schema["exclusiveMinimum"]):
                violations.append(
                    {
                        "path": path,
                        "keyword": "exclusiveMinimum",
                        "message": f"Value {instance} <= exclusiveMinimum {schema['exclusiveMinimum']}",
                    }
                )
            if "exclusiveMaximum" in schema and num >= float(schema["exclusiveMaximum"]):
                violations.append(
                    {
                        "path": path,
                        "keyword": "exclusiveMaximum",
                        "message": f"Value {instance} >= exclusiveMaximum {schema['exclusiveMaximum']}",
                    }
                )
            if "multipleOf" in schema:
                step = float(schema["multipleOf"])
                if step > 0:
                    quotient = num / step
                    if abs(quotient - round(quotient)) > 1e-9:
                        violations.append(
                            {
                                "path": path,
                                "keyword": "multipleOf",
                                "message": f"Value {instance} is not a multiple of {schema['multipleOf']}",
                            }
                        )

        if isinstance(instance, str):
            if "minLength" in schema and len(instance) < int(schema["minLength"]):
                violations.append(
                    {
                        "path": path,
                        "keyword": "minLength",
                        "message": f"String length {len(instance)} < minLength {schema['minLength']}",
                    }
                )
            if "maxLength" in schema and len(instance) > int(schema["maxLength"]):
                violations.append(
                    {
                        "path": path,
                        "keyword": "maxLength",
                        "message": f"String length {len(instance)} > maxLength {schema['maxLength']}",
                    }
                )
            if "pattern" in schema and re.search(str(schema["pattern"]), instance) is None:
                violations.append(
                    {
                        "path": path,
                        "keyword": "pattern",
                        "message": f"String does not match pattern {schema['pattern']!r}",
                    }
                )

        if isinstance(instance, list):
            if "minItems" in schema and len(instance) < int(schema["minItems"]):
                violations.append(
                    {
                        "path": path,
                        "keyword": "minItems",
                        "message": f"Array length {len(instance)} < minItems {schema['minItems']}",
                    }
                )
            if "maxItems" in schema and len(instance) > int(schema["maxItems"]):
                violations.append(
                    {
                        "path": path,
                        "keyword": "maxItems",
                        "message": f"Array length {len(instance)} > maxItems {schema['maxItems']}",
                    }
                )
            if schema.get("uniqueItems") is True:
                for i in range(len(instance)):
                    if any(_json_equal(instance[i], instance[j]) for j in range(i + 1, len(instance))):
                        violations.append(
                            {"path": path, "keyword": "uniqueItems", "message": "Array items must be unique"}
                        )
                        break

            prefix_len = 0
            if "prefixItems" in schema and isinstance(schema["prefixItems"], list):
                prefix_schemas = schema["prefixItems"]
                prefix_len = len(prefix_schemas)
                for idx, sub_schema in enumerate(prefix_schemas):
                    if idx < len(instance):
                        violations.extend(
                            cls.validate(instance[idx], sub_schema, path=f"{path}[{idx}]", _depth=_depth + 1)
                        )

            if "items" in schema:
                items_schema = schema["items"]
                if items_schema is False and len(instance) > prefix_len:
                    violations.append(
                        {
                            "path": f"{path}[{prefix_len}]",
                            "keyword": "items",
                            "message": f"Additional array items beyond index {prefix_len - 1} are not allowed",
                        }
                    )
                elif items_schema is not True:
                    for idx in range(prefix_len, len(instance)):
                        violations.extend(
                            cls.validate(instance[idx], items_schema, path=f"{path}[{idx}]", _depth=_depth + 1)
                        )

        if isinstance(instance, dict):
            if "minProperties" in schema and len(instance) < int(schema["minProperties"]):
                violations.append(
                    {
                        "path": path,
                        "keyword": "minProperties",
                        "message": f"Object has {len(instance)} properties, minProperties is {schema['minProperties']}",
                    }
                )
            if "maxProperties" in schema and len(instance) > int(schema["maxProperties"]):
                violations.append(
                    {
                        "path": path,
                        "keyword": "maxProperties",
                        "message": f"Object has {len(instance)} properties, maxProperties is {schema['maxProperties']}",
                    }
                )
            req_fields = schema.get("required")
            if isinstance(req_fields, list):
                for req_key in req_fields:
                    if req_key not in instance:
                        violations.append(
                            {
                                "path": f"{path}.{req_key}",
                                "keyword": "required",
                                "message": f"Missing required property '{req_key}'",
                            }
                        )
            props = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
            addl = schema.get("additionalProperties", True)
            for k, v in instance.items():
                child_path = f"{path}.{k}"
                if k in props:
                    violations.extend(cls.validate(v, props[k], path=child_path, _depth=_depth + 1))
                elif addl is False:
                    violations.append(
                        {
                            "path": child_path,
                            "keyword": "additionalProperties",
                            "message": f"Additional property '{k}' is not allowed",
                        }
                    )
                elif isinstance(addl, dict):
                    violations.extend(cls.validate(v, addl, path=child_path, _depth=_depth + 1))

        if "allOf" in schema and isinstance(schema["allOf"], list):
            for sub in schema["allOf"]:
                violations.extend(cls.validate(instance, sub, path=path, _depth=_depth + 1))

        if "anyOf" in schema and isinstance(schema["anyOf"], list) and not any(
            len(cls.validate(instance, sub, path=path, _depth=_depth + 1)) == 0 for sub in schema["anyOf"]
        ):
            violations.append(
                {"path": path, "keyword": "anyOf", "message": "Instance did not match any subschema in anyOf"}
            )

        if "oneOf" in schema and isinstance(schema["oneOf"], list):
            matches = sum(
                1 for sub in schema["oneOf"] if len(cls.validate(instance, sub, path=path, _depth=_depth + 1)) == 0
            )
            if matches != 1:
                violations.append(
                    {
                        "path": path,
                        "keyword": "oneOf",
                        "message": f"Expected exactly 1 matching subschema in oneOf, matched {matches}",
                    }
                )

        if "not" in schema and len(cls.validate(instance, schema["not"], path=path, _depth=_depth + 1)) == 0:
            violations.append({"path": path, "keyword": "not", "message": "Instance matched forbidden 'not' schema"})

        return violations


class OtlpTelemetryExporter:
    """Zero-dependency OpenTelemetry (OTLP/JSON) span exporter with W3C traceparent propagation, histograms & governance counters."""

    HISTOGRAM_BOUNDS_MS: tuple[float, ...] = (0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 50.0)
    _TRACEPARENT_RE = re.compile(r"^00-([0-9a-f]{32})-([0-9a-f]{16})-([0-9a-f]{2})$", re.IGNORECASE)

    def __init__(
        self,
        service_name: str = "mcp-moe-hybrid-sdk",
        service_version: str = "1.9.0",
        max_spans: int = 256,
        span_hook: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.service_name = service_name
        self.service_version = service_version
        self.spans: deque[dict[str, Any]] = deque(maxlen=max_spans)
        self.span_hook = span_hook
        self.cache_hits = 0
        self.cache_misses = 0
        self.rate_limit_allowed = 0
        self.rate_limit_rejected = 0
        self.schema_validations_passed = 0
        self.schema_validations_failed = 0
        self.requests_cancelled = 0
        self.progress_notifications_emitted = 0
        self.roots_checks_allowed = 0
        self.roots_checks_denied = 0
        self.elicitations_accepted = 0
        self.elicitations_declined = 0
        self.sampling_requests_governed = 0
        self.approval_gate_rejections = 0
        self.sessions_opened = 0
        self.sse_events_replayed = 0
        self.resource_links_emitted = 0
        self.annotation_rejections = 0
        self.oauth_tokens_verified = 0
        self.oauth_challenges_issued = 0
        self.oauth_audience_rejections = 0
        self.oauth_scope_rejections = 0
        self.pkce_verifications = 0
        self._histograms: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    @classmethod
    def parse_traceparent(cls, header: str | None) -> dict[str, str] | None:
        if not isinstance(header, str):
            return None
        match = cls._TRACEPARENT_RE.match(header.strip())
        if not match:
            return None
        trace_id = match.group(1).lower()
        parent_span_id = match.group(2).lower()
        trace_flags = match.group(3).lower()
        if trace_id == "0" * 32 or parent_span_id == "0" * 16:
            return None
        return {
            "version": "00",
            "trace_id": trace_id,
            "parent_span_id": parent_span_id,
            "trace_flags": trace_flags,
        }

    @staticmethod
    def format_traceparent(trace_id: str, span_id: str, trace_flags: str = "01") -> str:
        return f"00-{trace_id.lower()}-{span_id.lower()}-{trace_flags.lower()}"

    @classmethod
    def derive_child_traceparent(
        cls,
        parent_traceparent: str,
        *,
        step_seq: int = 0,
        agent_id: int = 1,
        label: str | bytes = "",
    ) -> str | None:
        """Derive a deterministic W3C child traceparent bit-identical to C11/Rust/WASM par_trace_derive_child."""
        parsed = cls.parse_traceparent(parent_traceparent)
        if parsed is None:
            return None
        trace_bytes = bytes.fromhex(parsed["trace_id"])
        span_bytes = bytes.fromhex(parsed["parent_span_id"])
        label_bytes = label.encode("utf-8") if isinstance(label, str) else bytes(label)
        label_crc = crc32c(label_bytes)
        prefix = (
            trace_bytes
            + span_bytes
            + struct.pack("<QII", step_seq & 0xFFFFFFFFFFFFFFFF, agent_id & 0xFFFFFFFF, label_crc)
        )
        hi32 = crc32c(prefix + b"HNPS")
        lo32 = crc32c(prefix + b"LNPS")
        if hi32 == 0 and lo32 == 0:
            lo32 = 1
        child_span_id = f"{hi32:08x}{lo32:08x}"
        return cls.format_traceparent(parsed["trace_id"], child_span_id, parsed["trace_flags"])

    def new_gateway_context(self, incoming_traceparent: str | None = None) -> dict[str, str]:
        parsed = self.parse_traceparent(incoming_traceparent)
        gateway_span_id = secrets.token_hex(8)
        if parsed is not None:
            trace_id = parsed["trace_id"]
            trace_flags = parsed["trace_flags"]
            child_parent_tp = self.format_traceparent(trace_id, parsed["parent_span_id"], trace_flags)
            response_tp = self.format_traceparent(trace_id, gateway_span_id, trace_flags)
            return {
                "trace_id": trace_id,
                "parent_span_id": parsed["parent_span_id"],
                "gateway_span_id": gateway_span_id,
                "trace_flags": trace_flags,
                "child_traceparent": child_parent_tp,
                "traceparent": response_tp,
            }
        trace_id = secrets.token_hex(16)
        trace_flags = "01"
        root_tp = self.format_traceparent(trace_id, gateway_span_id, trace_flags)
        return {
            "trace_id": trace_id,
            "parent_span_id": "",
            "gateway_span_id": gateway_span_id,
            "trace_flags": trace_flags,
            "child_traceparent": root_tp,
            "traceparent": root_tp,
        }

    def record_rate_limit_decision(self, allowed: bool) -> None:
        with self._lock:
            if allowed:
                self.rate_limit_allowed += 1
            else:
                self.rate_limit_rejected += 1

    def record_schema_validation(self, passed: bool) -> None:
        with self._lock:
            if passed:
                self.schema_validations_passed += 1
            else:
                self.schema_validations_failed += 1

    def record_cancellation(self) -> None:
        with self._lock:
            self.requests_cancelled += 1

    def record_progress_notification(self, count: int = 1) -> None:
        with self._lock:
            self.progress_notifications_emitted += max(0, int(count))

    def record_roots_check(self, allowed: bool) -> None:
        with self._lock:
            if allowed:
                self.roots_checks_allowed += 1
            else:
                self.roots_checks_denied += 1

    def record_elicitation_outcome(self, action: str) -> None:
        with self._lock:
            if action == "accept":
                self.elicitations_accepted += 1
            else:
                self.elicitations_declined += 1

    def record_sampling_governed(self) -> None:
        with self._lock:
            self.sampling_requests_governed += 1

    def record_approval_rejection(self) -> None:
        with self._lock:
            self.approval_gate_rejections += 1

    def record_session_opened(self) -> None:
        with self._lock:
            self.sessions_opened += 1

    def record_sse_events_replayed(self, count: int = 1) -> None:
        with self._lock:
            self.sse_events_replayed += max(0, int(count))

    def record_resource_link_emitted(self, count: int = 1) -> None:
        with self._lock:
            self.resource_links_emitted += max(0, int(count))

    def record_annotation_rejection(self) -> None:
        with self._lock:
            self.annotation_rejections += 1

    def record_oauth_token_verified(self) -> None:
        with self._lock:
            self.oauth_tokens_verified += 1

    def record_oauth_challenge_issued(self) -> None:
        with self._lock:
            self.oauth_challenges_issued += 1

    def record_oauth_audience_rejection(self) -> None:
        with self._lock:
            self.oauth_audience_rejections += 1

    def record_oauth_scope_rejection(self) -> None:
        with self._lock:
            self.oauth_scope_rejections += 1

    def record_pkce_verification(self) -> None:
        with self._lock:
            self.pkce_verifications += 1

    def record_tool_call(
        self,
        tool_name: str,
        start_ns: int,
        end_ns: int,
        *,
        cache_hit: bool = False,
        is_error: bool = False,
        attributes: dict[str, Any] | None = None,
        traceparent: str | None = None,
        parent_context: dict[str, str] | None = None,
        deterministic_seq: int | None = None,
        agent_id: int = 1,
    ) -> dict[str, Any]:
        parent = parent_context if parent_context is not None else self.parse_traceparent(traceparent)
        trace_id = parent["trace_id"] if parent else secrets.token_hex(16)
        parent_span_id = parent.get("parent_span_id", "") if parent else ""
        trace_flags = parent.get("trace_flags", "01") if parent else "01"
        clean_tool, _ = RedactionGuard.sanitize(tool_name)
        if parent and deterministic_seq is not None:
            parent_tp = self.format_traceparent(trace_id, parent_span_id, trace_flags)
            derived_tp = self.derive_child_traceparent(
                parent_tp,
                step_seq=deterministic_seq,
                agent_id=agent_id,
                label=f"mcp.tools.call/{clean_tool}",
            )
            span_id = derived_tp.split("-")[2] if derived_tp else secrets.token_hex(8)
        else:
            span_id = secrets.token_hex(8)
        span_tp = self.format_traceparent(trace_id, span_id, trace_flags)

        duration_ms = max(0.0, (end_ns - start_ns) / 1_000_000.0)

        otlp_attrs = [
            {"key": "mcp.tool.name", "value": {"stringValue": clean_tool}},
            {"key": "mcp.cache.hit", "value": {"boolValue": cache_hit}},
            {"key": "w3c.traceparent", "value": {"stringValue": span_tp}},
        ]
        for k, v in (attributes or {}).items():
            if isinstance(v, bool):
                otlp_attrs.append({"key": k, "value": {"boolValue": v}})
            elif isinstance(v, int):
                otlp_attrs.append({"key": k, "value": {"intValue": str(v)}})
            elif isinstance(v, float):
                otlp_attrs.append({"key": k, "value": {"doubleValue": round(v, 6)}})
            else:
                clean_val, _ = RedactionGuard.sanitize(str(v))
                otlp_attrs.append({"key": k, "value": {"stringValue": clean_val}})

        span: dict[str, Any] = {
            "traceId": trace_id,
            "spanId": span_id,
            "name": f"mcp.tools.call/{clean_tool}",
            "kind": 2,
            "traceFlags": int(trace_flags, 16),
            "startTimeUnixNano": str(start_ns),
            "endTimeUnixNano": str(max(start_ns, end_ns)),
            "attributes": otlp_attrs,
            "status": {"code": 2 if is_error else 1},
        }
        if parent_span_id:
            span["parentSpanId"] = parent_span_id

        with self._lock:
            if tool_name == "rag_hybrid_search":
                if cache_hit:
                    self.cache_hits += 1
                else:
                    self.cache_misses += 1

            hist = self._histograms.setdefault(
                clean_tool,
                {
                    "count": 0,
                    "sum_ms": 0.0,
                    "min_ms": duration_ms,
                    "max_ms": duration_ms,
                    "buckets": [0] * (len(self.HISTOGRAM_BOUNDS_MS) + 1),
                },
            )
            hist["count"] += 1
            hist["sum_ms"] += duration_ms
            hist["min_ms"] = min(hist["min_ms"], duration_ms)
            hist["max_ms"] = max(hist["max_ms"], duration_ms)
            bucket_idx = len(self.HISTOGRAM_BOUNDS_MS)
            for idx, bound in enumerate(self.HISTOGRAM_BOUNDS_MS):
                if duration_ms <= bound:
                    bucket_idx = idx
                    break
            hist["buckets"][bucket_idx] += 1
            self.spans.append(span)

        if self.span_hook is not None:
            self.span_hook(span)
        return span

    def export_otlp_json(self) -> dict[str, Any]:
        with self._lock:
            now_ns = str(time.time_ns())
            spans_snapshot = list(self.spans)
            cache_hits = self.cache_hits
            cache_misses = self.cache_misses
            rl_allowed = self.rate_limit_allowed
            rl_rejected = self.rate_limit_rejected
            sv_passed = self.schema_validations_passed
            sv_failed = self.schema_validations_failed
            req_cancelled = self.requests_cancelled
            prog_emitted = self.progress_notifications_emitted
            roots_allowed = self.roots_checks_allowed
            roots_denied = self.roots_checks_denied
            el_accepted = self.elicitations_accepted
            el_declined = self.elicitations_declined
            samp_governed = self.sampling_requests_governed
            app_rejected = self.approval_gate_rejections
            sess_opened = self.sessions_opened
            sse_replayed = self.sse_events_replayed
            rlinks_emitted = self.resource_links_emitted
            ann_rejected = self.annotation_rejections
            oauth_verified = self.oauth_tokens_verified
            oauth_challenges = self.oauth_challenges_issued
            oauth_aud_rejected = self.oauth_audience_rejections
            oauth_scope_rejected = self.oauth_scope_rejections
            pkce_checks = self.pkce_verifications
            dp_hist = []
            for tool_name, h in sorted(self._histograms.items()):
                dp_hist.append(
                    {
                        "attributes": [{"key": "mcp.tool.name", "value": {"stringValue": tool_name}}],
                        "timeUnixNano": now_ns,
                        "count": str(h["count"]),
                        "sum": round(h["sum_ms"], 6),
                        "min": round(h["min_ms"], 6),
                        "max": round(h["max_ms"], 6),
                        "bucketCounts": [str(c) for c in h["buckets"]],
                        "explicitBounds": list(self.HISTOGRAM_BOUNDS_MS),
                    }
                )

        resource = {
            "attributes": [
                {"key": "service.name", "value": {"stringValue": self.service_name}},
                {"key": "service.version", "value": {"stringValue": self.service_version}},
            ]
        }
        total_lookups = cache_hits + cache_misses
        hit_ratio = round(cache_hits / total_lookups, 4) if total_lookups else 0.0
        return {
            "resourceSpans": [
                {
                    "resource": resource,
                    "scopeSpans": [
                        {
                            "scope": {"name": "mcp_moe_sdk.otlp", "version": self.service_version},
                            "spans": spans_snapshot,
                        }
                    ],
                }
            ],
            "resourceMetrics": [
                {
                    "resource": resource,
                    "scopeMetrics": [
                        {
                            "scope": {"name": "mcp_moe_sdk.otlp", "version": self.service_version},
                            "metrics": [
                                {
                                    "name": "mcp.tool.duration",
                                    "unit": "ms",
                                    "histogram": {"aggregationTemporality": 2, "dataPoints": dp_hist},
                                },
                                {
                                    "name": "mcp.rag.cache.requests",
                                    "unit": "1",
                                    "sum": {
                                        "aggregationTemporality": 2,
                                        "isMonotonic": True,
                                        "dataPoints": [
                                            {
                                                "attributes": [{"key": "result", "value": {"stringValue": "hit"}}],
                                                "timeUnixNano": now_ns,
                                                "asInt": str(cache_hits),
                                            },
                                            {
                                                "attributes": [{"key": "result", "value": {"stringValue": "miss"}}],
                                                "timeUnixNano": now_ns,
                                                "asInt": str(cache_misses),
                                            },
                                        ],
                                    },
                                },
                                {
                                    "name": "mcp.http.ratelimit.decisions",
                                    "unit": "1",
                                    "sum": {
                                        "aggregationTemporality": 2,
                                        "isMonotonic": True,
                                        "dataPoints": [
                                            {
                                                "attributes": [{"key": "decision", "value": {"stringValue": "allowed"}}],
                                                "timeUnixNano": now_ns,
                                                "asInt": str(rl_allowed),
                                            },
                                            {
                                                "attributes": [{"key": "decision", "value": {"stringValue": "rejected"}}],
                                                "timeUnixNano": now_ns,
                                                "asInt": str(rl_rejected),
                                            },
                                        ],
                                    },
                                },
                            ],
                        }
                    ],
                }
            ],
            "summary": {
                "spans_buffered": len(spans_snapshot),
                "cache_hits": cache_hits,
                "cache_misses": cache_misses,
                "cache_hit_ratio": hit_ratio,
                "rate_limit_allowed": rl_allowed,
                "rate_limit_rejected": rl_rejected,
                "schema_validations_passed": sv_passed,
                "schema_validations_failed": sv_failed,
                "requests_cancelled": req_cancelled,
                "progress_notifications_emitted": prog_emitted,
                "roots_checks_allowed": roots_allowed,
                "roots_checks_denied": roots_denied,
                "elicitations_accepted": el_accepted,
                "elicitations_declined": el_declined,
                "sampling_requests_governed": samp_governed,
                "approval_gate_rejections": app_rejected,
                "sessions_opened": sess_opened,
                "sse_events_replayed": sse_replayed,
                "resource_links_emitted": rlinks_emitted,
                "annotation_rejections": ann_rejected,
                "oauth_tokens_verified": oauth_verified,
                "oauth_challenges_issued": oauth_challenges,
                "oauth_audience_rejections": oauth_aud_rejected,
                "oauth_scope_rejections": oauth_scope_rejected,
                "pkce_verifications": pkce_checks,
            },
        }


class ElicitationSchemaGuard:
    """MCP 2025-11-25 elicitation/create flat primitive requestedSchema validator & canonical CRC32C binder."""

    ERR_ELICITATION_SCHEMA: int = -35
    ALLOWED_PRIMITIVE_TYPES: tuple[str, ...] = ("string", "number", "integer", "boolean")
    ALLOWED_STRING_FORMATS: tuple[str, ...] = ("email", "uri", "date", "date-time")
    MAX_PROPERTIES: int = 64

    _TOP_KEYWORDS: frozenset[str] = frozenset(
        {"$schema", "type", "title", "description", "properties", "required", "additionalProperties"}
    )
    _COMMON_PROP_KEYWORDS: frozenset[str] = frozenset(
        {"type", "title", "description", "default", "enum", "enumNames"}
    )
    _STRING_PROP_KEYWORDS: frozenset[str] = frozenset({"minLength", "maxLength", "pattern", "format"})
    _NUMBER_PROP_KEYWORDS: frozenset[str] = frozenset({"minimum", "maximum"})
    _DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    _DATETIME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+\-]\d{2}:\d{2})$")
    _EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    _URI_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.\-]*://\S+$")
    _SECRET_KEY_RE = re.compile(r"\b(?:sk|rk|pk)-[A-Za-z0-9_\-]{12,}\b")
    _BEARER_TOKEN_RE = re.compile(r"Bearer\s+[A-Za-z0-9._\-+/=]{8,}", re.IGNORECASE)

    @classmethod
    def validate_requested_schema(cls, schema: Any) -> tuple[int, int, str]:
        """Validate that requestedSchema is a flat object of scalar primitive properties and return its CRC32C."""
        if not isinstance(schema, dict) or isinstance(schema, bool):
            return (cls.ERR_ELICITATION_SCHEMA, 0, "Invalid elicitation requestedSchema: must be a JSON object")
        unknown_top = set(schema.keys()) - cls._TOP_KEYWORDS
        if unknown_top:
            bad_kw = min(unknown_top)
            return (
                cls.ERR_ELICITATION_SCHEMA,
                0,
                f"Forbidden top-level keyword '{bad_kw}' in flat elicitation requestedSchema",
            )
        if schema.get("type") != "object":
            return (cls.ERR_ELICITATION_SCHEMA, 0, "Elicitation requestedSchema top-level 'type' must equal 'object'")
        if "additionalProperties" in schema and schema["additionalProperties"] is not False:
            return (
                cls.ERR_ELICITATION_SCHEMA,
                0,
                "Elicitation requestedSchema 'additionalProperties' must be false when specified",
            )
        props = schema.get("properties")
        if not isinstance(props, dict) or not props or len(props) > cls.MAX_PROPERTIES:
            return (
                cls.ERR_ELICITATION_SCHEMA,
                0,
                "Elicitation requestedSchema 'properties' must be a non-empty flat map (<= 64 fields)",
            )
        if "required" in schema:
            req = schema["required"]
            if (
                not isinstance(req, list)
                or any(not isinstance(k, str) or not k for k in req)
                or len(set(req)) != len(req)
                or any(k not in props for k in req)
            ):
                return (
                    cls.ERR_ELICITATION_SCHEMA,
                    0,
                    "Elicitation requestedSchema 'required' must list unique defined property names",
                )

        for prop_name, prop_spec in props.items():
            if not isinstance(prop_name, str) or not prop_name or not isinstance(prop_spec, dict):
                return (
                    cls.ERR_ELICITATION_SCHEMA,
                    0,
                    f"Property '{prop_name}' must be a flat JSON schema object",
                )
            ptype = prop_spec.get("type")
            if not isinstance(ptype, str) or ptype not in cls.ALLOWED_PRIMITIVE_TYPES:
                return (
                    cls.ERR_ELICITATION_SCHEMA,
                    0,
                    f"Property '{prop_name}' must use scalar primitive type {cls.ALLOWED_PRIMITIVE_TYPES}, got {ptype!r}",
                )
            allowed_kw = set(cls._COMMON_PROP_KEYWORDS)
            if ptype == "string":
                allowed_kw |= cls._STRING_PROP_KEYWORDS
            elif ptype in ("number", "integer"):
                allowed_kw |= cls._NUMBER_PROP_KEYWORDS
            extra_kw = set(prop_spec.keys()) - allowed_kw
            if extra_kw:
                bad_pkw = min(extra_kw)
                return (
                    cls.ERR_ELICITATION_SCHEMA,
                    0,
                    f"Forbidden nested/combinator keyword '{bad_pkw}' on property '{prop_name}'",
                )
            if "enum" in prop_spec:
                opts = prop_spec["enum"]
                if (
                    not isinstance(opts, list)
                    or not opts
                    or any(not JsonSchema2020Validator._matches_single_type(v, ptype) for v in opts)
                ):
                    return (
                        cls.ERR_ELICITATION_SCHEMA,
                        0,
                        f"Property '{prop_name}' enum values must be non-empty and match type '{ptype}'",
                    )
                if "enumNames" in prop_spec:
                    enames = prop_spec["enumNames"]
                    if not isinstance(enames, list) or len(enames) != len(opts) or any(not isinstance(n, str) for n in enames):
                        return (
                            cls.ERR_ELICITATION_SCHEMA,
                            0,
                            f"Property '{prop_name}' enumNames must match enum length with strings",
                        )
            elif "enumNames" in prop_spec:
                return (cls.ERR_ELICITATION_SCHEMA, 0, f"Property '{prop_name}' enumNames requires enum")

            if "default" in prop_spec and not JsonSchema2020Validator._matches_single_type(prop_spec["default"], ptype):
                return (
                    cls.ERR_ELICITATION_SCHEMA,
                    0,
                    f"Property '{prop_name}' default value does not match primitive type '{ptype}'",
                )
            if ptype == "string" and "format" in prop_spec:
                fmt = prop_spec["format"]
                if not isinstance(fmt, str) or fmt not in cls.ALLOWED_STRING_FORMATS:
                    return (
                        cls.ERR_ELICITATION_SCHEMA,
                        0,
                        f"Property '{prop_name}' format must be one of {cls.ALLOWED_STRING_FORMATS}",
                    )
            if ptype in ("number", "integer"):
                min_v = prop_spec.get("minimum")
                max_v = prop_spec.get("maximum")
                for bound_name, bound_val in (("minimum", min_v), ("maximum", max_v)):
                    if bound_val is not None and (
                        isinstance(bound_val, bool) or not isinstance(bound_val, (int, float)) or not math.isfinite(bound_val)
                    ):
                        return (
                            cls.ERR_ELICITATION_SCHEMA,
                            0,
                            f"Property '{prop_name}' {bound_name} must be a finite number",
                        )
                if min_v is not None and max_v is not None and float(min_v) > float(max_v):
                    return (
                        cls.ERR_ELICITATION_SCHEMA,
                        0,
                        f"Property '{prop_name}' minimum ({min_v}) > maximum ({max_v})",
                    )

        canonical = json.dumps(schema, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return (0, crc32c(canonical), "ok")

    @classmethod
    def _matches_format(cls, val: str, fmt: str) -> bool:
        if fmt == "date":
            return cls._DATE_RE.match(val) is not None
        if fmt == "date-time":
            return cls._DATETIME_RE.match(val) is not None
        if fmt == "email":
            return cls._EMAIL_RE.match(val) is not None
        if fmt == "uri":
            return cls._URI_RE.match(val) is not None
        return False

    @classmethod
    def validate_response(
        cls,
        action: Any,
        content: Any,
        requested_schema: dict[str, Any],
    ) -> tuple[int, dict[str, Any] | None, str]:
        """Validate an elicitation response ('accept' | 'decline' | 'cancel') and redact accepted string fields."""
        rc, _, msg = cls.validate_requested_schema(requested_schema)
        if rc != 0:
            return (rc, None, msg)
        if action not in ("accept", "decline", "cancel"):
            return (cls.ERR_ELICITATION_SCHEMA, None, "Elicitation action must be 'accept', 'decline', or 'cancel'")
        if action in ("decline", "cancel"):
            if content is not None:
                return (cls.ERR_ELICITATION_SCHEMA, None, f"Elicitation '{action}' must not carry 'content'")
            return (0, {"action": action}, "ok")

        if not isinstance(content, dict) or isinstance(content, bool):
            return (cls.ERR_ELICITATION_SCHEMA, None, "Elicitation 'accept' requires an object 'content' payload")

        closed_schema = {**requested_schema, "additionalProperties": False}
        violations = JsonSchema2020Validator.validate(content, closed_schema)
        if violations:
            return (
                cls.ERR_ELICITATION_SCHEMA,
                None,
                f"Elicitation content violated requestedSchema: {violations[0]['message']}",
            )

        props = requested_schema["properties"]
        sanitized_content: dict[str, Any] = {}
        for k, v in content.items():
            pspec = props[k]
            if isinstance(v, str):
                fmt = pspec.get("format")
                if isinstance(fmt, str) and not cls._matches_format(v, fmt):
                    return (
                        cls.ERR_ELICITATION_SCHEMA,
                        None,
                        f"Property '{k}' value failed format '{fmt}' check",
                    )
                if fmt == "email":
                    clean_v = cls._SECRET_KEY_RE.sub("[REDACTED_API_KEY]", v)
                    clean_v = cls._BEARER_TOKEN_RE.sub("Bearer [REDACTED_TOKEN]", clean_v)
                    sanitized_content[k] = clean_v
                else:
                    clean_v, _ = RedactionGuard.sanitize(v)
                    sanitized_content[k] = clean_v
            else:
                sanitized_content[k] = v

        return (0, {"action": "accept", "content": sanitized_content}, "ok")


class SamplingGovernanceGuard:
    """MCP 2025-11-25 sampling/createMessage token-budget, temperature, includeContext & modelPreferences fence."""

    ERR_SAMPLING_BUDGET: int = -34
    DEFAULT_CEILING_MAX_TOKENS: int = 4096

    @classmethod
    def check_budget(
        cls,
        requested_max_tokens: Any,
        ceiling_max_tokens: int = DEFAULT_CEILING_MAX_TOKENS,
        *,
        temperature: Any = 0.0,
        model_preferences: dict[str, Any] | None = None,
    ) -> tuple[int, dict[str, int], str]:
        """Bit-identical validation to C11/Rust/WASM par_sampling_check_budget (-34 on breach)."""
        if (
            isinstance(requested_max_tokens, bool)
            or not isinstance(requested_max_tokens, int)
            or isinstance(ceiling_max_tokens, bool)
            or not isinstance(ceiling_max_tokens, int)
            or ceiling_max_tokens <= 0
            or requested_max_tokens <= 0
            or requested_max_tokens > ceiling_max_tokens
        ):
            return (
                cls.ERR_SAMPLING_BUDGET,
                {},
                f"Sampling maxTokens ({requested_max_tokens!r}) must be an integer in 1..={ceiling_max_tokens}",
            )
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(temperature):
            return (cls.ERR_SAMPLING_BUDGET, {}, "Sampling temperature must be a finite number in [0.0, 2.0]")
        temp_milli = round(float(temperature) * 1000.0)
        if temp_milli < 0 or temp_milli > 2000:
            return (cls.ERR_SAMPLING_BUDGET, {}, f"Sampling temperature {temperature} outside [0.0, 2.0]")

        prefs = model_preferences if model_preferences is not None else {}
        if not isinstance(prefs, dict):
            return (cls.ERR_SAMPLING_BUDGET, {}, "Sampling modelPreferences must be an object")

        priorities_milli: dict[str, int] = {}
        for field in ("costPriority", "speedPriority", "intelligencePriority"):
            raw_p = prefs.get(field, 0.5)
            if isinstance(raw_p, bool) or not isinstance(raw_p, (int, float)) or not math.isfinite(raw_p):
                return (cls.ERR_SAMPLING_BUDGET, {}, f"Sampling {field} must be a finite number in [0.0, 1.0]")
            p_milli = round(float(raw_p) * 1000.0)
            if p_milli < 0 or p_milli > 1000:
                return (cls.ERR_SAMPLING_BUDGET, {}, f"Sampling {field} ({raw_p}) outside [0.0, 1.0]")
            priorities_milli[field] = p_milli

        return (
            0,
            {
                "requested_max_tokens": requested_max_tokens,
                "ceiling_max_tokens": ceiling_max_tokens,
                "temperature_milli": temp_milli,
                **priorities_milli,
            },
            "ok",
        )

    @classmethod
    def validate_request(
        cls,
        params: dict[str, Any],
        *,
        ceiling_max_tokens: int = DEFAULT_CEILING_MAX_TOKENS,
        allow_all_servers_context: bool = False,
    ) -> tuple[int, dict[str, Any] | None, str]:
        if not isinstance(params, dict):
            return (cls.ERR_SAMPLING_BUDGET, None, "Sampling params must be a JSON object")

        rc, budget_meta, msg = cls.check_budget(
            params.get("maxTokens"),
            ceiling_max_tokens,
            temperature=params.get("temperature", 0.0),
            model_preferences=params.get("modelPreferences"),
        )
        if rc != 0:
            return (rc, None, msg)

        include_ctx = params.get("includeContext", "none")
        allowed_ctx = ("none", "thisServer", "allServers") if allow_all_servers_context else ("none", "thisServer")
        if not isinstance(include_ctx, str) or include_ctx not in allowed_ctx:
            return (
                cls.ERR_SAMPLING_BUDGET,
                None,
                f"Sampling includeContext '{include_ctx}' is not permitted (allowed: {list(allowed_ctx)})",
            )

        raw_msgs = params.get("messages")
        if not isinstance(raw_msgs, list) or not raw_msgs or len(raw_msgs) > 64:
            return (cls.ERR_SAMPLING_BUDGET, None, "Sampling messages must be a non-empty array (<= 64 turns)")

        sanitized_msgs: list[dict[str, Any]] = []
        redactions = 0
        for m in raw_msgs:
            if not isinstance(m, dict) or m.get("role") not in ("user", "assistant"):
                return (cls.ERR_SAMPLING_BUDGET, None, "Each sampling message must specify role 'user' or 'assistant'")
            content = m.get("content")
            if not isinstance(content, dict) or content.get("type") != "text" or not isinstance(content.get("text"), str):
                return (cls.ERR_SAMPLING_BUDGET, None, "Each sampling message content must be {type:'text', text:str}")
            clean_txt, r_cnt = RedactionGuard.sanitize(content["text"])
            redactions += r_cnt
            sanitized_msgs.append({"role": m["role"], "content": {"type": "text", "text": clean_txt}})

        sys_prompt = params.get("systemPrompt")
        clean_sys: str | None = None
        if sys_prompt is not None:
            if not isinstance(sys_prompt, str):
                return (cls.ERR_SAMPLING_BUDGET, None, "Sampling systemPrompt must be a string")
            clean_sys, s_cnt = RedactionGuard.sanitize(sys_prompt)
            redactions += s_cnt

        contract_payload = {
            "includeContext": include_ctx,
            "maxTokens": budget_meta["requested_max_tokens"],
            "temperatureMilli": budget_meta["temperature_milli"],
            "costPriorityMilli": budget_meta["costPriority"],
            "speedPriorityMilli": budget_meta["speedPriority"],
            "intelligencePriorityMilli": budget_meta["intelligencePriority"],
        }
        schema_crc = crc32c(json.dumps(contract_payload, sort_keys=True, separators=(",", ":")))
        return (
            0,
            {
                "maxTokens": budget_meta["requested_max_tokens"],
                "includeContext": include_ctx,
                "temperatureMilli": budget_meta["temperature_milli"],
                "sanitizedMessages": sanitized_msgs,
                "systemPrompt": clean_sys,
                "redactionsApplied": redactions,
                "schemaCrc32c": schema_crc,
            },
            "ok",
        )


class HumanApprovalRegistry:
    """Thread-safe MCP 2025-11-25 Human-in-the-Loop approval gate matching C11/Rust/WASM ParApprovalTable (-29..-35)."""

    ERR_APPROVAL_DUPLICATE: int = -29
    ERR_APPROVAL_FULL: int = -30
    ERR_APPROVAL_KEY: int = -31
    ERR_APPROVAL_INACTIVE: int = -32
    ERR_APPROVAL_EXPIRED: int = -33
    ERR_SAMPLING_BUDGET: int = -34
    ERR_ELICITATION_SCHEMA: int = -35

    def __init__(
        self,
        max_slots: int = 64,
        *,
        default_ttl_ms: int = 30_000,
        ceiling_max_tokens: int = 4096,
        allow_all_servers_context: bool = False,
        now_ms_fn: Callable[[], int] | None = None,
    ) -> None:
        self.max_slots = max(1, int(max_slots))
        self.default_ttl_ms = max(1, int(default_ttl_ms))
        self.ceiling_max_tokens = max(1, int(ceiling_max_tokens))
        self.allow_all_servers_context = bool(allow_all_servers_context)
        self._now_ms_fn = now_ms_fn or (lambda: int(time.monotonic() * 1000.0))
        self._lock = threading.Lock()
        self._slots: OrderedDict[tuple[str, int, int], dict[str, Any]] = OrderedDict()
        self._requests_opened = 0
        self._accepted_count = 0
        self._declined_count = 0
        self._cancelled_count = 0
        self._expired_count = 0
        self._rejected_count = 0

    @classmethod
    def compute_key(cls, raw_id: Any) -> tuple[int, tuple[str, int, int] | None, int | str | None]:
        """Compute C11/Rust/WASM bit-identical ('APK1'/'APK2' dual CRC32C) approval slot key."""
        if raw_id is None or isinstance(raw_id, bool):
            return (cls.ERR_APPROVAL_KEY, None, None)
        if isinstance(raw_id, int):
            if raw_id < 0 or raw_id > 0xFFFFFFFFFFFFFFFF:
                return (cls.ERR_APPROVAL_KEY, None, None)
            return (0, ("int", raw_id, 0), raw_id)
        if isinstance(raw_id, str):
            raw_bytes = raw_id.encode("utf-8")
            if not raw_bytes or len(raw_bytes) > 256:
                return (cls.ERR_APPROVAL_KEY, None, None)
            h1 = crc32c(raw_bytes + b"APK1")
            h2 = crc32c(raw_bytes + b"APK2")
            primary = (h1 << 32) | h2
            return (0, ("str", primary, len(raw_bytes)), raw_id)
        return (cls.ERR_APPROVAL_KEY, None, None)

    def _sweep_expired_locked(self, now_ms: int) -> None:
        for slot in self._slots.values():
            if slot["state"] == "pending" and now_ms > slot["expires_ms"]:
                slot["state"] = "expired"
                self._expired_count += 1

    def _alloc_or_reuse_locked(self, key: tuple[str, int, int]) -> int:
        if key in self._slots:
            if self._slots[key]["state"] == "pending":
                return self.ERR_APPROVAL_DUPLICATE
            return 0
        if len(self._slots) < self.max_slots:
            return 0
        for existing_key, slot in list(self._slots.items()):
            if slot["state"] != "pending":
                del self._slots[existing_key]
                return 0
        return self.ERR_APPROVAL_FULL

    def begin_elicitation(
        self,
        approval_id: Any,
        message: Any,
        requested_schema: Any,
        *,
        ttl_ms: int | None = None,
    ) -> tuple[int, dict[str, Any] | None, str]:
        if not isinstance(message, str) or not message.strip():
            with self._lock:
                self._rejected_count += 1
            return (self.ERR_ELICITATION_SCHEMA, None, "Elicitation message must be a non-empty string")
        eff_ttl = self.default_ttl_ms if ttl_ms is None else int(ttl_ms)
        if isinstance(ttl_ms, bool) or eff_ttl <= 0:
            with self._lock:
                self._rejected_count += 1
            return (self.ERR_APPROVAL_KEY, None, "Approval ttl_ms must be > 0")
        rc_s, schema_crc, msg_s = ElicitationSchemaGuard.validate_requested_schema(requested_schema)
        if rc_s != 0:
            with self._lock:
                self._rejected_count += 1
            return (rc_s, None, msg_s)
        rc_k, key, norm_id = self.compute_key(approval_id)
        if rc_k != 0 or key is None or norm_id is None:
            with self._lock:
                self._rejected_count += 1
            return (self.ERR_APPROVAL_KEY, None, "Invalid approvalId: must be uint64 int or non-empty <=256B string")

        clean_msg, _ = RedactionGuard.sanitize(message.strip()[:512])
        with self._lock:
            now_ms = int(self._now_ms_fn())
            self._sweep_expired_locked(now_ms)
            alloc_rc = self._alloc_or_reuse_locked(key)
            if alloc_rc != 0:
                self._rejected_count += 1
                err_txt = (
                    f"Duplicate pending approval request {norm_id!r}"
                    if alloc_rc == self.ERR_APPROVAL_DUPLICATE
                    else "Approval gate table capacity (64 pending slots) exceeded"
                )
                return (alloc_rc, None, err_txt)
            exp_ms = now_ms + eff_ttl
            self._slots[key] = {
                "approval_id": norm_id,
                "kind": "elicitation",
                "state": "pending",
                "expires_ms": exp_ms,
                "max_tokens": 0,
                "schema_crc32c": schema_crc,
                "requested_schema": copy.deepcopy(requested_schema),
            }
            self._requests_opened += 1
        return (
            0,
            {
                "approvalId": norm_id,
                "kind": "elicitation",
                "status": "pending",
                "message": clean_msg,
                "requestedSchema": requested_schema,
                "schemaCrc32c": schema_crc,
                "expiresAtMs": exp_ms,
            },
            "ok",
        )

    def resolve_elicitation(
        self,
        approval_id: Any,
        action: Any,
        content: Any = None,
        *,
        expected_schema_crc32c: int | None = None,
    ) -> tuple[int, dict[str, Any] | None, str]:
        if action not in ("accept", "decline", "cancel"):
            with self._lock:
                self._rejected_count += 1
            return (self.ERR_APPROVAL_KEY, None, "Elicitation action must be 'accept', 'decline', or 'cancel'")
        rc_k, key, _ = self.compute_key(approval_id)
        if rc_k != 0 or key is None:
            with self._lock:
                self._rejected_count += 1
            return (self.ERR_APPROVAL_KEY, None, "Invalid approvalId")

        with self._lock:
            now_ms = int(self._now_ms_fn())
            slot = self._slots.get(key)
            if slot is None or slot["kind"] != "elicitation":
                self._rejected_count += 1
                return (self.ERR_APPROVAL_INACTIVE, None, "No matching elicitation approval request")
            if slot["state"] == "expired":
                self._rejected_count += 1
                return (self.ERR_APPROVAL_EXPIRED, None, "Elicitation approval window expired")
            if slot["state"] != "pending":
                self._rejected_count += 1
                return (self.ERR_APPROVAL_INACTIVE, None, "Elicitation approval request was already resolved")
            if now_ms > slot["expires_ms"]:
                slot["state"] = "expired"
                self._expired_count += 1
                self._rejected_count += 1
                return (self.ERR_APPROVAL_EXPIRED, None, "Elicitation approval window expired")
            if expected_schema_crc32c is not None and (expected_schema_crc32c & 0xFFFFFFFF) != slot["schema_crc32c"]:
                self._rejected_count += 1
                return (self.ERR_ELICITATION_SCHEMA, None, "Elicitation requestedSchema CRC32C mismatch")

            v_rc, validated, v_msg = ElicitationSchemaGuard.validate_response(
                action, content, slot["requested_schema"]
            )
            if v_rc != 0 or validated is None:
                self._rejected_count += 1
                return (v_rc, None, v_msg)

            if action == "accept":
                slot["state"] = "accepted"
                self._accepted_count += 1
            elif action == "decline":
                slot["state"] = "declined"
                self._declined_count += 1
            else:
                slot["state"] = "cancelled"
                self._cancelled_count += 1
            return (0, validated, "ok")

    def begin_sampling(
        self,
        approval_id: Any,
        params: dict[str, Any],
        *,
        ceiling_max_tokens: int | None = None,
        ttl_ms: int | None = None,
        allow_all_servers_context: bool | None = None,
    ) -> tuple[int, dict[str, Any] | None, str]:
        eff_ttl = self.default_ttl_ms if ttl_ms is None else int(ttl_ms)
        if isinstance(ttl_ms, bool) or eff_ttl <= 0:
            with self._lock:
                self._rejected_count += 1
            return (self.ERR_APPROVAL_KEY, None, "Approval ttl_ms must be > 0")
        eff_ceiling = self.ceiling_max_tokens if ceiling_max_tokens is None else int(ceiling_max_tokens)
        eff_allow_all = self.allow_all_servers_context if allow_all_servers_context is None else bool(allow_all_servers_context)

        s_rc, validated, s_msg = SamplingGovernanceGuard.validate_request(
            params,
            ceiling_max_tokens=eff_ceiling,
            allow_all_servers_context=eff_allow_all,
        )
        if s_rc != 0 or validated is None:
            with self._lock:
                self._rejected_count += 1
            return (s_rc, None, s_msg)

        rc_k, key, norm_id = self.compute_key(approval_id)
        if rc_k != 0 or key is None or norm_id is None:
            with self._lock:
                self._rejected_count += 1
            return (self.ERR_APPROVAL_KEY, None, "Invalid approvalId")

        with self._lock:
            now_ms = int(self._now_ms_fn())
            self._sweep_expired_locked(now_ms)
            alloc_rc = self._alloc_or_reuse_locked(key)
            if alloc_rc != 0:
                self._rejected_count += 1
                err_txt = (
                    f"Duplicate pending approval request {norm_id!r}"
                    if alloc_rc == self.ERR_APPROVAL_DUPLICATE
                    else "Approval gate table capacity (64 pending slots) exceeded"
                )
                return (alloc_rc, None, err_txt)
            exp_ms = now_ms + eff_ttl
            self._slots[key] = {
                "approval_id": norm_id,
                "kind": "sampling",
                "state": "pending",
                "expires_ms": exp_ms,
                "max_tokens": validated["maxTokens"],
                "schema_crc32c": validated["schemaCrc32c"],
            }
            self._requests_opened += 1
        return (
            0,
            {
                "approvalId": norm_id,
                "kind": "sampling",
                "status": "pending",
                **validated,
                "expiresAtMs": exp_ms,
            },
            "ok",
        )

    def resolve_sampling(
        self,
        approval_id: Any,
        action: Any,
        *,
        model: str = "local-qwen-moe",
        text: str = "",
        actual_tokens: int | None = None,
        stop_reason: str = "endTurn",
        expected_schema_crc32c: int | None = None,
    ) -> tuple[int, dict[str, Any] | None, str]:
        if action not in ("accept", "decline", "cancel"):
            with self._lock:
                self._rejected_count += 1
            return (self.ERR_APPROVAL_KEY, None, "Sampling action must be 'accept', 'decline', or 'cancel'")
        rc_k, key, norm_id = self.compute_key(approval_id)
        if rc_k != 0 or key is None or norm_id is None:
            with self._lock:
                self._rejected_count += 1
            return (self.ERR_APPROVAL_KEY, None, "Invalid approvalId")

        with self._lock:
            now_ms = int(self._now_ms_fn())
            slot = self._slots.get(key)
            if slot is None or slot["kind"] != "sampling":
                self._rejected_count += 1
                return (self.ERR_APPROVAL_INACTIVE, None, "No matching sampling approval request")
            if slot["state"] == "expired":
                self._rejected_count += 1
                return (self.ERR_APPROVAL_EXPIRED, None, "Sampling approval window expired")
            if slot["state"] != "pending":
                self._rejected_count += 1
                return (self.ERR_APPROVAL_INACTIVE, None, "Sampling approval request was already resolved")
            if now_ms > slot["expires_ms"]:
                slot["state"] = "expired"
                self._expired_count += 1
                self._rejected_count += 1
                return (self.ERR_APPROVAL_EXPIRED, None, "Sampling approval window expired")
            if expected_schema_crc32c is not None and (expected_schema_crc32c & 0xFFFFFFFF) != slot["schema_crc32c"]:
                self._rejected_count += 1
                return (self.ERR_SAMPLING_BUDGET, None, "Sampling budget contract CRC32C mismatch")

            if action == "decline":
                slot["state"] = "declined"
                self._declined_count += 1
                return (0, {"action": "decline", "approvalId": norm_id}, "User declined sampling request")
            if action == "cancel":
                slot["state"] = "cancelled"
                self._cancelled_count += 1
                return (0, {"action": "cancel", "approvalId": norm_id}, "User cancelled sampling request")

            clean_text, _ = RedactionGuard.sanitize(str(text))
            clean_model, _ = RedactionGuard.sanitize(str(model or "local-qwen-moe").strip()[:128])
            measured_toks = (
                actual_tokens
                if (actual_tokens is not None and not isinstance(actual_tokens, bool) and isinstance(actual_tokens, int))
                else max(1, (len(clean_text) + 3) // 4)
            )
            if measured_toks < 0 or measured_toks > slot["max_tokens"]:
                self._rejected_count += 1
                return (
                    self.ERR_SAMPLING_BUDGET,
                    None,
                    f"Sampling completion tokens ({measured_toks}) exceeded approved maxTokens ({slot['max_tokens']})",
                )
            slot["state"] = "accepted"
            self._accepted_count += 1
            return (
                0,
                {
                    "action": "accept",
                    "role": "assistant",
                    "content": {"type": "text", "text": clean_text},
                    "model": clean_model or "local-qwen-moe",
                    "stopReason": str(stop_reason or "endTurn"),
                    "actualTokens": measured_toks,
                },
                "ok",
            )

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            now_ms = int(self._now_ms_fn())
            self._sweep_expired_locked(now_ms)
            active_pending = sum(1 for s in self._slots.values() if s["state"] == "pending")
            return {
                "active_pending": active_pending,
                "tracked_slots": len(self._slots),
                "max_slots": self.max_slots,
                "ceiling_max_tokens": self.ceiling_max_tokens,
                "requests_opened": self._requests_opened,
                "accepted_count": self._accepted_count,
                "declined_count": self._declined_count,
                "cancelled_count": self._cancelled_count,
                "expired_count": self._expired_count,
                "rejected_count": self._rejected_count,
            }


class ContentAnnotationGuard:
    """MCP 2025-11-25 Content Annotations, audio/image media validator & 21-byte CRC32C ResourceLink provenance sealer (-42)."""

    ERR_CONTENT_ANNOTATION: int = -42
    AUDIENCE_USER: int = 0x01
    AUDIENCE_ASSISTANT: int = 0x02
    MAX_URI_LEN: int = 512
    MAX_MIME_LEN: int = 128

    ALLOWED_MEDIA_MIMES: ClassVar[dict[str, frozenset[str]]] = {
        "image": frozenset({"image/png", "image/jpeg", "image/webp", "image/gif", "image/svg+xml"}),
        "audio": frozenset(
            {"audio/wav", "audio/x-wav", "audio/mpeg", "audio/mp3", "audio/ogg", "audio/webm", "audio/aac", "audio/flac"}
        ),
    }
    _ANN_KEYS: frozenset[str] = frozenset({"audience", "priority", "lastModified"})

    @staticmethod
    def _days_in_month(year: int, month: int) -> int:
        if month < 1 or month > 12:
            return 0
        if month == 2:
            leap = ((year % 4 == 0) and (year % 100 != 0)) or (year % 400 == 0)
            return 29 if leap else 28
        return (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)[month - 1]

    @classmethod
    def validate_iso8601_utc(cls, ts: Any) -> bool:
        """Bit-identical leap-year-checked ISO-8601 UTC validator to C11 par_validate_content_annotations."""
        if not isinstance(ts, str):
            return False
        try:
            raw = ts.encode("ascii")
        except UnicodeEncodeError:
            return False
        n = len(raw)
        if n < 20 or n > 30:
            return False
        if (
            raw[4:5] != b"-"
            or raw[7:8] != b"-"
            or raw[10:11] != b"T"
            or raw[13:14] != b":"
            or raw[16:17] != b":"
            or raw[n - 1 : n] != b"Z"
        ):
            return False
        fields = (raw[0:4], raw[5:7], raw[8:10], raw[11:13], raw[14:16], raw[17:19])
        if any(not f.isdigit() for f in fields):
            return False
        year, month, day, hour, minute, second = (int(f.decode("ascii")) for f in fields)
        max_day = cls._days_in_month(year, month)
        if year < 1970 or max_day == 0 or day < 1 or day > max_day or hour > 23 or minute > 59 or second > 59:
            return False
        return not (n > 20 and (n == 21 or raw[19:20] != b"." or not raw[20 : n - 1].isdigit()))

    @classmethod
    def validate_annotations(cls, annotations: Any) -> tuple[int, dict[str, Any] | None, int, int, str]:
        if annotations is None:
            return (0, None, 0, -1, "ok")
        if not isinstance(annotations, dict) or isinstance(annotations, bool):
            return (cls.ERR_CONTENT_ANNOTATION, None, 0, -1, "Content annotations must be a JSON object")
        extra = set(annotations.keys()) - cls._ANN_KEYS
        if extra:
            bad_k = min(extra)
            return (cls.ERR_CONTENT_ANNOTATION, None, 0, -1, f"Unknown content annotation property '{bad_k}'")

        clean: dict[str, Any] = {}
        audience_mask = 0
        if "audience" in annotations and annotations["audience"] is not None:
            aud = annotations["audience"]
            if not isinstance(aud, list) or not aud or len(aud) > 2:
                return (
                    cls.ERR_CONTENT_ANNOTATION,
                    None,
                    0,
                    -1,
                    "Annotation 'audience' must be a non-empty array of ['user', 'assistant']",
                )
            seen_roles: list[str] = []
            for role in aud:
                if role == "user" and "user" not in seen_roles:
                    audience_mask |= cls.AUDIENCE_USER
                    seen_roles.append("user")
                elif role == "assistant" and "assistant" not in seen_roles:
                    audience_mask |= cls.AUDIENCE_ASSISTANT
                    seen_roles.append("assistant")
                else:
                    return (
                        cls.ERR_CONTENT_ANNOTATION,
                        None,
                        0,
                        -1,
                        f"Invalid or duplicate annotation audience role: {role!r}",
                    )
            clean["audience"] = seen_roles

        priority_milli = -1
        if "priority" in annotations and annotations["priority"] is not None:
            prio = annotations["priority"]
            if isinstance(prio, bool) or not isinstance(prio, (int, float)) or not math.isfinite(prio):
                return (cls.ERR_CONTENT_ANNOTATION, None, 0, -1, "Annotation 'priority' must be a finite number in [0.0, 1.0]")
            fprio = float(prio)
            if fprio < 0.0 or fprio > 1.0:
                return (cls.ERR_CONTENT_ANNOTATION, None, 0, -1, f"Annotation 'priority' ({prio}) outside [0.0, 1.0]")
            priority_milli = round(fprio * 1000.0)
            clean["priority"] = round(priority_milli / 1000.0, 3)

        if "lastModified" in annotations and annotations["lastModified"] is not None:
            lm = annotations["lastModified"]
            if not cls.validate_iso8601_utc(lm):
                return (
                    cls.ERR_CONTENT_ANNOTATION,
                    None,
                    0,
                    -1,
                    f"Annotation 'lastModified' ({lm!r}) must be valid leap-year-checked ISO-8601 UTC",
                )
            clean["lastModified"] = lm

        return (0, clean, audience_mask, priority_milli, "ok")

    @classmethod
    def compute_provenance_seal(
        cls,
        uri: Any,
        mime_type: Any = "",
        content_crc32c: int = 0,
        *,
        audience_mask: int = 0,
        priority_milli: int = -1,
        roots_guard: WorkspaceRootsGuard | None = None,
    ) -> tuple[int, int, str, str]:
        """Compute bit-identical 21-byte RLNK Castagnoli CRC32C seal matching C11/Rust/WASM par_seal_resource_link."""
        if (
            isinstance(audience_mask, bool)
            or not isinstance(audience_mask, int)
            or (audience_mask & ~0x03) != 0
            or isinstance(priority_milli, bool)
            or not isinstance(priority_milli, int)
            or priority_milli < -1
            or priority_milli > 1000
        ):
            return (cls.ERR_CONTENT_ANNOTATION, 0, "", "Invalid audience_mask or priority_milli for ResourceLink seal")
        if not isinstance(uri, str):
            return (cls.ERR_CONTENT_ANNOTATION, 0, "", "ResourceLink uri must be a non-empty visible ASCII string")
        raw_mime = "" if mime_type is None else mime_type
        if not isinstance(raw_mime, str):
            return (cls.ERR_CONTENT_ANNOTATION, 0, "", "ResourceLink mimeType must be a string")
        try:
            uri_bytes = uri.encode("ascii")
            mime_bytes = raw_mime.encode("ascii")
        except UnicodeEncodeError:
            return (cls.ERR_CONTENT_ANNOTATION, 0, "", "ResourceLink uri and mimeType must be visible ASCII")
        if not uri_bytes or len(uri_bytes) > cls.MAX_URI_LEN or len(mime_bytes) > cls.MAX_MIME_LEN:
            return (cls.ERR_CONTENT_ANNOTATION, 0, "", "ResourceLink uri or mimeType length out of bounds")
        if any(b < 0x21 or b > 0x7E for b in uri_bytes) or any(b < 0x21 or b > 0x7E for b in mime_bytes):
            return (cls.ERR_CONTENT_ANNOTATION, 0, "", "ResourceLink uri and mimeType must use visible ASCII (0x21..0x7E)")

        canon_uri = uri
        if uri.lower().startswith("file:"):
            if roots_guard is not None:
                rc_u, canon_uri, msg_u = roots_guard.check_uri(uri)
            else:
                rc_u, canon_uri, msg_u = WorkspaceRootsGuard.normalize_file_uri(uri)
            if rc_u != 0:
                return (rc_u, 0, canon_uri, msg_u)

        uri_crc = crc32c(uri_bytes)
        mime_crc = crc32c(mime_bytes) if mime_bytes else 0
        preimage = struct.pack(
            "<IIIBi4s",
            uri_crc & 0xFFFFFFFF,
            mime_crc & 0xFFFFFFFF,
            content_crc32c & 0xFFFFFFFF,
            audience_mask & 0xFF,
            priority_milli,
            b"RLNK",
        )
        return (0, crc32c(preimage), canon_uri, "ok")

    @classmethod
    def build_resource_link(
        cls,
        uri: str,
        name: str,
        *,
        description: str | None = None,
        mime_type: str = "application/json",
        size: int | None = None,
        content_text: str | bytes = b"",
        annotations: dict[str, Any] | None = None,
        roots_guard: WorkspaceRootsGuard | None = None,
    ) -> tuple[int, dict[str, Any] | None, str]:
        rc_a, clean_ann, aud_mask, prio_milli, msg_a = cls.validate_annotations(annotations)
        if rc_a != 0:
            return (rc_a, None, msg_a)
        if not isinstance(name, str) or not name.strip():
            return (cls.ERR_CONTENT_ANNOTATION, None, "ResourceLink 'name' must be a non-empty string")
        target_uri = uri
        if isinstance(uri, str) and uri.lower().startswith("file:"):
            if roots_guard is not None:
                rc_u, canon_uri, msg_u = roots_guard.check_uri(uri)
            else:
                rc_u, canon_uri, msg_u = WorkspaceRootsGuard.normalize_file_uri(uri)
            if rc_u != 0:
                return (rc_u, None, msg_u)
            target_uri = canon_uri

        content_crc = crc32c(content_text)
        rc_s, seal, _, msg_s = cls.compute_provenance_seal(
            target_uri,
            mime_type,
            content_crc,
            audience_mask=aud_mask,
            priority_milli=prio_milli,
        )
        if rc_s != 0:
            return (rc_s, None, msg_s)

        clean_name, _ = RedactionGuard.sanitize(name.strip()[:256])
        link: dict[str, Any] = {
            "type": "resource_link",
            "uri": target_uri,
            "name": clean_name or "resource",
        }
        if description is not None:
            clean_desc, _ = RedactionGuard.sanitize(str(description).strip()[:512])
            if clean_desc:
                link["description"] = clean_desc
        if mime_type:
            link["mimeType"] = mime_type
        if size is not None:
            if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                return (cls.ERR_CONTENT_ANNOTATION, None, "ResourceLink 'size' must be a non-negative integer")
            link["size"] = size
        if clean_ann is not None:
            link["annotations"] = clean_ann
        link["_meta"] = {
            "contentCrc32c": f"0x{content_crc:08X}",
            "provenanceCrc32c": f"0x{seal:08X}",
        }
        return (0, link, "ok")

    @classmethod
    def validate_media_block(
        cls,
        block: Any,
        *,
        max_decoded_bytes: int = 1_048_576,
    ) -> tuple[int, dict[str, Any] | None, str]:
        if not isinstance(block, dict) or isinstance(block, bool):
            return (cls.ERR_CONTENT_ANNOTATION, None, "Media content block must be a JSON object")
        btype = block.get("type")
        if btype not in ("image", "audio"):
            return (cls.ERR_CONTENT_ANNOTATION, None, f"Unsupported media block type {btype!r}")
        mime = block.get("mimeType")
        if not isinstance(mime, str) or mime.lower() not in cls.ALLOWED_MEDIA_MIMES[btype]:
            return (cls.ERR_CONTENT_ANNOTATION, None, f"Disallowed {btype} mimeType {mime!r}")
        raw_b64 = block.get("data")
        if not isinstance(raw_b64, str) or not raw_b64 or (len(raw_b64) % 4 != 0):
            return (cls.ERR_CONTENT_ANNOTATION, None, f"Media block '{btype}' data must be canonical RFC 4648 Base64")
        try:
            decoded = base64.b64decode(raw_b64, validate=True)
        except ValueError:
            return (cls.ERR_CONTENT_ANNOTATION, None, f"Malformed Base64 payload in '{btype}' block")
        if not decoded or len(decoded) > max_decoded_bytes:
            return (cls.ERR_CONTENT_ANNOTATION, None, f"Decoded '{btype}' byte length out of bounds")
        rc_a, clean_ann, _, _, msg_a = cls.validate_annotations(block.get("annotations"))
        if rc_a != 0:
            return (rc_a, None, msg_a)
        norm: dict[str, Any] = {
            "type": btype,
            "mimeType": mime.lower(),
            "data": base64.b64encode(decoded).decode("ascii"),
            "_meta": {"byteLength": len(decoded), "dataCrc32c": f"0x{crc32c(decoded):08X}"},
        }
        if clean_ann is not None:
            norm["annotations"] = clean_ann
        return (0, norm, "ok")


class StreamableSessionRegistry:
    """Thread-safe MCP 2025-11-25 Mcp-Session-Id & Last-Event-ID SSE replay ring matching C11/Rust/WASM ParSessionTable (-36..-41)."""

    ERR_SESSION_SYNTAX: int = -36
    ERR_SESSION_UNKNOWN: int = -37
    ERR_SESSION_EXPIRED: int = -38
    ERR_SESSION_BINDING: int = -39
    ERR_SESSION_FULL: int = -40
    ERR_REPLAY_GAP: int = -41

    MAX_ID_LEN: int = 128
    MAX_RING_CAP: int = 64

    def __init__(
        self,
        max_slots: int = 64,
        *,
        ring_capacity: int = 32,
        default_ttl_ms: int = 300_000,
        now_ms_fn: Callable[[], int] | None = None,
    ) -> None:
        self.max_slots = max(1, min(64, int(max_slots)))
        self.ring_capacity = max(1, min(self.MAX_RING_CAP, int(ring_capacity)))
        self.default_ttl_ms = max(1, int(default_ttl_ms))
        self._now_ms_fn = now_ms_fn or (lambda: int(time.monotonic() * 1000.0))
        self._lock = threading.Lock()
        self._slots: OrderedDict[tuple[int, int], dict[str, Any]] = OrderedDict()
        self._sessions_opened = 0
        self._events_appended = 0
        self._replays_served = 0
        self._sessions_terminated = 0
        self._sessions_expired = 0
        self._rejected_count = 0

    @classmethod
    def validate_session_id(cls, session_id: Any) -> tuple[int, bytes, int]:
        """Validate MCP 2025-11-25 visible-ASCII Mcp-Session-Id (0x21..0x7E, 1..128B) & compute SID1/SID2 hash64."""
        if not isinstance(session_id, str) or isinstance(session_id, bool):
            return (cls.ERR_SESSION_SYNTAX, b"", 0)
        try:
            raw = session_id.encode("ascii")
        except UnicodeEncodeError:
            return (cls.ERR_SESSION_SYNTAX, b"", 0)
        if not raw or len(raw) > cls.MAX_ID_LEN or any(b < 0x21 or b > 0x7E for b in raw):
            return (cls.ERR_SESSION_SYNTAX, b"", 0)
        h1 = crc32c(raw + b"SID1")
        h2 = crc32c(raw + b"SID2")
        return (0, raw, (h1 << 32) | h2)

    @staticmethod
    def client_fingerprint(client_id: str | None) -> int:
        norm = str(client_id or "anonymous").strip()[:128] or "anonymous"
        fp = crc32c(norm)
        return fp if fp != 0 else 1

    @staticmethod
    def mint_session_id() -> str:
        return f"mcp-sess-{secrets.token_hex(16)}"

    @classmethod
    def encode_event_id(cls, session_id: str, seq: int) -> str:
        rc, _, hash64 = cls.validate_session_id(session_id)
        if rc != 0 or seq <= 0:
            raise ValueError("Invalid session_id or non-positive seq for SSE event id")
        seal = crc32c(struct.pack("<QI4s", hash64, seq & 0xFFFFFFFF, b"SEVT"))
        return f"{seq}:{seal:08x}"

    @classmethod
    def parse_event_id(cls, session_id: str, raw_event_id: Any) -> tuple[int, int, str]:
        rc, _, hash64 = cls.validate_session_id(session_id)
        if rc != 0:
            return (rc, 0, "Invalid Mcp-Session-Id syntax")
        if isinstance(raw_event_id, int) and not isinstance(raw_event_id, bool):
            if raw_event_id < 0:
                return (cls.ERR_SESSION_SYNTAX, 0, "Negative Last-Event-ID sequence")
            if raw_event_id == 0:
                return (cls.ERR_REPLAY_GAP, 0, "Last-Event-ID sequence must be >= 1")
            return (0, raw_event_id, "ok")
        if not isinstance(raw_event_id, str) or ":" not in raw_event_id:
            return (cls.ERR_SESSION_SYNTAX, 0, "Malformed Last-Event-ID: expected '<seq>:<8hex-seal>'")
        seq_part, seal_part = raw_event_id.strip().split(":", 1)
        if not seq_part.isdigit() or len(seal_part) != 8 or any(ch not in "0123456789abcdefABCDEF" for ch in seal_part):
            return (cls.ERR_SESSION_SYNTAX, 0, "Malformed Last-Event-ID digits or CRC32C seal")
        seq = int(seq_part)
        if seq == 0 or seq > 0xFFFFFFFF:
            return (cls.ERR_REPLAY_GAP, 0, "Last-Event-ID sequence out of bounds")
        expected_seal = crc32c(struct.pack("<QI4s", hash64, seq & 0xFFFFFFFF, b"SEVT"))
        if int(seal_part, 16) != expected_seal:
            return (cls.ERR_REPLAY_GAP, 0, "Last-Event-ID session seal mismatch (cross-session or tampered cursor)")
        return (0, seq, "ok")

    def _sweep_expired_locked(self, now_ms: int) -> None:
        for slot in self._slots.values():
            if slot["state"] == "active" and now_ms > slot["expires_ms"]:
                slot["state"] = "expired"
                slot["events"].clear()
                self._sessions_expired += 1

    def open_session(
        self,
        session_id: Any,
        *,
        client_id: str | None = "anonymous",
        ring_capacity: int | None = None,
        ttl_ms: int | None = None,
    ) -> tuple[int, dict[str, Any] | None, str]:
        eff_cap = self.ring_capacity if ring_capacity is None else ring_capacity
        eff_ttl = self.default_ttl_ms if ttl_ms is None else ttl_ms
        if (
            isinstance(eff_cap, bool)
            or not isinstance(eff_cap, int)
            or eff_cap < 1
            or eff_cap > self.MAX_RING_CAP
            or isinstance(eff_ttl, bool)
            or not isinstance(eff_ttl, int)
            or eff_ttl <= 0
        ):
            with self._lock:
                self._rejected_count += 1
            return (self.ERR_SESSION_SYNTAX, None, "Invalid session ring_capacity (1..64) or ttl_ms")

        rc, raw, hash64 = self.validate_session_id(session_id)
        if rc != 0:
            with self._lock:
                self._rejected_count += 1
            return (rc, None, "Invalid Mcp-Session-Id: must be 1..128 visible ASCII bytes (0x21..0x7E)")

        key = (hash64, len(raw))
        client_crc = self.client_fingerprint(client_id)
        with self._lock:
            now_ms = int(self._now_ms_fn())
            self._sweep_expired_locked(now_ms)
            existing = self._slots.get(key)
            if existing is not None:
                if existing["state"] == "active":
                    self._rejected_count += 1
                    return (self.ERR_SESSION_BINDING, None, "Duplicate active Mcp-Session-Id")
                del self._slots[key]
            elif len(self._slots) >= self.max_slots:
                reclaimed = False
                for k_old, s_old in list(self._slots.items()):
                    if s_old["state"] != "active":
                        del self._slots[k_old]
                        reclaimed = True
                        break
                if not reclaimed:
                    self._rejected_count += 1
                    return (self.ERR_SESSION_FULL, None, "Active session table capacity (64 slots) exceeded")

            exp_ms = now_ms + eff_ttl
            self._slots[key] = {
                "session_id": str(session_id),
                "state": "active",
                "ring_capacity": eff_cap,
                "client_crc32c": client_crc,
                "expires_ms": exp_ms,
                "last_seq": 0,
                "min_retained_seq": 0,
                "events": deque(maxlen=eff_cap),
            }
            self._sessions_opened += 1
            return (
                0,
                {
                    "sessionId": str(session_id),
                    "ringCapacity": eff_cap,
                    "clientCrc32c": client_crc,
                    "expiresAtMs": exp_ms,
                },
                "ok",
            )

    def _find_active_locked(
        self,
        session_id: Any,
        client_id: str | None,
        now_ms: int,
    ) -> tuple[int, dict[str, Any] | None, str]:
        rc, raw, hash64 = self.validate_session_id(session_id)
        if rc != 0:
            self._rejected_count += 1
            return (rc, None, "Invalid Mcp-Session-Id header syntax")
        slot = self._slots.get((hash64, len(raw)))
        if slot is None or slot["state"] == "terminated":
            self._rejected_count += 1
            return (self.ERR_SESSION_UNKNOWN, None, "Unknown or terminated Mcp-Session-Id")
        if slot["state"] == "expired":
            self._rejected_count += 1
            return (self.ERR_SESSION_EXPIRED, None, "Mcp-Session-Id TTL window expired")
        if now_ms > slot["expires_ms"]:
            slot["state"] = "expired"
            slot["events"].clear()
            self._sessions_expired += 1
            self._rejected_count += 1
            return (self.ERR_SESSION_EXPIRED, None, "Mcp-Session-Id TTL window expired")
        if slot["client_crc32c"] != 0 and slot["client_crc32c"] != self.client_fingerprint(client_id):
            self._rejected_count += 1
            return (self.ERR_SESSION_BINDING, None, "Mcp-Session-Id client binding mismatch")
        return (0, slot, "ok")

    def verify_session(self, session_id: Any, *, client_id: str | None = "anonymous") -> tuple[int, str]:
        with self._lock:
            rc, _, msg = self._find_active_locked(session_id, client_id, int(self._now_ms_fn()))
            return (rc, msg)

    def append_event(
        self,
        session_id: str,
        payload: Any,
        *,
        client_id: str | None = "anonymous",
    ) -> tuple[int, dict[str, Any] | None, str]:
        with self._lock:
            rc, slot, msg = self._find_active_locked(session_id, client_id, int(self._now_ms_fn()))
            if rc != 0 or slot is None:
                return (rc, None, msg)
            seq = int(slot["last_seq"]) + 1
            event_id = self.encode_event_id(slot["session_id"], seq)
            slot["last_seq"] = seq
            cap = int(slot["ring_capacity"])
            slot["min_retained_seq"] = (seq - cap + 1) if seq > cap else 1
            slot["events"].append({"seq": seq, "eventId": event_id, "data": copy.deepcopy(payload)})
            self._events_appended += 1
            return (0, {"seq": seq, "eventId": event_id, "minRetainedSeq": slot["min_retained_seq"]}, "ok")

    def replay_events(
        self,
        session_id: Any,
        last_event_id: Any,
        *,
        client_id: str | None = "anonymous",
    ) -> tuple[int, list[dict[str, Any]], str]:
        with self._lock:
            rc, slot, msg = self._find_active_locked(session_id, client_id, int(self._now_ms_fn()))
            if rc != 0 or slot is None:
                return (rc, [], msg)
            p_rc, last_seq, p_msg = self.parse_event_id(slot["session_id"], last_event_id)
            if p_rc != 0:
                self._rejected_count += 1
                return (p_rc, [], p_msg)
            cur_last = int(slot["last_seq"])
            if cur_last == 0 or last_seq > cur_last:
                self._rejected_count += 1
                return (self.ERR_REPLAY_GAP, [], "Last-Event-ID is ahead of session stream")
            next_seq = last_seq + 1
            if next_seq == cur_last + 1:
                self._replays_served += 1
                return (0, [], "ok")
            if next_seq < int(slot["min_retained_seq"]):
                self._rejected_count += 1
                return (
                    self.ERR_REPLAY_GAP,
                    [],
                    f"Replay gap: event #{next_seq} was evicted (minRetainedSeq={slot['min_retained_seq']})",
                )
            replayed = [copy.deepcopy(ev) for ev in slot["events"] if ev["seq"] >= next_seq]
            self._replays_served += 1
            return (0, replayed, "ok")

    def terminate_session(self, session_id: Any, *, client_id: str | None = "anonymous") -> tuple[int, str]:
        with self._lock:
            rc, slot, msg = self._find_active_locked(session_id, client_id, int(self._now_ms_fn()))
            if rc != 0 or slot is None:
                return (rc, msg)
            slot["state"] = "terminated"
            slot["events"].clear()
            self._sessions_terminated += 1
            return (0, "ok")

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            self._sweep_expired_locked(int(self._now_ms_fn()))
            active = sum(1 for s in self._slots.values() if s["state"] == "active")
            return {
                "active_sessions": active,
                "tracked_slots": len(self._slots),
                "max_slots": self.max_slots,
                "ring_capacity": self.ring_capacity,
                "sessions_opened": self._sessions_opened,
                "events_appended": self._events_appended,
                "replays_served": self._replays_served,
                "sessions_terminated": self._sessions_terminated,
                "sessions_expired": self._sessions_expired,
                "rejected_count": self._rejected_count,
            }


_PKCE_UNRESERVED_BYTES: frozenset[int] = frozenset(
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~"
)
_B64URL_ALPHABET: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"


class OAuthProtectedResourceGuard:
    """MCP 2025-11-25 OAuth 2.1 Protected Resource Metadata (RFC 9728), PKCE S256 (RFC 7636), & RFC 8707/6750 Audience-Bound Bearer Token Introspection Guard."""

    ERR_PKCE_VERIFIER: int = -43
    ERR_PKCE_CHALLENGE: int = -44
    ERR_OAUTH_RESOURCE_URI: int = -45
    ERR_BEARER_SYNTAX: int = -46
    ERR_TOKEN_INACTIVE: int = -47
    ERR_TOKEN_AUDIENCE: int = -48
    ERR_TOKEN_SCOPE: int = -49
    ERR_TOKEN_FULL: int = -50

    SCOPE_READ: int = 0x01
    SCOPE_TOOLS: int = 0x02
    SCOPE_PROMPTS: int = 0x04
    SCOPE_SAMPLING: int = 0x08
    SCOPE_ALL: int = 0x0F

    SCOPE_DEFINITIONS: tuple[tuple[int, str], ...] = (
        (0x01, "mcp:read"),
        (0x02, "mcp:tools"),
        (0x04, "mcp:prompts"),
        (0x08, "mcp:sampling"),
    )

    MIN_PKCE_VERIFIER_LEN: int = 43
    MAX_PKCE_VERIFIER_LEN: int = 128
    PKCE_S256_B64_LEN: int = 43
    MAX_BEARER_LEN: int = 512
    MAX_RESOURCE_URI_LEN: int = 256

    def __init__(
        self,
        resource_uri: str = "https://mcp.example.com/mcp",
        *,
        authorization_servers: list[str] | None = None,
        max_slots: int = 64,
        default_ttl_ms: int = 300_000,
        now_ms_fn: Callable[[], int] | None = None,
    ) -> None:
        rc, canon_uri, aud_crc, msg = self.normalize_resource_uri(resource_uri)
        if rc != 0:
            raise ValueError(f"Invalid OAuth protected resource URI {resource_uri!r}: {msg} ({rc})")
        self._resource_uri = canon_uri
        self._audience_crc32c = aud_crc
        raw_as = authorization_servers or ["https://auth.example.com"]
        canon_as: list[str] = []
        for srv_uri in raw_as:
            as_rc, as_canon, _, as_msg = self.normalize_resource_uri(srv_uri)
            if as_rc != 0:
                raise ValueError(f"Invalid OAuth authorization server URI {srv_uri!r}: {as_msg} ({as_rc})")
            canon_as.append(as_canon)
        self._authorization_servers = canon_as
        self.max_slots = max(1, min(64, int(max_slots)))
        self.default_ttl_ms = max(1, int(default_ttl_ms))
        self._now_ms_fn: Callable[[], int] = now_ms_fn or (lambda: time.monotonic_ns() // 1_000_000)
        self._slots: OrderedDict[tuple[int, int], dict[str, Any]] = OrderedDict()
        self._lock = threading.Lock()
        self._tokens_registered = 0
        self._introspections_ok = 0
        self._tokens_revoked = 0
        self._tokens_expired = 0
        self._audience_rejections = 0
        self._scope_rejections = 0
        self._inactive_rejections = 0
        self._pkce_verifications = 0

    @property
    def resource_uri(self) -> str:
        return self._resource_uri

    @property
    def audience_crc32c(self) -> int:
        return self._audience_crc32c

    @property
    def authorization_servers(self) -> list[str]:
        return list(self._authorization_servers)

    def set_resource_uri(self, resource_uri: str) -> tuple[str, int]:
        rc, canon_uri, aud_crc, msg = self.normalize_resource_uri(resource_uri)
        if rc != 0:
            raise ValueError(f"Invalid OAuth protected resource URI {resource_uri!r}: {msg} ({rc})")
        with self._lock:
            self._resource_uri = canon_uri
            self._audience_crc32c = aud_crc
        return (canon_uri, aud_crc)

    @classmethod
    def compute_pkce_s256(cls, verifier: Any) -> tuple[int, str, str]:
        """Compute RFC 7636 S256 PKCE code_challenge bit-identical to C11 par_pkce_compute_s256."""
        if not isinstance(verifier, str):
            return (cls.ERR_PKCE_VERIFIER, "", "PKCE code_verifier must be an ASCII string")
        try:
            raw = verifier.encode("ascii")
        except UnicodeEncodeError:
            return (cls.ERR_PKCE_VERIFIER, "", "PKCE code_verifier contains non-ASCII characters")
        if len(raw) < cls.MIN_PKCE_VERIFIER_LEN or len(raw) > cls.MAX_PKCE_VERIFIER_LEN:
            return (cls.ERR_PKCE_VERIFIER, "", "PKCE code_verifier length must be in [43, 128]")
        for b in raw:
            if b not in _PKCE_UNRESERVED_BYTES:
                return (cls.ERR_PKCE_VERIFIER, "", "PKCE code_verifier contains forbidden character outside RFC 7636 §4.1")
        digest = hashlib.sha256(raw).digest()
        challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
        return (0, challenge, "ok")

    @classmethod
    def verify_pkce_s256(
        cls,
        verifier: Any,
        expected_challenge: Any,
        *,
        method: str = "S256",
    ) -> tuple[int, str]:
        """Verify RFC 7636 S256 PKCE challenge in constant time, rejecting OAuth 2.1 forbidden 'plain' method."""
        rc, computed, msg = cls.compute_pkce_s256(verifier)
        if rc != 0:
            return (rc, msg)
        if method != "S256" or not isinstance(expected_challenge, str) or len(expected_challenge) != cls.PKCE_S256_B64_LEN:
            return (cls.ERR_PKCE_CHALLENGE, "PKCE challenge must use S256 method with 43-char base64url digest")
        for idx, ch in enumerate(expected_challenge):
            sx = _B64URL_ALPHABET.find(ch)
            if sx < 0:
                return (cls.ERR_PKCE_CHALLENGE, "PKCE challenge contains non-base64url character")
            if idx == 42 and (sx & 0x03) != 0:
                return (cls.ERR_PKCE_CHALLENGE, "PKCE challenge has non-canonical trailing base64url bits")
        if not secrets.compare_digest(computed.encode("ascii"), expected_challenge.encode("ascii")):
            return (cls.ERR_PKCE_CHALLENGE, "PKCE S256 code_verifier mismatch")
        return (0, "ok")

    @classmethod
    def normalize_resource_uri(cls, uri: Any) -> tuple[int, str, int, str]:
        """Canonicalize RFC 9728 / RFC 8707 resource indicator URI bit-identical to C11 par_oauth_normalize_resource_uri."""
        if not isinstance(uri, str):
            return (cls.ERR_OAUTH_RESOURCE_URI, "", 0, "Resource URI must be a string")
        try:
            raw = uri.encode("ascii")
        except UnicodeEncodeError:
            return (cls.ERR_OAUTH_RESOURCE_URI, "", 0, "Resource URI must be printable ASCII")
        n = len(raw)
        if n == 0 or n > cls.MAX_RESOURCE_URI_LEN:
            return (cls.ERR_OAUTH_RESOURCE_URI, "", 0, "Resource URI length out of bounds")
        for b in raw:
            if b < 0x21 or b > 0x7E or b in (0x23, 0x3F, 0x5C, 0x40):  # '#', '?', '\\', '@'
                return (cls.ERR_OAUTH_RESOURCE_URI, "", 0, "Resource URI contains forbidden query, fragment, userinfo, or control byte")

        low_prefix = raw[:8].lower()
        if n > 8 and low_prefix == b"https://":
            is_https = True
            scheme_len = 8
        elif n > 7 and raw[:7].lower() == b"http://":
            is_https = False
            scheme_len = 7
        else:
            return (cls.ERR_OAUTH_RESOURCE_URI, "", 0, "Resource URI must start with https:// or loopback http://")

        slash_pos = raw.find(b"/", scheme_len)
        if slash_pos == -1:
            slash_pos = n
        if slash_pos == scheme_len:
            return (cls.ERR_OAUTH_RESOURCE_URI, "", 0, "Resource URI missing authority host")

        auth = raw[scheme_len:slash_pos]
        port_bytes: bytes | None = None
        if auth.startswith(b"["):
            close_br = auth.find(b"]", 1)
            if close_br <= 1:
                return (cls.ERR_OAUTH_RESOURCE_URI, "", 0, "Malformed IPv6 literal in resource URI")
            host_raw = auth[: close_br + 1]
            rest = auth[close_br + 1 :]
            if rest:
                if not rest.startswith(b":") or len(rest) == 1:
                    return (cls.ERR_OAUTH_RESOURCE_URI, "", 0, "Malformed port after IPv6 host")
                port_bytes = rest[1:]
        else:
            colon_idx = auth.find(b":")
            if colon_idx != -1:
                host_raw = auth[:colon_idx]
                port_bytes = auth[colon_idx + 1 :]
                if not host_raw or not port_bytes or b":" in port_bytes:
                    return (cls.ERR_OAUTH_RESOURCE_URI, "", 0, "Malformed host:port authority")
            else:
                host_raw = auth

        host_low = host_raw.lower()
        if host_low.startswith(b"["):
            for idx, b in enumerate(host_low):
                ok = (
                    (idx == 0 and b == ord("["))
                    or (idx == len(host_low) - 1 and b == ord("]"))
                    or (ord("0") <= b <= ord("9"))
                    or (ord("a") <= b <= ord("f"))
                    or b == ord(":")
                )
                if not ok:
                    return (cls.ERR_OAUTH_RESOURCE_URI, "", 0, "Invalid character in IPv6 literal")
        else:
            for idx, b in enumerate(host_low):
                ok = (ord("a") <= b <= ord("z")) or (ord("0") <= b <= ord("9")) or b in (ord("."), ord("-"))
                if not ok:
                    return (cls.ERR_OAUTH_RESOURCE_URI, "", 0, "Invalid character in resource host")
                if (idx == 0 or idx == len(host_low) - 1) and b in (ord("."), ord("-")):
                    return (cls.ERR_OAUTH_RESOURCE_URI, "", 0, "Invalid leading or trailing punctuation in host")
                if idx > 0 and b == ord(".") and host_low[idx - 1] == ord("."):
                    return (cls.ERR_OAUTH_RESOURCE_URI, "", 0, "Consecutive dots in resource host")

        if not is_https and host_low not in (b"localhost", b"127.0.0.1", b"[::1]"):
            return (cls.ERR_OAUTH_RESOURCE_URI, "", 0, "Non-HTTPS resource URI is restricted to loopback hosts")

        keep_port = False
        if port_bytes is not None:
            if len(port_bytes) > 5 or port_bytes[0] == ord("0") or not all(ord("0") <= b <= ord("9") for b in port_bytes):
                return (cls.ERR_OAUTH_RESOURCE_URI, "", 0, "Invalid port syntax in resource URI")
            port_val = int(port_bytes.decode("ascii"))
            if port_val < 1 or port_val > 65535:
                return (cls.ERR_OAUTH_RESOURCE_URI, "", 0, "Port out of range in resource URI")
            keep_port = port_val != (443 if is_https else 80)

        out = bytearray(b"https://" if is_https else b"http://")
        out.extend(host_low)
        if keep_port and port_bytes is not None:
            out.append(ord(":"))
            out.extend(port_bytes)

        p = slash_pos
        while p < n:
            while p < n and raw[p] == ord("/"):
                p += 1
            if p >= n:
                break
            seg_end = raw.find(b"/", p)
            if seg_end == -1:
                seg_end = n
            seg = raw[p:seg_end]
            if seg in (b".", b".."):
                return (cls.ERR_OAUTH_RESOURCE_URI, "", 0, "Dot-segment traversal forbidden in resource URI")
            low_seg = seg.lower()
            if b"%2f" in low_seg or b"%5c" in low_seg or b"%00" in low_seg:
                return (cls.ERR_OAUTH_RESOURCE_URI, "", 0, "Encoded slash or null byte forbidden in resource URI")
            out.append(ord("/"))
            out.extend(seg)
            p = seg_end

        if len(out) > cls.MAX_RESOURCE_URI_LEN:
            return (cls.ERR_OAUTH_RESOURCE_URI, "", 0, "Canonical resource URI exceeds 256 bytes")
        canon_bytes = bytes(out)
        return (0, canon_bytes.decode("ascii"), crc32c(canon_bytes), "ok")

    @classmethod
    def validate_bearer_syntax(cls, token: Any) -> tuple[int, bytes, int, str]:
        """Validate RFC 6750 §2.1 b64token syntax and derive zero-plaintext SHA-256('OAT1') hi64 slot key."""
        if not isinstance(token, str):
            return (cls.ERR_BEARER_SYNTAX, b"", 0, "Bearer token must be a string")
        try:
            raw = token.encode("ascii")
        except UnicodeEncodeError:
            return (cls.ERR_BEARER_SYNTAX, b"", 0, "Bearer token must be ASCII b64token")
        if len(raw) == 0 or len(raw) > cls.MAX_BEARER_LEN:
            return (cls.ERR_BEARER_SYNTAX, b"", 0, "Bearer token length must be in [1, 512]")
        seen_pad = False
        for idx, b in enumerate(raw):
            if b == ord("="):
                if idx == 0:
                    return (cls.ERR_BEARER_SYNTAX, b"", 0, "Bearer token cannot start with '=' padding")
                seen_pad = True
                continue
            if seen_pad:
                return (cls.ERR_BEARER_SYNTAX, b"", 0, "Non-padding byte after '=' in Bearer token")
            if b not in _PKCE_UNRESERVED_BYTES and b not in (ord("+"), ord("/")):
                return (cls.ERR_BEARER_SYNTAX, b"", 0, "Invalid character in RFC 6750 b64token")
        digest = hashlib.sha256(raw + b"OAT1").digest()
        hi64 = int.from_bytes(digest[:8], "big")
        return (0, raw, hi64, "ok")

    @classmethod
    def extract_bearer_header(cls, auth_header: Any) -> tuple[int, str, str]:
        """Parse HTTP Authorization header: returns (1, '', ...) when absent, (0, tok, 'ok') when syntactically valid, or (-46, '', ...)."""
        if auth_header is None:
            return (1, "", "Missing Authorization header")
        if not isinstance(auth_header, str) or not auth_header.strip():
            return (1, "", "Missing Authorization header")
        trimmed = auth_header.strip()
        if len(trimmed) <= 7 or trimmed[:7].lower() != "bearer ":
            return (cls.ERR_BEARER_SYNTAX, "", "Authorization header must use Bearer scheme")
        tok = trimmed[7:].strip()
        if not tok or " " in tok or "\t" in tok:
            return (cls.ERR_BEARER_SYNTAX, "", "Malformed Bearer credentials in Authorization header")
        rc, _, _, msg = cls.validate_bearer_syntax(tok)
        if rc != 0:
            return (rc, "", msg)
        return (0, tok, "ok")

    @staticmethod
    def has_forbidden_query_token(raw_query: str | None) -> bool:
        """OAuth 2.1 & MCP 2025-11-25 forbid passing access_token in URI query parameters."""
        if not raw_query:
            return False
        for part in raw_query.split("&"):
            if not part:
                continue
            key = part.split("=", 1)[0]
            if key == "access_token":
                return True
        return False

    @classmethod
    def parse_scope_mask(cls, scopes: int | str | list[str] | tuple[str, ...] | set[str] | None) -> int:
        if scopes is None:
            return 0
        if isinstance(scopes, int) and not isinstance(scopes, bool):
            return scopes & cls.SCOPE_ALL
        tokens: list[str]
        if isinstance(scopes, str):
            tokens = [s for s in scopes.strip().split() if s]
        elif isinstance(scopes, (list, tuple, set)):
            tokens = [str(s).strip() for s in scopes if str(s).strip()]
        else:
            return 0
        mask = 0
        by_name = {name: bit for bit, name in cls.SCOPE_DEFINITIONS}
        for tok in tokens:
            if tok in by_name:
                mask |= by_name[tok]
        return mask

    @classmethod
    def format_scope_list(cls, mask: int) -> list[str]:
        return [name for bit, name in cls.SCOPE_DEFINITIONS if (mask & bit) != 0]

    @classmethod
    def required_scope_for_method(cls, method: Any) -> int:
        if not isinstance(method, str):
            return cls.SCOPE_READ
        if method == "tools/call":
            return cls.SCOPE_READ | cls.SCOPE_TOOLS
        if method == "prompts/get":
            return cls.SCOPE_READ | cls.SCOPE_PROMPTS
        if method in ("sampling/createMessage", "sampling/respond", "elicitation/create", "elicitation/respond"):
            return cls.SCOPE_READ | cls.SCOPE_SAMPLING
        return cls.SCOPE_READ

    @classmethod
    def required_scope_for_payload(cls, payload: Any) -> int:
        if isinstance(payload, list):
            mask = cls.SCOPE_READ
            for item in payload:
                if isinstance(item, dict):
                    mask |= cls.required_scope_for_method(item.get("method"))
            return mask
        if isinstance(payload, dict):
            return cls.required_scope_for_method(payload.get("method"))
        return cls.SCOPE_READ

    @staticmethod
    def client_fingerprint(client_id: str | None) -> int:
        cleaned = str(client_id or "").strip()[:128]
        if not cleaned:
            return 0
        fp = crc32c(cleaned.encode("utf-8"))
        return fp if fp != 0 else 1

    def _sweep_expired_locked(self, now_ms: int) -> None:
        for slot in self._slots.values():
            if slot["state"] == "active" and now_ms > slot["expires_ms"]:
                slot["state"] = "expired"
                self._tokens_expired += 1

    def verify_pkce(self, verifier: Any, expected_challenge: Any, *, method: str = "S256") -> tuple[int, str]:
        with self._lock:
            self._pkce_verifications += 1
        return self.verify_pkce_s256(verifier, expected_challenge, method=method)

    def register_token(
        self,
        token: str,
        *,
        audience_uri: str | None = None,
        scopes: int | str | list[str] = SCOPE_ALL,
        client_id: str | None = None,
        ttl_ms: int | None = None,
    ) -> tuple[int, dict[str, Any] | None, str]:
        rc_tok, raw_tok, hi64, tok_msg = self.validate_bearer_syntax(token)
        if rc_tok != 0:
            return (rc_tok, None, tok_msg)
        target_uri = self._resource_uri if audience_uri is None else audience_uri
        rc_uri, canon_aud, aud_crc, uri_msg = self.normalize_resource_uri(target_uri)
        if rc_uri != 0 or aud_crc == 0:
            return (self.ERR_OAUTH_RESOURCE_URI, None, uri_msg or "Invalid audience resource URI")
        scope_mask = self.parse_scope_mask(scopes)
        eff_ttl = self.default_ttl_ms if ttl_ms is None else max(0, int(ttl_ms))
        client_crc = self.client_fingerprint(client_id) if client_id else 0

        with self._lock:
            now_ms = int(self._now_ms_fn())
            self._sweep_expired_locked(now_ms)
            key = (hi64, len(raw_tok))
            if key not in self._slots:
                active_count = sum(1 for s in self._slots.values() if s["state"] == "active")
                if active_count >= self.max_slots:
                    return (self.ERR_TOKEN_FULL, None, "OAuth token introspection table is full")
                if len(self._slots) >= self.max_slots:
                    for old_k, old_s in list(self._slots.items()):
                        if old_s["state"] != "active":
                            del self._slots[old_k]
                            break
            exp_ms = now_ms + eff_ttl
            self._slots[key] = {
                "state": "active",
                "token_len": len(raw_tok),
                "token_sha256_hi64": hi64,
                "audience_uri": canon_aud,
                "audience_crc32c": aud_crc,
                "scope_mask": scope_mask,
                "client_crc32c": client_crc,
                "expires_ms": exp_ms,
            }
            self._tokens_registered += 1
            return (
                0,
                {
                    "audienceUri": canon_aud,
                    "audienceCrc32c": aud_crc,
                    "scopeMask": scope_mask,
                    "scopes": self.format_scope_list(scope_mask),
                    "clientCrc32c": client_crc,
                    "expiresAtMs": exp_ms,
                },
                "ok",
            )

    def introspect_token(
        self,
        token: Any,
        *,
        expected_audience_uri: str | None = None,
        required_scopes: int | str | list[str] = SCOPE_READ,
        client_id: str | None = None,
    ) -> tuple[int, dict[str, Any] | None, str]:
        rc_tok, raw_tok, hi64, tok_msg = self.validate_bearer_syntax(token)
        if rc_tok != 0:
            return (rc_tok, None, tok_msg)
        if expected_audience_uri is None:
            exp_crc = self._audience_crc32c
        else:
            rc_uri, _, exp_crc, uri_msg = self.normalize_resource_uri(expected_audience_uri)
            if rc_uri != 0:
                return (self.ERR_OAUTH_RESOURCE_URI, None, uri_msg)
        req_mask = self.parse_scope_mask(required_scopes)

        with self._lock:
            now_ms = int(self._now_ms_fn())
            self._sweep_expired_locked(now_ms)
            slot = self._slots.get((hi64, len(raw_tok)))
            if slot is None or slot["state"] != "active":
                self._inactive_rejections += 1
                return (self.ERR_TOKEN_INACTIVE, None, "Bearer token is unknown, expired, or revoked")
            if slot["client_crc32c"] != 0 and slot["client_crc32c"] != self.client_fingerprint(client_id):
                self._inactive_rejections += 1
                return (self.ERR_TOKEN_INACTIVE, None, "Bearer token client binding mismatch")
            if slot["audience_crc32c"] != exp_crc:
                self._audience_rejections += 1
                return (
                    self.ERR_TOKEN_AUDIENCE,
                    None,
                    "Bearer token audience mismatch for this MCP protected resource (RFC 8707)",
                )
            if (slot["scope_mask"] & req_mask) != req_mask:
                self._scope_rejections += 1
                missing = self.format_scope_list(req_mask & ~slot["scope_mask"])
                return (
                    self.ERR_TOKEN_SCOPE,
                    None,
                    f"Bearer token lacks required scope(s): {' '.join(missing)}",
                )
            self._introspections_ok += 1
            return (
                0,
                {
                    "active": True,
                    "audienceUri": slot["audience_uri"],
                    "audienceCrc32c": slot["audience_crc32c"],
                    "grantedScopeMask": slot["scope_mask"],
                    "grantedScopes": self.format_scope_list(slot["scope_mask"]),
                    "expiresAtMs": slot["expires_ms"],
                },
                "ok",
            )

    def revoke_token(self, token: Any) -> tuple[int, str]:
        rc_tok, raw_tok, hi64, tok_msg = self.validate_bearer_syntax(token)
        if rc_tok != 0:
            return (rc_tok, tok_msg)
        with self._lock:
            self._sweep_expired_locked(int(self._now_ms_fn()))
            slot = self._slots.get((hi64, len(raw_tok)))
            if slot is None or slot["state"] != "active":
                return (self.ERR_TOKEN_INACTIVE, "Bearer token is not active")
            slot["state"] = "revoked"
            self._tokens_revoked += 1
            return (0, "ok")

    def resource_metadata_url(self, *, resource_uri: str | None = None) -> str:
        target = self._resource_uri if resource_uri is None else self.normalize_resource_uri(resource_uri)[1]
        scheme_end = target.find("://")
        slash_idx = target.find("/", scheme_end + 3)
        if slash_idx == -1:
            return f"{target}/.well-known/oauth-protected-resource"
        origin = target[:slash_idx]
        path = target[slash_idx:]
        return f"{origin}/.well-known/oauth-protected-resource{path}"

    def metadata_document(self, *, resource_uri: str | None = None) -> dict[str, Any]:
        target = self._resource_uri
        if resource_uri is not None:
            rc, canon, _, msg = self.normalize_resource_uri(resource_uri)
            if rc != 0:
                raise ValueError(f"Invalid resource_uri: {msg}")
            target = canon
        return {
            "resource": target,
            "authorization_servers": list(self._authorization_servers),
            "bearer_methods_supported": ["header"],
            "scopes_supported": [name for _, name in self.SCOPE_DEFINITIONS],
            "code_challenge_methods_supported": ["S256"],
            "resource_signing_alg_values_supported": ["RS256", "ES256"],
        }

    @classmethod
    def build_www_authenticate_header(
        cls,
        *,
        resource_metadata_url: str,
        required_scopes: int | str | list[str] | None = None,
        error: str | None = None,
        error_description: str | None = None,
    ) -> str:
        safe_meta = str(resource_metadata_url).replace('"', "").replace("\r", "").replace("\n", "")
        parts = [f'resource_metadata="{safe_meta}"']
        if required_scopes is not None:
            mask = cls.parse_scope_mask(required_scopes)
            scope_str = " ".join(cls.format_scope_list(mask))
            if scope_str:
                parts.append(f'scope="{scope_str}"')
        if error:
            safe_err = str(error).replace('"', "").replace("\r", "").replace("\n", "")
            parts.append(f'error="{safe_err}"')
        if error_description:
            clean_desc, _ = RedactionGuard.sanitize(str(error_description))
            safe_desc = clean_desc.replace('"', "'").replace("\r", " ").replace("\n", " ")[:160]
            parts.append(f'error_description="{safe_desc}"')
        return "Bearer " + ", ".join(parts)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            self._sweep_expired_locked(int(self._now_ms_fn()))
            active = sum(1 for s in self._slots.values() if s["state"] == "active")
            return {
                "resource_uri": self._resource_uri,
                "audience_crc32c": self._audience_crc32c,
                "authorization_servers": list(self._authorization_servers),
                "active_tokens": active,
                "max_slots": self.max_slots,
                "tokens_registered": self._tokens_registered,
                "introspections_ok": self._introspections_ok,
                "tokens_revoked": self._tokens_revoked,
                "tokens_expired": self._tokens_expired,
                "audience_rejections": self._audience_rejections,
                "scope_rejections": self._scope_rejections,
                "inactive_rejections": self._inactive_rejections,
                "pkce_verifications": self._pkce_verifications,
            }


_BUILTIN_TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "moe_route_query": {
        "name": "moe_route_query",
        "description": "Route a prompt across Sparse Top-2 MoE tiers under VRAM and latency SLA constraints.",
        "annotations": {
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        "inputSchema": {
            "$schema": JsonSchema2020Validator.SCHEMA_URI,
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "minLength": 1},
                "max_vram_gib": {"type": "number", "exclusiveMinimum": 0},
                "latency_sla_ms": {"type": "number", "exclusiveMinimum": 0},
            },
            "required": ["prompt"],
            "additionalProperties": False,
        },
        "outputSchema": {
            "$schema": JsonSchema2020Validator.SCHEMA_URI,
            "type": "object",
            "properties": {
                "selected_tier": {"type": "string", "minLength": 1},
                "selected_expert": {"type": "string", "minLength": 1},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "vram_budget_gib": {"type": "number", "exclusiveMinimum": 0},
                "est_cost_per_1k_tok_usd": {"type": "number", "minimum": 0},
                "top2_sparse_gates": {
                    "type": "array",
                    "minItems": 2,
                    "maxItems": 2,
                    "items": {
                        "type": "object",
                        "properties": {
                            "expert_id": {"type": "integer", "minimum": 0},
                            "tier": {"type": "string", "minLength": 1},
                            "gate_weight": {"type": "number", "minimum": 0, "maximum": 1},
                        },
                        "required": ["expert_id", "tier", "gate_weight"],
                        "additionalProperties": False,
                    },
                },
                "redactions_applied": {"type": "integer", "minimum": 0},
                "router_latency_ms": {"type": "number", "minimum": 0},
            },
            "required": [
                "selected_tier",
                "selected_expert",
                "confidence",
                "vram_budget_gib",
                "est_cost_per_1k_tok_usd",
                "top2_sparse_gates",
                "redactions_applied",
                "router_latency_ms",
            ],
            "additionalProperties": False,
        },
    },
    "rag_upsert_document": {
        "name": "rag_upsert_document",
        "description": "Sanitize and index a document into SQLite WAL FTS5 + Dense Vector store.",
        "annotations": {
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        "inputSchema": {
            "$schema": JsonSchema2020Validator.SCHEMA_URI,
            "type": "object",
            "properties": {
                "doc_id": {"type": "string", "minLength": 1},
                "title": {"type": "string", "minLength": 1},
                "body": {"type": "string", "minLength": 1},
                "tags": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["doc_id", "title", "body"],
            "additionalProperties": False,
        },
        "outputSchema": {
            "$schema": JsonSchema2020Validator.SCHEMA_URI,
            "type": "object",
            "properties": {
                "doc_id": {"type": "string", "minLength": 1},
                "redactions_applied": {"type": "integer", "minimum": 0},
                "embedding_dim": {"type": "integer", "minimum": 1},
            },
            "required": ["doc_id", "redactions_applied", "embedding_dim"],
            "additionalProperties": False,
        },
    },
    "rag_hybrid_search": {
        "name": "rag_hybrid_search",
        "description": "Execute hybrid BM25 FTS5 + cosine vector search fused with Reciprocal Rank Fusion (RRF) and highlighted citation spans.",
        "annotations": {
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        "inputSchema": {
            "$schema": JsonSchema2020Validator.SCHEMA_URI,
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 1},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 100},
                "include_resource_links": {"type": "boolean"},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        "outputSchema": {
            "$schema": JsonSchema2020Validator.SCHEMA_URI,
            "type": "object",
            "properties": {
                "hits": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "doc_id": {"type": "string", "minLength": 1},
                            "title": {"type": "string"},
                            "snippet": {"type": "string"},
                            "highlighted_snippet": {"type": "string"},
                            "citation_spans": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "field": {"type": "string", "enum": ["title", "body"]},
                                        "term": {"type": "string"},
                                        "start": {"type": "integer", "minimum": 0},
                                        "end": {"type": "integer", "minimum": 0},
                                    },
                                    "required": ["field", "term", "start", "end"],
                                    "additionalProperties": False,
                                },
                            },
                            "rrf_score": {"type": "number", "minimum": 0},
                            "lexical_rank": {"type": ["integer", "null"]},
                            "vector_rank": {"type": ["integer", "null"]},
                            "cosine": {"type": "number"},
                        },
                        "required": [
                            "doc_id",
                            "title",
                            "snippet",
                            "highlighted_snippet",
                            "citation_spans",
                            "rrf_score",
                            "lexical_rank",
                            "vector_rank",
                            "cosine",
                        ],
                        "additionalProperties": False,
                    },
                },
                "vault_documents": {"type": "integer", "minimum": 0},
                "cache_hit": {"type": "boolean"},
            },
            "required": ["hits", "vault_documents", "cache_hit"],
            "additionalProperties": False,
        },
    },
    "otlp_export_telemetry": {
        "name": "otlp_export_telemetry",
        "description": "Export OpenTelemetry (OTLP/JSON) tool-call spans, latency histograms, and RAG cache hit counters.",
        "annotations": {
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        "inputSchema": {
            "$schema": JsonSchema2020Validator.SCHEMA_URI,
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        "outputSchema": {
            "$schema": JsonSchema2020Validator.SCHEMA_URI,
            "type": "object",
            "properties": {
                "resourceSpans": {"type": "array"},
                "resourceMetrics": {"type": "array"},
                "summary": {
                    "type": "object",
                    "properties": {
                        "spans_buffered": {"type": "integer", "minimum": 0},
                        "cache_hits": {"type": "integer", "minimum": 0},
                        "cache_misses": {"type": "integer", "minimum": 0},
                        "cache_hit_ratio": {"type": "number", "minimum": 0, "maximum": 1},
                        "rate_limit_allowed": {"type": "integer", "minimum": 0},
                        "rate_limit_rejected": {"type": "integer", "minimum": 0},
                        "schema_validations_passed": {"type": "integer", "minimum": 0},
                        "schema_validations_failed": {"type": "integer", "minimum": 0},
                        "requests_cancelled": {"type": "integer", "minimum": 0},
                        "progress_notifications_emitted": {"type": "integer", "minimum": 0},
                        "roots_checks_allowed": {"type": "integer", "minimum": 0},
                        "roots_checks_denied": {"type": "integer", "minimum": 0},
                        "elicitations_accepted": {"type": "integer", "minimum": 0},
                        "elicitations_declined": {"type": "integer", "minimum": 0},
                        "sampling_requests_governed": {"type": "integer", "minimum": 0},
                        "approval_gate_rejections": {"type": "integer", "minimum": 0},
                        "sessions_opened": {"type": "integer", "minimum": 0},
                        "sse_events_replayed": {"type": "integer", "minimum": 0},
                        "resource_links_emitted": {"type": "integer", "minimum": 0},
                        "annotation_rejections": {"type": "integer", "minimum": 0},
                        "oauth_tokens_verified": {"type": "integer", "minimum": 0},
                        "oauth_challenges_issued": {"type": "integer", "minimum": 0},
                        "oauth_audience_rejections": {"type": "integer", "minimum": 0},
                        "oauth_scope_rejections": {"type": "integer", "minimum": 0},
                        "pkce_verifications": {"type": "integer", "minimum": 0},
                    },
                    "required": [
                        "spans_buffered",
                        "cache_hits",
                        "cache_misses",
                        "cache_hit_ratio",
                        "rate_limit_allowed",
                        "rate_limit_rejected",
                        "schema_validations_passed",
                        "schema_validations_failed",
                        "requests_cancelled",
                        "progress_notifications_emitted",
                        "roots_checks_allowed",
                        "roots_checks_denied",
                        "elicitations_accepted",
                        "elicitations_declined",
                        "sampling_requests_governed",
                        "approval_gate_rejections",
                        "sessions_opened",
                        "sse_events_replayed",
                        "resource_links_emitted",
                        "annotation_rejections",
                        "oauth_tokens_verified",
                        "oauth_challenges_issued",
                        "oauth_audience_rejections",
                        "oauth_scope_rejections",
                        "pkce_verifications",
                    ],
                    "additionalProperties": False,
                },
            },
            "required": ["resourceSpans", "resourceMetrics", "summary"],
            "additionalProperties": False,
        },
    },
}


class McpJsonRpcDispatcher:
    """Model Context Protocol (MCP 2025-11-25) JSON-RPC 2.0 server with structured tool schemas, pagination, completions, subscriptions, logging, cancellation, HITL elicitation/sampling governance & OTLP telemetry."""

    PROTOCOL_VERSION = "2025-11-25"
    REQUEST_CANCELLED_CODE = -32800
    CAPABILITY_REQUIRED_CODE = -32004
    USER_REJECTED_CODE = -1
    SUPPORTED_PROTOCOL_VERSIONS: tuple[str, ...] = (
        "2025-11-25",
        "2025-06-18",
        "2025-03-26",
        "2024-11-05",
    )
    RFC5424_LOG_LEVELS: tuple[str, ...] = (
        "debug",
        "info",
        "notice",
        "warning",
        "error",
        "critical",
        "alert",
        "emergency",
    )
    _LOG_SEVERITY: ClassVar[dict[str, int]] = {lvl: idx for idx, lvl in enumerate(RFC5424_LOG_LEVELS)}

    def __init__(
        self,
        vault: HybridKnowledgeVault | None = None,
        router: SparseTop2MoeRouter | None = None,
        telemetry: OtlpTelemetryExporter | None = None,
        cancellation: RequestCancellationRegistry | None = None,
        roots: WorkspaceRootsGuard | list[dict[str, str]] | None = None,
        progress: ProgressTokenRegistry | None = None,
        approvals: HumanApprovalRegistry | None = None,
        sessions: StreamableSessionRegistry | None = None,
        oauth: OAuthProtectedResourceGuard | None = None,
        enforce_approval_capabilities: bool = False,
        span_hook: Callable[[dict[str, Any]], None] | None = None,
        rag_cache_capacity: int = 128,
        max_batch_size: int = 32,
        page_size: int = 50,
        max_subscriptions: int = 256,
        max_notifications: int = 256,
    ) -> None:
        self.vault = vault or HybridKnowledgeVault(":memory:")
        self.router = router or SparseTop2MoeRouter()
        self.telemetry = telemetry or OtlpTelemetryExporter(span_hook=span_hook)
        if span_hook is not None and self.telemetry.span_hook is None:
            self.telemetry.span_hook = span_hook
        self.cancellation = cancellation or RequestCancellationRegistry()
        if isinstance(roots, WorkspaceRootsGuard):
            self.roots = roots
        else:
            self.roots = WorkspaceRootsGuard(roots=roots)
        self.progress = progress or ProgressTokenRegistry()
        self.approvals = approvals or HumanApprovalRegistry()
        self.sessions = sessions or StreamableSessionRegistry()
        self.oauth = oauth or OAuthProtectedResourceGuard()
        self.enforce_approval_capabilities = bool(enforce_approval_capabilities)
        self._rag_cache: OrderedDict[tuple[str, int], dict[str, Any]] = OrderedDict()
        self._rag_cache_capacity = max(0, rag_cache_capacity)
        self.max_batch_size = max(1, int(max_batch_size))
        self.page_size = max(1, int(page_size))
        self.max_subscriptions = max(1, int(max_subscriptions))
        self.log_level = "info"
        self.negotiated_client_capabilities: dict[str, Any] = {}
        self._custom_tools: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._custom_resources: OrderedDict[str, tuple[dict[str, Any], Callable[[], str | dict[str, Any]]]] = OrderedDict()
        self._custom_prompts: OrderedDict[
            str,
            tuple[dict[str, Any], Callable[[dict[str, Any]], dict[str, Any]], Callable[[str, str], list[str]] | None],
        ] = OrderedDict()
        self._subscriptions: OrderedDict[str, float] = OrderedDict()
        self._notification_outbox: deque[dict[str, Any]] = deque(maxlen=max(1, int(max_notifications)))
        self._sub_lock = threading.Lock()

    def set_roots(self, roots: list[dict[str, str]]) -> list[dict[str, str]]:
        updated = self.roots.set_roots(roots)
        with self._sub_lock:
            self._notification_outbox.append(
                {
                    "jsonrpc": "2.0",
                    "method": "notifications/roots/list_changed",
                }
            )
        self._notify_resource_updated("mcp://governance/roots")
        return updated

    def register_tool(
        self,
        name: str,
        description: str,
        input_schema: dict[str, Any],
        handler: Callable[..., dict[str, Any]],
        *,
        output_schema: dict[str, Any] | None = None,
        annotations: dict[str, bool] | None = None,
        accepts_cancel_token: bool = False,
        accepts_progress: bool = False,
    ) -> None:
        meta: dict[str, Any] = {
            "name": name,
            "description": description,
            "inputSchema": input_schema,
        }
        if annotations is not None:
            meta["annotations"] = dict(annotations)
        if output_schema is not None:
            meta["outputSchema"] = output_schema
        self._custom_tools[name] = {
            "meta": meta,
            "handler": handler,
            "accepts_cancel_token": accepts_cancel_token,
            "accepts_progress": accepts_progress,
        }

    def register_resource(
        self,
        uri: str,
        name: str,
        description: str,
        reader: Callable[[], str | dict[str, Any]],
        mime_type: str = "application/json",
    ) -> None:
        stored_uri = uri
        if uri.lower().startswith("file:"):
            rc, canon_uri, msg = WorkspaceRootsGuard.normalize_file_uri(uri)
            if rc != 0:
                raise ValueError(f"Invalid file:// resource URI {uri!r}: {msg} ({rc})")
            stored_uri = canon_uri
        self._custom_resources[stored_uri] = (
            {"uri": stored_uri, "name": name, "description": description, "mimeType": mime_type},
            reader,
        )

    def register_prompt(
        self,
        name: str,
        description: str,
        arguments: list[dict[str, Any]],
        renderer: Callable[[dict[str, Any]], dict[str, Any]],
        *,
        completer: Callable[[str, str], list[str]] | None = None,
    ) -> None:
        self._custom_prompts[name] = (
            {"name": name, "description": description, "arguments": arguments},
            renderer,
            completer,
        )

    def subscription_snapshot(self) -> dict[str, Any]:
        with self._sub_lock:
            return {
                "subscribed_uris": list(self._subscriptions.keys()),
                "active_subscriptions": len(self._subscriptions),
                "queued_notifications": len(self._notification_outbox),
                "log_level": self.log_level,
            }

    def drain_notifications(self) -> list[dict[str, Any]]:
        with self._sub_lock:
            items = list(self._notification_outbox)
            self._notification_outbox.clear()
            return items

    def emit_log(self, level: str, data: Any, *, logger: str = "mcp-moe-hybrid-sdk") -> bool:
        norm = str(level or "").strip().lower()
        sev = self._LOG_SEVERITY.get(norm)
        if sev is None:
            raise ValueError(f"Unsupported RFC 5424 log level: {level!r}")
        clean_logger, _ = RedactionGuard.sanitize(str(logger or "mcp-moe-hybrid-sdk")[:128])
        if isinstance(data, str):
            clean_data: Any = RedactionGuard.sanitize(data)[0]
        else:
            sanitized_json, _ = RedactionGuard.sanitize(json.dumps(data, ensure_ascii=False))
            clean_data = json.loads(sanitized_json)
        with self._sub_lock:
            if sev < self._LOG_SEVERITY[self.log_level]:
                return False
            self._notification_outbox.append(
                {
                    "jsonrpc": "2.0",
                    "method": "notifications/message",
                    "params": {"level": norm, "logger": clean_logger, "data": clean_data},
                }
            )
            return True

    def _notify_resource_updated(self, uri: str) -> bool:
        with self._sub_lock:
            if uri not in self._subscriptions:
                return False
            self._notification_outbox.append(
                {
                    "jsonrpc": "2.0",
                    "method": "notifications/resources/updated",
                    "params": {"uri": uri},
                }
            )
            return True

    def _emit_progress_notification(
        self,
        notif: dict[str, Any],
        progress_sink: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        with self._sub_lock:
            self._notification_outbox.append(notif)
        self.telemetry.record_progress_notification(1)
        if progress_sink is not None:
            progress_sink(notif)

    def _check_file_uri_access(self, uri: str, req_id: Any) -> tuple[str | None, dict[str, Any] | None]:
        rc, canon_uri, msg = self.roots.check_uri(uri)
        self.telemetry.record_roots_check(rc == 0)
        if rc == 0:
            return (canon_uri, None)
        err_code = -32003 if rc == WorkspaceRootsGuard.ERR_URI_OUTSIDE_ROOTS else -32602
        return (
            None,
            {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {
                    "code": err_code,
                    "message": msg,
                    "data": {"uri": uri, "code": rc},
                },
            },
        )

    def _is_subscribable_uri(self, uri: str) -> bool:
        if uri in (
            "mcp://moe-router/tiers",
            "mcp://telemetry/otlp",
            "mcp://governance/cancellation",
            "mcp://governance/subscriptions",
            "mcp://governance/roots",
            "mcp://governance/progress",
            "mcp://governance/approvals",
            "mcp://governance/sessions",
            "mcp://governance/oauth",
        ) or uri in self._custom_resources:
            return True
        if uri.startswith("vault://documents/"):
            doc_id = uri[len("vault://documents/") :].strip()
            return bool(doc_id)
        return False

    def _paginate_items(
        self,
        items: list[dict[str, Any]],
        *,
        key_field: str,
        result_field: str,
        scope: int,
        cursor_param: Any,
        req_id: Any,
    ) -> dict[str, Any]:
        snap_crc = McpCursorCodec.snapshot_hash([str(it[key_field]) for it in items])
        start_idx = 0
        if cursor_param is not None:
            code, offset, msg = McpCursorCodec.decode_and_verify(cursor_param, scope, snap_crc, len(items))
            if code != 0:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {
                        "code": -32602,
                        "message": msg,
                        "data": {"code": code, "scope": scope},
                    },
                }
            start_idx = offset
        end_idx = start_idx + self.page_size
        res_body: dict[str, Any] = {result_field: items[start_idx:end_idx]}
        if end_idx < len(items):
            res_body["nextCursor"] = McpCursorCodec.encode(scope, end_idx, snap_crc)
        return {"jsonrpc": "2.0", "id": req_id, "result": res_body}

    def _complete_argument(
        self,
        ref: Any,
        argument: Any,
        *,
        cancel_token: CancellationToken,
    ) -> tuple[list[str] | None, str | None]:
        cancel_token.raise_if_cancelled()
        if not isinstance(ref, dict) or not isinstance(argument, dict):
            return (None, "Invalid params: 'ref' and 'argument' objects are required")
        ref_type = str(ref.get("type") or "").strip()
        arg_name = str(argument.get("name") or "").strip()
        raw_val = argument.get("value")
        if not arg_name or not isinstance(raw_val, str):
            return (None, "Invalid params: 'argument.name' and string 'argument.value' are required")
        clean_prefix, _ = RedactionGuard.sanitize(raw_val.strip())
        low_prefix = clean_prefix.lower()

        candidates: list[str] = []
        seen: set[str] = set()

        def add_candidate(val: str) -> None:
            if val and val not in seen:
                seen.add(val)
                candidates.append(val)

        if ref_type == "ref/resource":
            uri_tmpl = str(ref.get("uri") or "").strip()
            if uri_tmpl != "vault://documents/{doc_id}":
                return (None, f"Unknown resource template for completion: {uri_tmpl}")
            docs = self.vault.list_documents()
            for doc in docs:
                did = doc["doc_id"]
                if not low_prefix or did.lower().startswith(low_prefix):
                    add_candidate(did)
            if clean_prefix:
                for hit in self.vault.hybrid_search(clean_prefix, top_k=25, cancel_token=cancel_token):
                    if low_prefix in hit.doc_id.lower() or low_prefix in hit.title.lower() or (hit.lexical_rank is not None):
                        add_candidate(hit.doc_id)
            return (candidates, None)

        if ref_type == "ref/prompt":
            pname = str(ref.get("name") or "").strip()
            if pname == "grounded_rag_answer":
                if arg_name == "query":
                    for doc in self.vault.list_documents():
                        title = doc["title"]
                        if not low_prefix or low_prefix in title.lower():
                            add_candidate(title)
                        for tg in doc["tags"]:
                            if not low_prefix or tg.lower().startswith(low_prefix):
                                add_candidate(tg)
                    if clean_prefix:
                        for hit in self.vault.hybrid_search(clean_prefix, top_k=15, cancel_token=cancel_token):
                            if hit.lexical_rank is not None or hit.cosine > 0.15:
                                add_candidate(hit.title)
                elif arg_name == "top_k":
                    for opt in ("1", "3", "5", "10", "20"):
                        if opt.startswith(low_prefix):
                            add_candidate(opt)
                return (candidates, None)

            if pname == "moe_sla_routing_plan":
                if arg_name == "task":
                    exemplars = (
                        "Return strict JSON schema health status",
                        "Summarize SQLite WAL FTS5 hybrid RAG citations",
                        "Verify Rust and C11 FFI pointer alignment under ASan",
                        "Prove Black-Scholes convexity theorem and MoE invariant",
                    )
                    for ex in exemplars:
                        if not low_prefix or low_prefix in ex.lower():
                            add_candidate(ex)
                elif arg_name == "max_vram_gib":
                    for opt in ("1.0", "3.5", "3.8", "4.8", "6.0"):
                        if opt.startswith(low_prefix):
                            add_candidate(opt)
                elif arg_name == "latency_sla_ms":
                    for opt in ("200", "350", "750", "1500"):
                        if opt.startswith(low_prefix):
                            add_candidate(opt)
                return (candidates, None)

            if pname in self._custom_prompts:
                _, _, completer = self._custom_prompts[pname]
                if completer is not None:
                    for item in completer(arg_name, clean_prefix):
                        clean_item, _ = RedactionGuard.sanitize(str(item))
                        add_candidate(clean_item)
                return (candidates, None)

            return (None, f"Unknown prompt for completion: {pname}")

        return (None, f"Unsupported completion reference type: {ref_type}")

    def _list_resources(self) -> list[dict[str, Any]]:
        resources: list[dict[str, Any]] = [
            {
                "uri": "mcp://moe-router/tiers",
                "name": "Sparse Top-2 MoE Tier Table & SLA Telemetry",
                "description": "Hardware VRAM budgets, per-token cost estimates, and p50/p95 router latency statistics.",
                "mimeType": "application/json",
            },
            {
                "uri": "mcp://telemetry/otlp",
                "name": "OpenTelemetry (OTLP/JSON) Spans & Histograms",
                "description": "Live OTLP resourceSpans, latency histograms, and RAG LRU cache hit counters.",
                "mimeType": "application/json",
            },
            {
                "uri": "mcp://governance/cancellation",
                "name": "MCP Request Cancellation Registry",
                "description": "Live in-flight cancellation tokens, pre-arrival tombstones, and cancelled request counters.",
                "mimeType": "application/json",
            },
            {
                "uri": "mcp://governance/subscriptions",
                "name": "MCP Resource Subscriptions & RFC 5424 Logging State",
                "description": "Active subscribed URIs, queued change/log notifications, and server log severity threshold.",
                "mimeType": "application/json",
            },
            {
                "uri": "mcp://governance/roots",
                "name": "MCP Workspace Roots & RFC 8089 URI Boundary Guard",
                "description": "Configured workspace file:// roots, canonical boundary confinement counters, and listChanged epoch.",
                "mimeType": "application/json",
            },
            {
                "uri": "mcp://governance/progress",
                "name": "MCP Monotonic Progress Token Tracker",
                "description": "Active _meta.progressToken streams, monotonic notifications/progress counters, and lifecycle stats.",
                "mimeType": "application/json",
            },
            {
                "uri": "mcp://governance/approvals",
                "name": "MCP Human-in-the-Loop Elicitation & Sampling Approval Gate",
                "description": "Pending elicitation/create and sampling/createMessage gates, token ceilings, and replay/expiry counters.",
                "mimeType": "application/json",
            },
            {
                "uri": "mcp://governance/sessions",
                "name": "MCP Streamable HTTP Session & Last-Event-ID Replay Ring",
                "description": "Active Mcp-Session-Id bindings, bounded SSE event replay rings, and replay/termination counters.",
                "mimeType": "application/json",
            },
            {
                "uri": "mcp://governance/oauth",
                "name": "MCP OAuth 2.1 Protected Resource & Audience Token Guard",
                "description": "RFC 9728 protected resource metadata, PKCE S256 verification counters, and RFC 8707 audience-bound token introspection state.",
                "mimeType": "application/json",
            },
        ]
        allowed_roots = [r["uri"] for r in self.roots.list_roots()]
        for meta, _ in self._custom_resources.values():
            curi = meta["uri"]
            if (
                curi.lower().startswith("file:")
                and allowed_roots
                and not any(r == "file:///" or curi == r or curi.startswith(r + "/") for r in allowed_roots)
            ):
                continue
            resources.append(dict(meta))
        for doc in self.vault.list_documents():
            tag_label = ", ".join(doc["tags"]) if doc["tags"] else "untagged"
            resources.append(
                {
                    "uri": f"vault://documents/{doc['doc_id']}",
                    "name": doc["title"],
                    "description": f"Hybrid RAG indexed document [{tag_label}] (redactions={doc['redactions']}).",
                    "mimeType": "application/json",
                }
            )
        return resources

    def _list_prompts(self) -> list[dict[str, Any]]:
        prompts: list[dict[str, Any]] = [
            {
                "name": "grounded_rag_answer",
                "description": "Synthesize a citation-grounded response using SQLite FTS5 highlighted spans and RRF retrieval.",
                "arguments": [
                    {"name": "query", "description": "Question or topic to ground against the vault.", "required": True},
                    {"name": "top_k", "description": "Maximum passages to cite (default: 3).", "required": False},
                ],
            },
            {
                "name": "moe_sla_routing_plan",
                "description": "Generate a VRAM- and SLA-constrained Sparse Top-2 Mixture-of-Experts execution plan.",
                "arguments": [
                    {"name": "task", "description": "Prompt or workload to classify across MoE tiers.", "required": True},
                    {"name": "max_vram_gib", "description": "Maximum VRAM budget in GiB (default: 6.0).", "required": False},
                    {"name": "latency_sla_ms", "description": "Latency SLA target in milliseconds (default: 1500.0).", "required": False},
                ],
            },
        ]
        for meta, _, _ in self._custom_prompts.values():
            prompts.append(dict(meta))
        return prompts

    @staticmethod
    def _is_valid_notification(item: Any) -> bool:
        return (
            isinstance(item, dict)
            and item.get("jsonrpc") == "2.0"
            and isinstance(item.get("method"), str)
            and bool(item["method"])
            and "id" not in item
        )

    @classmethod
    def _is_cancelled_response(cls, resp: Any) -> bool:
        return (
            isinstance(resp, dict)
            and isinstance(resp.get("error"), dict)
            and resp["error"].get("code") == cls.REQUEST_CANCELLED_CODE
        )

    def handle_batch(
        self,
        batch: list[Any],
        *,
        traceparent: str | None = None,
        deterministic_child_spans: bool = False,
        agent_id: int = 1,
        elide_cancelled: bool = True,
        progress_sink: Callable[[dict[str, Any]], None] | None = None,
    ) -> list[dict[str, Any]]:
        if not batch:
            return [{"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "Invalid Request: empty JSON-RPC batch"}}]
        if len(batch) > self.max_batch_size:
            return [
                {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {
                        "code": -32600,
                        "message": f"Invalid Request: batch size {len(batch)} exceeds max {self.max_batch_size}",
                    },
                }
            ]
        # Pre-apply notifications/cancelled across the batch so matching sibling requests do not execute side effects
        preapplied_cancel_indices: set[int] = set()
        for idx, item in enumerate(batch):
            if self._is_valid_notification(item) and item.get("method") == "notifications/cancelled":
                det_seq = idx if deterministic_child_spans else None
                self.handle(item, traceparent=traceparent, deterministic_seq=det_seq, agent_id=agent_id)
                preapplied_cancel_indices.add(idx)

        responses: list[dict[str, Any]] = []
        for idx, item in enumerate(batch):
            if not isinstance(item, dict):
                responses.append(
                    {"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "Invalid JSON-RPC 2.0 Request"}}
                )
                continue
            det_seq = idx if deterministic_child_spans else None
            if self._is_valid_notification(item):
                if idx not in preapplied_cancel_indices:
                    self.handle(
                        item,
                        traceparent=traceparent,
                        deterministic_seq=det_seq,
                        agent_id=agent_id,
                        progress_sink=progress_sink,
                    )
                continue
            resp = self.handle(
                item,
                traceparent=traceparent,
                deterministic_seq=det_seq,
                agent_id=agent_id,
                progress_sink=progress_sink,
            )
            if elide_cancelled and self._is_cancelled_response(resp):
                continue
            responses.append(resp)
        return responses

    def handle_http_stream(
        self,
        payload: Any,
        *,
        traceparent: str | None = None,
    ) -> tuple[int, dict[str, Any] | list[dict[str, Any]] | None, str, list[dict[str, Any]]]:
        gw_ctx = self.telemetry.new_gateway_context(traceparent)
        child_tp = gw_ctx["child_traceparent"]
        outbound_tp = gw_ctx["traceparent"]
        streamed_progress: list[dict[str, Any]] = []

        if isinstance(payload, list):
            if not payload:
                return (
                    400,
                    {"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "Invalid Request: empty JSON-RPC batch"}},
                    outbound_tp,
                    [],
                )
            if len(payload) > self.max_batch_size:
                return (
                    400,
                    {
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {
                            "code": -32600,
                            "message": f"Invalid Request: batch size {len(payload)} exceeds max {self.max_batch_size}",
                        },
                    },
                    outbound_tp,
                    [],
                )
            responses = self.handle_batch(
                payload,
                traceparent=child_tp,
                elide_cancelled=True,
                progress_sink=streamed_progress.append,
            )
            if not responses:
                return (202, None, outbound_tp, [])
            return (200, responses, outbound_tp, streamed_progress)

        if isinstance(payload, dict):
            if self._is_valid_notification(payload):
                self.handle(payload, traceparent=child_tp, progress_sink=streamed_progress.append)
                return (202, None, outbound_tp, [])
            resp = self.handle(payload, traceparent=child_tp, progress_sink=streamed_progress.append)
            if self._is_cancelled_response(resp):
                return (202, None, outbound_tp, [])
            return (200, resp, outbound_tp, streamed_progress)

        return (
            400,
            {"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "Invalid JSON-RPC 2.0 Request"}},
            outbound_tp,
            [],
        )

    def handle_http_payload(
        self,
        payload: Any,
        *,
        traceparent: str | None = None,
    ) -> tuple[int, dict[str, Any] | list[dict[str, Any]] | None, str]:
        status, body, tp, _ = self.handle_http_stream(payload, traceparent=traceparent)
        return (status, body, tp)

    def handle(
        self,
        req: dict[str, Any],
        *,
        traceparent: str | None = None,
        deterministic_seq: int | None = None,
        agent_id: int = 1,
        progress_sink: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        if not isinstance(req, dict):
            return {"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "Invalid JSON-RPC 2.0 Request"}}
        req_id = req.get("id")
        if req.get("jsonrpc") != "2.0" or not isinstance(req.get("method"), str) or not req.get("method"):
            return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32600, "message": "Invalid JSON-RPC 2.0 Request"}}

        method = req["method"]
        params = req.get("params") if isinstance(req.get("params"), dict) else {}

        if method == "notifications/cancelled":
            target_id = params.get("requestId")
            reason = str(params.get("reason") or "Cancelled by client")
            outcome = self.cancellation.cancel(target_id, reason)
            return {"jsonrpc": "2.0", "id": req_id, "result": outcome}

        if method == "notifications/roots/list_changed":
            self.roots.record_client_list_changed()
            return {"jsonrpc": "2.0", "id": req_id, "result": {}}

        token = self.cancellation.begin(req_id, non_cancellable=(method == "initialize"))
        reporter: ProgressReporter | None = None
        req_meta = params.get("_meta") if isinstance(params.get("_meta"), dict) else {}
        try:
            token.raise_if_cancelled()

            if "progressToken" in req_meta:
                p_rc, reporter, p_msg = self.progress.begin(
                    req_meta["progressToken"],
                    cancel_token=token,
                    emit_fn=lambda n: self._emit_progress_notification(n, progress_sink),
                )
                if p_rc != 0:
                    return {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "error": {"code": -32602, "message": p_msg, "data": {"code": p_rc}},
                    }

            def finish_response(resp: dict[str, Any]) -> dict[str, Any]:
                if reporter is not None:
                    if "error" in resp:
                        reporter.abort()
                    else:
                        reporter.complete()
                return resp

            if method == "initialize":
                requested_version = str(params.get("protocolVersion") or self.PROTOCOL_VERSION)
                negotiated_version = (
                    requested_version if requested_version in self.SUPPORTED_PROTOCOL_VERSIONS else self.PROTOCOL_VERSION
                )
                client_caps = params.get("capabilities")
                if isinstance(client_caps, dict):
                    self.negotiated_client_capabilities = dict(client_caps)
                return finish_response(
                    {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "result": {
                            "protocolVersion": negotiated_version,
                            "serverInfo": {"name": "mcp-moe-hybrid-sdk", "version": "1.9.0"},
                            "capabilities": {
                                "tools": {"listChanged": False},
                                "resources": {"subscribe": True, "listChanged": False},
                                "prompts": {"listChanged": False},
                                "completions": {},
                                "logging": {},
                                "roots": {"listChanged": True},
                            },
                        },
                    }
                )
            if method in ("notifications/initialized", "ping"):
                return finish_response({"jsonrpc": "2.0", "id": req_id, "result": {}})
            if method == "roots/list":
                return finish_response({"jsonrpc": "2.0", "id": req_id, "result": {"roots": self.roots.list_roots()}})
            if method == "elicitation/create":
                require_cap = bool(params.get("requireClientCapability", self.enforce_approval_capabilities))
                if require_cap and not isinstance(self.negotiated_client_capabilities.get("elicitation"), dict):
                    self.telemetry.record_approval_rejection()
                    return finish_response(
                        {
                            "jsonrpc": "2.0",
                            "id": req_id,
                            "error": {
                                "code": self.CAPABILITY_REQUIRED_CODE,
                                "message": "Client did not declare 'elicitation' capability during initialize",
                                "data": {"capability": "elicitation"},
                            },
                        }
                    )
                app_id = params.get("approvalId", req_id)
                e_rc, env, e_msg = self.approvals.begin_elicitation(
                    app_id,
                    params.get("message"),
                    params.get("requestedSchema"),
                    ttl_ms=params.get("ttlMs"),
                )
                if e_rc != 0 or env is None:
                    self.telemetry.record_approval_rejection()
                    return finish_response(
                        {
                            "jsonrpc": "2.0",
                            "id": req_id,
                            "error": {"code": -32602, "message": e_msg, "data": {"code": e_rc}},
                        }
                    )
                self._notify_resource_updated("mcp://governance/approvals")
                return finish_response({"jsonrpc": "2.0", "id": req_id, "result": env})
            if method == "elicitation/respond":
                app_id = params.get("approvalId")
                r_rc, resolved, r_msg = self.approvals.resolve_elicitation(
                    app_id,
                    params.get("action"),
                    params.get("content"),
                    expected_schema_crc32c=params.get("schemaCrc32c"),
                )
                if r_rc != 0 or resolved is None:
                    self.telemetry.record_approval_rejection()
                    return finish_response(
                        {
                            "jsonrpc": "2.0",
                            "id": req_id,
                            "error": {"code": -32602, "message": r_msg, "data": {"code": r_rc}},
                        }
                    )
                self.telemetry.record_elicitation_outcome(resolved["action"])
                self._notify_resource_updated("mcp://governance/approvals")
                return finish_response({"jsonrpc": "2.0", "id": req_id, "result": resolved})
            if method == "sampling/createMessage":
                require_cap = bool(params.get("requireClientCapability", self.enforce_approval_capabilities))
                if require_cap and not isinstance(self.negotiated_client_capabilities.get("sampling"), dict):
                    self.telemetry.record_approval_rejection()
                    return finish_response(
                        {
                            "jsonrpc": "2.0",
                            "id": req_id,
                            "error": {
                                "code": self.CAPABILITY_REQUIRED_CODE,
                                "message": "Client did not declare 'sampling' capability during initialize",
                                "data": {"capability": "sampling"},
                            },
                        }
                    )
                app_id = params.get("approvalId", req_id)
                s_rc, env, s_msg = self.approvals.begin_sampling(
                    app_id,
                    params,
                    ceiling_max_tokens=params.get("ceilingMaxTokens"),
                    ttl_ms=params.get("ttlMs"),
                    allow_all_servers_context=params.get("allowAllServersContext"),
                )
                if s_rc != 0 or env is None:
                    self.telemetry.record_approval_rejection()
                    return finish_response(
                        {
                            "jsonrpc": "2.0",
                            "id": req_id,
                            "error": {"code": -32602, "message": s_msg, "data": {"code": s_rc}},
                        }
                    )
                self._notify_resource_updated("mcp://governance/approvals")
                return finish_response({"jsonrpc": "2.0", "id": req_id, "result": env})
            if method == "sampling/respond":
                app_id = params.get("approvalId")
                raw_content = params.get("content")
                resp_text = (
                    raw_content.get("text", "")
                    if isinstance(raw_content, dict)
                    else str(params.get("text") or "")
                )
                sr_rc, s_res, sr_msg = self.approvals.resolve_sampling(
                    app_id,
                    params.get("action"),
                    model=str(params.get("model") or "local-qwen-moe"),
                    text=resp_text,
                    actual_tokens=params.get("actualTokens"),
                    stop_reason=str(params.get("stopReason") or "endTurn"),
                    expected_schema_crc32c=params.get("schemaCrc32c"),
                )
                if sr_rc != 0 or s_res is None:
                    self.telemetry.record_approval_rejection()
                    return finish_response(
                        {
                            "jsonrpc": "2.0",
                            "id": req_id,
                            "error": {"code": -32602, "message": sr_msg, "data": {"code": sr_rc}},
                        }
                    )
                self._notify_resource_updated("mcp://governance/approvals")
                if s_res["action"] != "accept":
                    self.telemetry.record_approval_rejection()
                    return finish_response(
                        {
                            "jsonrpc": "2.0",
                            "id": req_id,
                            "error": {
                                "code": self.USER_REJECTED_CODE,
                                "message": sr_msg,
                                "data": s_res,
                            },
                        }
                    )
                self.telemetry.record_sampling_governed()
                out_msg = {k: v for k, v in s_res.items() if k != "action"}
                return finish_response({"jsonrpc": "2.0", "id": req_id, "result": out_msg})
            if method == "logging/setLevel":
                raw_level = params.get("level")
                norm_level = str(raw_level or "").strip().lower() if isinstance(raw_level, str) else ""
                if norm_level not in self._LOG_SEVERITY:
                    return finish_response(
                        {
                            "jsonrpc": "2.0",
                            "id": req_id,
                            "error": {
                                "code": -32602,
                                "message": f"Invalid params: level must be one of {list(self.RFC5424_LOG_LEVELS)}",
                            },
                        }
                    )
                with self._sub_lock:
                    self.log_level = norm_level
                return finish_response({"jsonrpc": "2.0", "id": req_id, "result": {}})
            if method == "resources/subscribe":
                uri = str(params.get("uri") or "").strip() if isinstance(params.get("uri"), str) else ""
                if uri.lower().startswith("file:"):
                    canon_uri, file_err = self._check_file_uri_access(uri, req_id)
                    if file_err is not None or canon_uri is None:
                        return finish_response(file_err or {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32602, "message": "Invalid file URI"}})
                    uri = canon_uri
                if not uri or not self._is_subscribable_uri(uri):
                    return finish_response(
                        {
                            "jsonrpc": "2.0",
                            "id": req_id,
                            "error": {"code": -32602, "message": f"Invalid params: unsubscribable or missing URI '{uri}'"},
                        }
                    )
                with self._sub_lock:
                    if uri in self._subscriptions:
                        self._subscriptions.move_to_end(uri)
                    self._subscriptions[uri] = time.time()
                    while len(self._subscriptions) > self.max_subscriptions:
                        self._subscriptions.popitem(last=False)
                return finish_response({"jsonrpc": "2.0", "id": req_id, "result": {}})
            if method == "resources/unsubscribe":
                uri = str(params.get("uri") or "").strip() if isinstance(params.get("uri"), str) else ""
                if not uri:
                    return finish_response({"jsonrpc": "2.0", "id": req_id, "error": {"code": -32602, "message": "Invalid params: 'uri' is required"}})
                if uri.lower().startswith("file:"):
                    rc, canon_uri, _ = WorkspaceRootsGuard.normalize_file_uri(uri)
                    if rc == 0:
                        uri = canon_uri
                with self._sub_lock:
                    self._subscriptions.pop(uri, None)
                return finish_response({"jsonrpc": "2.0", "id": req_id, "result": {}})
            if method == "completion/complete":
                if reporter is not None:
                    reporter.report(1, 2, "Ranking completion candidates")
                vals, err_msg = self._complete_argument(params.get("ref"), params.get("argument"), cancel_token=token)
                if err_msg is not None or vals is None:
                    return finish_response({"jsonrpc": "2.0", "id": req_id, "error": {"code": -32602, "message": err_msg or "Invalid completion params"}})
                capped = vals[:100]
                if reporter is not None:
                    reporter.report(2, 2, f"Matched {len(capped)} completion candidates")
                return finish_response(
                    {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "result": {
                            "completion": {
                                "values": capped,
                                "total": len(vals),
                                "hasMore": len(vals) > 100,
                            }
                        },
                    }
                )
            if method == "resources/list":
                return finish_response(
                    self._paginate_items(
                        self._list_resources(),
                        key_field="uri",
                        result_field="resources",
                        scope=McpCursorCodec.SCOPE_RESOURCES,
                        cursor_param=params.get("cursor"),
                        req_id=req_id,
                    )
                )
            if method == "resources/templates/list":
                templates = [
                    {
                        "uriTemplate": "vault://documents/{doc_id}",
                        "name": "Hybrid Knowledge Vault Document",
                        "description": "Retrieve a sanitized document, tags, and redaction metadata by doc_id.",
                        "mimeType": "application/json",
                    }
                ]
                return finish_response(
                    self._paginate_items(
                        templates,
                        key_field="uriTemplate",
                        result_field="resourceTemplates",
                        scope=McpCursorCodec.SCOPE_TEMPLATES,
                        cursor_param=params.get("cursor"),
                        req_id=req_id,
                    )
                )
            if method == "resources/read":
                uri = str(params.get("uri") or "").strip()
                if not uri:
                    return finish_response({"jsonrpc": "2.0", "id": req_id, "error": {"code": -32602, "message": "Invalid params: 'uri' is required"}})
                if uri.lower().startswith("file:"):
                    canon_uri, file_err = self._check_file_uri_access(uri, req_id)
                    if file_err is not None or canon_uri is None:
                        return finish_response(file_err or {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32602, "message": "Invalid file URI"}})
                    uri = canon_uri
                if uri == "mcp://moe-router/tiers":
                    text = json.dumps({"tiers": self.router.TIERS, "stats": self.router.stats()})
                    mime = "application/json"
                elif uri == "mcp://telemetry/otlp":
                    text = json.dumps(self.telemetry.export_otlp_json())
                    mime = "application/json"
                elif uri == "mcp://governance/cancellation":
                    text = json.dumps(self.cancellation.snapshot())
                    mime = "application/json"
                elif uri == "mcp://governance/subscriptions":
                    text = json.dumps(self.subscription_snapshot())
                    mime = "application/json"
                elif uri == "mcp://governance/roots":
                    text = json.dumps(self.roots.snapshot())
                    mime = "application/json"
                elif uri == "mcp://governance/progress":
                    text = json.dumps(self.progress.snapshot())
                    mime = "application/json"
                elif uri == "mcp://governance/approvals":
                    text = json.dumps(self.approvals.snapshot())
                    mime = "application/json"
                elif uri == "mcp://governance/sessions":
                    text = json.dumps(self.sessions.snapshot())
                    mime = "application/json"
                elif uri == "mcp://governance/oauth":
                    text = json.dumps(self.oauth.snapshot())
                    mime = "application/json"
                elif uri in self._custom_resources:
                    meta, reader = self._custom_resources[uri]
                    raw_val = reader()
                    text = raw_val if isinstance(raw_val, str) else json.dumps(raw_val)
                    mime = meta["mimeType"]
                elif uri.startswith("vault://documents/"):
                    doc_id = uri[len("vault://documents/") :]
                    doc = self.vault.get_document(doc_id)
                    if doc is None:
                        return finish_response(
                            {
                                "jsonrpc": "2.0",
                                "id": req_id,
                                "error": {"code": -32002, "message": f"Resource not found: {uri}", "data": {"uri": uri}},
                            }
                        )
                    text = json.dumps(doc)
                    mime = "application/json"
                else:
                    return finish_response(
                        {
                            "jsonrpc": "2.0",
                            "id": req_id,
                            "error": {"code": -32002, "message": f"Resource not found: {uri}", "data": {"uri": uri}},
                        }
                    )
                return finish_response(
                    {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "result": {"contents": [{"uri": uri, "mimeType": mime, "text": text}]},
                    }
                )
            if method == "prompts/list":
                return finish_response(
                    self._paginate_items(
                        self._list_prompts(),
                        key_field="name",
                        result_field="prompts",
                        scope=McpCursorCodec.SCOPE_PROMPTS,
                        cursor_param=params.get("cursor"),
                        req_id=req_id,
                    )
                )
            if method == "prompts/get":
                pname = str(params.get("name") or "").strip()
                pargs = params.get("arguments") or {}
                if pname == "grounded_rag_answer":
                    raw_query = str(pargs.get("query") or "").strip()
                    if not raw_query:
                        return finish_response(
                            {
                                "jsonrpc": "2.0",
                                "id": req_id,
                                "error": {"code": -32602, "message": "Invalid params: missing required argument 'query'"},
                            }
                        )
                    clean_q, _ = RedactionGuard.sanitize(raw_query)
                    top_k = max(1, int(pargs.get("top_k") or 3))
                    hits = self.vault.hybrid_search(clean_q, top_k=top_k, cancel_token=token, progress_reporter=reporter)
                    evidence_lines = [
                        f"[{h.doc_id}] {h.title} (rrf={h.rrf_score}, spans={len(h.citation_spans)}): {h.highlighted_snippet}"
                        for h in hits
                    ]
                    context_block = "\n".join(evidence_lines) if evidence_lines else "No indexed passages matched."
                    prompt_text = (
                        f"Answer the query strictly from the cited evidence passages and cite [doc_id] inline.\n"
                        f"Query: {clean_q}\n"
                        f"Evidence:\n{context_block}"
                    )
                    return finish_response(
                        {
                            "jsonrpc": "2.0",
                            "id": req_id,
                            "result": {
                                "description": f"Citation-grounded synthesis prompt ({len(hits)} FTS5+RRF passages)",
                                "messages": [{"role": "user", "content": {"type": "text", "text": prompt_text}}],
                            },
                        }
                    )
                if pname == "moe_sla_routing_plan":
                    raw_task = str(pargs.get("task") or "").strip()
                    if not raw_task:
                        return finish_response(
                            {
                                "jsonrpc": "2.0",
                                "id": req_id,
                                "error": {"code": -32602, "message": "Invalid params: missing required argument 'task'"},
                            }
                        )
                    if reporter is not None:
                        reporter.report(1, 2, "Scoring Sparse Top-2 MoE routing plan")
                    vram = float(pargs.get("max_vram_gib") or 6.0)
                    sla = float(pargs.get("latency_sla_ms") or 1500.0)
                    decision = self.router.route(raw_task, max_vram_gib=vram, latency_sla_ms=sla, cancel_token=token)
                    if reporter is not None:
                        reporter.report(2, 2, f"Selected {decision['selected_tier']}")
                    plan_text = (
                        f"Execute workload via tier '{decision['selected_tier']}' ({decision['selected_expert']}) "
                        f"under {vram:.1f} GiB VRAM / {sla:.0f} ms SLA.\n"
                        f"Top-2 Sparse Gates: {json.dumps(decision['top2_sparse_gates'])}"
                    )
                    return finish_response(
                        {
                            "jsonrpc": "2.0",
                            "id": req_id,
                            "result": {
                                "description": f"Sparse Top-2 MoE execution plan -> {decision['selected_tier']}",
                                "messages": [{"role": "user", "content": {"type": "text", "text": plan_text}}],
                            },
                        }
                    )
                if pname in self._custom_prompts:
                    meta, renderer, _ = self._custom_prompts[pname]
                    for arg_def in meta.get("arguments", []):
                        if arg_def.get("required") and not str(pargs.get(arg_def["name"]) or "").strip():
                            return finish_response(
                                {
                                    "jsonrpc": "2.0",
                                    "id": req_id,
                                    "error": {"code": -32602, "message": f"Invalid params: missing required argument '{arg_def['name']}'"},
                                }
                            )
                    return finish_response({"jsonrpc": "2.0", "id": req_id, "result": renderer(pargs)})
                return finish_response({"jsonrpc": "2.0", "id": req_id, "error": {"code": -32602, "message": f"Unknown prompt: {pname}"}})
            if method == "tools/list":
                tools_list = [dict(t) for t in _BUILTIN_TOOL_SCHEMAS.values()]
                for ctool in self._custom_tools.values():
                    tools_list.append(dict(ctool["meta"]))
                return finish_response(
                    self._paginate_items(
                        tools_list,
                        key_field="name",
                        result_field="tools",
                        scope=McpCursorCodec.SCOPE_TOOLS,
                        cursor_param=params.get("cursor"),
                        req_id=req_id,
                    )
                )
            if method == "tools/call":
                name = str(params.get("name", ""))
                raw_args = params.get("arguments")
                args = {} if raw_args is None else raw_args
                meta = req_meta
                effective_tp = traceparent if traceparent is not None else (
                    str(meta["traceparent"]) if isinstance(meta.get("traceparent"), str) else None
                )
                t_start = time.time_ns()
                cache_hit = False
                attrs: dict[str, Any] = {}

                tool_spec = _BUILTIN_TOOL_SCHEMAS.get(name) or (
                    self._custom_tools[name]["meta"] if name in self._custom_tools else None
                )
                if tool_spec is None:
                    self.telemetry.record_tool_call(
                        name or "unknown",
                        t_start,
                        time.time_ns(),
                        is_error=True,
                        traceparent=effective_tp,
                        deterministic_seq=deterministic_seq,
                        agent_id=agent_id,
                    )
                    return finish_response({"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"Unknown tool: {name}"}})

                in_violations = JsonSchema2020Validator.validate(args, tool_spec["inputSchema"])
                if in_violations:
                    self.telemetry.record_schema_validation(False)
                    self.telemetry.record_tool_call(
                        name,
                        t_start,
                        time.time_ns(),
                        is_error=True,
                        attributes={"mcp.schema.phase": "input", "mcp.schema.violations": len(in_violations)},
                        traceparent=effective_tp,
                        deterministic_seq=deterministic_seq,
                        agent_id=agent_id,
                    )
                    return finish_response(
                        {
                            "jsonrpc": "2.0",
                            "id": req_id,
                            "error": {
                                "code": -32602,
                                "message": f"Invalid params for tool '{name}': {in_violations[0]['message']}",
                                "data": {"tool": name, "phase": "inputSchema", "violations": in_violations},
                            },
                        }
                    )
                self.telemetry.record_schema_validation(True)

                try:
                    token.raise_if_cancelled()
                    if name == "moe_route_query":
                        if reporter is not None:
                            reporter.report(1, 2, "Sanitizing prompt and scoring Sparse MoE logits")
                        payload = self.router.route(
                            str(args["prompt"]),
                            float(args.get("max_vram_gib", 6.0)),
                            float(args.get("latency_sla_ms", 1500.0)),
                            cancel_token=token,
                        )
                        if reporter is not None:
                            reporter.report(2, 2, f"Selected {payload['selected_tier']}")
                        attrs["mcp.selected_tier"] = payload["selected_tier"]
                    elif name == "rag_upsert_document":
                        doc_id_str = str(args["doc_id"])
                        if reporter is not None:
                            reporter.report(1, 2, "Sanitizing document and computing 64-D embedding")
                        payload = self.vault.upsert(
                            doc_id_str,
                            str(args["title"]),
                            str(args["body"]),
                            list(args.get("tags") or []),
                        )
                        self._rag_cache.clear()
                        self._notify_resource_updated(f"vault://documents/{doc_id_str}")
                        if reporter is not None:
                            reporter.report(2, 2, f"Indexed document {doc_id_str} into SQLite WAL FTS5")
                        attrs["mcp.redactions"] = payload["redactions_applied"]
                    elif name == "rag_hybrid_search":
                        clean_q, _ = RedactionGuard.sanitize(str(args["query"]))
                        top_k = int(args.get("top_k", 5))
                        ckey = (" ".join(clean_q.lower().split()), top_k)
                        if self._rag_cache_capacity > 0 and ckey in self._rag_cache:
                            self._rag_cache.move_to_end(ckey)
                            payload = {**self._rag_cache[ckey], "cache_hit": True}
                            cache_hit = True
                            if reporter is not None:
                                reporter.report(1, 1, "Served hybrid RRF hits from LRU cache")
                        else:
                            hits = self.vault.hybrid_search(
                                clean_q,
                                top_k,
                                cancel_token=token,
                                progress_reporter=reporter,
                            )
                            base_payload = {"hits": [h.to_dict() for h in hits], "vault_documents": self.vault.count()}
                            if self._rag_cache_capacity > 0:
                                self._rag_cache[ckey] = base_payload
                                if len(self._rag_cache) > self._rag_cache_capacity:
                                    self._rag_cache.popitem(last=False)
                            payload = {**base_payload, "cache_hit": False}
                        attrs["mcp.hits"] = len(payload["hits"])
                    elif name == "otlp_export_telemetry":
                        if reporter is not None:
                            reporter.report(1, 1, "Exported OTLP telemetry snapshot")
                        payload = self.telemetry.export_otlp_json()
                    else:
                        ctool = self._custom_tools[name]
                        if ctool["accepts_cancel_token"] and ctool["accepts_progress"]:
                            payload = ctool["handler"](args, token, reporter)
                        elif ctool["accepts_cancel_token"]:
                            payload = ctool["handler"](args, token)
                        elif ctool["accepts_progress"]:
                            payload = ctool["handler"](args, reporter)
                        else:
                            payload = ctool["handler"](args)
                    token.raise_if_cancelled()
                except RequestCancelledError as cancel_err:
                    if reporter is not None:
                        reporter.abort()
                    self.telemetry.record_cancellation()
                    self.telemetry.record_tool_call(
                        name,
                        t_start,
                        time.time_ns(),
                        is_error=True,
                        attributes={"mcp.cancelled": True, "mcp.cancel.reason": cancel_err.reason},
                        traceparent=effective_tp,
                        deterministic_seq=deterministic_seq,
                        agent_id=agent_id,
                    )
                    return {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "error": {
                            "code": self.REQUEST_CANCELLED_CODE,
                            "message": f"Request cancelled: {cancel_err.reason}",
                            "data": {"requestId": req_id, "reason": cancel_err.reason, "cancelled": True},
                        },
                    }

                output_schema = tool_spec.get("outputSchema")
                if output_schema is not None:
                    out_violations = JsonSchema2020Validator.validate(payload, output_schema)
                    if out_violations:
                        self.telemetry.record_schema_validation(False)
                        self.telemetry.record_tool_call(
                            name,
                            t_start,
                            time.time_ns(),
                            is_error=True,
                            attributes={"mcp.schema.phase": "output", "mcp.schema.violations": len(out_violations)},
                            traceparent=effective_tp,
                            deterministic_seq=deterministic_seq,
                            agent_id=agent_id,
                        )
                        return finish_response(
                            {
                                "jsonrpc": "2.0",
                                "id": req_id,
                                "error": {
                                    "code": -32603,
                                    "message": f"Tool '{name}' output violated outputSchema: {out_violations[0]['message']}",
                                    "data": {"tool": name, "phase": "outputSchema", "violations": out_violations},
                                },
                            }
                        )
                    self.telemetry.record_schema_validation(True)

                extra_content_blocks: list[dict[str, Any]] = []
                if name == "rag_hybrid_search" and args.get("include_resource_links") is True:
                    for rank_idx, hit_item in enumerate(payload.get("hits", []), start=1):
                        did = str(hit_item["doc_id"])
                        doc_obj = self.vault.get_document(did)
                        doc_serialized = json.dumps(doc_obj) if doc_obj is not None else str(hit_item.get("snippet", ""))
                        prio = round(min(1.0, max(0.1, 1.0 - (rank_idx - 1) * 0.15)), 3)
                        rl_rc, rl_block, rl_msg = ContentAnnotationGuard.build_resource_link(
                            f"vault://documents/{did}",
                            str(hit_item.get("title") or did),
                            description=str(hit_item.get("snippet") or "")[:120],
                            mime_type="application/json",
                            size=len(doc_serialized.encode("utf-8")),
                            content_text=doc_serialized,
                            annotations={
                                "audience": ["user", "assistant"],
                                "priority": prio,
                                "lastModified": "2026-11-25T00:00:00Z",
                            },
                            roots_guard=self.roots,
                        )
                        if rl_rc != 0 or rl_block is None:
                            self.telemetry.record_annotation_rejection()
                            return finish_response(
                                {
                                    "jsonrpc": "2.0",
                                    "id": req_id,
                                    "error": {"code": -32603, "message": rl_msg, "data": {"code": rl_rc}},
                                }
                            )
                        extra_content_blocks.append(rl_block)
                    if extra_content_blocks:
                        self.telemetry.record_resource_link_emitted(len(extra_content_blocks))
                        attrs["mcp.resource_links"] = len(extra_content_blocks)

                if reporter is not None and reporter.updates_count > 0:
                    attrs["mcp.progress.updates"] = reporter.updates_count

                self.telemetry.record_tool_call(
                    name,
                    t_start,
                    time.time_ns(),
                    cache_hit=cache_hit,
                    attributes=attrs,
                    traceparent=effective_tp,
                    deterministic_seq=deterministic_seq,
                    agent_id=agent_id,
                )
                result_obj: dict[str, Any] = {
                    "content": [{"type": "text", "text": json.dumps(payload)}, *extra_content_blocks],
                    "isError": False,
                }
                if output_schema is not None:
                    result_obj["structuredContent"] = payload
                return finish_response(
                    {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "result": result_obj,
                    }
                )

            return finish_response({"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"Method not found: {method}"}})
        except RequestCancelledError as cancel_err:
            if reporter is not None:
                reporter.abort()
            self.telemetry.record_cancellation()
            meta = req_meta
            effective_tp = traceparent if traceparent is not None else (
                str(meta["traceparent"]) if isinstance(meta.get("traceparent"), str) else None
            )
            span_label = str(params.get("name") or method) if method == "tools/call" else method
            self.telemetry.record_tool_call(
                span_label,
                time.time_ns(),
                time.time_ns(),
                is_error=True,
                attributes={"mcp.cancelled": True, "mcp.cancel.reason": cancel_err.reason},
                traceparent=effective_tp,
                deterministic_seq=deterministic_seq,
                agent_id=agent_id,
            )
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {
                    "code": self.REQUEST_CANCELLED_CODE,
                    "message": f"Request cancelled: {cancel_err.reason}",
                    "data": {"requestId": req_id, "reason": cancel_err.reason, "cancelled": True},
                },
            }
        finally:
            if reporter is not None:
                reporter.abort()
            self.cancellation.finish(req_id, token)
