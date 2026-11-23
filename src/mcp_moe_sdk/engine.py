"""Core engine: PII Redaction Guard, SQLite WAL FTS5 + Cosine Vector RRF Vault, Sparse Top-2 MoE Router & MCP 2025-11-25 Dispatcher."""

from __future__ import annotations

import hashlib
import html
import json
import math
import re
import secrets
import sqlite3
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

_FTS_OPEN = "\x01"
_FTS_CLOSE = "\x02"


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

    _RULES: list[tuple[str, re.Pattern[str]]] = [
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

    def hybrid_search(self, query: str, top_k: int = 5, rrf_k: int = 60) -> list[SearchHit]:
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

        q_vec = deterministic_embed(clean_q)
        all_docs = self.conn.execute("SELECT doc_id, title, body, embedding_json FROM documents").fetchall()
        scored_vec: list[tuple[str, str, str, float]] = []
        for did, title, body, emb_json in all_docs:
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

        candidates.sort(key=lambda c: (c[0], c[1]), reverse=True)
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

    TIERS: list[dict[str, Any]] = [
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

    def route(self, prompt: str, max_vram_gib: float = 6.0, latency_sla_ms: float = 1500.0) -> dict[str, Any]:
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

    def _evaluate(self, raw_client_id: str, raw_tokens: int | float, *, mutate: bool) -> dict[str, Any]:
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
        "prompts/list",
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


class OtlpTelemetryExporter:
    """Zero-dependency OpenTelemetry (OTLP/JSON) span exporter with W3C traceparent propagation, histograms & rate-limit counters."""

    HISTOGRAM_BOUNDS_MS: tuple[float, ...] = (0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 50.0)
    _TRACEPARENT_RE = re.compile(r"^00-([0-9a-f]{32})-([0-9a-f]{16})-([0-9a-f]{2})$", re.IGNORECASE)

    def __init__(
        self,
        service_name: str = "mcp-moe-hybrid-sdk",
        service_version: str = "1.3.0",
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
    ) -> dict[str, Any]:
        parent = parent_context if parent_context is not None else self.parse_traceparent(traceparent)
        trace_id = parent["trace_id"] if parent else secrets.token_hex(16)
        parent_span_id = parent.get("parent_span_id", "") if parent else ""
        trace_flags = parent.get("trace_flags", "01") if parent else "01"
        span_id = secrets.token_hex(8)
        span_tp = self.format_traceparent(trace_id, span_id, trace_flags)

        clean_tool, _ = RedactionGuard.sanitize(tool_name)
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
            },
        }


class McpJsonRpcDispatcher:
    """Model Context Protocol (MCP 2025-11-25) JSON-RPC 2.0 server with tools, resources, prompts & OTLP telemetry."""

    PROTOCOL_VERSION = "2025-11-25"
    SUPPORTED_PROTOCOL_VERSIONS: tuple[str, ...] = (
        "2025-11-25",
        "2025-06-18",
        "2025-03-26",
        "2024-11-05",
    )

    def __init__(
        self,
        vault: HybridKnowledgeVault | None = None,
        router: SparseTop2MoeRouter | None = None,
        telemetry: OtlpTelemetryExporter | None = None,
        span_hook: Callable[[dict[str, Any]], None] | None = None,
        rag_cache_capacity: int = 128,
        max_batch_size: int = 32,
    ) -> None:
        self.vault = vault or HybridKnowledgeVault(":memory:")
        self.router = router or SparseTop2MoeRouter()
        self.telemetry = telemetry or OtlpTelemetryExporter(span_hook=span_hook)
        if span_hook is not None and self.telemetry.span_hook is None:
            self.telemetry.span_hook = span_hook
        self._rag_cache: OrderedDict[tuple[str, int], dict[str, Any]] = OrderedDict()
        self._rag_cache_capacity = max(0, rag_cache_capacity)
        self.max_batch_size = max(1, int(max_batch_size))
        self.negotiated_client_capabilities: dict[str, Any] = {}
        self._custom_resources: OrderedDict[str, tuple[dict[str, Any], Callable[[], str | dict[str, Any]]]] = OrderedDict()
        self._custom_prompts: OrderedDict[str, tuple[dict[str, Any], Callable[[dict[str, Any]], dict[str, Any]]]] = OrderedDict()

    def register_resource(
        self,
        uri: str,
        name: str,
        description: str,
        reader: Callable[[], str | dict[str, Any]],
        mime_type: str = "application/json",
    ) -> None:
        self._custom_resources[uri] = (
            {"uri": uri, "name": name, "description": description, "mimeType": mime_type},
            reader,
        )

    def register_prompt(
        self,
        name: str,
        description: str,
        arguments: list[dict[str, Any]],
        renderer: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> None:
        self._custom_prompts[name] = (
            {"name": name, "description": description, "arguments": arguments},
            renderer,
        )

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
        ]
        for meta, _ in self._custom_resources.values():
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
        for meta, _ in self._custom_prompts.values():
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

    def handle_batch(self, batch: list[Any], *, traceparent: str | None = None) -> list[dict[str, Any]]:
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
        responses: list[dict[str, Any]] = []
        for item in batch:
            if not isinstance(item, dict):
                responses.append(
                    {"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "Invalid JSON-RPC 2.0 Request"}}
                )
                continue
            if self._is_valid_notification(item):
                self.handle(item, traceparent=traceparent)
                continue
            responses.append(self.handle(item, traceparent=traceparent))
        return responses

    def handle_http_payload(
        self,
        payload: Any,
        *,
        traceparent: str | None = None,
    ) -> tuple[int, dict[str, Any] | list[dict[str, Any]] | None, str]:
        gw_ctx = self.telemetry.new_gateway_context(traceparent)
        child_tp = gw_ctx["child_traceparent"]
        outbound_tp = gw_ctx["traceparent"]

        if isinstance(payload, list):
            if not payload:
                return (
                    400,
                    {"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "Invalid Request: empty JSON-RPC batch"}},
                    outbound_tp,
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
                )
            responses = self.handle_batch(payload, traceparent=child_tp)
            if not responses:
                return (202, None, outbound_tp)
            return (200, responses, outbound_tp)

        if isinstance(payload, dict):
            if self._is_valid_notification(payload):
                self.handle(payload, traceparent=child_tp)
                return (202, None, outbound_tp)
            return (200, self.handle(payload, traceparent=child_tp), outbound_tp)

        return (
            400,
            {"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "Invalid JSON-RPC 2.0 Request"}},
            outbound_tp,
        )

    def handle(self, req: dict[str, Any], *, traceparent: str | None = None) -> dict[str, Any]:
        if not isinstance(req, dict):
            return {"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "Invalid JSON-RPC 2.0 Request"}}
        req_id = req.get("id")
        if req.get("jsonrpc") != "2.0" or not isinstance(req.get("method"), str) or not req.get("method"):
            return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32600, "message": "Invalid JSON-RPC 2.0 Request"}}

        method = req["method"]
        params = req.get("params") if isinstance(req.get("params"), dict) else {}

        if method == "initialize":
            requested_version = str(params.get("protocolVersion") or self.PROTOCOL_VERSION)
            negotiated_version = (
                requested_version if requested_version in self.SUPPORTED_PROTOCOL_VERSIONS else self.PROTOCOL_VERSION
            )
            client_caps = params.get("capabilities")
            if isinstance(client_caps, dict):
                self.negotiated_client_capabilities = dict(client_caps)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": negotiated_version,
                    "serverInfo": {"name": "mcp-moe-hybrid-sdk", "version": "1.3.0"},
                    "capabilities": {
                        "tools": {"listChanged": False},
                        "resources": {"subscribe": False, "listChanged": False},
                        "prompts": {"listChanged": False},
                    },
                },
            }
        if method in ("notifications/initialized", "ping"):
            return {"jsonrpc": "2.0", "id": req_id, "result": {}}
        if method == "resources/list":
            return {"jsonrpc": "2.0", "id": req_id, "result": {"resources": self._list_resources()}}
        if method == "resources/templates/list":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "resourceTemplates": [
                        {
                            "uriTemplate": "vault://documents/{doc_id}",
                            "name": "Hybrid Knowledge Vault Document",
                            "description": "Retrieve a sanitized document, tags, and redaction metadata by doc_id.",
                            "mimeType": "application/json",
                        }
                    ]
                },
            }
        if method == "resources/read":
            uri = str(params.get("uri") or "").strip()
            if not uri:
                return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32602, "message": "Invalid params: 'uri' is required"}}
            if uri == "mcp://moe-router/tiers":
                text = json.dumps({"tiers": self.router.TIERS, "stats": self.router.stats()})
                mime = "application/json"
            elif uri == "mcp://telemetry/otlp":
                text = json.dumps(self.telemetry.export_otlp_json())
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
                    return {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "error": {"code": -32002, "message": f"Resource not found: {uri}", "data": {"uri": uri}},
                    }
                text = json.dumps(doc)
                mime = "application/json"
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32002, "message": f"Resource not found: {uri}", "data": {"uri": uri}},
                }
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"contents": [{"uri": uri, "mimeType": mime, "text": text}]},
            }
        if method == "prompts/list":
            return {"jsonrpc": "2.0", "id": req_id, "result": {"prompts": self._list_prompts()}}
        if method == "prompts/get":
            pname = str(params.get("name") or "").strip()
            pargs = params.get("arguments") or {}
            if pname == "grounded_rag_answer":
                raw_query = str(pargs.get("query") or "").strip()
                if not raw_query:
                    return {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "error": {"code": -32602, "message": "Invalid params: missing required argument 'query'"},
                    }
                clean_q, _ = RedactionGuard.sanitize(raw_query)
                top_k = max(1, int(pargs.get("top_k") or 3))
                hits = self.vault.hybrid_search(clean_q, top_k=top_k)
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
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "description": f"Citation-grounded synthesis prompt ({len(hits)} FTS5+RRF passages)",
                        "messages": [{"role": "user", "content": {"type": "text", "text": prompt_text}}],
                    },
                }
            if pname == "moe_sla_routing_plan":
                raw_task = str(pargs.get("task") or "").strip()
                if not raw_task:
                    return {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "error": {"code": -32602, "message": "Invalid params: missing required argument 'task'"},
                    }
                vram = float(pargs.get("max_vram_gib") or 6.0)
                sla = float(pargs.get("latency_sla_ms") or 1500.0)
                decision = self.router.route(raw_task, max_vram_gib=vram, latency_sla_ms=sla)
                plan_text = (
                    f"Execute workload via tier '{decision['selected_tier']}' ({decision['selected_expert']}) "
                    f"under {vram:.1f} GiB VRAM / {sla:.0f} ms SLA.\n"
                    f"Top-2 Sparse Gates: {json.dumps(decision['top2_sparse_gates'])}"
                )
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "description": f"Sparse Top-2 MoE execution plan -> {decision['selected_tier']}",
                        "messages": [{"role": "user", "content": {"type": "text", "text": plan_text}}],
                    },
                }
            if pname in self._custom_prompts:
                meta, renderer = self._custom_prompts[pname]
                for arg_def in meta.get("arguments", []):
                    if arg_def.get("required") and not str(pargs.get(arg_def["name"]) or "").strip():
                        return {
                            "jsonrpc": "2.0",
                            "id": req_id,
                            "error": {"code": -32602, "message": f"Invalid params: missing required argument '{arg_def['name']}'"},
                        }
                return {"jsonrpc": "2.0", "id": req_id, "result": renderer(pargs)}
            return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32602, "message": f"Unknown prompt: {pname}"}}
        if method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "tools": [
                        {
                            "name": "moe_route_query",
                            "description": "Route a prompt across Sparse Top-2 MoE tiers under VRAM and latency SLA constraints.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "prompt": {"type": "string"},
                                    "max_vram_gib": {"type": "number"},
                                    "latency_sla_ms": {"type": "number"},
                                },
                                "required": ["prompt"],
                            },
                        },
                        {
                            "name": "rag_upsert_document",
                            "description": "Sanitize and index a document into SQLite WAL FTS5 + Dense Vector store.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "doc_id": {"type": "string"},
                                    "title": {"type": "string"},
                                    "body": {"type": "string"},
                                    "tags": {"type": "array", "items": {"type": "string"}},
                                },
                                "required": ["doc_id", "title", "body"],
                            },
                        },
                        {
                            "name": "rag_hybrid_search",
                            "description": "Execute hybrid BM25 FTS5 + cosine vector search fused with Reciprocal Rank Fusion (RRF) and highlighted citation spans.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "query": {"type": "string"},
                                    "top_k": {"type": "integer"},
                                },
                                "required": ["query"],
                            },
                        },
                        {
                            "name": "otlp_export_telemetry",
                            "description": "Export OpenTelemetry (OTLP/JSON) tool-call spans, latency histograms, and RAG cache hit counters.",
                            "inputSchema": {"type": "object", "properties": {}},
                        },
                    ]
                },
            }
        if method == "tools/call":
            name = str(params.get("name", ""))
            args = params.get("arguments") if isinstance(params.get("arguments"), dict) else {}
            meta = params.get("_meta") if isinstance(params.get("_meta"), dict) else {}
            effective_tp = traceparent if traceparent is not None else (
                str(meta["traceparent"]) if isinstance(meta.get("traceparent"), str) else None
            )
            t_start = time.time_ns()
            cache_hit = False
            attrs: dict[str, Any] = {}
            if name == "moe_route_query":
                payload = self.router.route(
                    str(args.get("prompt", "")),
                    float(args.get("max_vram_gib", 6.0)),
                    float(args.get("latency_sla_ms", 1500.0)),
                )
                attrs["mcp.selected_tier"] = payload["selected_tier"]
            elif name == "rag_upsert_document":
                if "doc_id" not in args or "title" not in args or "body" not in args:
                    self.telemetry.record_tool_call(
                        name, t_start, time.time_ns(), is_error=True, traceparent=effective_tp
                    )
                    return {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "error": {"code": -32602, "message": "Invalid params: doc_id, title, and body are required"},
                    }
                payload = self.vault.upsert(
                    str(args["doc_id"]),
                    str(args["title"]),
                    str(args["body"]),
                    list(args.get("tags") or []),
                )
                self._rag_cache.clear()
                attrs["mcp.redactions"] = payload["redactions_applied"]
            elif name == "rag_hybrid_search":
                clean_q, _ = RedactionGuard.sanitize(str(args.get("query", "")))
                top_k = int(args.get("top_k", 5))
                ckey = (" ".join(clean_q.lower().split()), top_k)
                if self._rag_cache_capacity > 0 and ckey in self._rag_cache:
                    self._rag_cache.move_to_end(ckey)
                    payload = {**self._rag_cache[ckey], "cache_hit": True}
                    cache_hit = True
                else:
                    hits = self.vault.hybrid_search(clean_q, top_k)
                    base_payload = {"hits": [h.to_dict() for h in hits], "vault_documents": self.vault.count()}
                    if self._rag_cache_capacity > 0:
                        self._rag_cache[ckey] = base_payload
                        if len(self._rag_cache) > self._rag_cache_capacity:
                            self._rag_cache.popitem(last=False)
                    payload = {**base_payload, "cache_hit": False}
                attrs["mcp.hits"] = len(payload["hits"])
            elif name == "otlp_export_telemetry":
                payload = self.telemetry.export_otlp_json()
            else:
                self.telemetry.record_tool_call(
                    name or "unknown", t_start, time.time_ns(), is_error=True, traceparent=effective_tp
                )
                return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"Unknown tool: {name}"}}
            self.telemetry.record_tool_call(
                name, t_start, time.time_ns(), cache_hit=cache_hit, attributes=attrs, traceparent=effective_tp
            )
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"content": [{"type": "text", "text": json.dumps(payload)}], "isError": False},
            }

        return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"Method not found: {method}"}}
