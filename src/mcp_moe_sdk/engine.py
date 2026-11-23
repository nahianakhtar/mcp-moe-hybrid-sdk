"""Core engine: PII Redaction Guard, SQLite WAL FTS5 + Cosine Vector RRF Vault, Sparse Top-2 MoE Router & MCP 2025-11-25 Dispatcher."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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
        clean_title, r1 = RedactionGuard.sanitize(title)
        clean_body, r2 = RedactionGuard.sanitize(body)
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
        if q_tokens:
            fts_expr = " OR ".join(f'"{t}"' for t in q_tokens[:12])
            rows = self.conn.execute(
                "SELECT doc_id FROM documents_fts WHERE documents_fts MATCH ? ORDER BY bm25(documents_fts) LIMIT 50",
                (fts_expr,),
            ).fetchall()
            for rank_idx, (did,) in enumerate(rows, start=1):
                lex_ranks[did] = rank_idx

        q_vec = deterministic_embed(clean_q)
        all_docs = self.conn.execute("SELECT doc_id, title, body, embedding_json FROM documents").fetchall()
        scored_vec: list[tuple[str, str, str, float]] = []
        for did, title, body, emb_json in all_docs:
            sim = cosine_sim(q_vec, json.loads(emb_json))
            scored_vec.append((did, title, body, sim))
        scored_vec.sort(key=lambda item: item[3], reverse=True)
        vec_ranks = {item[0]: idx for idx, item in enumerate(scored_vec, start=1)}

        fused: list[SearchHit] = []
        for did, title, body, sim in scored_vec:
            l_rank = lex_ranks.get(did)
            v_rank = vec_ranks[did]
            score = (1.0 / (rrf_k + v_rank)) + ((1.25 / (rrf_k + l_rank)) if l_rank is not None else 0.0)
            snippet = body[:180] + ("..." if len(body) > 180 else "")
            fused.append(
                SearchHit(
                    doc_id=did,
                    title=title,
                    snippet=snippet,
                    rrf_score=round(score, 6),
                    lexical_rank=l_rank,
                    vector_rank=v_rank,
                    cosine=round(sim, 4),
                )
            )
        fused.sort(key=lambda h: (h.rrf_score, h.cosine), reverse=True)
        return fused[: max(1, top_k)]

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


class McpJsonRpcDispatcher:
    """Model Context Protocol (MCP 2025-11-25) JSON-RPC 2.0 compliant tool server."""

    PROTOCOL_VERSION = "2025-11-25"

    def __init__(self, vault: HybridKnowledgeVault | None = None, router: SparseTop2MoeRouter | None = None) -> None:
        self.vault = vault or HybridKnowledgeVault(":memory:")
        self.router = router or SparseTop2MoeRouter()

    def handle(self, req: dict[str, Any]) -> dict[str, Any]:
        req_id = req.get("id")
        if req.get("jsonrpc") != "2.0" or "method" not in req:
            return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32600, "message": "Invalid JSON-RPC 2.0 Request"}}

        method = req["method"]
        params = req.get("params") or {}

        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": self.PROTOCOL_VERSION,
                    "serverInfo": {"name": "mcp-moe-hybrid-sdk", "version": "1.0.0"},
                    "capabilities": {"tools": {"listChanged": False}},
                },
            }
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
                            "description": "Execute hybrid BM25 FTS5 + cosine vector search fused with Reciprocal Rank Fusion (RRF).",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "query": {"type": "string"},
                                    "top_k": {"type": "integer"},
                                },
                                "required": ["query"],
                            },
                        },
                    ]
                },
            }
        if method == "tools/call":
            name = params.get("name")
            args = params.get("arguments") or {}
            if name == "moe_route_query":
                payload = self.router.route(
                    str(args.get("prompt", "")),
                    float(args.get("max_vram_gib", 6.0)),
                    float(args.get("latency_sla_ms", 1500.0)),
                )
            elif name == "rag_upsert_document":
                payload = self.vault.upsert(
                    str(args["doc_id"]),
                    str(args["title"]),
                    str(args["body"]),
                    list(args.get("tags") or []),
                )
            elif name == "rag_hybrid_search":
                hits = self.vault.hybrid_search(str(args.get("query", "")), int(args.get("top_k", 5)))
                payload = {"hits": [h.__dict__ for h in hits], "vault_documents": self.vault.count()}
            else:
                return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"Unknown tool: {name}"}}
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"content": [{"type": "text", "text": json.dumps(payload)}], "isError": False},
            }

        return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"Method not found: {method}"}}
