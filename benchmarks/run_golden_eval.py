#!/usr/bin/env python3
"""Deterministic golden-dataset benchmark measuring Recall@1, Router Accuracy, and p50/p95 latencies."""

from __future__ import annotations

import json
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from mcp_moe_sdk.engine import HybridKnowledgeVault, SparseTop2MoeRouter


def main() -> int:
    vault = HybridKnowledgeVault(":memory:")
    corpus = [
        ("doc-mcp", "MCP 2025-11-25 JSON-RPC Tool Protocol", "Defines initialize, tools/list, and tools/call over JSON-RPC 2.0.", ["mcp", "protocol"]),
        ("doc-rag", "SQLite WAL FTS5 + Vector Cosine RRF", "Combines BM25 full-text search with normalized vector similarity via Reciprocal Rank Fusion.", ["sqlite", "fts5", "rag"]),
        ("doc-moe", "Sparse Top-2 Mixture of Experts Routing", "Dispatches queries across quantized Vulkan and hybrid CPU/GPU MoE tiers under VRAM caps.", ["moe", "vram", "vulkan"]),
        ("doc-sec", "Deterministic Zero-Leak Redaction Guard", "Strips API keys, bearer tokens, and email addresses prior to storage and logging.", ["security", "redaction"]),
        ("doc-ci", "Automated Golden Evaluation & Regression CI", "Verifies Recall@K, p95 latency budgets, and schema conformance on every commit.", ["ci", "eval"]),
    ]
    for did, title, body, tags in corpus:
        vault.upsert(did, title, body, tags)

    rag_queries = [
        ("JSON-RPC 2.0 tools/call protocol", "doc-mcp"),
        ("BM25 FTS5 Reciprocal Rank Fusion", "doc-rag"),
        ("Vulkan hybrid VRAM Mixture of Experts", "doc-moe"),
        ("strip bearer tokens and API keys", "doc-sec"),
        ("Recall@K regression schema CI", "doc-ci"),
    ]
    rag_lat_ms = []
    hits_at_1 = 0
    for q, expected_id in rag_queries * 20:
        t0 = time.perf_counter()
        res = vault.hybrid_search(q, top_k=3)
        rag_lat_ms.append((time.perf_counter() - t0) * 1000.0)
        if res and res[0].doc_id == expected_id:
            hits_at_1 += 1

    router = SparseTop2MoeRouter()
    route_cases = [
        ("Emit JSON health status schema ping", "edge_fast_json"),
        ("Search FTS5 sqlite document policy and summarize", "rag_grounded_synthesis"),
        ("Debug C++ pointer bug and run pytest in docker", "code_and_systems_verifier"),
        ("Prove Black-Scholes convexity theorem and MoE invariant", "frontier_moe_math_architect"),
    ]
    route_ok = 0
    for prompt, exp_tier in route_cases * 25:
        out = router.route(prompt, max_vram_gib=6.0, latency_sla_ms=1200.0)
        if out["selected_tier"] == exp_tier:
            route_ok += 1

    rag_sorted = sorted(rag_lat_ms)
    report = {
        "rag_eval_queries": len(rag_lat_ms),
        "rag_recall_at_1": round(hits_at_1 / len(rag_lat_ms), 4),
        "rag_p50_ms": round(rag_sorted[len(rag_sorted) // 2], 4),
        "rag_p95_ms": round(rag_sorted[int(len(rag_sorted) * 0.95)], 4),
        "router_eval_queries": 100,
        "router_accuracy": round(route_ok / 100.0, 4),
        "router_stats": router.stats(),
    }
    print(json.dumps(report, indent=2))
    assert report["rag_recall_at_1"] == 1.0 and report["router_accuracy"] == 1.0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
