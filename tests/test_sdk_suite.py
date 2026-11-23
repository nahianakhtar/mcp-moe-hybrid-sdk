"""Automated golden evaluation, zero-leak redaction, MCP JSON-RPC 2.0, and HTTP server test suite."""

from __future__ import annotations

import json
import threading
import urllib.request
import unittest

from mcp_moe_sdk.engine import (
    HybridKnowledgeVault,
    McpJsonRpcDispatcher,
    OtlpTelemetryExporter,
    RedactionGuard,
    SparseTop2MoeRouter,
)
from mcp_moe_sdk.server import build_server


class TestMcpMoeHybridSdk(unittest.TestCase):
    def test_redaction_guard_strips_secrets(self) -> None:
        raw = "Contact dev@example.com with key sk-1234567890abcdefghijklmnop and Bearer secret.jwt.token"
        clean, count = RedactionGuard.sanitize(raw)
        self.assertEqual(count, 3)
        self.assertNotIn("dev@example.com", clean)
        self.assertNotIn("sk-1234567890", clean)
        self.assertNotIn("secret.jwt.token", clean)

    def test_hybrid_rag_rrf_precision(self) -> None:
        vault = HybridKnowledgeVault(":memory:")
        vault.upsert("d1", "SQLite WAL FTS5 Hybrid Index", "Uses BM25 lexical search fused with vector cosine RRF.", ["db", "rag"])
        vault.upsert("d2", "Rust UEFI Framebuffer", "Configures IA32_PAT write-combining memory type.", ["os", "uefi"])
        vault.upsert("d3", "Sparse Top-2 MoE Router", "Dispatches JSON tool queries under VRAM budgets.", ["moe", "ai"])

        hits = vault.hybrid_search("BM25 FTS5 vector RRF index", top_k=2)
        self.assertEqual(hits[0].doc_id, "d1")
        self.assertEqual(hits[0].lexical_rank, 1)
        self.assertGreater(hits[0].rrf_score, hits[1].rrf_score)

    def test_sparse_moe_routing_and_vram_constraint(self) -> None:
        router = SparseTop2MoeRouter()
        r_json = router.route("Return strict JSON health status ping flag", max_vram_gib=6.0, latency_sla_ms=200.0)
        self.assertEqual(r_json["selected_tier"], "edge_fast_json")
        self.assertEqual(len(r_json["top2_sparse_gates"]), 2)

        r_math = router.route("Prove Black-Scholes convexity theorem and quantum MoE invariant", max_vram_gib=6.0)
        self.assertEqual(r_math["selected_tier"], "frontier_moe_math_architect")

        # Tight 2.0 GiB VRAM cap forces fallback away from 4.8 GiB frontier tier
        r_capped = router.route("Prove Black-Scholes convexity theorem", max_vram_gib=2.0, latency_sla_ms=250.0)
        self.assertEqual(r_capped["selected_tier"], "edge_fast_json")
        self.assertGreaterEqual(router.stats()["calls"], 3)

    def test_mcp_dispatcher_and_live_http_server(self) -> None:
        srv = build_server("127.0.0.1", 0)
        port = srv.server_address[1]
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            with opener.open(f"http://127.0.0.1:{port}/health", timeout=5) as resp:
                health = json.loads(resp.read().decode("utf-8"))
                self.assertEqual(health["status"], "ok")
                self.assertEqual(health["protocol"], "2025-11-25")

            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/mcp",
                data=json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": 42,
                        "method": "tools/call",
                        "params": {"name": "rag_hybrid_search", "arguments": {"query": "Reciprocal Rank Fusion BM25", "top_k": 2}},
                    }
                ).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with opener.open(req, timeout=5) as mresp:
                body = json.loads(mresp.read().decode("utf-8"))
                self.assertEqual(body["id"], 42)
                inner = json.loads(body["result"]["content"][0]["text"])
                self.assertEqual(inner["hits"][0]["doc_id"], "arch-sqlite-rrf")
                self.assertFalse(inner["cache_hit"])

            with opener.open(req, timeout=5) as mresp2:
                inner2 = json.loads(json.loads(mresp2.read().decode("utf-8"))["result"]["content"][0]["text"])
                self.assertTrue(inner2["cache_hit"])

            with opener.open(f"http://127.0.0.1:{port}/metrics/otlp", timeout=5) as oresp:
                otlp = json.loads(oresp.read().decode("utf-8"))
                self.assertEqual(otlp["summary"]["cache_hits"], 1)
                self.assertEqual(otlp["summary"]["cache_misses"], 1)
                self.assertGreaterEqual(len(otlp["resourceSpans"][0]["scopeSpans"][0]["spans"]), 2)
        finally:
            srv.shutdown()
            srv.server_close()

    def test_otlp_span_hook_histogram_and_cache_invalidation(self) -> None:
        captured_spans: list[dict] = []
        exporter = OtlpTelemetryExporter(span_hook=captured_spans.append)
        disp = McpJsonRpcDispatcher(telemetry=exporter)

        disp.handle(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "rag_upsert_document",
                    "arguments": {"doc_id": "otlp-1", "title": "OTLP Histogram Spec", "body": "Exports tool latency buckets.", "tags": ["otlp"]},
                },
            }
        )
        r1 = disp.handle(
            {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "rag_hybrid_search", "arguments": {"query": "OTLP Histogram", "top_k": 1}}}
        )
        r2 = disp.handle(
            {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "rag_hybrid_search", "arguments": {"query": "  otlp   histogram ", "top_k": 1}}}
        )
        self.assertFalse(json.loads(r1["result"]["content"][0]["text"])["cache_hit"])
        self.assertTrue(json.loads(r2["result"]["content"][0]["text"])["cache_hit"])

        # Upsert must invalidate cached RAG search entries cleanly
        disp.handle(
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {
                    "name": "rag_upsert_document",
                    "arguments": {"doc_id": "otlp-2", "title": "OTLP Cache Reset", "body": "Invalidates stale RAG cache.", "tags": ["otlp"]},
                },
            }
        )
        r3 = disp.handle(
            {"jsonrpc": "2.0", "id": 5, "method": "tools/call", "params": {"name": "rag_hybrid_search", "arguments": {"query": "OTLP Histogram", "top_k": 2}}}
        )
        self.assertFalse(json.loads(r3["result"]["content"][0]["text"])["cache_hit"])
        self.assertEqual(len(captured_spans), 5)

        otlp = exporter.export_otlp_json()
        self.assertEqual(otlp["summary"]["cache_hits"], 1)
        self.assertEqual(otlp["summary"]["cache_misses"], 2)
        metrics = otlp["resourceMetrics"][0]["scopeMetrics"][0]["metrics"]
        self.assertEqual(metrics[0]["name"], "mcp.tool.duration")
        self.assertGreaterEqual(len(metrics[0]["histogram"]["dataPoints"]), 2)


if __name__ == "__main__":
    unittest.main()
