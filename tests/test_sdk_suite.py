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

            init_req = urllib.request.Request(
                f"http://127.0.0.1:{port}/mcp",
                data=json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": 43,
                        "method": "initialize",
                        "params": {"protocolVersion": "2025-11-25", "capabilities": {"sampling": {}}},
                    }
                ).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with opener.open(init_req, timeout=5) as iresp:
                init_body = json.loads(iresp.read().decode("utf-8"))["result"]
                self.assertEqual(init_body["serverInfo"]["version"], "1.2.0")
                self.assertIn("resources", init_body["capabilities"])
                self.assertIn("prompts", init_body["capabilities"])
        finally:
            srv.shutdown()
            srv.server_close()

    def test_fts5_highlighted_citation_spans_and_xss_safety(self) -> None:
        vault = HybridKnowledgeVault(":memory:")
        doc_body = "SQLite WAL <script>alert(1)</script> fuses BM25 lexical rankings with cosine RRF vectors."
        vault.upsert("cite-1", "FTS5 Citation Engine", doc_body, ["fts5", "citations"])
        vault.upsert("cite-2", "Pure Dense Orthogonal Note", "Unrelated memory allocator slab geometry.", ["os"])

        hits = vault.hybrid_search("BM25 RRF Citation", top_k=2)
        top = hits[0]
        self.assertEqual(top.doc_id, "cite-1")
        self.assertIn("<mark>BM25</mark>", top.highlighted_snippet)
        self.assertIn("<mark>RRF</mark>", top.highlighted_snippet)
        self.assertNotIn("<script>", top.highlighted_snippet)
        self.assertIn("&lt;script&gt;", top.highlighted_snippet)

        clean_body, _ = RedactionGuard.sanitize(doc_body)
        body_spans = [s for s in top.citation_spans if s["field"] == "body"]
        title_spans = [s for s in top.citation_spans if s["field"] == "title"]
        self.assertGreaterEqual(len(body_spans), 2)
        self.assertEqual(len(title_spans), 1)
        for sp in body_spans:
            self.assertEqual(clean_body[sp["start"] : sp["end"]].lower(), sp["term"].lower())
        for sp in title_spans:
            self.assertEqual("FTS5 Citation Engine"[sp["start"] : sp["end"]].lower(), sp["term"].lower())

        # Unmatched lexical hit falls back cleanly with empty citation_spans
        self.assertIsNone(hits[1].lexical_rank)
        self.assertEqual(len(hits[1].citation_spans), 0)

    def test_mcp_capability_negotiation_and_resources_lifecycle(self) -> None:
        vault = HybridKnowledgeVault(":memory:")
        vault.upsert("res-doc-1", "Zero-Leak Security Spec", "Strips sk-abcdefghijklmnopqrstuvwx tokens.", ["sec"])
        disp = McpJsonRpcDispatcher(vault=vault)
        disp.register_resource(
            "mcp://custom/build-info",
            "SDK Build Matrix",
            "Deterministic build metadata.",
            lambda: {"sdk": "mcp-moe-hybrid-sdk", "deterministic": True},
        )

        # Supported older protocol version negotiates cleanly; unknown version falls back to latest 2025-11-25
        init_compat = disp.handle(
            {"jsonrpc": "2.0", "id": 10, "method": "initialize", "params": {"protocolVersion": "2024-11-05", "capabilities": {"roots": {}}}}
        )["result"]
        self.assertEqual(init_compat["protocolVersion"], "2024-11-05")
        self.assertIn("roots", disp.negotiated_client_capabilities)

        init_fallback = disp.handle(
            {"jsonrpc": "2.0", "id": 11, "method": "initialize", "params": {"protocolVersion": "1999-01-01"}}
        )["result"]
        self.assertEqual(init_fallback["protocolVersion"], "2025-11-25")
        self.assertEqual(set(init_fallback["capabilities"].keys()), {"tools", "resources", "prompts"})

        res_list = disp.handle({"jsonrpc": "2.0", "id": 12, "method": "resources/list"})["result"]["resources"]
        uris = {r["uri"] for r in res_list}
        self.assertIn("mcp://moe-router/tiers", uris)
        self.assertIn("mcp://telemetry/otlp", uris)
        self.assertIn("mcp://custom/build-info", uris)
        self.assertIn("vault://documents/res-doc-1", uris)

        tmpl_list = disp.handle({"jsonrpc": "2.0", "id": 13, "method": "resources/templates/list"})["result"]["resourceTemplates"]
        self.assertEqual(tmpl_list[0]["uriTemplate"], "vault://documents/{doc_id}")

        doc_read = disp.handle(
            {"jsonrpc": "2.0", "id": 14, "method": "resources/read", "params": {"uri": "vault://documents/res-doc-1"}}
        )["result"]["contents"][0]
        parsed_doc = json.loads(doc_read["text"])
        self.assertEqual(parsed_doc["doc_id"], "res-doc-1")
        self.assertEqual(parsed_doc["redactions"], 1)
        self.assertIn("[REDACTED_API_KEY]", parsed_doc["body"])

        missing = disp.handle(
            {"jsonrpc": "2.0", "id": 15, "method": "resources/read", "params": {"uri": "vault://documents/nonexistent"}}
        )
        self.assertEqual(missing["error"]["code"], -32002)

    def test_mcp_prompts_list_and_grounded_prompt_rendering(self) -> None:
        vault = HybridKnowledgeVault(":memory:")
        vault.upsert("prm-1", "SQLite WAL Checkpointing", "WAL frames flush sequentially with FTS5 BM25.", ["sqlite"])
        disp = McpJsonRpcDispatcher(vault=vault)

        plist = disp.handle({"jsonrpc": "2.0", "id": 20, "method": "prompts/list"})["result"]["prompts"]
        pnames = {p["name"] for p in plist}
        self.assertIn("grounded_rag_answer", pnames)
        self.assertIn("moe_sla_routing_plan", pnames)

        grounded = disp.handle(
            {
                "jsonrpc": "2.0",
                "id": 21,
                "method": "prompts/get",
                "params": {"name": "grounded_rag_answer", "arguments": {"query": "FTS5 BM25 user@corp.io", "top_k": 1}},
            }
        )["result"]
        msg_txt = grounded["messages"][0]["content"]["text"]
        self.assertIn("[prm-1]", msg_txt)
        self.assertIn("<mark>FTS5</mark>", msg_txt)
        self.assertNotIn("user@corp.io", msg_txt)

        plan = disp.handle(
            {
                "jsonrpc": "2.0",
                "id": 22,
                "method": "prompts/get",
                "params": {
                    "name": "moe_sla_routing_plan",
                    "arguments": {"task": "Prove Black-Scholes convexity theorem", "max_vram_gib": 6.0, "latency_sla_ms": 1200},
                },
            }
        )["result"]
        self.assertIn("frontier_moe_math_architect", plan["messages"][0]["content"]["text"])

        err_missing = disp.handle(
            {"jsonrpc": "2.0", "id": 23, "method": "prompts/get", "params": {"name": "grounded_rag_answer", "arguments": {}}}
        )
        self.assertEqual(err_missing["error"]["code"], -32602)

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
