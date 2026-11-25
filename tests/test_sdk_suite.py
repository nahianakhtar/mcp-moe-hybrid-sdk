"""Automated golden evaluation, zero-leak redaction, MCP JSON-RPC 2.0, and HTTP server test suite."""

from __future__ import annotations

import json
import threading
import time
import unittest
import urllib.error
import urllib.request

from mcp_moe_sdk.engine import (
    CancellationToken,
    ElicitationSchemaGuard,
    HumanApprovalRegistry,
    HybridKnowledgeVault,
    JsonSchema2020Validator,
    McpCursorCodec,
    McpJsonRpcDispatcher,
    OAuthProtectedResourceGuard,
    OtlpTelemetryExporter,
    ProgressReporter,
    ProgressTokenRegistry,
    RedactionGuard,
    RequestCancellationRegistry,
    SamplingGovernanceGuard,
    SlidingWindowTokenRateLimiter,
    SparseTop2MoeRouter,
    WorkspaceRootsGuard,
    crc32c,
    estimate_mcp_tokens,
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
                self.assertEqual(init_body["serverInfo"]["version"], "1.9.0")
                self.assertIsNotNone(iresp.headers.get("Mcp-Session-Id"))
                self.assertIn("resources", init_body["capabilities"])
                self.assertIn("prompts", init_body["capabilities"])
                self.assertIn("completions", init_body["capabilities"])
                self.assertIn("logging", init_body["capabilities"])
                self.assertIn("roots", init_body["capabilities"])
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
        self.assertEqual(
            set(init_fallback["capabilities"].keys()),
            {"tools", "resources", "prompts", "completions", "logging", "roots"},
        )
        self.assertTrue(init_fallback["capabilities"]["resources"]["subscribe"])

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

    def test_w3c_traceparent_parsing_and_otlp_span_propagation(self) -> None:
        valid_tp = "00-4BF92F3577B34DA6A3CE929D0E0E4736-00F067AA0BA902B7-01"
        parsed = OtlpTelemetryExporter.parse_traceparent(valid_tp)
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed["trace_id"], "4bf92f3577b34da6a3ce929d0e0e4736")
        self.assertEqual(parsed["parent_span_id"], "00f067aa0ba902b7")
        self.assertEqual(parsed["trace_flags"], "01")

        # Reject all-zero trace_id or parent_span_id and malformed headers per W3C Trace Context Level 1
        self.assertIsNone(OtlpTelemetryExporter.parse_traceparent("00-" + "0" * 32 + "-00f067aa0ba902b7-01"))
        self.assertIsNone(OtlpTelemetryExporter.parse_traceparent("00-4bf92f3577b34da6a3ce929d0e0e4736-" + "0" * 16 + "-01"))
        self.assertIsNone(OtlpTelemetryExporter.parse_traceparent("99-invalid"))

        captured: list[dict] = []
        disp = McpJsonRpcDispatcher(telemetry=OtlpTelemetryExporter(span_hook=captured.append))
        disp.handle(
            {
                "jsonrpc": "2.0",
                "id": 301,
                "method": "tools/call",
                "params": {"name": "moe_route_query", "arguments": {"prompt": "Emit JSON health ping"}},
            },
            traceparent=valid_tp,
        )
        disp.handle(
            {
                "jsonrpc": "2.0",
                "id": 302,
                "method": "tools/call",
                "params": {
                    "name": "moe_route_query",
                    "arguments": {"prompt": "Prove convexity theorem"},
                    "_meta": {"traceparent": "00-11112222333344445555666677778888-aaaabbbbccccdddd-00"},
                },
            }
        )
        self.assertEqual(len(captured), 2)
        self.assertEqual(captured[0]["traceId"], "4bf92f3577b34da6a3ce929d0e0e4736")
        self.assertEqual(captured[0]["parentSpanId"], "00f067aa0ba902b7")
        self.assertEqual(captured[0]["traceFlags"], 1)
        self.assertEqual(captured[1]["traceId"], "11112222333344445555666677778888")
        self.assertEqual(captured[1]["parentSpanId"], "aaaabbbbccccdddd")
        self.assertEqual(captured[1]["traceFlags"], 0)

    def test_sliding_window_token_rate_limiter_invariants_and_headers(self) -> None:
        clock_ms = [1_000]
        limiter = SlidingWindowTokenRateLimiter(
            window_ms=10_000,
            max_requests_per_window=3,
            max_tokens_per_window=20,
            max_tracked_clients=2,
            now_fn=lambda: clock_ms[0],
        )

        d1 = limiter.consume("client-a", 12)
        self.assertTrue(d1["allowed"])
        self.assertEqual(d1["remaining_requests"], 2)
        self.assertEqual(d1["remaining_tokens"], 8)
        h1 = limiter.to_headers(d1)
        self.assertEqual(h1["X-RateLimit-Remaining-Tokens"], "8")
        self.assertNotIn("Retry-After", h1)

        clock_ms[0] = 3_500
        d2 = limiter.consume("client-a", 10)
        self.assertFalse(d2["allowed"])
        self.assertEqual(d2["reason"], "sliding_window_tokens_exceeded")
        # First entry (12 tokens at t=1000) expires at t=11000 -> 7500 ms remaining -> Retry-After = ceil(7.5) = 8s
        self.assertEqual(d2["retry_after_ms"], 7_500)
        h2 = limiter.to_headers(d2)
        self.assertEqual(h2["Retry-After"], "8")

        # Non-penalizing rejection: advancing past t=11000 frees the 12-token entry cleanly
        clock_ms[0] = 11_001
        d3 = limiter.consume("client-a", 10)
        self.assertTrue(d3["allowed"])
        self.assertEqual(d3["used_tokens"], 10)

        # Bounded LRU client eviction keeps active_clients <= max_tracked_clients (2)
        limiter.consume("client-b", 2)
        limiter.consume("client-c", 2)
        snap = limiter.snapshot()
        self.assertLessEqual(snap["active_clients"], 2)
        self.assertEqual(snap["total_allowed"], 4)
        self.assertEqual(snap["total_rejected"], 1)

    def test_mcp_jsonrpc_batch_notifications_and_error_isolation(self) -> None:
        disp = McpJsonRpcDispatcher(max_batch_size=4)
        batch = [
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 1, "method": "ping"},
            "malformed-item",
            {"jsonrpc": "2.0", "id": 2, "method": "unknown/method"},
        ]
        status, responses, resp_tp = disp.handle_http_payload(batch)
        self.assertEqual(status, 200)
        self.assertIsNotNone(OtlpTelemetryExporter.parse_traceparent(resp_tp))
        assert isinstance(responses, list)
        # Notification must be omitted -> exactly 3 response items
        self.assertEqual(len(responses), 3)
        self.assertEqual(responses[0]["id"], 1)
        self.assertEqual(responses[0]["result"], {})
        self.assertEqual(responses[1]["error"]["code"], -32600)
        self.assertEqual(responses[2]["error"]["code"], -32601)

        # Notification-only batch returns 202 Accepted with None body
        status_notif, body_notif, _ = disp.handle_http_payload(
            [{"jsonrpc": "2.0", "method": "notifications/initialized"}]
        )
        self.assertEqual(status_notif, 202)
        self.assertIsNone(body_notif)

        # Empty batch and oversized batch return 400 Invalid Request (-32600)
        status_empty, body_empty, _ = disp.handle_http_payload([])
        self.assertEqual(status_empty, 400)
        assert isinstance(body_empty, dict)
        self.assertEqual(body_empty["error"]["code"], -32600)

        status_over, body_over, _ = disp.handle_http_payload(
            [{"jsonrpc": "2.0", "id": i, "method": "ping"} for i in range(5)]
        )
        self.assertEqual(status_over, 400)
        assert isinstance(body_over, dict)
        self.assertEqual(body_over["error"]["code"], -32600)

        self.assertGreater(
            estimate_mcp_tokens([{"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"query": "x" * 80}}]),
            estimate_mcp_tokens({"jsonrpc": "2.0", "id": 1, "method": "ping"}),
        )

    def test_http_server_streamable_batch_traceparent_sse_and_429_governance(self) -> None:
        limiter = SlidingWindowTokenRateLimiter(window_ms=60_000, max_requests_per_window=4, max_tokens_per_window=50)
        srv = build_server("127.0.0.1", 0, rate_limiter=limiter, max_batch_size=8)
        port = srv.server_address[1]
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        incoming_tp = "00-abcdef0123456789abcdef0123456789-1122334455667788-01"
        try:
            batch_payload = [
                {"jsonrpc": "2.0", "method": "notifications/initialized"},
                {
                    "jsonrpc": "2.0",
                    "id": 901,
                    "method": "tools/call",
                    "params": {"name": "moe_route_query", "arguments": {"prompt": "Return JSON status ping"}},
                },
                {
                    "jsonrpc": "2.0",
                    "id": 902,
                    "method": "tools/call",
                    "params": {"name": "rag_hybrid_search", "arguments": {"query": "BM25 RRF", "top_k": 1}},
                },
            ]
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/mcp",
                data=json.dumps(batch_payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "traceparent": incoming_tp,
                    "X-Client-Id": "tenant-alpha",
                },
            )
            with opener.open(req, timeout=5) as resp:
                self.assertEqual(resp.status, 200)
                echo_tp = resp.headers.get("traceparent") or ""
                self.assertTrue(echo_tp.startswith("00-abcdef0123456789abcdef0123456789-"))
                self.assertEqual(resp.headers.get("X-RateLimit-Limit-Requests"), "4")
                self.assertEqual(resp.headers.get("X-RateLimit-Remaining-Requests"), "3")
                arr = json.loads(resp.read().decode("utf-8"))
                self.assertEqual([item["id"] for item in arr], [901, 902])

            # Verify SSE framing when Accept: text/event-stream is requested
            sse_req = urllib.request.Request(
                f"http://127.0.0.1:{port}/mcp",
                data=json.dumps({"jsonrpc": "2.0", "id": 903, "method": "ping"}).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Accept": "text/event-stream",
                    "X-Client-Id": "tenant-alpha",
                },
            )
            with opener.open(sse_req, timeout=5) as sresp:
                self.assertIn("text/event-stream", sresp.headers.get("Content-Type", ""))
                sse_text = sresp.read().decode("utf-8")
                self.assertIn("event: message\ndata: ", sse_text)
                self.assertIn('"id": 903', sse_text)

            # Verify OTLP spans share incoming W3C traceId and parentSpanId
            with opener.open(f"http://127.0.0.1:{port}/metrics/otlp", timeout=5) as oresp:
                otlp = json.loads(oresp.read().decode("utf-8"))
                spans = otlp["resourceSpans"][0]["scopeSpans"][0]["spans"]
                self.assertEqual(len(spans), 2)
                for sp in spans:
                    self.assertEqual(sp["traceId"], "abcdef0123456789abcdef0123456789")
                    self.assertEqual(sp["parentSpanId"], "1122334455667788")

            # Exceed tenant-alpha's 30-token window budget -> HTTP 429 + Retry-After + -32029
            heavy_req = urllib.request.Request(
                f"http://127.0.0.1:{port}/mcp",
                data=json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": 904,
                        "method": "tools/call",
                        "params": {"name": "rag_hybrid_search", "arguments": {"query": "A" * 160, "top_k": 3}},
                    }
                ).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "traceparent": incoming_tp,
                    "X-Client-Id": "tenant-alpha",
                },
            )
            with self.assertRaises(urllib.error.HTTPError) as ctx:
                opener.open(heavy_req, timeout=5)
            err_resp = ctx.exception
            try:
                self.assertEqual(err_resp.code, 429)
                self.assertIsNotNone(err_resp.headers.get("Retry-After"))
                self.assertTrue((err_resp.headers.get("traceparent") or "").startswith("00-abcdef0123456789abcdef0123456789-"))
                err_json = json.loads(err_resp.read().decode("utf-8"))
                self.assertEqual(err_json["error"]["code"], -32029)
                self.assertEqual(err_json["id"], 904)
            finally:
                err_resp.close()

            with opener.open(f"http://127.0.0.1:{port}/health", timeout=5) as hresp:
                health = json.loads(hresp.read().decode("utf-8"))
                self.assertEqual(health["version"], "1.9.0")
                self.assertEqual(health["rate_limit"]["total_allowed"], 2)
                self.assertEqual(health["rate_limit"]["total_rejected"], 1)
        finally:
            srv.shutdown()
            srv.server_close()

    def test_crc32c_and_c11_ffi_bit_identical_child_traceparent_vectors(self) -> None:
        self.assertEqual(crc32c(b""), 0)
        self.assertEqual(crc32c("123456789"), 0xE3069283)

        c0 = OtlpTelemetryExporter.derive_child_traceparent(
            "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
            step_seq=0,
            agent_id=1,
            label="rag.retrieval.rrf",
        )
        self.assertEqual(c0, "00-4bf92f3577b34da6a3ce929d0e0e4736-ccabe778b789d68b-01")

        c1 = OtlpTelemetryExporter.derive_child_traceparent(
            "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
            step_seq=1,
            agent_id=1,
            label="mcp.tools.call/rag_query",
        )
        self.assertEqual(c1, "00-4bf92f3577b34da6a3ce929d0e0e4736-222a56995908676a-01")

        c2 = OtlpTelemetryExporter.derive_child_traceparent(
            "00-abcdef0123456789abcdef0123456789-1122334455667788-01",
            step_seq=2,
            agent_id=7,
            label="rag.diff.compare",
        )
        self.assertEqual(c2, "00-abcdef0123456789abcdef0123456789-1f870e9064a53f63-01")
        self.assertIsNone(OtlpTelemetryExporter.derive_child_traceparent("invalid-header"))

    def test_deterministic_batch_child_span_lineage_across_mcp_calls(self) -> None:
        captured: list[dict] = []
        disp = McpJsonRpcDispatcher(telemetry=OtlpTelemetryExporter(span_hook=captured.append))
        parent_tp = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
        batch = [
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "moe_route_query", "arguments": {"prompt": "Return JSON ping"}},
            },
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "rag_hybrid_search", "arguments": {"query": "SQLite WAL", "top_k": 1}},
            },
        ]
        responses = disp.handle_batch(batch, traceparent=parent_tp, deterministic_child_spans=True, agent_id=3)
        self.assertEqual(len(responses), 2)
        self.assertEqual(len(captured), 2)
        exp0 = OtlpTelemetryExporter.derive_child_traceparent(
            parent_tp, step_seq=1, agent_id=3, label="mcp.tools.call/moe_route_query"
        )
        exp1 = OtlpTelemetryExporter.derive_child_traceparent(
            parent_tp, step_seq=2, agent_id=3, label="mcp.tools.call/rag_hybrid_search"
        )
        assert exp0 is not None and exp1 is not None
        self.assertEqual(captured[0]["spanId"], exp0.split("-")[2])
        self.assertEqual(captured[1]["spanId"], exp1.split("-")[2])
        self.assertNotEqual(captured[0]["spanId"], captured[1]["spanId"])

    def test_json_schema_2020_12_validator_edge_cases_and_bool_int_strictness(self) -> None:
        # Strict Python bool vs int/number separation (isinstance(True, int) == True trap)
        self.assertFalse(JsonSchema2020Validator.is_valid(True, {"type": "integer"}))
        self.assertFalse(JsonSchema2020Validator.is_valid(False, {"type": "number"}))
        self.assertFalse(JsonSchema2020Validator.is_valid(float("nan"), {"type": "number"}))
        self.assertFalse(JsonSchema2020Validator.is_valid(float("inf"), {"type": "number"}))
        self.assertTrue(JsonSchema2020Validator.is_valid(42, {"type": "integer", "minimum": 1, "multipleOf": 6}))

        # Draft 2020-12 prefixItems + items: false tuple validation & enum bool/int distinction
        tuple_schema = {
            "$schema": JsonSchema2020Validator.SCHEMA_URI,
            "type": "array",
            "prefixItems": [{"type": "string", "pattern": "^doc-[0-9]+$"}, {"type": ["integer", "null"]}],
            "items": False,
        }
        self.assertTrue(JsonSchema2020Validator.is_valid(["doc-07", None], tuple_schema))
        self.assertTrue(JsonSchema2020Validator.is_valid(["doc-07", 3], tuple_schema))
        self.assertFalse(JsonSchema2020Validator.is_valid(["doc-07", 3, "extra"], tuple_schema))
        self.assertFalse(JsonSchema2020Validator.is_valid(["bad", 3], tuple_schema))

        # Enum / const / uniqueItems strict JSON equality (1 != True)
        self.assertFalse(JsonSchema2020Validator.is_valid(True, {"enum": [1, 2, 3]}))
        self.assertTrue(JsonSchema2020Validator.is_valid([1, True], {"type": "array", "uniqueItems": True}))
        self.assertFalse(JsonSchema2020Validator.is_valid([1, 1.0], {"type": "array", "uniqueItems": True}))

        # Composition keywords (oneOf / anyOf / allOf / not)
        one_of_schema = {"oneOf": [{"type": "integer", "multipleOf": 2}, {"type": "integer", "multipleOf": 3}]}
        self.assertTrue(JsonSchema2020Validator.is_valid(4, one_of_schema))
        self.assertFalse(JsonSchema2020Validator.is_valid(6, one_of_schema))  # matches both -> violates oneOf

    def test_mcp_tools_output_schema_structured_content_and_contract_violation(self) -> None:
        disp = McpJsonRpcDispatcher()
        tools = disp.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})["result"]["tools"]
        by_name = {t["name"]: t for t in tools}
        for builtin in ("moe_route_query", "rag_upsert_document", "rag_hybrid_search", "otlp_export_telemetry"):
            self.assertIn("inputSchema", by_name[builtin])
            self.assertIn("outputSchema", by_name[builtin])
            self.assertEqual(by_name[builtin]["outputSchema"]["$schema"], JsonSchema2020Validator.SCHEMA_URI)

        # Valid tools/call returns both backwards-compatible text content and MCP 2025-11-25 structuredContent
        ok_resp = disp.handle(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "moe_route_query", "arguments": {"prompt": "Prove convexity theorem"}},
            }
        )["result"]
        self.assertFalse(ok_resp["isError"])
        self.assertIn("structuredContent", ok_resp)
        self.assertEqual(ok_resp["structuredContent"], json.loads(ok_resp["content"][0]["text"]))
        self.assertEqual(ok_resp["structuredContent"]["selected_tier"], "frontier_moe_math_architect")

        # Strict inputSchema enforcement rejects unknown property and bool-as-integer top_k with -32602
        bad_in = disp.handle(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": "rag_hybrid_search", "arguments": {"query": "BM25", "top_k": True}},
            }
        )
        self.assertEqual(bad_in["error"]["code"], -32602)
        self.assertEqual(bad_in["error"]["data"]["phase"], "inputSchema")

        # Faulty custom tool violating its outputSchema is caught deterministically with -32603
        disp.register_tool(
            "broken_metric_emitter",
            "Test tool returning malformed output to verify outputSchema contract gate.",
            input_schema={"type": "object", "properties": {}, "additionalProperties": False},
            output_schema={
                "$schema": JsonSchema2020Validator.SCHEMA_URI,
                "type": "object",
                "properties": {"score": {"type": "number", "minimum": 0, "maximum": 1}},
                "required": ["score"],
                "additionalProperties": False,
            },
            handler=lambda _args: {"score": 1.75},
        )
        bad_out = disp.handle(
            {"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "broken_metric_emitter", "arguments": {}}}
        )
        self.assertEqual(bad_out["error"]["code"], -32603)
        self.assertEqual(bad_out["error"]["data"]["phase"], "outputSchema")

        otlp_resp = disp.handle(
            {"jsonrpc": "2.0", "id": 5, "method": "tools/call", "params": {"name": "otlp_export_telemetry"}}
        )["result"]
        self.assertGreaterEqual(otlp_resp["structuredContent"]["summary"]["schema_validations_passed"], 2)
        self.assertEqual(otlp_resp["structuredContent"]["summary"]["schema_validations_failed"], 2)

    def test_mcp_notifications_cancelled_prearrival_batch_and_initialize_immunity(self) -> None:
        clock = [10.0]
        reg = RequestCancellationRegistry(max_tombstones=4, tombstone_ttl_sec=5.0, now_fn=lambda: clock[0])
        disp = McpJsonRpcDispatcher(cancellation=reg)

        # 1. Pre-arrival cancellation prevents cancelled rag_upsert_document from mutating SQLite WAL vault
        disp.handle(
            {
                "jsonrpc": "2.0",
                "method": "notifications/cancelled",
                "params": {"requestId": "upsert-99", "reason": "User aborted with key sk-1234567890abcdefghijklmnop"},
            }
        )
        cancelled_direct = disp.handle(
            {
                "jsonrpc": "2.0",
                "id": "upsert-99",
                "method": "tools/call",
                "params": {
                    "name": "rag_upsert_document",
                    "arguments": {"doc_id": "never-written", "title": "Aborted", "body": "Should never persist."},
                },
            }
        )
        self.assertEqual(cancelled_direct["error"]["code"], -32800)
        self.assertTrue(cancelled_direct["error"]["data"]["cancelled"])
        self.assertNotIn("sk-1234567890", cancelled_direct["error"]["message"])
        self.assertEqual(disp.vault.count(), 0)

        # 2. MCP spec immunity: initialize MUST NOT be cancelled even if pre-cancelled
        disp.handle({"jsonrpc": "2.0", "method": "notifications/cancelled", "params": {"requestId": 777}})
        init_ok = disp.handle({"jsonrpc": "2.0", "id": 777, "method": "initialize", "params": {"protocolVersion": "2025-11-25"}})
        self.assertIn("result", init_ok)
        self.assertEqual(init_ok["result"]["protocolVersion"], "2025-11-25")

        # 3. Same-batch notifications/cancelled cancels matching sibling request and elides its response
        batch = [
            {
                "jsonrpc": "2.0",
                "id": 501,
                "method": "tools/call",
                "params": {"name": "moe_route_query", "arguments": {"prompt": "Return JSON health status"}},
            },
            {
                "jsonrpc": "2.0",
                "id": 502,
                "method": "tools/call",
                "params": {
                    "name": "rag_upsert_document",
                    "arguments": {"doc_id": "batch-aborted", "title": "Skip", "body": "Cancelled in batch."},
                },
            },
            {"jsonrpc": "2.0", "method": "notifications/cancelled", "params": {"requestId": 502, "reason": "Superseded"}},
        ]
        status_b, resp_b, _ = disp.handle_http_payload(batch)
        self.assertEqual(status_b, 200)
        assert isinstance(resp_b, list)
        self.assertEqual([r["id"] for r in resp_b], [501])
        self.assertEqual(disp.vault.count(), 0)

    def test_concurrent_inflight_http_tool_cancellation_via_cancellation_token(self) -> None:
        srv = build_server("127.0.0.1", 0)
        port = srv.server_address[1]
        entered = threading.Event()

        def cancellable_slow_worker(_args: dict, token: CancellationToken) -> dict:
            entered.set()
            token.wait(timeout=3.0)
            token.raise_if_cancelled()
            return {"completed": True}

        srv.dispatcher.register_tool(  # type: ignore[attr-defined]
            "cancellable_long_job",
            "Cooperative cancellable job for live Streamable HTTP cancellation verification.",
            input_schema={"type": "object", "properties": {}, "additionalProperties": False},
            output_schema={
                "type": "object",
                "properties": {"completed": {"type": "boolean"}},
                "required": ["completed"],
                "additionalProperties": False,
            },
            handler=cancellable_slow_worker,
            accepts_cancel_token=True,
        )

        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        worker_status: list[int] = []

        def run_inflight_call() -> None:
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/mcp",
                data=json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": "slow-42",
                        "method": "tools/call",
                        "params": {"name": "cancellable_long_job", "arguments": {}},
                    }
                ).encode("utf-8"),
                headers={"Content-Type": "application/json", "X-Client-Id": "cancel-tenant"},
            )
            with opener.open(req, timeout=5) as resp:
                worker_status.append(resp.status)

        t0 = time.perf_counter()
        call_thread = threading.Thread(target=run_inflight_call, daemon=True)
        call_thread.start()
        try:
            self.assertTrue(entered.wait(timeout=2.0))
            cancel_req = urllib.request.Request(
                f"http://127.0.0.1:{port}/mcp",
                data=json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "method": "notifications/cancelled",
                        "params": {"requestId": "slow-42", "reason": "SLA deadline exceeded"},
                    }
                ).encode("utf-8"),
                headers={"Content-Type": "application/json", "X-Client-Id": "cancel-tenant"},
            )
            with opener.open(cancel_req, timeout=5) as cresp:
                self.assertEqual(cresp.status, 202)

            call_thread.join(timeout=2.0)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self.assertFalse(call_thread.is_alive())
            self.assertEqual(worker_status, [202])
            self.assertLess(elapsed_ms, 1000.0)

            with opener.open(f"http://127.0.0.1:{port}/metrics/otlp", timeout=5) as oresp:
                otlp = json.loads(oresp.read().decode("utf-8"))
                self.assertEqual(otlp["summary"]["requests_cancelled"], 1)
                spans = otlp["resourceSpans"][0]["scopeSpans"][0]["spans"]
                cancel_spans = [s for s in spans if s["name"] == "mcp.tools.call/cancellable_long_job"]
                self.assertEqual(len(cancel_spans), 1)
                attr_map = {a["key"]: a["value"] for a in cancel_spans[0]["attributes"]}
                self.assertEqual(attr_map["mcp.cancelled"]["boolValue"], True)
                self.assertEqual(attr_map["mcp.cancel.reason"]["stringValue"], "SLA deadline exceeded")
        finally:
            srv.shutdown()
            srv.server_close()

    def test_mcp_cursor_codec_c11_canonical_vectors_and_tamper_invariants(self) -> None:
        import base64

        # Canonical C11 FFI par_cursor_seal(2, 50, 0xE3069283) check vector
        seal = McpCursorCodec.compute_seal(McpCursorCodec.MAGIC, McpCursorCodec.VERSION, 2, 50, 0xE3069283)
        self.assertEqual(seal, 0xF70B1A24)
        tok = McpCursorCodec.encode(McpCursorCodec.SCOPE_RESOURCES, 50, 0xE3069283)
        self.assertEqual(tok, "Q1ABAjIAAACDkgbjJBoL9w")
        code, off, _ = McpCursorCodec.decode_and_verify(tok, McpCursorCodec.SCOPE_RESOURCES, 0xE3069283, 120)
        self.assertEqual((code, off), (0, 50))

        # Cross-scope, snapshot-drift, and out-of-bounds checks
        self.assertEqual(
            McpCursorCodec.decode_and_verify(tok, McpCursorCodec.SCOPE_TOOLS, 0xE3069283, 120)[0],
            McpCursorCodec.ERR_SCOPE,
        )
        self.assertEqual(
            McpCursorCodec.decode_and_verify(tok, McpCursorCodec.SCOPE_RESOURCES, 0xE3069282, 120)[0],
            McpCursorCodec.ERR_SNAPSHOT,
        )
        self.assertEqual(
            McpCursorCodec.decode_and_verify(tok, McpCursorCodec.SCOPE_RESOURCES, 0xE3069283, 50)[0],
            McpCursorCodec.ERR_BOUNDS,
        )

        # Exhaustive 128-bit single-bit flip tamper check -> ERR_SEAL (-17)
        raw16 = bytearray(base64.urlsafe_b64decode(tok + "=="))
        for bit in range(128):
            mutated = bytearray(raw16)
            mutated[bit >> 3] ^= 1 << (bit & 7)
            mut_tok = base64.urlsafe_b64encode(bytes(mutated)).rstrip(b"=").decode("ascii")
            self.assertEqual(
                McpCursorCodec.decode_and_verify(mut_tok, McpCursorCodec.SCOPE_RESOURCES, 0xE3069283, 120)[0],
                McpCursorCodec.ERR_SEAL,
            )

        # Strict RFC 4648 §5 canonical trailing-bit check on 22nd sextet -> ERR_B64 (-13)
        alph = McpCursorCodec._ALPHABET
        base_s21 = alph.index(tok[21]) & 0x30
        for low_bits in range(1, 16):
            non_canon = tok[:21] + alph[base_s21 | low_bits]
            self.assertEqual(
                McpCursorCodec.decode_and_verify(non_canon, McpCursorCodec.SCOPE_RESOURCES, 0xE3069283, 120)[0],
                McpCursorCodec.ERR_B64,
            )

    def test_mcp_paginated_lists_cursor_iteration_and_snapshot_drift_guard(self) -> None:
        vault = HybridKnowledgeVault(":memory:")
        for idx in range(5):
            vault.upsert(f"doc-{idx:02d}", f"Doc {idx}", f"Body content for document {idx}.", ["rag"])
        disp = McpJsonRpcDispatcher(vault=vault, page_size=2)

        # Paginate resources/list (9 built-in governance/telemetry resources + 5 vault docs = 14 total)
        collected_uris: list[str] = []
        cursor: str | None = None
        pages = 0
        while True:
            params = {"cursor": cursor} if cursor is not None else {}
            resp = disp.handle({"jsonrpc": "2.0", "id": 100 + pages, "method": "resources/list", "params": params})
            self.assertIn("result", resp)
            page_items = resp["result"]["resources"]
            self.assertLessEqual(len(page_items), 2)
            collected_uris.extend(r["uri"] for r in page_items)
            pages += 1
            cursor = resp["result"].get("nextCursor")
            if cursor is None:
                break
        self.assertEqual(pages, 7)
        self.assertEqual(len(collected_uris), 14)
        self.assertEqual(len(set(collected_uris)), 14)

        # Cross-endpoint replay: passing a tools/list cursor into resources/list must fail with -32602
        t_page1 = disp.handle({"jsonrpc": "2.0", "id": 201, "method": "tools/list"})["result"]
        self.assertEqual(len(t_page1["tools"]), 2)
        tools_cursor = t_page1["nextCursor"]
        replay_err = disp.handle(
            {"jsonrpc": "2.0", "id": 202, "method": "resources/list", "params": {"cursor": tools_cursor}}
        )
        self.assertEqual(replay_err["error"]["code"], -32602)
        self.assertEqual(replay_err["error"]["data"]["code"], McpCursorCodec.ERR_SCOPE)

        # Mutating the vault mid-pagination invalidates stale resources/list snapshot cursor with -32602
        r_page1 = disp.handle({"jsonrpc": "2.0", "id": 203, "method": "resources/list"})["result"]
        stale_cursor = r_page1["nextCursor"]
        disp.handle(
            {
                "jsonrpc": "2.0",
                "id": 204,
                "method": "tools/call",
                "params": {
                    "name": "rag_upsert_document",
                    "arguments": {"doc_id": "doc-99", "title": "Mutated", "body": "Epoch changed."},
                },
            }
        )
        stale_err = disp.handle(
            {"jsonrpc": "2.0", "id": 205, "method": "resources/list", "params": {"cursor": stale_cursor}}
        )
        self.assertEqual(stale_err["error"]["code"], -32602)
        self.assertEqual(stale_err["error"]["data"]["code"], McpCursorCodec.ERR_SNAPSHOT)

    def test_mcp_completion_complete_resource_and_prompt_ranking_plus_redaction(self) -> None:
        vault = HybridKnowledgeVault(":memory:")
        for i in range(115):
            vault.upsert(f"spec-{i:03d}", f"Protocol Spec {i}", "SQLite WAL FTS5 hybrid retrieval.", ["mcp", "sqlite"])
        vault.upsert("moe-gpu-01", "Sparse Top-2 MoE Kernel", "Routes Vulkan Q8 slots.", ["moe", "vulkan"])
        disp = McpJsonRpcDispatcher(vault=vault)

        # 1. Resource template completion caps at <= 100 items and sets total=115, hasMore=True
        res_comp = disp.handle(
            {
                "jsonrpc": "2.0",
                "id": 301,
                "method": "completion/complete",
                "params": {
                    "ref": {"type": "ref/resource", "uri": "vault://documents/{doc_id}"},
                    "argument": {"name": "doc_id", "value": "spec-"},
                },
            }
        )["result"]["completion"]
        self.assertEqual(len(res_comp["values"]), 100)
        self.assertEqual(res_comp["total"], 115)
        self.assertTrue(res_comp["hasMore"])
        self.assertEqual(res_comp["values"][0], "spec-000")

        # 2. Prompt completion for moe_sla_routing_plan & custom prompt with secret scrubbing
        moe_comp = disp.handle(
            {
                "jsonrpc": "2.0",
                "id": 302,
                "method": "completion/complete",
                "params": {
                    "ref": {"type": "ref/prompt", "name": "moe_sla_routing_plan"},
                    "argument": {"name": "max_vram_gib", "value": "4."},
                },
            }
        )["result"]["completion"]
        self.assertEqual(moe_comp["values"], ["4.8"])
        self.assertFalse(moe_comp["hasMore"])

        disp.register_prompt(
            "tenant_runbook",
            "Custom runbook prompt with completion.",
            [{"name": "env", "required": True}],
            lambda args: {"description": "Runbook", "messages": []},
            completer=lambda _arg, pref: [f"{pref}-prod-contact-ops@corp.io", f"{pref}-staging"],
        )
        cust_comp = disp.handle(
            {
                "jsonrpc": "2.0",
                "id": 303,
                "method": "completion/complete",
                "params": {
                    "ref": {"type": "ref/prompt", "name": "tenant_runbook"},
                    "argument": {"name": "env", "value": "us-east"},
                },
            }
        )["result"]["completion"]
        self.assertEqual(len(cust_comp["values"]), 2)
        self.assertNotIn("ops@corp.io", cust_comp["values"][0])
        self.assertIn("[REDACTED_EMAIL]", cust_comp["values"][0])

    def test_mcp_resource_subscriptions_and_change_notification_outbox(self) -> None:
        srv = build_server("127.0.0.1", 0)
        port = srv.server_address[1]
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            def rpc_post(body: dict) -> dict:
                req = urllib.request.Request(
                    f"http://127.0.0.1:{port}/mcp",
                    data=json.dumps(body).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                )
                with opener.open(req, timeout=5) as r:
                    return json.loads(r.read().decode("utf-8"))

            # Subscribe to vault://documents/live-rfc, ignore unsubscribed doc updates
            sub_ok = rpc_post(
                {"jsonrpc": "2.0", "id": 401, "method": "resources/subscribe", "params": {"uri": "vault://documents/live-rfc"}}
            )
            self.assertEqual(sub_ok["result"], {})

            sub_bad = rpc_post(
                {"jsonrpc": "2.0", "id": 402, "method": "resources/subscribe", "params": {"uri": "ftp://invalid"}}
            )
            self.assertEqual(sub_bad["error"]["code"], -32602)

            rpc_post(
                {
                    "jsonrpc": "2.0",
                    "id": 403,
                    "method": "tools/call",
                    "params": {
                        "name": "rag_upsert_document",
                        "arguments": {"doc_id": "other-doc", "title": "Other", "body": "No subscriber."},
                    },
                }
            )
            rpc_post(
                {
                    "jsonrpc": "2.0",
                    "id": 404,
                    "method": "tools/call",
                    "params": {
                        "name": "rag_upsert_document",
                        "arguments": {"doc_id": "live-rfc", "title": "Subscribed", "body": "Triggers update event."},
                    },
                }
            )

            with opener.open(f"http://127.0.0.1:{port}/mcp/notifications", timeout=5) as nresp:
                notifs = json.loads(nresp.read().decode("utf-8"))["notifications"]
                self.assertEqual(len(notifs), 1)
                self.assertEqual(notifs[0]["method"], "notifications/resources/updated")
                self.assertEqual(notifs[0]["params"]["uri"], "vault://documents/live-rfc")

            # Unsubscribe silences subsequent updates
            rpc_post(
                {"jsonrpc": "2.0", "id": 405, "method": "resources/unsubscribe", "params": {"uri": "vault://documents/live-rfc"}}
            )
            rpc_post(
                {
                    "jsonrpc": "2.0",
                    "id": 406,
                    "method": "tools/call",
                    "params": {
                        "name": "rag_upsert_document",
                        "arguments": {"doc_id": "live-rfc", "title": "Subscribed v2", "body": "Muted after unsubscribe."},
                    },
                }
            )
            with opener.open(f"http://127.0.0.1:{port}/mcp/notifications", timeout=5) as nresp2:
                self.assertEqual(json.loads(nresp2.read().decode("utf-8"))["notifications"], [])
        finally:
            srv.shutdown()
            srv.server_close()

    def test_mcp_rfc5424_logging_set_level_and_redacted_message_notifications(self) -> None:
        disp = McpJsonRpcDispatcher()
        bad_lvl = disp.handle({"jsonrpc": "2.0", "id": 501, "method": "logging/setLevel", "params": {"level": "verbose"}})
        self.assertEqual(bad_lvl["error"]["code"], -32602)

        ok_lvl = disp.handle({"jsonrpc": "2.0", "id": 502, "method": "logging/setLevel", "params": {"level": "warning"}})
        self.assertEqual(ok_lvl["result"], {})

        # Below threshold ('info' < 'warning') dropped; >= 'warning' queued with PII/API-key redaction
        self.assertFalse(disp.emit_log("info", {"msg": "Routine heartbeat"}))
        self.assertTrue(
            disp.emit_log("error", {"msg": "Upstream token sk-1234567890abcdefghijklmnop for dev@corp.io rejected"})
        )
        drained = disp.drain_notifications()
        self.assertEqual(len(drained), 1)
        self.assertEqual(drained[0]["method"], "notifications/message")
        self.assertEqual(drained[0]["params"]["level"], "error")
        msg_str = drained[0]["params"]["data"]["msg"]
        self.assertNotIn("sk-1234567890", msg_str)
        self.assertNotIn("dev@corp.io", msg_str)
        self.assertIn("[REDACTED_API_KEY]", msg_str)
        self.assertIn("[REDACTED_EMAIL]", msg_str)

    def test_mcp_tool_annotations_on_builtin_and_custom_tools(self) -> None:
        disp = McpJsonRpcDispatcher()
        disp.register_tool(
            "readonly_probe",
            "Read-only diagnostic probe.",
            input_schema={"type": "object", "properties": {}, "additionalProperties": False},
            annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
            handler=lambda _a: {"ok": True},
        )
        tools = {t["name"]: t for t in disp.handle({"jsonrpc": "2.0", "id": 601, "method": "tools/list"})["result"]["tools"]}
        self.assertTrue(tools["moe_route_query"]["annotations"]["readOnlyHint"])
        self.assertTrue(tools["rag_hybrid_search"]["annotations"]["readOnlyHint"])
        self.assertFalse(tools["rag_upsert_document"]["annotations"]["readOnlyHint"])
        self.assertTrue(tools["rag_upsert_document"]["annotations"]["idempotentHint"])
        self.assertTrue(tools["readonly_probe"]["annotations"]["readOnlyHint"])

    def test_workspace_roots_guard_rfc8089_canonicalization_and_traversal_defense(self) -> None:
        # Canonical normalization of localhost, redundant slashes, dot segments, and %20
        rc, canon, _ = WorkspaceRootsGuard.normalize_file_uri(
            "FILE://LOCALHOST/workspace/app//./docs/../specs/design%20v1.md/"
        )
        self.assertEqual(rc, 0)
        self.assertEqual(canon, "file:///workspace/app/specs/design%20v1.md")

        # Exact root match and nested child match succeed; sibling-prefix confusion fails with -22
        self.assertEqual(
            WorkspaceRootsGuard.is_within_root("file:///workspace/app", "file:///workspace/app/")[0],
            0,
        )
        self.assertEqual(
            WorkspaceRootsGuard.is_within_root("file:///workspace/app/src/main.py", "file:///workspace/app")[0],
            0,
        )
        self.assertEqual(
            WorkspaceRootsGuard.is_within_root("file:///workspace/app-secrets/keys.pem", "file:///workspace/app")[0],
            WorkspaceRootsGuard.ERR_URI_OUTSIDE_ROOTS,
        )

        # Dot-dot escaping root vs escaping filesystem '/'
        self.assertEqual(
            WorkspaceRootsGuard.is_within_root(
                "file:///workspace/app/%2e%2e/%2e%2e/etc/passwd", "file:///workspace/app"
            )[0],
            WorkspaceRootsGuard.ERR_URI_OUTSIDE_ROOTS,
        )
        self.assertEqual(
            WorkspaceRootsGuard.normalize_file_uri("file:///workspace/../../etc/shadow")[0],
            WorkspaceRootsGuard.ERR_URI_TRAVERSAL,
        )

        # Reject remote UNC/SMB authority (-21) and encoded slash/backslash/NUL/query/fragment (-19)
        self.assertEqual(
            WorkspaceRootsGuard.normalize_file_uri("file://evil.example.com/workspace/app/a.txt")[0],
            WorkspaceRootsGuard.ERR_URI_AUTHORITY,
        )
        for bad_syntax in (
            "file:///workspace/app/..%2fsecret.txt",
            "file:///workspace/app\\..\\secret.txt",
            "file:///workspace/app/%5csecret.txt",
            "file:///workspace/app/doc%00.txt",
            "file:///workspace/app/doc.txt?raw=1",
            "file:///workspace/app/doc.txt#frag",
            "https://example.com/file.txt",
        ):
            self.assertEqual(
                WorkspaceRootsGuard.normalize_file_uri(bad_syntax)[0],
                WorkspaceRootsGuard.ERR_URI_SYNTAX,
                msg=f"Expected ERR_URI_SYNTAX for {bad_syntax!r}",
            )

    def test_mcp_roots_list_set_roots_notification_and_file_resource_confinement(self) -> None:
        disp = McpJsonRpcDispatcher(
            roots=[{"uri": "file:///workspace/allowed/", "name": "Allowed Project dev@corp.io"}]
        )
        disp.register_resource(
            "file:///workspace/allowed/config.json",
            "Allowed Config",
            "Inside workspace boundary.",
            lambda: {"env": "prod"},
        )
        disp.register_resource(
            "file:///workspace/allowed-private/secret.json",
            "Sibling Secret",
            "Outside workspace boundary (sibling prefix trap).",
            lambda: {"secret": "forbidden"},
        )

        # roots/list returns canonicalized root with PII-scrubbed name
        rlist = disp.handle({"jsonrpc": "2.0", "id": 701, "method": "roots/list"})["result"]["roots"]
        self.assertEqual(len(rlist), 1)
        self.assertEqual(rlist[0]["uri"], "file:///workspace/allowed")
        self.assertNotIn("dev@corp.io", rlist[0]["name"])

        # resources/list hides out-of-root file:// resources while keeping allowed ones
        listed_uris = {
            r["uri"] for r in disp.handle({"jsonrpc": "2.0", "id": 702, "method": "resources/list"})["result"]["resources"]
        }
        self.assertIn("file:///workspace/allowed/config.json", listed_uris)
        self.assertNotIn("file:///workspace/allowed-private/secret.json", listed_uris)

        # resources/read allows localhost/dot canonical alias inside root, blocks sibling prefix (-32003 / -22) & traversal (-32602 / -20)
        ok_read = disp.handle(
            {
                "jsonrpc": "2.0",
                "id": 703,
                "method": "resources/read",
                "params": {"uri": "file://localhost/workspace/allowed/./sub/../config.json"},
            }
        )
        self.assertEqual(json.loads(ok_read["result"]["contents"][0]["text"])["env"], "prod")

        denied_sibling = disp.handle(
            {
                "jsonrpc": "2.0",
                "id": 704,
                "method": "resources/read",
                "params": {"uri": "file:///workspace/allowed-private/secret.json"},
            }
        )
        self.assertEqual(denied_sibling["error"]["code"], -32003)
        self.assertEqual(denied_sibling["error"]["data"]["code"], WorkspaceRootsGuard.ERR_URI_OUTSIDE_ROOTS)

        denied_trav = disp.handle(
            {
                "jsonrpc": "2.0",
                "id": 705,
                "method": "resources/subscribe",
                "params": {"uri": "file:///workspace/../../etc/passwd"},
            }
        )
        self.assertEqual(denied_trav["error"]["code"], -32602)
        self.assertEqual(denied_trav["error"]["data"]["code"], WorkspaceRootsGuard.ERR_URI_TRAVERSAL)

        # Updating roots emits notifications/roots/list_changed and unlocks newly authorized root
        disp.set_roots(
            [
                {"uri": "file:///workspace/allowed", "name": "Main"},
                {"uri": "file:///workspace/allowed-private", "name": "Unlocked"},
            ]
        )
        notifs = disp.drain_notifications()
        self.assertEqual([n["method"] for n in notifs], ["notifications/roots/list_changed"])
        unlocked = disp.handle(
            {
                "jsonrpc": "2.0",
                "id": 706,
                "method": "resources/read",
                "params": {"uri": "file:///workspace/allowed-private/secret.json"},
            }
        )
        self.assertEqual(json.loads(unlocked["result"]["contents"][0]["text"])["secret"], "forbidden")

    def test_progress_token_registry_monotonicity_duplicate_and_lifecycle_fencing(self) -> None:
        reg = ProgressTokenRegistry(max_active=2)
        # Reject bool, empty string, and negative int keys (-25)
        self.assertEqual(reg.begin(True)[0], ProgressTokenRegistry.ERR_PROGRESS_KEY)
        self.assertEqual(reg.begin("")[0], ProgressTokenRegistry.ERR_PROGRESS_KEY)
        self.assertEqual(reg.begin(-1)[0], ProgressTokenRegistry.ERR_PROGRESS_KEY)

        # Integer 7 and string "7" are isolated concurrent progressTokens
        rc_i, rep_i, _ = reg.begin(7)
        rc_s, rep_s, _ = reg.begin("7")
        self.assertEqual((rc_i, rc_s), (0, 0))
        assert rep_i is not None and rep_s is not None

        # Duplicate active token rejected with -23; 3rd concurrent stream hits max_active=2 -> -24
        self.assertEqual(reg.begin(7)[0], ProgressTokenRegistry.ERR_PROGRESS_DUPLICATE)
        self.assertEqual(reg.begin("third")[0], ProgressTokenRegistry.ERR_PROGRESS_FULL)

        # Strict monotonicity & total bounds
        self.assertEqual(rep_i.report(25, total=100, message="Phase 1 sk-1234567890abcdefghijklmnop")[0], 0)
        self.assertNotIn("sk-1234567890", rep_i.emitted_notifications[0]["params"]["message"])
        self.assertEqual(rep_i.report(25, total=100)[0], ProgressTokenRegistry.ERR_PROGRESS_NON_MONOTONIC)
        self.assertEqual(rep_i.report(20, total=100)[0], ProgressTokenRegistry.ERR_PROGRESS_NON_MONOTONIC)
        self.assertEqual(rep_i.report(105, total=100)[0], ProgressTokenRegistry.ERR_PROGRESS_EXCEEDS_TOTAL)
        self.assertEqual(rep_i.report(50, total=80)[0], ProgressTokenRegistry.ERR_PROGRESS_EXCEEDS_TOTAL)
        self.assertEqual(rep_i.report(100, total=100, message="Complete")[0], 0)

        # Post-completion updates suppressed (-26), but token 7 can be reused on a fresh request
        self.assertEqual(rep_i.complete(), 0)
        self.assertEqual(rep_i.report(100)[0], ProgressTokenRegistry.ERR_PROGRESS_INACTIVE)
        rc_reuse, rep_reuse, _ = reg.begin(7)
        self.assertEqual(rc_reuse, 0)
        assert rep_reuse is not None
        self.assertEqual(rep_reuse.abort(), 0)
        self.assertEqual(rep_reuse.report(10)[0], ProgressTokenRegistry.ERR_PROGRESS_INACTIVE)
        rep_s.complete()

    def test_mcp_streamable_http_sse_progress_frames_and_cancellation_suppression(self) -> None:
        srv = build_server("127.0.0.1", 0)
        port = srv.server_address[1]

        def custom_progressive_tool(args: dict, token: CancellationToken, prog: ProgressReporter | None) -> dict:
            if prog is not None:
                prog.report(10, 100, f"Step 1 for {args['label']}")
                prog.report(60, 100, "Step 2")
            token.raise_if_cancelled()
            if prog is not None:
                prog.report(100, 100, "Done")
            return {"ok": True}

        srv.dispatcher.register_tool(  # type: ignore[attr-defined]
            "progressive_pipeline",
            "Emits monotonic MCP notifications/progress frames.",
            input_schema={
                "type": "object",
                "properties": {"label": {"type": "string"}},
                "required": ["label"],
                "additionalProperties": False,
            },
            output_schema={
                "type": "object",
                "properties": {"ok": {"type": "boolean"}},
                "required": ["ok"],
                "additionalProperties": False,
            },
            handler=custom_progressive_tool,
            accepts_cancel_token=True,
            accepts_progress=True,
        )

        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            # 1. Built-in rag_hybrid_search over Accept: text/event-stream streams 3 monotonic progress frames + final result
            sse_req = urllib.request.Request(
                f"http://127.0.0.1:{port}/mcp",
                data=json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": 801,
                        "method": "tools/call",
                        "params": {
                            "name": "rag_hybrid_search",
                            "arguments": {"query": "Reciprocal Rank Fusion", "top_k": 2},
                            "_meta": {"progressToken": "rag-tok-01"},
                        },
                    }
                ).encode("utf-8"),
                headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
            )
            with opener.open(sse_req, timeout=5) as sresp:
                raw_sse = sresp.read().decode("utf-8")
            events = [
                json.loads(block.split("data: ", 1)[1])
                for block in raw_sse.strip().split("\n\n")
                if "data: " in block
            ]
            self.assertEqual(len(events), 4)
            prog_vals = [ev["params"]["progress"] for ev in events[:3]]
            self.assertEqual(prog_vals, [1, 2, 3])
            for ev in events[:3]:
                self.assertEqual(ev["method"], "notifications/progress")
                self.assertEqual(ev["params"]["progressToken"], "rag-tok-01")
                self.assertEqual(ev["params"]["total"], 3)
            self.assertEqual(events[3]["id"], 801)
            self.assertFalse(events[3]["result"]["structuredContent"]["cache_hit"])

            # 2. Custom tool with PII in progress message scrubs PII & records OTLP mcp.progress.updates
            srv.dispatcher.drain_notifications()  # type: ignore[attr-defined]
            cust_resp = srv.dispatcher.handle(  # type: ignore[attr-defined]
                {
                    "jsonrpc": "2.0",
                    "id": 802,
                    "method": "tools/call",
                    "params": {
                        "name": "progressive_pipeline",
                        "arguments": {"label": "owner@corp.io"},
                        "_meta": {"progressToken": 9001},
                    },
                }
            )
            self.assertTrue(cust_resp["result"]["structuredContent"]["ok"])
            queued = srv.dispatcher.drain_notifications()  # type: ignore[attr-defined]
            self.assertEqual([n["params"]["progress"] for n in queued], [10, 60, 100])
            self.assertNotIn("owner@corp.io", queued[0]["params"]["message"])

            # 3. Pre-cancelled request emits zero progress frames
            srv.dispatcher.handle(  # type: ignore[attr-defined]
                {"jsonrpc": "2.0", "method": "notifications/cancelled", "params": {"requestId": 803}}
            )
            aborted = srv.dispatcher.handle(  # type: ignore[attr-defined]
                {
                    "jsonrpc": "2.0",
                    "id": 803,
                    "method": "tools/call",
                    "params": {
                        "name": "rag_hybrid_search",
                        "arguments": {"query": "SQLite WAL", "top_k": 1},
                        "_meta": {"progressToken": "aborted-tok"},
                    },
                }
            )
            self.assertEqual(aborted["error"]["code"], -32800)
            self.assertEqual(srv.dispatcher.drain_notifications(), [])  # type: ignore[attr-defined]

            with opener.open(f"http://127.0.0.1:{port}/metrics/otlp", timeout=5) as oresp:
                otlp = json.loads(oresp.read().decode("utf-8"))
                self.assertEqual(otlp["summary"]["progress_notifications_emitted"], 6)
        finally:
            srv.shutdown()
            srv.server_close()

    def test_v16_canonical_uri_and_progress_error_code_parity_invariants(self) -> None:
        # Verify exact negative PAR_ERR_* integer codes shared across C11, Rust, WASM, and Python
        self.assertEqual(WorkspaceRootsGuard.ERR_URI_SYNTAX, -19)
        self.assertEqual(WorkspaceRootsGuard.ERR_URI_TRAVERSAL, -20)
        self.assertEqual(WorkspaceRootsGuard.ERR_URI_AUTHORITY, -21)
        self.assertEqual(WorkspaceRootsGuard.ERR_URI_OUTSIDE_ROOTS, -22)
        self.assertEqual(ProgressTokenRegistry.ERR_PROGRESS_DUPLICATE, -23)
        self.assertEqual(ProgressTokenRegistry.ERR_PROGRESS_FULL, -24)
        self.assertEqual(ProgressTokenRegistry.ERR_PROGRESS_KEY, -25)
        self.assertEqual(ProgressTokenRegistry.ERR_PROGRESS_INACTIVE, -26)
        self.assertEqual(ProgressTokenRegistry.ERR_PROGRESS_NON_MONOTONIC, -27)
        self.assertEqual(ProgressTokenRegistry.ERR_PROGRESS_EXCEEDS_TOTAL, -28)

    def test_v17_elicitation_flat_schema_guard_and_response_redaction(self) -> None:
        valid_flat = {
            "type": "object",
            "properties": {
                "approved_tier": {
                    "type": "string",
                    "enum": ["edge_fast_json", "frontier_moe_math_architect"],
                    "enumNames": ["Edge Fast JSON", "Frontier MoE Math"],
                },
                "max_vram_gib": {"type": "integer", "minimum": 1, "maximum": 24, "default": 6},
                "operator_email": {"type": "string", "format": "email"},
                "confirm_deploy": {"type": "boolean"},
                "justification": {"type": "string", "minLength": 3, "maxLength": 160},
            },
            "required": ["approved_tier", "confirm_deploy"],
            "additionalProperties": False,
        }
        rc, crc, _ = ElicitationSchemaGuard.validate_requested_schema(valid_flat)
        self.assertEqual(rc, 0)
        self.assertGreater(crc, 0)

        # Reject nested objects, arrays, $ref, combinators, and unsupported formats (-35)
        for forbidden_schema in (
            {"type": "object", "properties": {"nested": {"type": "object", "properties": {"x": {"type": "string"}}}}},
            {"type": "object", "properties": {"items_list": {"type": "array", "items": {"type": "string"}}}},
            {"type": "object", "properties": {"ref_field": {"type": "string", "$ref": "#/$defs/x"}}},
            {"type": "object", "allOf": [], "properties": {"a": {"type": "string"}}},
            {"type": "object", "properties": {"u": {"type": ["string", "null"]}}},
            {"type": "object", "properties": {"bad_fmt": {"type": "string", "format": "ipv4"}}},
            {"type": "object", "properties": {"bad_range": {"type": "integer", "minimum": 10, "maximum": 4}}},
        ):
            self.assertEqual(
                ElicitationSchemaGuard.validate_requested_schema(forbidden_schema)[0],
                ElicitationSchemaGuard.ERR_ELICITATION_SCHEMA,
            )

        # Valid accept scrubs Bearer/API keys in free-text while enforcing format="email" and closed properties
        ok_rc, ok_body, _ = ElicitationSchemaGuard.validate_response(
            "accept",
            {
                "approved_tier": "edge_fast_json",
                "max_vram_gib": 6,
                "operator_email": "lead@example.org",
                "confirm_deploy": True,
                "justification": "Approved with token sk-1234567890abcdefghijklmnop",
            },
            valid_flat,
        )
        self.assertEqual(ok_rc, 0)
        assert ok_body is not None
        self.assertEqual(ok_body["content"]["operator_email"], "lead@example.org")
        self.assertNotIn("sk-1234567890", ok_body["content"]["justification"])
        self.assertIn("[REDACTED_API_KEY]", ok_body["content"]["justification"])

        # Reject bool-as-integer, bad email format, extra property, or content on 'decline'
        self.assertEqual(
            ElicitationSchemaGuard.validate_response(
                "accept",
                {"approved_tier": "edge_fast_json", "confirm_deploy": True, "max_vram_gib": True},
                valid_flat,
            )[0],
            ElicitationSchemaGuard.ERR_ELICITATION_SCHEMA,
        )
        self.assertEqual(
            ElicitationSchemaGuard.validate_response(
                "accept",
                {"approved_tier": "edge_fast_json", "confirm_deploy": True, "operator_email": "not-an-email"},
                valid_flat,
            )[0],
            ElicitationSchemaGuard.ERR_ELICITATION_SCHEMA,
        )
        self.assertEqual(
            ElicitationSchemaGuard.validate_response("decline", {"unexpected": 1}, valid_flat)[0],
            ElicitationSchemaGuard.ERR_ELICITATION_SCHEMA,
        )
        self.assertEqual(
            ElicitationSchemaGuard.validate_response("cancel", None, valid_flat),
            (0, {"action": "cancel"}, "ok"),
        )

    def test_v17_mcp_sampling_and_elicitation_dispatcher_lifecycle_and_c11_parity(self) -> None:
        clock_ms = [1_000]
        approvals = HumanApprovalRegistry(
            max_slots=4,
            default_ttl_ms=5_000,
            ceiling_max_tokens=1_024,
            now_ms_fn=lambda: clock_ms[0],
        )
        disp = McpJsonRpcDispatcher(approvals=approvals, enforce_approval_capabilities=True)

        # 1. Missing client capabilities during initialize -> -32004 CAPABILITY_REQUIRED_CODE
        disp.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"capabilities": {}}})
        no_cap_el = disp.handle(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "elicitation/create",
                "params": {
                    "approvalId": 10,
                    "message": "Approve deployment?",
                    "requestedSchema": {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]},
                },
            }
        )
        self.assertEqual(no_cap_el["error"]["code"], -32004)

        # Re-initialize with elicitation + sampling capabilities
        disp.handle(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "initialize",
                "params": {"capabilities": {"elicitation": {}, "sampling": {}}},
            }
        )
        disp.handle(
            {"jsonrpc": "2.0", "id": 4, "method": "resources/subscribe", "params": {"uri": "mcp://governance/approvals"}}
        )

        # 2. Integer 77 (sampling) and string "77" (elicitation) isolated; duplicate pending -> -29
        samp_open = disp.handle(
            {
                "jsonrpc": "2.0",
                "id": 5,
                "method": "sampling/createMessage",
                "params": {
                    "approvalId": 77,
                    "maxTokens": 512,
                    "temperature": 0.25,
                    "includeContext": "thisServer",
                    "modelPreferences": {"costPriority": 0.4, "speedPriority": 0.7, "intelligencePriority": 0.9},
                    "messages": [{"role": "user", "content": {"type": "text", "text": "Summarize Bearer secretJwtToken99"}}],
                },
            }
        )
        self.assertEqual(samp_open["result"]["status"], "pending")
        self.assertNotIn("secretJwtToken99", samp_open["result"]["sanitizedMessages"][0]["content"]["text"])
        samp_crc = samp_open["result"]["schemaCrc32c"]

        dup_samp = disp.handle(
            {
                "jsonrpc": "2.0",
                "id": 6,
                "method": "sampling/createMessage",
                "params": {
                    "approvalId": 77,
                    "maxTokens": 512,
                    "messages": [{"role": "user", "content": {"type": "text", "text": "Dup"}}],
                },
            }
        )
        self.assertEqual(dup_samp["error"]["data"]["code"], HumanApprovalRegistry.ERR_APPROVAL_DUPLICATE)

        # Oversized completion tokens (600 > 512) -> -34, slot remains pending for valid completion (320 <= 512)
        over_tok = disp.handle(
            {
                "jsonrpc": "2.0",
                "id": 7,
                "method": "sampling/respond",
                "params": {"approvalId": 77, "action": "accept", "text": "Too long", "actualTokens": 600, "schemaCrc32c": samp_crc},
            }
        )
        self.assertEqual(over_tok["error"]["data"]["code"], HumanApprovalRegistry.ERR_SAMPLING_BUDGET)

        ok_samp = disp.handle(
            {
                "jsonrpc": "2.0",
                "id": 8,
                "method": "sampling/respond",
                "params": {"approvalId": 77, "action": "accept", "text": "Grounded summary.", "actualTokens": 320, "schemaCrc32c": samp_crc},
            }
        )
        self.assertEqual(ok_samp["result"]["actualTokens"], 320)

        # Second replay on settled slot -> -32
        replay_samp = disp.handle(
            {
                "jsonrpc": "2.0",
                "id": 9,
                "method": "sampling/respond",
                "params": {"approvalId": 77, "action": "accept", "text": "Replay", "actualTokens": 100},
            }
        )
        self.assertEqual(replay_samp["error"]["data"]["code"], HumanApprovalRegistry.ERR_APPROVAL_INACTIVE)

        # User decline on sampling -> MCP -1 error code
        disp.handle(
            {
                "jsonrpc": "2.0",
                "id": 10,
                "method": "sampling/createMessage",
                "params": {
                    "approvalId": 78,
                    "maxTokens": 256,
                    "messages": [{"role": "user", "content": {"type": "text", "text": "Decline me"}}],
                },
            }
        )
        declined_samp = disp.handle(
            {"jsonrpc": "2.0", "id": 11, "method": "sampling/respond", "params": {"approvalId": 78, "action": "decline"}}
        )
        self.assertEqual(declined_samp["error"]["code"], -1)

        # 3. Elicitation string key "77" with TTL expiry (-33) and OTLP export outputSchema check
        el_open = disp.handle(
            {
                "jsonrpc": "2.0",
                "id": 12,
                "method": "elicitation/create",
                "params": {
                    "approvalId": "77",
                    "message": "Confirm tier?",
                    "requestedSchema": {"type": "object", "properties": {"tier": {"type": "string"}}, "required": ["tier"]},
                },
            }
        )
        self.assertEqual(el_open["result"]["status"], "pending")
        clock_ms[0] = 7_500  # past 1000 + 5000 TTL
        exp_el = disp.handle(
            {
                "jsonrpc": "2.0",
                "id": 13,
                "method": "elicitation/respond",
                "params": {"approvalId": "77", "action": "accept", "content": {"tier": "edge"}},
            }
        )
        self.assertEqual(exp_el["error"]["data"]["code"], HumanApprovalRegistry.ERR_APPROVAL_EXPIRED)

        otlp_call = disp.handle(
            {"jsonrpc": "2.0", "id": 14, "method": "tools/call", "params": {"name": "otlp_export_telemetry", "arguments": {}}}
        )
        self.assertFalse(otlp_call["result"]["isError"])
        summary = otlp_call["result"]["structuredContent"]["summary"]
        self.assertEqual(summary["sampling_requests_governed"], 1)
        self.assertGreaterEqual(summary["approval_gate_rejections"], 4)
        notifs = disp.drain_notifications()
        self.assertTrue(any(n.get("params", {}).get("uri") == "mcp://governance/approvals" for n in notifs))

        # 4. Cross-language C11 ctypes parity check when sibling ring_crc.c + cc exist
        import ctypes
        import pathlib
        import shutil
        import subprocess
        import tempfile

        c_src = pathlib.Path(__file__).resolve().parents[2] / "polyglot-portfolio-2026" / "polyglot-agent-runtime-rs" / "c_src" / "ring_crc.c"
        if c_src.is_file() and shutil.which("cc"):
            with tempfile.TemporaryDirectory() as tmpd:
                so_path = pathlib.Path(tmpd) / "libring_crc_v17.so"
                subprocess.run(
                    ["cc", "-std=c11", "-O2", "-shared", "-fPIC", str(c_src), "-o", str(so_path)],
                    check=True,
                )
                lib = ctypes.CDLL(str(so_path))
                lib.par_sampling_check_budget.argtypes = [
                    ctypes.c_uint32,
                    ctypes.c_uint32,
                    ctypes.c_int32,
                    ctypes.c_int32,
                    ctypes.c_int32,
                    ctypes.c_int32,
                ]
                lib.par_sampling_check_budget.restype = ctypes.c_int32
                for req_tok, ceil_tok, temp_m, cost_m in (
                    (512, 1024, 250, 400),
                    (2048, 1024, 250, 400),
                    (512, 1024, 2500, 400),
                    (512, 1024, 250, 1200),
                ):
                    c_rc = lib.par_sampling_check_budget(req_tok, ceil_tok, temp_m, cost_m, 500, 500)
                    py_rc = SamplingGovernanceGuard.check_budget(
                        req_tok,
                        ceil_tok,
                        temperature=temp_m / 1000.0,
                        model_preferences={"costPriority": cost_m / 1000.0},
                    )[0]
                    self.assertEqual(c_rc, py_rc)

    def test_v18_content_annotations_media_and_resourcelink_provenance_c11_parity(self) -> None:
        import base64
        import ctypes
        import pathlib
        import shutil
        import subprocess
        import tempfile

        from mcp_moe_sdk import ContentAnnotationGuard

        # 1. Canonical C11/Rust/WASM ResourceLink provenance vector -> 0xF28542A7
        rc_s, seal, _, _ = ContentAnnotationGuard.compute_provenance_seal(
            "vault://documents/arch-mcp-2025",
            "application/json",
            0xE3069283,
            audience_mask=0x03,
            priority_milli=850,
        )
        self.assertEqual(rc_s, 0)
        self.assertEqual(seal, 0xF28542A7)

        # 2. Leap-year ISO-8601 UTC & audience/priority guards (-42)
        self.assertEqual(
            ContentAnnotationGuard.validate_annotations(
                {"audience": ["user", "assistant"], "priority": 0.85, "lastModified": "2024-02-29T12:30:00.125Z"}
            )[:4],
            (0, {"audience": ["user", "assistant"], "priority": 0.85, "lastModified": "2024-02-29T12:30:00.125Z"}, 3, 850),
        )
        for bad_ann in (
            {"audience": []},
            {"audience": ["system"]},
            {"audience": ["user", "user"]},
            {"priority": 1.05},
            {"priority": -0.01},
            {"priority": True},
            {"lastModified": "2026-02-29T00:00:00Z"},  # 2026 is not a leap year
            {"lastModified": "2026-11-25T10:00:00+00:00"},  # must terminate with Z
            {"unknownField": 1},
        ):
            self.assertEqual(
                ContentAnnotationGuard.validate_annotations(bad_ann)[0],
                ContentAnnotationGuard.ERR_CONTENT_ANNOTATION,
            )

        # 3. RFC 8089 file:// confinement on ResourceLink
        roots = WorkspaceRootsGuard(roots=[{"uri": "file:///workspace/docs", "name": "Docs"}])
        ok_rl, link_ok, _ = ContentAnnotationGuard.build_resource_link(
            "file://localhost/workspace/docs/./guide.md",
            "Guide owner@corp.io",
            mime_type="text/markdown",
            content_text="# Guide",
            annotations={"audience": ["user"], "priority": 0.9, "lastModified": "2026-11-25T00:00:00Z"},
            roots_guard=roots,
        )
        self.assertEqual(ok_rl, 0)
        assert link_ok is not None
        self.assertEqual(link_ok["uri"], "file:///workspace/docs/guide.md")
        self.assertNotIn("owner@corp.io", link_ok["name"])
        self.assertEqual(
            ContentAnnotationGuard.build_resource_link(
                "file:///workspace/docs-secrets/key.pem",
                "Secret",
                roots_guard=roots,
            )[0],
            WorkspaceRootsGuard.ERR_URI_OUTSIDE_ROOTS,
        )

        # 4. Audio & Image media block Base64 & MIME validation
        tiny_wav = base64.b64encode(b"RIFF\x24\x00\x00\x00WAVEfmt ").decode("ascii")
        m_rc, m_blk, _ = ContentAnnotationGuard.validate_media_block(
            {
                "type": "audio",
                "mimeType": "audio/wav",
                "data": tiny_wav,
                "annotations": {"audience": ["user"], "priority": 0.5},
            }
        )
        self.assertEqual(m_rc, 0)
        assert m_blk is not None
        self.assertEqual(m_blk["_meta"]["byteLength"], 16)
        self.assertEqual(
            ContentAnnotationGuard.validate_media_block({"type": "audio", "mimeType": "audio/x-ms-wma", "data": tiny_wav})[0],
            ContentAnnotationGuard.ERR_CONTENT_ANNOTATION,
        )
        self.assertEqual(
            ContentAnnotationGuard.validate_media_block({"type": "image", "mimeType": "image/png", "data": "not!base64"})[0],
            ContentAnnotationGuard.ERR_CONTENT_ANNOTATION,
        )

        # 5. Built-in rag_hybrid_search emits CRC32C-sealed resource_link blocks when include_resource_links=True
        vault = HybridKnowledgeVault(":memory:")
        vault.upsert("doc-rlink-1", "MCP 2025-11-25 ResourceLink Spec", "ResourceLink carries URI and annotations.", ["mcp"])
        disp = McpJsonRpcDispatcher(vault=vault)
        call_res = disp.handle(
            {
                "jsonrpc": "2.0",
                "id": 910,
                "method": "tools/call",
                "params": {
                    "name": "rag_hybrid_search",
                    "arguments": {"query": "ResourceLink annotations", "top_k": 1, "include_resource_links": True},
                },
            }
        )["result"]
        self.assertEqual(len(call_res["content"]), 2)
        rlink = call_res["content"][1]
        self.assertEqual(rlink["type"], "resource_link")
        self.assertEqual(rlink["uri"], "vault://documents/doc-rlink-1")
        self.assertTrue(rlink["_meta"]["provenanceCrc32c"].startswith("0x"))
        self.assertEqual(disp.telemetry.export_otlp_json()["summary"]["resource_links_emitted"], 1)

        # 6. Direct C11 ctypes parity check on par_seal_resource_link & par_validate_content_annotations
        c_src = pathlib.Path(__file__).resolve().parents[2] / "polyglot-portfolio-2026" / "polyglot-agent-runtime-rs" / "c_src" / "ring_crc.c"
        if c_src.is_file() and shutil.which("cc"):
            with tempfile.TemporaryDirectory() as tmpd:
                so_path = pathlib.Path(tmpd) / "libring_crc_v18.so"
                subprocess.run(
                    ["cc", "-std=c11", "-O2", "-shared", "-fPIC", str(c_src), "-o", str(so_path)],
                    check=True,
                )
                lib = ctypes.CDLL(str(so_path))
                lib.par_seal_resource_link.argtypes = [
                    ctypes.c_char_p,
                    ctypes.c_size_t,
                    ctypes.c_char_p,
                    ctypes.c_size_t,
                    ctypes.c_uint32,
                    ctypes.c_uint8,
                    ctypes.c_int32,
                    ctypes.POINTER(ctypes.c_uint32),
                ]
                lib.par_seal_resource_link.restype = ctypes.c_int32
                c_seal = ctypes.c_uint32(0)
                u_b = rlink["uri"].encode("ascii")
                m_b = rlink["mimeType"].encode("ascii")
                c_crc = int(rlink["_meta"]["contentCrc32c"], 16)
                c_rc = lib.par_seal_resource_link(u_b, len(u_b), m_b, len(m_b), c_crc, 0x03, 1000, ctypes.byref(c_seal))
                self.assertEqual(c_rc, 0)
                self.assertEqual(f"0x{c_seal.value:08X}", rlink["_meta"]["provenanceCrc32c"])

    def test_v18_streamable_http_mcp_session_id_and_last_event_id_sse_replay(self) -> None:
        import urllib.error

        from mcp_moe_sdk import StreamableSessionRegistry

        srv = build_server("127.0.0.1", 0, require_session=True, session_ring_capacity=3)
        port = srv.server_address[1]
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            # 1. Missing Mcp-Session-Id when require_session=True -> 400 (-36)
            no_sid_req = urllib.request.Request(
                f"http://127.0.0.1:{port}/mcp",
                data=json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping"}).encode("utf-8"),
                headers={"Content-Type": "application/json", "X-Client-Id": "tenant-s1"},
            )
            with self.assertRaises(urllib.error.HTTPError) as ctx_nosid:
                opener.open(no_sid_req, timeout=5)
            self.assertEqual(ctx_nosid.exception.code, 400)
            ctx_nosid.exception.close()

            # 2. initialize mints Mcp-Session-Id bound to tenant-s1
            init_req = urllib.request.Request(
                f"http://127.0.0.1:{port}/mcp",
                data=json.dumps(
                    {"jsonrpc": "2.0", "id": 2, "method": "initialize", "params": {"protocolVersion": "2025-11-25"}}
                ).encode("utf-8"),
                headers={"Content-Type": "application/json", "X-Client-Id": "tenant-s1"},
            )
            with opener.open(init_req, timeout=5) as iresp:
                sid = iresp.headers.get("Mcp-Session-Id")
                self.assertIsNotNone(sid)
                assert sid is not None

            # 3. Hijack attempt with another X-Client-Id -> 403 (-39 ERR_SESSION_BINDING)
            hijack_req = urllib.request.Request(
                f"http://127.0.0.1:{port}/mcp",
                data=json.dumps({"jsonrpc": "2.0", "id": 3, "method": "ping"}).encode("utf-8"),
                headers={"Content-Type": "application/json", "X-Client-Id": "attacker", "Mcp-Session-Id": sid},
            )
            with self.assertRaises(urllib.error.HTTPError) as ctx_hijack:
                opener.open(hijack_req, timeout=5)
            self.assertEqual(ctx_hijack.exception.code, 403)
            err_hijack = json.loads(ctx_hijack.exception.read().decode("utf-8"))
            self.assertEqual(err_hijack["error"]["data"]["code"], StreamableSessionRegistry.ERR_SESSION_BINDING)
            ctx_hijack.exception.close()

            # 4. POST /mcp over SSE emits 3 id:-stamped frames (2 progress + 1 result) within ring_capacity=3
            sse_req = urllib.request.Request(
                f"http://127.0.0.1:{port}/mcp",
                data=json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": 4,
                        "method": "tools/call",
                        "params": {
                            "name": "moe_route_query",
                            "arguments": {"prompt": "Verify MoE invariant"},
                            "_meta": {"progressToken": "sess-prog-1"},
                        },
                    }
                ).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Accept": "text/event-stream",
                    "X-Client-Id": "tenant-s1",
                    "Mcp-Session-Id": sid,
                },
            )
            with opener.open(sse_req, timeout=5) as sresp:
                sse_blocks = [b for b in sresp.read().decode("utf-8").strip().split("\n\n") if b.strip()]
            self.assertEqual(len(sse_blocks), 3)
            event_ids = [b.splitlines()[0].split("id: ", 1)[1] for b in sse_blocks]
            self.assertTrue(event_ids[0].startswith("1:"))

            # 5. Reconnect GET /mcp with Last-Event-ID = event_ids[0] -> replays exact frames 2 and 3
            replay_req = urllib.request.Request(
                f"http://127.0.0.1:{port}/mcp",
                method="GET",
                headers={"X-Client-Id": "tenant-s1", "Mcp-Session-Id": sid, "Last-Event-ID": event_ids[0]},
            )
            with opener.open(replay_req, timeout=5) as rresp:
                self.assertEqual(rresp.status, 200)
                self.assertEqual(rresp.headers.get("X-Mcp-Replayed-Count"), "2")
                replayed_blocks = [b for b in rresp.read().decode("utf-8").strip().split("\n\n") if "data: " in b]
            self.assertEqual(len(replayed_blocks), 2)
            self.assertIn(f"id: {event_ids[1]}", replayed_blocks[0])
            self.assertIn(f"id: {event_ids[2]}", replayed_blocks[1])

            # 6. Push 2 more SSE frames so seq reaches 5 (minRetainedSeq=3 with ring_capacity=3), then Last-Event-ID=1:... -> 409 (-41)
            for extra_id in (5, 6):
                extra_req = urllib.request.Request(
                    f"http://127.0.0.1:{port}/mcp",
                    data=json.dumps({"jsonrpc": "2.0", "id": extra_id, "method": "ping"}).encode("utf-8"),
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "text/event-stream",
                        "X-Client-Id": "tenant-s1",
                        "Mcp-Session-Id": sid,
                    },
                )
                with opener.open(extra_req, timeout=5) as eresp:
                    eresp.read()

            gap_req = urllib.request.Request(
                f"http://127.0.0.1:{port}/mcp",
                method="GET",
                headers={"X-Client-Id": "tenant-s1", "Mcp-Session-Id": sid, "Last-Event-ID": event_ids[0]},
            )
            with self.assertRaises(urllib.error.HTTPError) as ctx_gap:
                opener.open(gap_req, timeout=5)
            self.assertEqual(ctx_gap.exception.code, 409)
            err_gap = json.loads(ctx_gap.exception.read().decode("utf-8"))
            self.assertEqual(err_gap["error"]["data"]["code"], StreamableSessionRegistry.ERR_REPLAY_GAP)
            ctx_gap.exception.close()

            # 7. DELETE /mcp terminates session (204 No Content), subsequent POST returns 404 (-37)
            del_req = urllib.request.Request(
                f"http://127.0.0.1:{port}/mcp",
                method="DELETE",
                headers={"X-Client-Id": "tenant-s1", "Mcp-Session-Id": sid},
            )
            with opener.open(del_req, timeout=5) as dresp:
                self.assertEqual(dresp.status, 204)

            dead_req = urllib.request.Request(
                f"http://127.0.0.1:{port}/mcp",
                data=json.dumps({"jsonrpc": "2.0", "id": 7, "method": "ping"}).encode("utf-8"),
                headers={"Content-Type": "application/json", "X-Client-Id": "tenant-s1", "Mcp-Session-Id": sid},
            )
            with self.assertRaises(urllib.error.HTTPError) as ctx_dead:
                opener.open(dead_req, timeout=5)
            self.assertEqual(ctx_dead.exception.code, 404)
            ctx_dead.exception.close()

            with opener.open(f"http://127.0.0.1:{port}/health", timeout=5) as hresp:
                h = json.loads(hresp.read().decode("utf-8"))
                self.assertEqual(h["sessions"]["active_sessions"], 0)
                self.assertEqual(h["sessions"]["sessions_terminated"], 1)
                self.assertEqual(h["telemetry"]["sessions_opened"], 1)
                self.assertEqual(h["telemetry"]["sse_events_replayed"], 2)
        finally:
            srv.shutdown()
            srv.server_close()

    def test_v19_oauth21_pkce_s256_resource_uri_and_c11_ctypes_parity(self) -> None:
        import ctypes
        import shutil
        import subprocess
        import tempfile
        from pathlib import Path

        # 1. RFC 7636 Appendix B canonical S256 vector & plain-method rejection (-43 / -44)
        rfc_verifier = "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"
        rfc_challenge = "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"
        rc_pkce, ch_out, _ = OAuthProtectedResourceGuard.compute_pkce_s256(rfc_verifier)
        self.assertEqual((rc_pkce, ch_out), (0, rfc_challenge))
        self.assertEqual(OAuthProtectedResourceGuard.verify_pkce_s256(rfc_verifier, rfc_challenge)[0], 0)
        self.assertEqual(
            OAuthProtectedResourceGuard.verify_pkce_s256(rfc_verifier, rfc_challenge, method="plain")[0],
            OAuthProtectedResourceGuard.ERR_PKCE_CHALLENGE,
        )
        self.assertEqual(
            OAuthProtectedResourceGuard.compute_pkce_s256("short-verifier")[0],
            OAuthProtectedResourceGuard.ERR_PKCE_VERIFIER,
        )
        # Non-canonical 43rd sextet low-2-bit check -> -44
        self.assertEqual(
            OAuthProtectedResourceGuard.verify_pkce_s256(rfc_verifier, rfc_challenge[:-1] + "N")[0],
            OAuthProtectedResourceGuard.ERR_PKCE_CHALLENGE,
        )

        # 2. RFC 9728 / RFC 8707 canonical resource URI & C11 audience CRC32C vector (0xC4EC1DE9)
        rc_u, canon_u, aud_crc, _ = OAuthProtectedResourceGuard.normalize_resource_uri(
            "HTTPS://MCP.Example.COM:443/v1//mcp/"
        )
        self.assertEqual((rc_u, canon_u, aud_crc), (0, "https://mcp.example.com/v1/mcp", 0xC4EC1DE9))
        self.assertEqual(
            OAuthProtectedResourceGuard.normalize_resource_uri("http://127.0.0.1:80/mcp/")[1],
            "http://127.0.0.1/mcp",
        )
        for bad_uri in (
            "http://mcp.example.com/v1/mcp",
            "https://mcp.example.com/v1/mcp#frag",
            "https://mcp.example.com/v1/mcp?x=1",
            "https://user@mcp.example.com/v1/mcp",
            "https://mcp.example.com/v1/../admin",
            "https://mcp.example.com/v1/%2fadmin",
        ):
            self.assertEqual(
                OAuthProtectedResourceGuard.normalize_resource_uri(bad_uri)[0],
                OAuthProtectedResourceGuard.ERR_OAUTH_RESOURCE_URI,
            )

        # 3. Token registration (zero plaintext storage), TTL expiry, revocation, audience & scope checks
        clock_ms = [1_000]
        guard = OAuthProtectedResourceGuard(
            "https://mcp.example.com/v1/mcp",
            authorization_servers=["https://auth.example.com/"],
            max_slots=2,
            default_ttl_ms=5_000,
            now_ms_fn=lambda: clock_ms[0],
        )
        raw_secret = "mcp_live_tok_9876543210abcdef=="
        reg_rc, reg_meta, _ = guard.register_token(
            raw_secret,
            scopes=["mcp:read", "mcp:tools"],
            client_id="tenant-o1",
        )
        self.assertEqual(reg_rc, 0)
        assert reg_meta is not None
        self.assertEqual(reg_meta["audienceCrc32c"], 0xC4EC1DE9)
        # Ensure raw plaintext token is NEVER stored anywhere in guard._slots
        self.assertNotIn(raw_secret, repr(guard._slots))

        # Valid introspection -> 0; wrong audience -> -48; missing scope -> -49; wrong client -> -47
        self.assertEqual(
            guard.introspect_token(raw_secret, required_scopes="mcp:read mcp:tools", client_id="tenant-o1")[0],
            0,
        )
        self.assertEqual(
            guard.introspect_token(
                raw_secret,
                expected_audience_uri="https://evil.example.org/v1/mcp",
                required_scopes="mcp:read",
                client_id="tenant-o1",
            )[0],
            OAuthProtectedResourceGuard.ERR_TOKEN_AUDIENCE,
        )
        self.assertEqual(
            guard.introspect_token(raw_secret, required_scopes="mcp:sampling", client_id="tenant-o1")[0],
            OAuthProtectedResourceGuard.ERR_TOKEN_SCOPE,
        )
        self.assertEqual(
            guard.introspect_token(raw_secret, required_scopes="mcp:read", client_id="attacker")[0],
            OAuthProtectedResourceGuard.ERR_TOKEN_INACTIVE,
        )

        # Advance past 5_000 ms TTL -> -47 ERR_TOKEN_INACTIVE
        clock_ms[0] = 6_001
        self.assertEqual(
            guard.introspect_token(raw_secret, required_scopes="mcp:read", client_id="tenant-o1")[0],
            OAuthProtectedResourceGuard.ERR_TOKEN_INACTIVE,
        )

        # 4. Live C11 FFI ctypes parity check when sibling C11 kernel source and cc are present
        c_src = (
            Path(__file__).resolve().parents[2]
            / ".."
            / "polyglot-portfolio-2026"
            / "polyglot-agent-runtime-rs"
            / "c_src"
            / "ring_crc.c"
        ).resolve()
        cc_bin = shutil.which("cc")
        if c_src.is_file() and cc_bin:
            with tempfile.TemporaryDirectory() as tmpdir:
                so_path = Path(tmpdir) / "libring_crc.so"
                subprocess.run(
                    [cc_bin, "-std=c11", "-O2", "-shared", "-fPIC", str(c_src), "-o", str(so_path)],
                    check=True,
                )
                lib = ctypes.CDLL(str(so_path))
                out_b64 = ctypes.create_string_buffer(64)
                v_bytes = rfc_verifier.encode("ascii")
                c_rc = lib.par_pkce_compute_s256(v_bytes, len(v_bytes), out_b64, 64)
                self.assertEqual(c_rc, 0)
                self.assertEqual(out_b64.raw[:43].decode("ascii"), ch_out)

                uri_in = b"HTTPS://MCP.Example.COM:443/v1//mcp/"
                canon_buf = ctypes.create_string_buffer(256)
                out_len = ctypes.c_size_t(0)
                out_crc = ctypes.c_uint32(0)
                u_rc = lib.par_oauth_normalize_resource_uri(
                    uri_in,
                    len(uri_in),
                    canon_buf,
                    256,
                    ctypes.byref(out_len),
                    ctypes.byref(out_crc),
                )
                self.assertEqual(u_rc, 0)
                self.assertEqual(canon_buf.raw[: out_len.value].decode("ascii"), canon_u)
                self.assertEqual(out_crc.value, aud_crc)

    def test_v19_http_well_known_protected_resource_bearer_audience_and_scope_step_up(self) -> None:
        srv = build_server(
            "127.0.0.1",
            0,
            require_oauth=True,
            resource_uri="https://mcp.example.com/mcp",
            authorization_servers=["https://auth.example.com"],
        )
        port = srv.server_address[1]
        oauth: OAuthProtectedResourceGuard = srv.dispatcher.oauth  # type: ignore[attr-defined]
        read_tok = "tok_read_only_1234567890abcdef"
        tools_tok = "tok_read_tools_1234567890abcdef"
        wrong_aud_tok = "tok_other_server_1234567890abcdef"
        oauth.register_token(read_tok, scopes=["mcp:read"], client_id="oauth-client-1")
        oauth.register_token(tools_tok, scopes=["mcp:read", "mcp:tools"], client_id="oauth-client-1")
        oauth.register_token(
            wrong_aud_tok,
            audience_uri="https://other-mcp.example.org/mcp",
            scopes=["mcp:read", "mcp:tools"],
            client_id="oauth-client-1",
        )

        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            # 1. RFC 9728 well-known Protected Resource Metadata discovery (both root and /mcp suffix)
            for wk in ("/.well-known/oauth-protected-resource", "/.well-known/oauth-protected-resource/mcp"):
                with opener.open(f"http://127.0.0.1:{port}{wk}", timeout=5) as wresp:
                    self.assertEqual(wresp.status, 200)
                    meta = json.loads(wresp.read().decode("utf-8"))
                    self.assertEqual(meta["resource"], "https://mcp.example.com/mcp")
                    self.assertEqual(meta["authorization_servers"], ["https://auth.example.com"])
                    self.assertEqual(meta["bearer_methods_supported"], ["header"])
                    self.assertEqual(meta["code_challenge_methods_supported"], ["S256"])

            # 2. Query-string ?access_token= leak attempt -> 400 (-46)
            leak_req = urllib.request.Request(
                f"http://127.0.0.1:{port}/mcp?access_token={tools_tok}",
                data=json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping"}).encode("utf-8"),
                headers={"Content-Type": "application/json", "X-Client-Id": "oauth-client-1"},
            )
            with self.assertRaises(urllib.error.HTTPError) as ctx_leak:
                opener.open(leak_req, timeout=5)
            self.assertEqual(ctx_leak.exception.code, 400)
            self.assertIn('error="invalid_request"', ctx_leak.exception.headers.get("WWW-Authenticate", ""))
            ctx_leak.exception.close()

            # 3. Missing Authorization header -> 401 with WWW-Authenticate resource_metadata (no error= per RFC 6750 §3.1)
            unauth_req = urllib.request.Request(
                f"http://127.0.0.1:{port}/mcp",
                data=json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}).encode("utf-8"),
                headers={"Content-Type": "application/json", "X-Client-Id": "oauth-client-1"},
            )
            with self.assertRaises(urllib.error.HTTPError) as ctx_unauth:
                opener.open(unauth_req, timeout=5)
            self.assertEqual(ctx_unauth.exception.code, 401)
            www_unauth = ctx_unauth.exception.headers.get("WWW-Authenticate", "")
            self.assertIn(
                'resource_metadata="https://mcp.example.com/.well-known/oauth-protected-resource/mcp"',
                www_unauth,
            )
            self.assertNotIn("error=", www_unauth)
            ctx_unauth.exception.close()

            # 4. Confused-deputy replay of token minted for another MCP server -> 401 (-48 ERR_TOKEN_AUDIENCE)
            aud_req = urllib.request.Request(
                f"http://127.0.0.1:{port}/mcp",
                data=json.dumps({"jsonrpc": "2.0", "id": 3, "method": "tools/list"}).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "X-Client-Id": "oauth-client-1",
                    "Authorization": f"Bearer {wrong_aud_tok}",
                },
            )
            with self.assertRaises(urllib.error.HTTPError) as ctx_aud:
                opener.open(aud_req, timeout=5)
            self.assertEqual(ctx_aud.exception.code, 401)
            aud_err = json.loads(ctx_aud.exception.read().decode("utf-8"))
            self.assertEqual(aud_err["error"]["data"]["code"], OAuthProtectedResourceGuard.ERR_TOKEN_AUDIENCE)
            ctx_aud.exception.close()

            # 5. mcp:read token succeeds on tools/list (200), then receives 403 insufficient_scope (-49) on tools/call
            list_req = urllib.request.Request(
                f"http://127.0.0.1:{port}/mcp",
                data=json.dumps({"jsonrpc": "2.0", "id": 4, "method": "tools/list"}).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "X-Client-Id": "oauth-client-1",
                    "Authorization": f"Bearer {read_tok}",
                },
            )
            with opener.open(list_req, timeout=5) as lresp:
                self.assertEqual(lresp.status, 200)

            stepup_req = urllib.request.Request(
                f"http://127.0.0.1:{port}/mcp",
                data=json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": 5,
                        "method": "tools/call",
                        "params": {"name": "moe_route_query", "arguments": {"prompt": "Return JSON ping"}},
                    }
                ).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "X-Client-Id": "oauth-client-1",
                    "Authorization": f"Bearer {read_tok}",
                },
            )
            with self.assertRaises(urllib.error.HTTPError) as ctx_scope:
                opener.open(stepup_req, timeout=5)
            self.assertEqual(ctx_scope.exception.code, 403)
            www_scope = ctx_scope.exception.headers.get("WWW-Authenticate", "")
            self.assertIn('error="insufficient_scope"', www_scope)
            self.assertIn('scope="mcp:read mcp:tools"', www_scope)
            scope_err = json.loads(ctx_scope.exception.read().decode("utf-8"))
            self.assertEqual(scope_err["error"]["data"]["code"], OAuthProtectedResourceGuard.ERR_TOKEN_SCOPE)
            ctx_scope.exception.close()

            # 6. Step-up to mcp:read + mcp:tools token -> tools/call & resources/read mcp://governance/oauth succeed (200)
            call_ok_req = urllib.request.Request(
                f"http://127.0.0.1:{port}/mcp",
                data=json.dumps(
                    [
                        {
                            "jsonrpc": "2.0",
                            "id": 6,
                            "method": "tools/call",
                            "params": {"name": "moe_route_query", "arguments": {"prompt": "Return JSON ping"}},
                        },
                        {
                            "jsonrpc": "2.0",
                            "id": 7,
                            "method": "resources/read",
                            "params": {"uri": "mcp://governance/oauth"},
                        },
                    ]
                ).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "X-Client-Id": "oauth-client-1",
                    "Authorization": f"Bearer {tools_tok}",
                },
            )
            with opener.open(call_ok_req, timeout=5) as ok_resp:
                self.assertEqual(ok_resp.status, 200)
                batch_out = json.loads(ok_resp.read().decode("utf-8"))
                self.assertEqual(len(batch_out), 2)
                oauth_res_snap = json.loads(batch_out[1]["result"]["contents"][0]["text"])
                self.assertEqual(oauth_res_snap["audience_crc32c"], crc32c(b"https://mcp.example.com/mcp"))

            # 7. Revoke tools_tok -> subsequent call fails with 401 (-47 ERR_TOKEN_INACTIVE)
            self.assertEqual(oauth.revoke_token(tools_tok)[0], 0)
            revoked_req = urllib.request.Request(
                f"http://127.0.0.1:{port}/mcp",
                data=json.dumps({"jsonrpc": "2.0", "id": 8, "method": "ping"}).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "X-Client-Id": "oauth-client-1",
                    "Authorization": f"Bearer {tools_tok}",
                },
            )
            with self.assertRaises(urllib.error.HTTPError) as ctx_rev:
                opener.open(revoked_req, timeout=5)
            self.assertEqual(ctx_rev.exception.code, 401)
            ctx_rev.exception.close()

            with opener.open(f"http://127.0.0.1:{port}/health", timeout=5) as hresp:
                h = json.loads(hresp.read().decode("utf-8"))
                self.assertEqual(h["version"], "1.9.0")
                self.assertEqual(h["telemetry"]["oauth_tokens_verified"], 2)
                self.assertEqual(h["telemetry"]["oauth_audience_rejections"], 1)
                self.assertEqual(h["telemetry"]["oauth_scope_rejections"], 1)
                self.assertGreaterEqual(h["telemetry"]["oauth_challenges_issued"], 5)
        finally:
            srv.shutdown()
            srv.server_close()


if __name__ == "__main__":
    unittest.main()
