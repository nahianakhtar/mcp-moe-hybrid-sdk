"""Deterministic offline unit, routing, negative-input, and edge-case tests."""

from __future__ import annotations

import base64
import json
import math
import struct
import threading
import unittest
import urllib.error
import urllib.request
from typing import Any

from mcp_moe_sdk import (
    CancellationToken,
    HybridKnowledgeVault,
    JsonSchema2020Validator,
    McpCursorCodec,
    McpJsonRpcDispatcher,
    ProgressTokenRegistry,
    RequestCancellationRegistry,
    RequestCancelledError,
    SamplingGovernanceGuard,
    SlidingWindowTokenRateLimiter,
    SparseTop2MoeRouter,
    StreamableSessionRegistry,
)
from mcp_moe_sdk.server import build_server


class TestRoutingErrorsAndEdgeCases(unittest.TestCase):
    def test_moe_router_all_tiers_edge_inputs_sla_and_cancellation(self) -> None:
        router = SparseTop2MoeRouter()
        self.assertEqual(router.stats(), {"calls": 0, "p50_ms": 0.0, "p95_ms": 0.0})

        cases = [
            ("Return strict JSON health status ping flag", "edge_fast_json", 6.0, 200.0),
            ("Retrieve and summarize SQLite FTS5 RAG document policy", "rag_grounded_synthesis", 6.0, 1500.0),
            ("Fix Python pytest SQL Docker API bug", "code_and_systems_verifier", 6.0, 1500.0),
            (
                "Prove Black-Scholes convexity theorem and quantum MoE invariant",
                "frontier_moe_math_architect",
                6.0,
                1500.0,
            ),
        ]
        for prompt, expected_tier, max_vram, sla_ms in cases:
            res = router.route(prompt, max_vram_gib=max_vram, latency_sla_ms=sla_ms)
            self.assertEqual(res["selected_tier"], expected_tier)
            self.assertEqual(len(res["top2_sparse_gates"]), 2)
            self.assertTrue(math.isfinite(res["confidence"]))
            self.assertGreater(res["confidence"], 0.0)
            self.assertLessEqual(res["confidence"], 1.0)
            gate_sum = sum(g["gate_weight"] for g in res["top2_sparse_gates"])
            self.assertLessEqual(abs(gate_sum - 1.0), 0.002)

        calls_before_edge = router.stats()["calls"]
        for edge_prompt in ("", "   ", "!!!???"):
            res = router.route(edge_prompt, max_vram_gib=0.5, latency_sla_ms=100.0)
            self.assertEqual(len(res["top2_sparse_gates"]), 2)
            gate_sum = sum(g["gate_weight"] for g in res["top2_sparse_gates"])
            self.assertLessEqual(abs(gate_sum - 1.0), 0.002)
            self.assertTrue(math.isfinite(res["confidence"]))
            self.assertGreater(res["confidence"], 0.0)
            self.assertLessEqual(res["confidence"], 1.0)

        secret_res = router.route(
            "Email sec@example.com with sk-1234567890abcdefghijklmnop and Bearer tokenValue999"
        )
        self.assertEqual(secret_res["redactions_applied"], 3)

        calls_before_cancel = router.stats()["calls"]
        self.assertEqual(calls_before_cancel, calls_before_edge + 4)
        tok = CancellationToken("route-cancel")
        tok.cancel("aborted")
        with self.assertRaises(RequestCancelledError):
            router.route("Return JSON status", cancel_token=tok)
        self.assertEqual(router.stats()["calls"], calls_before_cancel)

    def test_hybrid_vault_empty_punctuation_fts_injection_sentinel_and_upsert_edges(self) -> None:
        vault = HybridKnowledgeVault(":memory:")
        self.assertEqual(vault.count(), 0)
        self.assertIsNone(vault.get_document("missing"))
        self.assertEqual(vault.hybrid_search("anything"), [])

        vault.upsert(
            "edge-1",
            "Sentinel\x01Title\x02 <b onclick=evil()>",
            "Body\x01with\x02 sentinels <script>xss()</script> and alpha_unique_v1.",
            ["alpha"],
        )
        self.assertEqual(vault.count(), 1)
        stored = vault.get_document("edge-1")
        assert stored is not None
        self.assertNotIn("\x01", stored["title"])
        self.assertNotIn("\x02", stored["title"])
        self.assertNotIn("\x01", stored["body"])
        self.assertNotIn("\x02", stored["body"])

        hits_xss = vault.hybrid_search("alpha_unique_v1", top_k=2)
        self.assertEqual(len(hits_xss), 1)
        self.assertNotIn("<script>", hits_xss[0].highlighted_snippet)
        self.assertIn("&lt;script&gt;", hits_xss[0].highlighted_snippet)

        # Re-upserting same doc_id overwrites cleanly without duplicating or keeping stale lexical terms
        vault.upsert("edge-1", "Updated Title", "Replaced with beta_unique_v2 only.", ["beta"])
        self.assertEqual(vault.count(), 1)
        stale_hits = vault.hybrid_search("alpha_unique_v1", top_k=2)
        self.assertEqual(len(stale_hits), 1)
        self.assertIsNone(stale_hits[0].lexical_rank)

        vault.upsert("edge-2", "Second Doc", "SQLite FTS5 vector index.", ["sqlite"])
        for edge_query in (
            "",
            "   ",
            "--- ???",
            'OR AND NOT (sqlite*) "unterminated title:foo',
        ):
            hits = vault.hybrid_search(edge_query, top_k=2)
            self.assertEqual(len(hits), 2)

        # top_k <= 0 clamps to 1 result when vault is non-empty
        self.assertEqual(len(vault.hybrid_search("SQLite", top_k=0)), 1)
        self.assertEqual(len(vault.hybrid_search("SQLite", top_k=-3)), 1)

        tok = CancellationToken("vault-cancel")
        tok.cancel("stop")
        with self.assertRaises(RequestCancelledError):
            vault.hybrid_search("SQLite", cancel_token=tok)

    def test_mcp_dispatcher_jsonrpc_framing_and_routing_negative_inputs(self) -> None:
        disp = McpJsonRpcDispatcher(max_batch_size=3)

        for bad_req in (None, "bad", 123, []):
            resp = disp.handle(bad_req)  # type: ignore[arg-type]
            self.assertIsNone(resp["id"])
            self.assertEqual(resp["error"]["code"], -32600)

        for bad_frame in (
            {"jsonrpc": "1.0", "id": 11, "method": "ping"},
            {"id": "str-12", "method": "ping"},
            {"jsonrpc": "2.0", "id": 13, "method": ""},
            {"jsonrpc": "2.0", "id": 14, "method": 42},
        ):
            resp = disp.handle(bad_frame)  # type: ignore[arg-type]
            self.assertEqual(resp["id"], bad_frame["id"])
            self.assertEqual(resp["error"]["code"], -32600)

        unk_method = disp.handle({"jsonrpc": "2.0", "id": 15, "method": "nonexistent/method"})
        self.assertEqual(unk_method["error"]["code"], -32601)

        unk_tool = disp.handle(
            {"jsonrpc": "2.0", "id": 16, "method": "tools/call", "params": {"name": "no_such_tool"}}
        )
        self.assertEqual(unk_tool["error"]["code"], -32601)
        spans = disp.telemetry.export_otlp_json()["resourceSpans"][0]["scopeSpans"][0]["spans"]
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0]["status"]["code"], 2)

        for bad_args_params in (
            {"name": "moe_route_query", "arguments": "oops"},
            {"name": "moe_route_query", "arguments": [1]},
            {"name": "moe_route_query", "arguments": {}},
            {"name": "moe_route_query", "arguments": {"prompt": "hi", "unknownExtra": 1}},
        ):
            bad_call = disp.handle(
                {"jsonrpc": "2.0", "id": 17, "method": "tools/call", "params": bad_args_params}
            )
            self.assertEqual(bad_call["error"]["code"], -32602)
            self.assertEqual(bad_call["error"]["data"]["phase"], "inputSchema")

        for primitive in (123, "hi", None):
            st, body, tp = disp.handle_http_payload(primitive)
            self.assertEqual(st, 400)
            assert isinstance(body, dict)
            self.assertEqual(body["error"]["code"], -32600)
            self.assertTrue(tp.startswith("00-"))

        empty_batch = disp.handle_batch([])
        self.assertEqual(len(empty_batch), 1)
        self.assertEqual(empty_batch[0]["error"]["code"], -32600)

        over_batch = disp.handle_batch([{"jsonrpc": "2.0", "id": i, "method": "ping"} for i in range(4)])
        self.assertEqual(len(over_batch), 1)
        self.assertEqual(over_batch[0]["error"]["code"], -32600)

        preserved = disp.handle_batch(
            [
                {"jsonrpc": "2.0", "id": 91, "method": "ping"},
                {"jsonrpc": "2.0", "method": "notifications/cancelled", "params": {"requestId": 91}},
            ],
            elide_cancelled=False,
        )
        self.assertEqual(len(preserved), 1)
        self.assertEqual(preserved[0]["id"], 91)
        self.assertEqual(preserved[0]["error"]["code"], -32800)

        for bad_ptok in (True, "", -5):
            bad_prog = disp.handle(
                {
                    "jsonrpc": "2.0",
                    "id": 92,
                    "method": "tools/call",
                    "params": {
                        "name": "moe_route_query",
                        "arguments": {"prompt": "Return JSON status"},
                        "_meta": {"progressToken": bad_ptok},
                    },
                }
            )
            self.assertEqual(bad_prog["error"]["code"], -32602)
            self.assertEqual(bad_prog["error"]["data"]["code"], ProgressTokenRegistry.ERR_PROGRESS_KEY)

    def test_mcp_resources_prompts_completions_logging_negative_and_edge_cases(self) -> None:
        disp = McpJsonRpcDispatcher()

        for missing_uri_params in ({}, {"uri": ""}, {"uri": "   "}):
            err = disp.handle({"jsonrpc": "2.0", "id": 1, "method": "resources/read", "params": missing_uri_params})
            self.assertEqual(err["error"]["code"], -32602)

        not_found = disp.handle(
            {"jsonrpc": "2.0", "id": 2, "method": "resources/read", "params": {"uri": "mcp://does-not-exist"}}
        )
        self.assertEqual(not_found["error"]["code"], -32002)

        builtin_uris = (
            "mcp://moe-router/tiers",
            "mcp://telemetry/otlp",
            "mcp://governance/cancellation",
            "mcp://governance/subscriptions",
            "mcp://governance/roots",
            "mcp://governance/progress",
            "mcp://governance/approvals",
            "mcp://governance/sessions",
        )
        for idx, uri in enumerate(builtin_uris, start=10):
            res = disp.handle({"jsonrpc": "2.0", "id": idx, "method": "resources/read", "params": {"uri": uri}})
            self.assertIn("result", res)
            parsed = json.loads(res["result"]["contents"][0]["text"])
            self.assertIsInstance(parsed, dict)

        for bad_unsub in ({}, {"uri": ""}):
            err = disp.handle({"jsonrpc": "2.0", "id": 20, "method": "resources/unsubscribe", "params": bad_unsub})
            self.assertEqual(err["error"]["code"], -32602)
        unsub_ok = disp.handle(
            {
                "jsonrpc": "2.0",
                "id": 21,
                "method": "resources/unsubscribe",
                "params": {"uri": "vault://documents/never-subscribed"},
            }
        )
        self.assertEqual(unsub_ok["result"], {})

        disp.register_prompt(
            "needs_env",
            "Prompt with required argument.",
            [{"name": "env", "required": True}],
            lambda args: {"description": "ok", "messages": [{"role": "user", "content": {"type": "text", "text": args["env"]}}]},
        )
        for bad_prompt in (
            {"name": "unknown_prompt", "arguments": {}},
            {"name": "moe_sla_routing_plan", "arguments": {}},
            {"name": "needs_env", "arguments": {}},
        ):
            err = disp.handle({"jsonrpc": "2.0", "id": 30, "method": "prompts/get", "params": bad_prompt})
            self.assertEqual(err["error"]["code"], -32602)

        empty_rag_prompt = disp.handle(
            {
                "jsonrpc": "2.0",
                "id": 31,
                "method": "prompts/get",
                "params": {"name": "grounded_rag_answer", "arguments": {"query": "anything"}},
            }
        )
        self.assertIn("No indexed passages matched.", empty_rag_prompt["result"]["messages"][0]["content"]["text"])

        for bad_comp in (
            {"ref": "bad", "argument": {"name": "doc_id", "value": ""}},
            {"ref": {"type": "ref/resource", "uri": "vault://documents/{doc_id}"}, "argument": "bad"},
            {"ref": {"type": "ref/resource", "uri": "vault://documents/{doc_id}"}, "argument": {"name": "", "value": ""}},
            {"ref": {"type": "ref/resource", "uri": "vault://documents/{doc_id}"}, "argument": {"name": "doc_id", "value": 123}},
            {"ref": {"type": "ref/unknown"}, "argument": {"name": "x", "value": ""}},
            {"ref": {"type": "ref/resource", "uri": "vault://unknown/{id}"}, "argument": {"name": "id", "value": ""}},
            {"ref": {"type": "ref/prompt", "name": "no_such_prompt"}, "argument": {"name": "x", "value": ""}},
        ):
            err = disp.handle({"jsonrpc": "2.0", "id": 40, "method": "completion/complete", "params": bad_comp})
            self.assertEqual(err["error"]["code"], -32602)

        for bad_lvl in (123, None):
            err = disp.handle({"jsonrpc": "2.0", "id": 50, "method": "logging/setLevel", "params": {"level": bad_lvl}})
            self.assertEqual(err["error"]["code"], -32602)
        with self.assertRaises(ValueError):
            disp.emit_log("invalid", "msg")

        with self.assertRaises(ValueError):
            disp.register_resource("file://evil.host/a.txt", "Evil", "Disallowed host", lambda: "{}")

    def test_rate_limiter_cursor_codec_and_cancellation_edge_cases(self) -> None:
        clock_ms = [10_000]
        limiter = SlidingWindowTokenRateLimiter(
            window_ms=5_000,
            max_requests_per_window=2,
            max_tokens_per_window=100,
            now_fn=lambda: clock_ms[0],
        )

        # check() must not mutate counters
        chk = limiter.check(None, 5)  # type: ignore[arg-type]
        self.assertTrue(chk["allowed"])
        self.assertEqual(chk["client_id"], "anonymous")
        self.assertEqual(limiter.snapshot()["total_allowed"], 0)

        # Negative token cost clamps to 0 and whitespace client_id normalizes to anonymous
        d_neg = limiter.consume("   ", -10)
        self.assertTrue(d_neg["allowed"])
        self.assertEqual(d_neg["client_id"], "anonymous")
        self.assertEqual(d_neg["requested_tokens"], 0)

        # Single request exceeding entire token capacity
        d_cap = limiter.consume("tenant-cap", 101)
        self.assertFalse(d_cap["allowed"])
        self.assertEqual(d_cap["reason"], "request_exceeds_window_capacity")
        self.assertEqual(d_cap["retry_after_ms"], 5_000)

        # Request-count exhaustion while well below token budget
        self.assertTrue(limiter.consume("tenant-req", 5)["allowed"])
        clock_ms[0] = 11_000
        self.assertTrue(limiter.consume("tenant-req", 5)["allowed"])
        clock_ms[0] = 12_000
        d_req_exceeded = limiter.consume("tenant-req", 5)
        self.assertFalse(d_req_exceeded["allowed"])
        self.assertEqual(d_req_exceeded["reason"], "sliding_window_requests_exceeded")
        self.assertEqual(d_req_exceeded["retry_after_ms"], 3_000)

        # McpCursorCodec negative cases
        for bad_scope in (0, 5, True):
            with self.assertRaises(ValueError):
                McpCursorCodec.encode(bad_scope, 1, 0)  # type: ignore[arg-type]
        for bad_offset in (0, -1, True, McpCursorCodec.MAX_OFFSET + 1):
            with self.assertRaises(ValueError):
                McpCursorCodec.encode(McpCursorCodec.SCOPE_TOOLS, bad_offset, 0)  # type: ignore[arg-type]

        for bad_cursor in (None, 123, "short", "A" * 21 + "+", "A" * 21 + "/", "A" * 21 + "="):
            self.assertEqual(
                McpCursorCodec.decode_and_verify(bad_cursor, McpCursorCodec.SCOPE_TOOLS, 0, 10)[0],
                McpCursorCodec.ERR_B64,
            )

        bad_magic = 0x1234
        bad_magic_seal = McpCursorCodec.compute_seal(bad_magic, McpCursorCodec.VERSION, McpCursorCodec.SCOPE_TOOLS, 1, 0)
        bad_magic_wire = struct.pack("<HBBIII", bad_magic, McpCursorCodec.VERSION, McpCursorCodec.SCOPE_TOOLS, 1, 0, bad_magic_seal)
        bad_magic_tok = base64.urlsafe_b64encode(bad_magic_wire).rstrip(b"=").decode("ascii")
        self.assertEqual(
            McpCursorCodec.decode_and_verify(bad_magic_tok, McpCursorCodec.SCOPE_TOOLS, 0, 10)[0],
            McpCursorCodec.ERR_MAGIC,
        )

        # RequestCancellationRegistry negative & TTL/eviction edges
        clock_sec = [100.0]
        creg = RequestCancellationRegistry(max_tombstones=2, tombstone_ttl_sec=5.0, now_fn=lambda: clock_sec[0])
        for bad_id in (None, True, "", 1.5):
            self.assertEqual(creg.cancel(bad_id), {"cancelled": False, "status": "invalid_request_id"})

        creg.cancel("req-ttl", "early")
        clock_sec[0] = 105.1
        tok_after_ttl = creg.begin("req-ttl")
        self.assertFalse(tok_after_ttl.is_cancelled)
        creg.finish("req-ttl", tok_after_ttl)

        creg.cancel("t-1", "first")
        creg.cancel("t-2", "second")
        creg.cancel("t-3", "third")
        self.assertFalse(creg.begin("t-1").is_cancelled)
        self.assertTrue(creg.begin("t-2").is_cancelled)
        self.assertTrue(creg.begin("t-3").is_cancelled)

    def test_session_registry_sampling_guard_and_schema_validator_negative_cases(self) -> None:
        clock_ms = [1_000]
        reg = StreamableSessionRegistry(
            max_slots=2,
            ring_capacity=2,
            default_ttl_ms=5_000,
            now_ms_fn=lambda: clock_ms[0],
        )

        for bad_sid in ("", "has space", "caf\u00e9", "a" * 129, True, None):
            rc, _, _ = reg.open_session(bad_sid)
            self.assertEqual(rc, StreamableSessionRegistry.ERR_SESSION_SYNTAX)

        for bad_cap in (0, 65, True):
            self.assertEqual(reg.open_session("sid-cap", ring_capacity=bad_cap)[0], StreamableSessionRegistry.ERR_SESSION_SYNTAX)
        for bad_ttl in (0, -1, False):
            self.assertEqual(reg.open_session("sid-ttl", ttl_ms=bad_ttl)[0], StreamableSessionRegistry.ERR_SESSION_SYNTAX)

        self.assertEqual(reg.open_session("sid-1", client_id="c1")[0], 0)
        self.assertEqual(reg.open_session("sid-1", client_id="c1")[0], StreamableSessionRegistry.ERR_SESSION_BINDING)
        self.assertEqual(reg.open_session("sid-2", client_id="c1")[0], 0)
        self.assertEqual(reg.open_session("sid-3", client_id="c1")[0], StreamableSessionRegistry.ERR_SESSION_FULL)

        clock_ms[0] = 6_001
        self.assertEqual(reg.verify_session("sid-1", client_id="c1")[0], StreamableSessionRegistry.ERR_SESSION_EXPIRED)
        # Opening a new session reclaims an expired slot
        self.assertEqual(reg.open_session("sid-3", client_id="c1")[0], 0)

        with self.assertRaises(ValueError):
            StreamableSessionRegistry.encode_event_id("bad sid", 1)
        with self.assertRaises(ValueError):
            StreamableSessionRegistry.encode_event_id("sid-3", 0)

        ev1 = StreamableSessionRegistry.encode_event_id("sid-3", 1)
        self.assertEqual(StreamableSessionRegistry.parse_event_id("sid-3", "malformed")[0], StreamableSessionRegistry.ERR_SESSION_SYNTAX)
        self.assertEqual(StreamableSessionRegistry.parse_event_id("sid-3", -1)[0], StreamableSessionRegistry.ERR_SESSION_SYNTAX)
        self.assertEqual(StreamableSessionRegistry.parse_event_id("sid-3", 0)[0], StreamableSessionRegistry.ERR_REPLAY_GAP)
        self.assertEqual(StreamableSessionRegistry.parse_event_id("other-sid", ev1)[0], StreamableSessionRegistry.ERR_REPLAY_GAP)

        # Replay ahead-of-stream and caught-up checks
        rc_app, meta1, _ = reg.append_event("sid-3", {"n": 1}, client_id="c1")
        self.assertEqual(rc_app, 0)
        assert meta1 is not None
        self.assertEqual(reg.replay_events("sid-3", meta1["eventId"], client_id="c1"), (0, [], "ok"))
        ahead_id = StreamableSessionRegistry.encode_event_id("sid-3", 5)
        self.assertEqual(reg.replay_events("sid-3", ahead_id, client_id="c1")[0], StreamableSessionRegistry.ERR_REPLAY_GAP)

        # SamplingGovernanceGuard negative & includeContext checks
        base_sampling: dict[str, Any] = {
            "maxTokens": 128,
            "includeContext": "allServers",
            "messages": [{"role": "user", "content": {"type": "text", "text": "Hello"}}],
        }
        self.assertEqual(
            SamplingGovernanceGuard.validate_request(base_sampling, allow_all_servers_context=False)[0],
            SamplingGovernanceGuard.ERR_SAMPLING_BUDGET,
        )
        self.assertEqual(
            SamplingGovernanceGuard.validate_request(base_sampling, allow_all_servers_context=True)[0],
            0,
        )

        bad_sampling_payloads: list[dict[str, Any]] = [
            {"maxTokens": 128, "messages": [{"role": "system", "content": {"type": "text", "text": "x"}}]},
            {"maxTokens": 128, "messages": []},
            {"maxTokens": 128, "messages": [{"role": "user", "content": {"type": "image", "data": "abc"}}]},
            {"maxTokens": 128, "systemPrompt": 123, "messages": [{"role": "user", "content": {"type": "text", "text": "x"}}]},
            {"maxTokens": 128, "temperature": float("nan"), "messages": [{"role": "user", "content": {"type": "text", "text": "x"}}]},
            {"maxTokens": 128, "temperature": float("inf"), "messages": [{"role": "user", "content": {"type": "text", "text": "x"}}]},
            {"maxTokens": 128, "temperature": True, "messages": [{"role": "user", "content": {"type": "text", "text": "x"}}]},
            {
                "maxTokens": 128,
                "modelPreferences": {"costPriority": True},
                "messages": [{"role": "user", "content": {"type": "text", "text": "x"}}],
            },
        ]
        for bad_samp in bad_sampling_payloads:
            self.assertEqual(
                SamplingGovernanceGuard.validate_request(bad_samp)[0],
                SamplingGovernanceGuard.ERR_SAMPLING_BUDGET,
            )

        disp_cap = McpJsonRpcDispatcher(enforce_approval_capabilities=True)
        disp_cap.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"capabilities": {}}})
        no_samp_cap = disp_cap.handle(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "sampling/createMessage",
                "params": {
                    "approvalId": 1,
                    "maxTokens": 64,
                    "messages": [{"role": "user", "content": {"type": "text", "text": "Summarize"}}],
                },
            }
        )
        self.assertEqual(no_samp_cap["error"]["code"], -32004)

        # JsonSchema2020Validator edge cases
        self.assertTrue(JsonSchema2020Validator.is_valid({"anything": 1}, True))
        self.assertFalse(JsonSchema2020Validator.is_valid({"anything": 1}, False))

        deep_viol = JsonSchema2020Validator.validate({}, {"type": "object"}, _depth=JsonSchema2020Validator.MAX_DEPTH + 1)
        self.assertEqual(deep_viol[0]["keyword"], "maxDepth")

        strict_obj = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "minLength": 2, "maxLength": 5},
                "score": {"type": "number", "exclusiveMinimum": 0, "exclusiveMaximum": 10},
            },
            "additionalProperties": False,
        }
        self.assertTrue(JsonSchema2020Validator.is_valid({"name": "ok", "score": 5}, strict_obj))
        self.assertFalse(JsonSchema2020Validator.is_valid({"name": "ok", "extra": 1}, strict_obj))
        self.assertFalse(JsonSchema2020Validator.is_valid({"name": "a"}, strict_obj))
        self.assertFalse(JsonSchema2020Validator.is_valid({"name": "toolong"}, strict_obj))
        self.assertFalse(JsonSchema2020Validator.is_valid({"score": 0}, strict_obj))
        self.assertFalse(JsonSchema2020Validator.is_valid({"score": 10}, strict_obj))

        comp_schema = {
            "allOf": [{"type": "integer"}, {"minimum": 2}],
            "anyOf": [{"multipleOf": 2}, {"multipleOf": 3}],
            "not": {"const": 6},
        }
        self.assertTrue(JsonSchema2020Validator.is_valid(4, comp_schema))
        self.assertFalse(JsonSchema2020Validator.is_valid(6, comp_schema))
        self.assertFalse(JsonSchema2020Validator.is_valid(5, comp_schema))

    def test_http_server_negative_paths_and_caught_up_sse(self) -> None:
        srv = build_server("127.0.0.1", 0)
        port = srv.server_address[1]
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            with opener.open(f"http://127.0.0.1:{port}/", timeout=5) as root_resp:
                self.assertEqual(root_resp.status, 200)
                self.assertIn("text/html", root_resp.headers.get("Content-Type", ""))
                self.assertIn("mcp-moe-hybrid-sdk", root_resp.read().decode("utf-8"))

            for method_name in ("POST", "DELETE"):
                req_404 = urllib.request.Request(
                    f"http://127.0.0.1:{port}/nope",
                    data=b"{}" if method_name == "POST" else None,
                    method=method_name,
                    headers={"Content-Type": "application/json"},
                )
                with self.assertRaises(urllib.error.HTTPError) as ctx_404:
                    opener.open(req_404, timeout=5)
                self.assertEqual(ctx_404.exception.code, 404)
                ctx_404.exception.close()

            for bad_raw in (b"{bad", b"\xff\xfe"):
                bad_req = urllib.request.Request(
                    f"http://127.0.0.1:{port}/mcp",
                    data=bad_raw,
                    headers={"Content-Type": "application/json"},
                )
                with self.assertRaises(urllib.error.HTTPError) as ctx_parse:
                    opener.open(bad_req, timeout=5)
                err_p = ctx_parse.exception
                try:
                    self.assertEqual(err_p.code, 400)
                    self.assertIsNotNone(err_p.headers.get("traceparent"))
                    self.assertIsNotNone(err_p.headers.get("X-RateLimit-Limit-Requests"))
                    err_body = json.loads(err_p.read().decode("utf-8"))
                    self.assertEqual(err_body["error"]["code"], -32700)
                finally:
                    err_p.close()

            for method_name in ("GET", "DELETE"):
                no_sid_req = urllib.request.Request(f"http://127.0.0.1:{port}/mcp", method=method_name)
                with self.assertRaises(urllib.error.HTTPError) as ctx_nosid:
                    opener.open(no_sid_req, timeout=5)
                try:
                    self.assertEqual(ctx_nosid.exception.code, 400)
                    err_body = json.loads(ctx_nosid.exception.read().decode("utf-8"))
                    self.assertEqual(err_body["error"]["code"], -32600)
                finally:
                    ctx_nosid.exception.close()

            unk_sid_req = urllib.request.Request(
                f"http://127.0.0.1:{port}/mcp",
                method="GET",
                headers={"Mcp-Session-Id": "mcp-sess-unknown-valid-syntax"},
            )
            with self.assertRaises(urllib.error.HTTPError) as ctx_unk:
                opener.open(unk_sid_req, timeout=5)
            try:
                self.assertEqual(ctx_unk.exception.code, 404)
                err_body = json.loads(ctx_unk.exception.read().decode("utf-8"))
                self.assertEqual(err_body["error"]["code"], -32001)
            finally:
                ctx_unk.exception.close()

            notif_req = urllib.request.Request(
                f"http://127.0.0.1:{port}/mcp",
                data=json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with opener.open(notif_req, timeout=5) as nresp:
                self.assertEqual(nresp.status, 202)
                self.assertEqual(nresp.read(), b"")

            init_req = urllib.request.Request(
                f"http://127.0.0.1:{port}/mcp",
                data=json.dumps(
                    {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-11-25"}}
                ).encode("utf-8"),
                headers={"Content-Type": "application/json", "X-Client-Id": "caughtup-client"},
            )
            with opener.open(init_req, timeout=5) as iresp:
                sid = iresp.headers.get("Mcp-Session-Id")
                self.assertIsNotNone(sid)
                assert sid is not None

            sse_req = urllib.request.Request(
                f"http://127.0.0.1:{port}/mcp",
                data=json.dumps({"jsonrpc": "2.0", "id": 2, "method": "ping"}).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Accept": "text/event-stream",
                    "X-Client-Id": "caughtup-client",
                    "Mcp-Session-Id": sid,
                },
            )
            with opener.open(sse_req, timeout=5) as sresp:
                sse_text = sresp.read().decode("utf-8")
            first_line = sse_text.strip().splitlines()[0]
            self.assertTrue(first_line.startswith("id: "))
            event_id = first_line.split("id: ", 1)[1]

            caught_up_req = urllib.request.Request(
                f"http://127.0.0.1:{port}/mcp",
                method="GET",
                headers={
                    "X-Client-Id": "caughtup-client",
                    "Mcp-Session-Id": sid,
                    "Last-Event-ID": event_id,
                },
            )
            with opener.open(caught_up_req, timeout=5) as cresp:
                self.assertEqual(cresp.status, 200)
                self.assertIn("text/event-stream", cresp.headers.get("Content-Type", ""))
                self.assertEqual(cresp.headers.get("X-Mcp-Replayed-Count"), "0")
                self.assertIn(": session-caught-up", cresp.read().decode("utf-8"))
        finally:
            srv.shutdown()
            srv.server_close()


if __name__ == "__main__":
    unittest.main()
