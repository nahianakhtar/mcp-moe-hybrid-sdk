"""Full-Stack HTTP + MCP Streamable JSON-RPC Server + Interactive Dark Bento Web Showcase."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .engine import (
    HybridKnowledgeVault,
    McpJsonRpcDispatcher,
    SlidingWindowTokenRateLimiter,
    SparseTop2MoeRouter,
    estimate_mcp_tokens,
)

DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>MCP + Sparse Top-2 MoE + SQLite Hybrid RAG SDK</title>
<style>
  :root{--bg:#090d16;--card:#111827;--border:#1f2937;--text:#f3f4f6;--muted:#9ca3af;--acc:#10b981;--gold:#f59e0b}
  body{margin:0;font-family:system-ui,-apple-system,sans-serif;background:var(--bg);color:var(--text);padding:24px}
  .wrap{max-width:1040px;margin:0 auto;display:grid;gap:18px;grid-template-columns:1fr 1fr}
  .full{grid-column:1 / -1}
  .card{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:18px}
  h1{margin:0 0 6px;font-size:1.45rem;color:var(--acc)}
  h2{margin:0 0 10px;font-size:1.05rem;color:var(--gold)}
  textarea,input{width:100%;box-sizing:border-box;background:#0b1120;color:var(--text);border:1px solid var(--border);border-radius:8px;padding:10px;margin-bottom:10px}
  button{background:var(--acc);color:#04130d;font-weight:700;border:0;border-radius:8px;padding:9px 16px;cursor:pointer}
  pre{background:#050811;border:1px solid var(--border);border-radius:8px;padding:12px;overflow:auto;font-size:.84rem}
</style>
</head>
<body>
<div class="wrap">
  <div class="card full">
    <h1>mcp-moe-hybrid-sdk v1.3.0</h1>
    <div style="color:var(--muted)">MCP 2025-11-25 Streamable HTTP Batching &bull; W3C traceparent OTLP Spans &bull; X-RateLimit-* Token Governance &bull; Sparse Top-2 MoE &bull; SQLite WAL FTS5 RRF</div>
  </div>
  <div class="card">
    <h2>1. Sparse Top-2 MoE Router</h2>
    <textarea id="p" rows="3">Architect a 20B MoE hybrid CPU-offload pipeline and verify bond convexity invariants</textarea>
    <button onclick="runRoute()">Route Query</button>
    <pre id="rout">Ready.</pre>
  </div>
  <div class="card">
    <h2>2. SQLite FTS5 Highlighted + Vector RRF Search</h2>
    <input id="q" value="MCP JSON-RPC protocol hybrid retrieval"/>
    <button onclick="runSearch()">Hybrid Search</button>
    <pre id="sout">Ready.</pre>
  </div>
  <div class="card">
    <h2>3. MCP 2025-11-25 Capability Negotiation</h2>
    <button onclick="runCaps()">Inspect Negotiated Capabilities</button>
    <pre id="cout">Ready.</pre>
  </div>
  <div class="card">
    <h2>4. Streamable JSON-RPC 2.0 Batch + W3C traceparent &amp; Rate-Limit Headers</h2>
    <button onclick="runBatch()">Execute Atomic Batch + Inspect Governance Headers</button>
    <pre id="bout">Ready.</pre>
  </div>
</div>
<script>
async function mcpRpc(method, params={}){
  const r = await fetch('/mcp',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({jsonrpc:'2.0',id:1,method,params})});
  return (await r.json()).result;
}
async function mcpCall(name, args){
  const res = await mcpRpc('tools/call',{name,arguments:args});
  return JSON.parse(res.content[0].text);
}
async function runRoute(){ document.getElementById('rout').textContent = JSON.stringify(await mcpCall('moe_route_query',{prompt:document.getElementById('p').value}),null,2); }
async function runSearch(){ document.getElementById('sout').textContent = JSON.stringify(await mcpCall('rag_hybrid_search',{query:document.getElementById('q').value,top_k:3}),null,2); }
async function runCaps(){
  const [init, resources, prompts] = await Promise.all([
    mcpRpc('initialize',{protocolVersion:'2025-11-25',capabilities:{roots:{listChanged:false}}}),
    mcpRpc('resources/list'),
    mcpRpc('prompts/list')
  ]);
  document.getElementById('cout').textContent = JSON.stringify({initialize:init,resources:resources.resources,prompts:prompts.prompts},null,2);
}
async function runBatch(){
  const tp = '00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01';
  const batch = [
    {jsonrpc:'2.0',method:'notifications/initialized'},
    {jsonrpc:'2.0',id:101,method:'tools/call',params:{name:'moe_route_query',arguments:{prompt:'Verify SQLite WAL FTS5 RRF'}}},
    {jsonrpc:'2.0',id:102,method:'tools/call',params:{name:'rag_hybrid_search',arguments:{query:'Streamable JSON-RPC',top_k:1}}}
  ];
  const r = await fetch('/mcp',{method:'POST',headers:{'Content-Type':'application/json','traceparent':tp},body:JSON.stringify(batch)});
  const headers = {};
  for (const k of ['traceparent','X-RateLimit-Limit-Requests','X-RateLimit-Remaining-Requests','X-RateLimit-Limit-Tokens','X-RateLimit-Remaining-Tokens','X-RateLimit-Reset-Ms']){
    headers[k] = r.headers.get(k);
  }
  document.getElementById('bout').textContent = JSON.stringify({status:r.status,headers,batchResponses:await r.json()},null,2);
}
</script>
</body>
</html>"""


def create_seeded_dispatcher(*, max_batch_size: int = 32) -> McpJsonRpcDispatcher:
    vault = HybridKnowledgeVault(":memory:")
    vault.upsert(
        "arch-mcp-2025",
        "Model Context Protocol 2025-11-25 Streamable JSON-RPC",
        "Deterministic tool discovery, JSON Schema input validation, and structured error handling for agentic SDKs.",
        ["mcp", "jsonrpc", "sdk"],
    )
    vault.upsert(
        "arch-sqlite-rrf",
        "SQLite WAL FTS5 BM25 + Vector Cosine Reciprocal Rank Fusion",
        "Combines exact lexical BM25 keyword precision with L2-normalized dense embeddings via k=60 RRF.",
        ["sqlite", "fts5", "rag", "vector"],
    )
    vault.upsert(
        "arch-sparse-moe",
        "Sparse Top-2 Mixture-of-Experts Cost & Latency Governance",
        "Routes requests dynamically across edge Q8 Vulkan slots and 20B hybrid CPU/GPU MoE tiers under VRAM budgets.",
        ["moe", "routing", "vram", "latency"],
    )
    return McpJsonRpcDispatcher(vault=vault, router=SparseTop2MoeRouter(), max_batch_size=max_batch_size)


def build_server(
    host: str = "127.0.0.1",
    port: int = 8090,
    *,
    rate_limiter: SlidingWindowTokenRateLimiter | None = None,
    max_batch_size: int = 32,
) -> ThreadingHTTPServer:
    dispatcher = create_seeded_dispatcher(max_batch_size=max_batch_size)
    limiter = rate_limiter or SlidingWindowTokenRateLimiter()
    dispatcher.register_resource(
        "mcp://governance/ratelimit",
        "Sliding-Window Token-Budget Rate Limiter",
        "Live per-client sliding-window request/token counters and active bucket snapshot.",
        limiter.snapshot,
    )

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            return

        def _send_json(
            self,
            code: int,
            obj: dict[str, Any] | list[dict[str, Any]],
            *,
            extra_headers: dict[str, str] | None = None,
        ) -> None:
            raw = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            for hk, hv in (extra_headers or {}).items():
                self.send_header(hk, hv)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _send_empty(self, code: int, *, extra_headers: dict[str, str] | None = None) -> None:
            self.send_response(code)
            for hk, hv in (extra_headers or {}).items():
                self.send_header(hk, hv)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def _send_sse(
            self,
            obj: dict[str, Any] | list[dict[str, Any]],
            *,
            extra_headers: dict[str, str] | None = None,
        ) -> None:
            events = obj if isinstance(obj, list) else [obj]
            frames = "".join(f"event: message\ndata: {json.dumps(ev)}\n\n" for ev in events).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            for hk, hv in (extra_headers or {}).items():
                self.send_header(hk, hv)
            self.send_header("Content-Length", str(len(frames)))
            self.end_headers()
            self.wfile.write(frames)

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send_json(
                    200,
                    {
                        "status": "ok",
                        "version": "1.3.0",
                        "protocol": dispatcher.PROTOCOL_VERSION,
                        "documents": dispatcher.vault.count(),
                        "router": dispatcher.router.stats(),
                        "telemetry": dispatcher.telemetry.export_otlp_json()["summary"],
                        "rate_limit": limiter.snapshot(),
                    },
                )
                return
            if self.path == "/metrics/otlp":
                self._send_json(200, dispatcher.telemetry.export_otlp_json())
                return
            html_bytes = DASHBOARD_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html_bytes)))
            self.end_headers()
            self.wfile.write(html_bytes)

        def do_POST(self) -> None:
            length = max(0, int(self.headers.get("Content-Length", "0") or "0"))
            raw_bytes = self.rfile.read(length) if length > 0 else b""
            if self.path != "/mcp":
                self._send_json(404, {"error": "Not found"})
                return

            incoming_tp = self.headers.get("traceparent")
            client_id = (
                self.headers.get("X-Client-Id")
                or (self.client_address[0] if self.client_address else None)
                or "anonymous"
            )

            try:
                body = json.loads(raw_bytes.decode("utf-8") if raw_bytes else "{}")
            except (UnicodeDecodeError, json.JSONDecodeError):
                gw_ctx = dispatcher.telemetry.new_gateway_context(incoming_tp)
                decision = limiter.consume(client_id, 1)
                dispatcher.telemetry.record_rate_limit_decision(decision["allowed"])
                hdrs = {**limiter.to_headers(decision), "traceparent": gw_ctx["traceparent"]}
                self._send_json(
                    400,
                    {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error: invalid JSON payload"}},
                    extra_headers=hdrs,
                )
                return

            est_tokens = estimate_mcp_tokens(body)
            decision = limiter.consume(client_id, est_tokens)
            dispatcher.telemetry.record_rate_limit_decision(decision["allowed"])
            rl_headers = limiter.to_headers(decision)

            if not decision["allowed"]:
                gw_ctx = dispatcher.telemetry.new_gateway_context(incoming_tp)
                req_id = body.get("id") if isinstance(body, dict) else None
                self._send_json(
                    429,
                    {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "error": {
                            "code": -32029,
                            "message": "Rate limit exceeded",
                            "data": {
                                "reason": decision["reason"],
                                "retryAfterMs": decision["retry_after_ms"],
                                "requestedTokens": decision["requested_tokens"],
                            },
                        },
                    },
                    extra_headers={**rl_headers, "traceparent": gw_ctx["traceparent"]},
                )
                return

            status_code, resp_obj, resp_tp = dispatcher.handle_http_payload(body, traceparent=incoming_tp)
            out_headers = {**rl_headers, "traceparent": resp_tp}

            if status_code == 202 or resp_obj is None:
                self._send_empty(202, extra_headers=out_headers)
                return

            accept_hdr = (self.headers.get("Accept") or "").lower()
            wants_sse = "text/event-stream" in accept_hdr and "application/json" not in accept_hdr
            if status_code == 200 and wants_sse:
                self._send_sse(resp_obj, extra_headers=out_headers)
                return

            self._send_json(status_code, resp_obj, extra_headers=out_headers)

    return ThreadingHTTPServer((host, port), Handler)


if __name__ == "__main__":
    srv = build_server()
    print("Serving mcp-moe-hybrid-sdk on http://127.0.0.1:8090")
    srv.serve_forever()
