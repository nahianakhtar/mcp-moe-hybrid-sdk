"""Full-Stack HTTP + MCP Streamable JSON-RPC Server + Interactive Dark Bento Web Showcase."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .engine import HybridKnowledgeVault, McpJsonRpcDispatcher, SparseTop2MoeRouter

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
    <h1>mcp-moe-hybrid-sdk v1.0.0</h1>
    <div style="color:var(--muted)">MCP 2025-11-25 JSON-RPC 2.0 &bull; Sparse Top-2 MoE SLA Router &bull; SQLite WAL FTS5 + Vector RRF Hybrid RAG</div>
  </div>
  <div class="card">
    <h2>1. Sparse Top-2 MoE Router</h2>
    <textarea id="p" rows="3">Architect a 20B MoE hybrid CPU-offload pipeline and verify bond convexity invariants</textarea>
    <button onclick="runRoute()">Route Query</button>
    <pre id="rout">Ready.</pre>
  </div>
  <div class="card">
    <h2>2. SQLite FTS5 + Vector RRF Search</h2>
    <input id="q" value="MCP JSON-RPC protocol hybrid retrieval"/>
    <button onclick="runSearch()">Hybrid Search</button>
    <pre id="sout">Ready.</pre>
  </div>
</div>
<script>
async function mcpCall(name, args){
  const r = await fetch('/mcp',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({jsonrpc:'2.0',id:1,method:'tools/call',params:{name,arguments:args}})});
  const j = await r.json();
  return JSON.parse(j.result.content[0].text);
}
async function runRoute(){ document.getElementById('rout').textContent = JSON.stringify(await mcpCall('moe_route_query',{prompt:document.getElementById('p').value}),null,2); }
async function runSearch(){ document.getElementById('sout').textContent = JSON.stringify(await mcpCall('rag_hybrid_search',{query:document.getElementById('q').value,top_k:3}),null,2); }
</script>
</body>
</html>"""


def create_seeded_dispatcher() -> McpJsonRpcDispatcher:
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
    return McpJsonRpcDispatcher(vault=vault, router=SparseTop2MoeRouter())


def build_server(host: str = "127.0.0.1", port: int = 8090) -> ThreadingHTTPServer:
    dispatcher = create_seeded_dispatcher()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            return

        def _send_json(self, code: int, obj: dict[str, Any]) -> None:
            raw = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send_json(
                    200,
                    {
                        "status": "ok",
                        "protocol": dispatcher.PROTOCOL_VERSION,
                        "documents": dispatcher.vault.count(),
                        "router": dispatcher.router.stats(),
                    },
                )
                return
            html = DASHBOARD_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            if self.path == "/mcp":
                self._send_json(200, dispatcher.handle(body))
            else:
                self._send_json(404, {"error": "Not found"})

    return ThreadingHTTPServer((host, port), Handler)


if __name__ == "__main__":
    srv = build_server()
    print("Serving mcp-moe-hybrid-sdk on http://127.0.0.1:8090")
    srv.serve_forever()
