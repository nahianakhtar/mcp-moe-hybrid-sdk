"""mcp-moe-hybrid-sdk: MCP 2025-11-25 + Sparse Top-2 MoE Router + SQLite WAL Hybrid RAG."""

from .engine import (
    HybridKnowledgeVault,
    McpJsonRpcDispatcher,
    OtlpTelemetryExporter,
    RedactionGuard,
    SearchHit,
    SlidingWindowTokenRateLimiter,
    SparseTop2MoeRouter,
    estimate_mcp_tokens,
)

__all__ = [
    "HybridKnowledgeVault",
    "McpJsonRpcDispatcher",
    "OtlpTelemetryExporter",
    "RedactionGuard",
    "SearchHit",
    "SlidingWindowTokenRateLimiter",
    "SparseTop2MoeRouter",
    "estimate_mcp_tokens",
]
__version__ = "1.3.0"
