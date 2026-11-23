"""mcp-moe-hybrid-sdk: MCP 2025-11-25 + Sparse Top-2 MoE Router + SQLite WAL Hybrid RAG."""

from .engine import (
    HybridKnowledgeVault,
    McpJsonRpcDispatcher,
    OtlpTelemetryExporter,
    RedactionGuard,
    SparseTop2MoeRouter,
)

__all__ = [
    "HybridKnowledgeVault",
    "McpJsonRpcDispatcher",
    "OtlpTelemetryExporter",
    "RedactionGuard",
    "SparseTop2MoeRouter",
]
__version__ = "1.1.0"
