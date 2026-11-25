FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH" \
    MCP_HOST="0.0.0.0" \
    MCP_PORT="8090"

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY tests ./tests
COPY benchmarks ./benchmarks

RUN uv venv --python python3 .venv \
    && uv pip install --python .venv/bin/python --no-cache -e ".[dev]" \
    && .venv/bin/ruff check . \
    && PYTHONPATH=src .venv/bin/pytest -q \
    && PYTHONPATH=src .venv/bin/python benchmarks/run_golden_eval.py

EXPOSE 8090
CMD [".venv/bin/python", "-m", "mcp_moe_sdk.server"]
