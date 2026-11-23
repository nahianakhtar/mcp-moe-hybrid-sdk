FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY tests ./tests
RUN pip install --no-cache-dir -e . && python -m unittest discover -s tests
EXPOSE 8090
CMD ["python", "-m", "mcp_moe_sdk.server"]
