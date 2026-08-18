FROM python:3.12-slim

# uv is copied from its own image rather than installed, which keeps the layer
# small and the version pinned.
COPY --from=ghcr.io/astral-sh/uv:0.12.3 /uv /uvx /bin/

WORKDIR /app

# Dependencies are installed before the source is copied, so editing code does
# not invalidate the dependency layer.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# README.md is named by pyproject.toml as the package readme, so the build
# backend needs it present to install the project itself.
COPY README.md ./
COPY src/ ./src/
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

CMD ["uvicorn", "analyst.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
