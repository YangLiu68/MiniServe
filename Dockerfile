# syntax=docker/dockerfile:1.7
# CPU image (CPU-only torch wheels via [tool.uv.sources]); the model is downloaded at first start
# into $HF_HOME — mount a volume there to keep it across restarts.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy
COPY --from=ghcr.io/astral-sh/uv:0.9.2 /uv /usr/local/bin/uv

WORKDIR /app
# Dependencies first (cached layer), then the project itself.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --extra hf --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev --extra hf

RUN useradd --create-home --uid 10001 miniserve
USER miniserve

ENV PATH=/app/.venv/bin:$PATH \
    HF_HOME=/home/miniserve/.cache/huggingface \
    MINISERVE_HOST=0.0.0.0 \
    MINISERVE_PORT=8000 \
    MINISERVE_DEVICE=cpu \
    MINISERVE_LOG_FORMAT=json
# Binding 0.0.0.0 requires MINISERVE_API_KEY (or MINISERVE_API_KEYS); the server refuses to
# start unauthenticated on a non-loopback address.
EXPOSE 8000
STOPSIGNAL SIGTERM
HEALTHCHECK --interval=15s --timeout=3s --start-period=600s --retries=3 \
  CMD python -c "import sys,urllib.request as u; sys.exit(u.urlopen('http://127.0.0.1:8000/healthz', timeout=2).status != 200)"
CMD ["miniserve"]
