FROM python:3.12-slim AS base

WORKDIR /app

# Haystack telemetry tries to mkdir under $HOME on import; the appuser has no
# home directory (--no-create-home), so opt out at the Dockerfile level.
ENV HAYSTACK_TELEMETRY_ENABLED=False

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .

# --- Dev target: includes test/lint tools ---
FROM base AS dev
RUN pip install --no-cache-dir ".[dev]"
COPY app/ app/
RUN adduser --system --no-create-home appuser \
 # /cache is the HuggingFace + fastembed model cache mount point. Docker's
 # named-volume first-mount semantics copy this directory's ownership into
 # the volume, so creating it as appuser here is what lets the non-root
 # uvicorn process write the BM42 + tokenizer caches inside the volume.
 && mkdir -p /cache/hf /cache/fastembed \
 && chown -R appuser /cache
USER appuser
EXPOSE 8000
HEALTHCHECK CMD curl -f http://localhost:8000/health || exit 1
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# --- Prod target: runtime deps only ---
FROM base AS prod
RUN pip install --no-cache-dir .
COPY app/ app/
RUN adduser --system --no-create-home appuser \
 # See dev-target comment — same ownership setup is required in prod for
 # the named model-cache volume to be writable by the non-root user.
 && mkdir -p /cache/hf /cache/fastembed \
 && chown -R appuser /cache
USER appuser
EXPOSE 8000
HEALTHCHECK CMD curl -f http://localhost:8000/health || exit 1
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
