FROM python:3.12-slim AS base

WORKDIR /app

# Haystack telemetry tries to mkdir under $HOME on import; the appuser has no
# home directory (--no-create-home), so opt out at the Dockerfile level.
ENV HAYSTACK_TELEMETRY_ENABLED=False

# Pin the in-container appuser to a uid/gid that match the host developer's
# user. /app is bind-mounted in dev, and tools like ruff/pytest need to
# create cache dirs there — running as a mismatched uid makes those writes
# fail with EACCES. Defaults of 1000 cover the common Linux desktop case;
# override at build time for CI / other-uid hosts:
#   docker build --build-arg APP_UID=$(id -u) --build-arg APP_GID=$(id -g) ...
ARG APP_UID=1000
ARG APP_GID=1000

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .

# --- Dev target: includes test/lint tools ---
FROM base AS dev
ARG APP_UID
ARG APP_GID
RUN pip install --no-cache-dir ".[dev]"
COPY app/ app/
RUN addgroup --system --gid ${APP_GID} appuser \
 && adduser --system --no-create-home --uid ${APP_UID} --ingroup appuser appuser \
 # /cache is the HuggingFace + fastembed model cache mount point. Docker's
 # named-volume first-mount semantics copy this directory's ownership into
 # the volume, so creating it as appuser here is what lets the non-root
 # uvicorn process write the BM42 + tokenizer caches inside the volume.
 && mkdir -p /cache/hf /cache/fastembed \
 && chown -R appuser:appuser /cache
USER appuser
EXPOSE 8000
HEALTHCHECK CMD curl -f http://localhost:8000/health || exit 1
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# --- Prod target: runtime deps only ---
FROM base AS prod
ARG APP_UID
ARG APP_GID
RUN pip install --no-cache-dir .
COPY app/ app/
RUN addgroup --system --gid ${APP_GID} appuser \
 && adduser --system --no-create-home --uid ${APP_UID} --ingroup appuser appuser \
 # See dev-target comment — same ownership setup is required in prod for
 # the named model-cache volume to be writable by the non-root user.
 && mkdir -p /cache/hf /cache/fastembed \
 && chown -R appuser:appuser /cache
USER appuser
EXPOSE 8000
HEALTHCHECK CMD curl -f http://localhost:8000/health || exit 1
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
