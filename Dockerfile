FROM python:3.12-slim AS base

WORKDIR /app

# Haystack telemetry tries to mkdir under $HOME on import; the appuser has no
# home directory (--no-create-home), so opt out at the Dockerfile level.
ENV HAYSTACK_TELEMETRY_ENABLED=False

# Backstop for OpenMP/BLAS kernels inside onnxruntime, which read these at
# native-library load time (before Python sets anything). The authoritative cap
# is the in-code `threads` arg (EMBEDDING_THREADS) on the fastembed embedders;
# this just stops OpenMP from over-subscribing cores on the common 8-core host.
# Override via compose for differently-sized hosts.
ENV OMP_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4

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

# Create appuser BEFORE installing dependencies so the venv is built as that
# user. A `chown -R /opt/venv` after the fact would rewrite every file into a
# second multi-GB layer (slow build, slow "exporting layers"). Every chown
# below is non-recursive and runs on an empty directory, so it is instant.
#  - /opt/venv: owned by appuser so `task install` can re-sync the venv inside
#    the container.
#  - /app: owned by appuser so ruff/pytest can create cache dirs there.
#  - /cache: HuggingFace + fastembed model cache mount point. Docker's
#    named-volume first-mount semantics copy this directory's ownership into
#    the volume, so creating it as appuser here is what lets the non-root
#    uvicorn process write the BM42 + tokenizer caches inside the volume.
RUN addgroup --gid ${APP_GID} appuser \
 && adduser --disabled-password --gecos "" --no-create-home \
      --uid ${APP_UID} --ingroup appuser appuser \
 && mkdir -p /opt/venv /cache/hf /cache/fastembed \
 && chown appuser:appuser /app /opt/venv /cache /cache/hf /cache/fastembed

# Dependencies come from uv.lock, so dev and prod install exactly what CI tested.
# The venv lives outside /app because the dev bind mount (./:/app) would hide it.
COPY --from=ghcr.io/astral-sh/uv:0.9.30 /uv /usr/local/bin/uv
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_CACHE_DIR=/tmp/uv-cache \
    PATH=/opt/venv/bin:$PATH

# Everything from here on runs as appuser. Anything that needs root (apt-get,
# etc.) must go above this line.
USER appuser

COPY --chown=appuser:appuser pyproject.toml uv.lock ./

# --- Dev target: includes test/lint tools ---
FROM base AS dev
RUN uv sync --frozen --no-cache --no-install-project --extra dev
COPY --chown=appuser:appuser app/ app/
EXPOSE 8000
HEALTHCHECK CMD curl -f http://localhost:8000/health || exit 1
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# --- Prod target: runtime deps only ---
FROM base AS prod
RUN uv sync --frozen --no-cache --no-install-project
COPY --chown=appuser:appuser app/ app/
EXPOSE 8000
HEALTHCHECK CMD curl -f http://localhost:8000/health || exit 1
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]