# --- Temporary stage used only during the build process
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    wget \
    libgomp1 \
    libgfortran5 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install production dependencies only
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project --compile

# --- Final stage that becomes the actual shipped Docker image (starts fresh)
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS runtime

ENV MLFLOW_TRACKING="False"
ENV TMP_PATH=/tmp

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

RUN useradd --create-home --shell /bin/bash app
USER app
COPY --chown=app:app src/ ./src/
COPY api/ ./api/
COPY config/ ./config/
COPY models/ ./models/
COPY prompts/ ./prompts/
COPY main.py ./

EXPOSE 8000

ENTRYPOINT ["fastapi", "run", "api/api.py", "--host", "0.0.0.0"]