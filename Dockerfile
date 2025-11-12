FROM python:3.13-slim-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    wget \
    libgomp1 \
    libgfortran5 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml ./

# Install production dependencies only
RUN python -m pip install --root-user-action=ignore --no-cache-dir --upgrade pip setuptools wheel \
 && pip install --use-pep517 --root-user-action=ignore --no-cache-dir --prefix=/install .


FROM python:3.13-slim-bookworm AS runtime

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

COPY --from=builder /install /usr/local

COPY --from=builder /models ./models
COPY src/ ./src/
COPY api/ ./api/
COPY config/ ./config/
COPY models/ ./models/
COPY prompts/ ./prompts/
COPY main.py ./

RUN useradd --create-home --shell /bin/bash app \
 && chown -R app:app /app
USER app

EXPOSE 8000

ENTRYPOINT ["fastapi", "run", "api/api.py", "--host", "0.0.0.0"]