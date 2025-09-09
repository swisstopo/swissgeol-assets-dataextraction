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

RUN python -m pip install --root-user-action=ignore --no-build-isolation --upgrade pip setuptools wheel \
 && pip install --use-pep517 --root-user-action=ignore --quiet --prefix=/install .

RUN mkdir -p /models/FastText \
 && wget -O /models/FastText/lid.176.bin https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin


FROM python:3.13-slim-bookworm AS runtime

ENV FASTTEXT_MODEL_PATH="models/FastText/lid.176.bin"

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
COPY . .

RUN useradd --create-home --shell /bin/bash app \
 && chown -R app:app /app
USER app

EXPOSE 8000

ENTRYPOINT ["fastapi", "run", "api/api.py", "--host", "0.0.0.0"]