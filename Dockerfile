FROM python:3.13-slim-bookworm

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgfortran5 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better caching
COPY pyproject.toml ./

RUN mkdir -p models/FastText \
 && wget -O models/FastText/lid.176.bin https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin

RUN python -m pip install --root-user-action=ignore --no-build-isolation --upgrade pip setuptools \
 && pip install --use-pep517 --root-user-action=ignore --quiet .

COPY . .

RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

ENTRYPOINT ["fastapi", "run", "api/api.py", "--host", "0.0.0.0"]