FROM python:3.11-slim

# Copy the uv binary from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    gcc \
    g++ \
    libpq-dev \
    libgl1 \
    mesa-utils \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install packages using uv (still very fast without cache mounts)
RUN uv pip install --system -r requirements.txt

COPY . .

# Ensure directories exist
RUN mkdir -p /app/uploads /app/processed

EXPOSE 8000

# Run with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]