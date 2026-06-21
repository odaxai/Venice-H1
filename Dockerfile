FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

LABEL maintainer="nicolo.savioli@odaxai.com"
LABEL description="Venice-H1: Failure-Aware Query Re-Ranking for RIS"
LABEL org.opencontainers.image.source="https://github.com/odaxai/Venice-H1"

WORKDIR /workspace

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY . .

# Default command: show help
CMD ["python", "train.py", "--help"]
