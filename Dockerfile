# SCAFAD Reproducible Experiments Docker Container
# =================================================
# This Dockerfile creates a complete environment for running SCAFAD experiments
# with all dependencies, proper isolation, and reproducible results.

FROM python:3.9-slim

# Metadata
LABEL maintainer="SCAFAD Research Team"
LABEL description="SCAFAD (Serverless Context-Aware Fusion Anomaly Detection) Experimental Environment"
LABEL version="1.0"

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive
ENV SCAFAD_ENVIRONMENT=DOCKER
ENV SCAFAD_REPRODUCIBLE=true

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    curl \
    jq \
    vim \
    htop \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /scafad

# Copy requirements first for better caching
COPY requirements.txt .
COPY docker/requirements-docker.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r docker/requirements-docker.txt

# Copy the entire SCAFAD codebase
COPY . .

# Create necessary directories
RUN mkdir -p /scafad/experiments/results \
             /scafad/experiments/logs \
             /scafad/experiments/datasets \
             /scafad/experiments/checkpoints \
             /scafad/experiments/reports

# Create experiment runner script
COPY docker/run_experiments.sh /scafad/run_experiments.sh
RUN chmod +x /scafad/run_experiments.sh

# Set up Python path
ENV PYTHONPATH="/scafad:$PYTHONPATH"

# Run basic validation on build
RUN python -c "import sys; print(f'Python version: {sys.version}')"
RUN python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"
RUN python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
RUN python -c "import networkx; print(f'NetworkX version: {networkx.__version__}')"

# Create non-root user for security
RUN useradd -m -u 1000 scafad && \
    chown -R scafad:scafad /scafad
USER scafad

# Default command
CMD ["python", "experiments/run_reproducible_experiments.py", "--help"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import app_config; print('SCAFAD container healthy')"