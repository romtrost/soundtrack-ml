# Use Python 3.13 slim image
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install uv package manager
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install system dependencies (if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY search/pyproject.toml search/uv.lock ./

# Install dependencies using uv
RUN uv sync --frozen --no-dev

# Copy the application code
COPY search/app/ ./app/

# Copy the embeddings models (required by the app)
COPY search/embeddings/ ./embeddings/

# Update config.yaml paths using sed (replace absolute paths with container paths)
RUN sed -i 's|/home/romain/Documents/Personal/Projects/Soundtrack_ML/soundtrack-ml/search|/app|g' \
    app/retrievers/config.yaml

# Set Python path to include both /app and /app/app for imports
ENV PYTHONPATH=/app:/app/app

# Change working directory to the app directory where the code lives
WORKDIR /app/app

# Expose the port FastAPI runs on
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Run the FastAPI application using uvicorn from the app directory
CMD ["uv", "run", "uvicorn", "search_app:app", "--host", "0.0.0.0", "--port", "8000"]