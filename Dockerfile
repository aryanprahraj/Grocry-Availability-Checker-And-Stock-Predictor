# Use Python 3.10 slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY setup.py .

# Install the package
RUN pip install -e .

# Create directory for data and models
RUN mkdir -p /app/data /app/saved_models

# Expose port for API (if you add FastAPI/Flask later)
EXPOSE 8000

# Default command (can be overridden)
CMD ["python", "-c", "print('Grocery Stock Predictor Container Ready')"]
