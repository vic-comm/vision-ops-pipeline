# MLOps Dockerfile
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    postgresql-client \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY mlops/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY mlops/ .
COPY scripts/ ./scripts/

# Create necessary directories
RUN mkdir -p logs models data/raw data/processed

# Expose ports (if needed for local development)
EXPOSE 5000 4200

# Default command (can be overridden)
CMD ["python", "-c", "print('MLOps container ready')"]