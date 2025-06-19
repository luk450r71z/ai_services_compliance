# AI Image Analysis Agent Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    bash \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY .env ./

# Create necessary directories
RUN mkdir -p assets/images/processed envs/data

# Make the run script executable
COPY envs/run_analysis.sh ./run_analysis.sh
RUN chmod +x ./run_analysis.sh

# Set environment variables
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["./run_analysis.sh"] 