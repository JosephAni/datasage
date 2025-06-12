# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080 \
    FLASK_APP=app.py

# Set working directory
WORKDIR /datasage

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies (ensure gunicorn is in requirements.txt)
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port (for Cloud Run or similar platforms)
EXPOSE ${PORT}

# Command to run the application using Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:${PORT}", "${FLASK_APP%.py}:app"]
