# Use Python 3.11.2-slim image as base for a smaller image
FROM python:3.11.2-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080 \
    FLASK_APP=app.py

# Set working directory inside the container
WORKDIR /datasage

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to leverage Docker's build cache.
COPY requirements.txt .

# Install Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application code into the working directory
COPY . .

# Expose the port your application will listen on.
EXPOSE ${PORT}

# Command to run the application using Gunicorn (with env var expansion)
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT} ${FLASK_APP%.py}:app"]
