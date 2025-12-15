FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements_docker.txt .

# Install python dependencies
RUN pip install --no-cache-dir -r requirements_docker.txt

# Copy the rest of the application
COPY . .

# Expose ports
EXPOSE 5000
EXPOSE 8501

# Default command (will be overridden in docker-compose)
CMD ["python", "Rest_api/flask_app.py"]
