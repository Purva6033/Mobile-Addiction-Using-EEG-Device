FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir flask==3.0.3 numpy==1.24.3 pandas==2.0.3 scikit-learn==1.3.0 werkzeug==3.0.0

# Copy application files
COPY mobile_Addiction_model/mobile_Addiction_model /app

# Create directories
RUN mkdir -p /app/uploads /app/models

# Expose port
EXPOSE 5000

# Set environment
ENV FLASK_APP=app_no_tf.py
ENV PYTHONUNBUFFERED=1

# Run application
CMD ["python", "app_no_tf.py"]
