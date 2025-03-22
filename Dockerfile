FROM python:3.10.0-slim

# Ensure pip is installed and upgraded
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libc-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python -m ensurepip --upgrade

# Environment variables to improve Python behavior in Docker
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH="/app"

# Set the working directory
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt .

# Remove Windows-specific packages and install dependencies
RUN pip install --upgrade pip && \
    # grep -v "pywin32" requirements.txt > requirements_docker.txt && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install kfp==2.12.1 kfp-server-api==2.4.0 kfp-pipeline-spec==0.6.0

# Copy the project files
COPY data/ /app/data/
COPY models/ /app/models/
COPY pipeline/ /app/pipeline/
COPY *.py /app/
COPY *.md /app/
COPY *.yaml /app/
COPY *.ipynb /app/

# Make pipeline scripts executable
RUN chmod +x /app/pipeline/*.py
RUN chmod +x /app/kubeflow_pipeline.py

# The container will run the Kubeflow pipeline by default
CMD ["python", "kubeflow_pipeline.py"]