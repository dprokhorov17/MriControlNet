# Backend Dockerfile for MRI ControlNet
# This Dockerfile sets up the FastAPI backend service with CUDA support for GPU acceleration.
# It includes all necessary system dependencies for OpenCV and PyTorch.

# Use an official NVIDIA CUDA runtime as a parent image
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies
# - libgl1-mesa-glx: Required for OpenCV
# - libglib2.0-0: Required for OpenCV
# - python3-pip: Python package manager
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Copy code
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r backend/requirements.txt

# Expose the port the app runs on
EXPOSE 8000

# Command to run the FastAPI application
CMD ["sh", "-c", "cd backend && uvicorn main:app --host 0.0.0.0 --port 8000"]
