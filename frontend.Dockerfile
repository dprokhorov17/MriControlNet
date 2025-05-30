# Frontend Dockerfile for MRI ControlNet
# This Dockerfile sets up the Gradio web interface for the application.
# It uses a lightweight Python image as it doesn't require GPU support.

# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy only the frontend requirements first for better layer caching
COPY frontend/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the frontend code
COPY frontend/ .

# Expose the Gradio interface port
EXPOSE 7860

# Command to run the Gradio interface
CMD ["python", "gradio_ui.py"]
