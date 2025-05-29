# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy only the frontend requirements
COPY frontend/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the frontend code
COPY frontend/ .

# Expose the port the app runs on
EXPOSE 7860

# Command to run the application
CMD ["python", "gradio_ui.py"]
