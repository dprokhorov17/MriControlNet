# MRI ControlNet Image Processing

This repository contains a frontend and backend implementation for processing MRI images using ControlNet. The frontend is built with Gradio, while the backend is powered by FastAPI.

## Features
- Upload MRI images and adjust parameters for image processing.
- Generate synthetic brain scan images using ControlNet.
- Apply color transfer modes to processed images.

## Prerequisites
- Docker and Docker Compose installed.
- Python 3.10 or higher installed.

## Installation

### Clone the Repository
```bash
git clone https://github.com/dprokhorov17/MriControlNet.git
cd MriControlNet
```

### Build and Deploy with Docker

1. Build the Docker images:
   ```bash
   docker-compose build
   ```

2. Start the containers:
   ```bash
   docker-compose up
   ```

3. Access the Gradio UI at `http://localhost:7860`.

### Manual Deployment (not recommended!)

#### Backend
1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Start the FastAPI server:
   ```bash
   cd frontend
   python gradio_ui.py
   ```

#### Frontend
1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Start the Gradio UI:
   ```bash
   python gradio_ui.py
   ```

4. Access the Gradio UI at `http://localhost:7860`.

## Configuration
- The backend API URL can be configured using the `API_URL` environment variable. Default is `http://localhost:8000/process`.

## Example Usage
1. Upload an MRI image.
2. Adjust parameters such as inference steps, guidance scale, and color transfer mode.
3. View the processed images including original, control, generated, and color-transferred versions.

## License
This project is licensed under the Apache License. See the LICENSE file for details.