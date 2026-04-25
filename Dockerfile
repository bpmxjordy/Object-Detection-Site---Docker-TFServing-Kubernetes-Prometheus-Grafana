# Dockerfile - Flask Object Detection Web Application

# This Dockerfile containerises the Flask frontend and backend,
# packaging the application with all its Python dependencies
# into a Docker image.

FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .

# Installing Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copying the entire Flask application into the container
COPY . .

# Creating directories for uploaded and processed images
RUN mkdir -p static/originals static/results

# Exposing port 5000 for the Flask development server
EXPOSE 5000

# Setting environment variables for the Flask app
ENV TF_SERVING_HOST=localhost
ENV TF_SERVING_PORT=8500
ENV MODEL_NAME=ssd_mobilenet_v2
ENV CONFIDENCE_THRESHOLD=0.5

# Running the Flask application
CMD ["python", "app.py"]
