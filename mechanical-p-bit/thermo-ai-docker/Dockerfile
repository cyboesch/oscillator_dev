# Use a slim Python image
FROM python:3.10-slim

# Avoid prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory inside the container
WORKDIR /app

# Copy the cleaned requirements file
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install jax==0.4.33 jaxlib==0.4.33 protobuf