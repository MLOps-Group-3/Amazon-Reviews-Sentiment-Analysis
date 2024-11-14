# # Start with the official Apache Airflow image
# FROM apache/airflow:2.10.2

# # Set the environment variable for the data file path
# ENV FILE_PATH=/opt/airflow/data/sampled_data_2018_2019.csv

# # Copy the requirements.txt file and install dependencies
# COPY requirements.txt /requirements.txt
# RUN pip install --no-cache-dir -r /requirements.txt

# # Install MLflow and any additional dependencies
# RUN pip install --no-cache-dir mlflow psycopg2-binary

# # Copy DAGs, plugins, config, and data
# COPY dags /opt/airflow/dags
# COPY src /opt/airflow/src

# # COPY plugins /opt/airflow/plugins
# # COPY config /opt/airflow/config
# # COPY data /opt/airflow/data

# # Set the working directory for MLflow
# WORKDIR /opt/airflow

# # Expose port 8080 for the Airflow webserver and 5000 for MLflow tracking server
# EXPOSE 8080 5000
# Use NVIDIA's PyTorch container as a base
FROM nvcr.io/nvidia/pytorch:23.08-py3

# Set environment variable to avoid interactive timezone configuration
ENV DEBIAN_FRONTEND=noninteractive

# Install additional dependencies if needed
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3-dev \
    libpq-dev \
    build-essential \
    cmake \
    git && \
    rm -rf /var/lib/apt/lists/*

# Install Python packages for transformers, torch, and other dependencies
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# Copy any necessary files for your Airflow and MLflow setup
COPY dags /opt/airflow/dags
COPY src /opt/airflow/src

# Set working directory
WORKDIR /opt/airflow

# Expose port for Airflow and MLflow
EXPOSE 8080 5000
