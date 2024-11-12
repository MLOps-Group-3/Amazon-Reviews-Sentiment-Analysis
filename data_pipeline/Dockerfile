# Start with the official Apache Airflow image with Python 3.11
FROM apache/airflow:2.10.1-python3.11

# Switch to root to install system dependencies
USER root

# Install any system dependencies if needed
RUN apt-get update && apt-get install -y \
    # Add any system packages you need here \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Add the project root to the Python path
ENV PYTHONPATH=/opt/airflow/dags
# ENV PYTHONPATH="${PYTHONPATH}:/opt/airflow"

# Copy the requirements.txt file
COPY requirements.txt /requirements.txt

# Switch to the airflow user for pip installations
USER airflow

# Install Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# Install Dask and its dependencies with specific versions
RUN pip install --no-cache-dir \
    dask[complete]==2024.9.1 \
    distributed==2024.9.1 \
    cloudpickle==3.0.0 \
    numpy==2.1.1 \
    pandas==2.2.3 \
    toolz==0.12.0

# Switch back to root to create directories
USER root

# Create the necessary folders in the /opt/airflow/data directory
RUN mkdir -p /opt/airflow/data/cleaned \
             /opt/airflow/data/labeled \
             /opt/airflow/data/sampled \
             /opt/airflow/data/validation \
             /opt/airflow/data/raw \
    && chown -R airflow:root /opt/airflow/data

# Copy DAGs, plugins, config, and tests
COPY --chown=airflow:root dags /opt/airflow/dags
#COPY --chown=airflow:root plugins /opt/airflow/plugins
#COPY --chown=airflow:root config /opt/airflow/config
COPY --chown=airflow:root tests /opt/airflow/tests

# Expose port 8080 for the Airflow webserver
EXPOSE 8080

# Switch back to the airflow user
USER airflow
