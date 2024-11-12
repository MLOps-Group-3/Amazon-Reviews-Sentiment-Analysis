# Stage 1: Base image with system dependencies
FROM apache/airflow:2.10.1-python3.11 AS base
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Stage 2: Python dependencies
FROM base AS python-deps
COPY requirements.txt /requirements.txt
USER airflow
RUN pip install --no-cache-dir -r /requirements.txt

# Stage 3: Final image
FROM python-deps AS final
USER root
ENV PYTHONPATH=/opt/airflow/dags

# Create necessary folders
RUN mkdir -p /opt/airflow/data/{cleaned,labeled,sampled,validation,raw} \
    && chown -R airflow:root /opt/airflow/data

# Copy DAGs and tests
COPY --chown=airflow:root dags /opt/airflow/dags
COPY --chown=airflow:root tests /opt/airflow/tests

# Expose port 8080 for the Airflow webserver
EXPOSE 8080

# Switch back to the airflow user
USER airflow
