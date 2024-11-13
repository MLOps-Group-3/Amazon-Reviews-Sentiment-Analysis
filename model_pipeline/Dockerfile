# Start with the official Apache Airflow image
FROM apache/airflow:2.10.2

# Set the environment variable for the data file path
ENV FILE_PATH=/opt/airflow/data/sampled_data_2018_2019.csv

# Copy the requirements.txt file and install dependencies
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# Install MLflow and any additional dependencies
RUN pip install --no-cache-dir mlflow psycopg2-binary

# Copy DAGs, plugins, config, and data
COPY dags /opt/airflow/dags
COPY src /opt/airflow/src

# COPY plugins /opt/airflow/plugins
# COPY config /opt/airflow/config
# COPY data /opt/airflow/data

# Set the working directory for MLflow
WORKDIR /opt/airflow

# Expose port 8080 for the Airflow webserver and 5000 for MLflow tracking server
EXPOSE 8080 5000
