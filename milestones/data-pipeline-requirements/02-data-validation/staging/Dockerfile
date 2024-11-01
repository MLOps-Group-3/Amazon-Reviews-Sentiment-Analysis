# Start with the official Apache Airflow image
FROM apache/airflow:2.10.2

# Copy the requirements.txt file and install dependencies
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# Copy DAGs, plugins, config, and tests
COPY dags /opt/airflow/dags
COPY plugins /opt/airflow/plugins
COPY config /opt/airflow/config
COPY tests /opt/airflow/tests

# Expose port 8080 for the Airflow webserver
EXPOSE 8080