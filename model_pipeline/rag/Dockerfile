# Start with the official Apache Airflow image
FROM apache/airflow:2.10.2

# Set the environment variable for the data file path
ENV FILE_PATH=/opt/airflow/data/sampled_data_2018_2019.csv

# Copy the requirements.txt file and install dependencies
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# Create the necessary folders in the /opt/airflow/data directory
RUN mkdir -p /opt/airflow/data/cleaned \
             /opt/airflow/data/labeled \
             /opt/airflow/data/sampled \
             /opt/airflow/data/validation \
             /opt/airflow/data/raw

# Copy DAGs, plugins, config, and tests
COPY dags /opt/airflow/dags
#COPY plugins /opt/airflow/plugins
#COPY config /opt/airflow/config
COPY tests /opt/airflow/tests


# Copy .env file into the container
COPY .env /opt/airflow/.env

# Set environment variable to point to the .env file
ENV AIRFLOW__CORE__LOAD_DEFAULT_CONNECTIONS=False

# Expose port 8080 for the Airflow webserver
EXPOSE 8080


# Start with the official Apache Airflow image
FROM apache/airflow:2.10.2

# Set the environment variable for the data file path
ENV FILE_PATH=/opt/airflow/data/sampled_data_2018_2019.csv

# Install Google Cloud Storage client and Apache Airflow GCP provider
RUN pip install --no-cache-dir \
    apache-airflow-providers-google \
    google-cloud-storage

# Copy the requirements.txt file and install additional dependencies
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# Create the necessary folders in the /opt/airflow/data directory
RUN mkdir -p /opt/airflow/data/cleaned \
             /opt/airflow/data/labeled \
             /opt/airflow/data/sampled \
             /opt/airflow/data/validation \
             /opt/airflow/data/raw

# Copy DAGs, plugins, config, and tests
COPY dags /opt/airflow/dags
#COPY plugins /opt/airflow/plugins
COPY config /opt/airflow/config
#COPY tests /opt/airflow/tests

# Copy .env file into the container
COPY .env /opt/airflow/.env

# Set environment variable to point to the .env file
ENV AIRFLOW__CORE__LOAD_DEFAULT_CONNECTIONS=False

# Expose port 8080 for the Airflow webserver
EXPOSE 8080
