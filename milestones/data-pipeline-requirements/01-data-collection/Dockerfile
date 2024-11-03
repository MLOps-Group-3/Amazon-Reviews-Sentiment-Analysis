# Base Airflow image with Python 3.12
FROM apache/airflow:2.9.1-python3.12

# Switch to root to install Java and other dependencies
USER root

# Install OpenJDK-17 for PySpark compatibility
RUN apt update && \
    apt install -y openjdk-17-jdk && \
    apt clean

# Set JAVA_HOME environment variable
ENV JAVA_HOME /usr/lib/jvm/java-17-openjdk-amd64/
RUN export JAVA_HOME

# Switch back to the airflow user
USER airflow

# Install necessary Python packages for Spark and data processing
RUN pip install \
    apache-airflow-providers-apache-spark==4.8.0 \
    pyspark \
    pandas

# Additional Airflow configs and requirements can be added as needed
