# Amazon Reviews Data Pipeline

This project uses Apache Airflow to manage a data pipeline for Amazon product reviews and metadata.

## Folder Structure

- dags/: Contains Airflow DAG files and related Python scripts
- scripts/: For any additional scripts
- plugins/: For custom Airflow plugins
- logs/: Airflow logs
- data/: 
  - raw/: Raw data downloaded from Amazon
- docker-compose.yaml: Docker Compose configuration file
- requirements.txt: Python package requirements

## Setup

1. Ensure Docker and Docker Compose are installed on your system.
2. Run `docker-compose up -d` to start Airflow services.
3. Access the Airflow UI at http://localhost:8080 (username: airflow068, password: airflow068).

## Usage

1. The main DAG is `amazon_reviews_data_pipeline` in the Airflow UI.
2. Enable and trigger the DAG to start the data acquisition process.
3. Downloaded data will be stored in the `data/raw` directory.

## Configuration

- Modify `dags/config.py` to adjust data acquisition parameters.
- Update `docker-compose.yaml` for any changes to the Airflow setup.

