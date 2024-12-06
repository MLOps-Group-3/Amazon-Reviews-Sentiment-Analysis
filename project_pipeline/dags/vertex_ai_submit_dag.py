import os
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.email import EmailOperator
from datetime import datetime, timedelta
from google.cloud import aiplatform
import sys
sys.path.append("/opt/airflow/dags")
sys.path.append("/opt/airflow/dags/model_utils")
# from model_utils.pipeline_archive import run_and_monitor_pipeline  # Import the pipeline function
from model_utils.pipeline_CI_CD import run_pipeline  # Import the pipeline function

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Fetch environment variables
# SMTP_USER = os.getenv("SMTP_USER", "mlopsgrp3@gmail.com")
# SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
# SMTP_MAIL_FROM = os.getenv("SMTP_MAIL_FROM", SMTP_USER)
# SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
# SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
# GCS_SERVICE_ACCOUNT_KEY = os.getenv("GCS_SERVICE_ACCOUNT_KEY", "/opt/airflow/config/amazonreviewssentimentanalysis-8dfde6e21c1d.json")
# GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME_MODEL", "model-deployment-from-airflow")
# GCS_PROJECT_ID = os.getenv("GCS_PROJECT_ID", "amazonreviewssentimentanalysis")
# GCS_REGION = os.getenv("GCS_REGION", "us-central1")

# Email configuration
# EMAIL_RECIPIENTS = [SMTP_MAIL_FROM]

# Default DAG arguments
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 10, 22),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'email_on_failure': True,         # Send email on failure
    'email_on_retry': True,           # Send email on retry
    'email_on_success': False,        # Optional: email on success
    'email': 'vallimeenaavellaiyan@gmail.com'  # Global recipient for all tasks
}

# Define the DAG
with DAG(
    dag_id="06_vertex_ai_pipeline_job_submission_with_run",
    default_args=default_args,
    description="Run and monitor Vertex AI pipeline using pipeline.py",
    schedule_interval=None,
    start_date=datetime(2024, 12, 1),
    catchup=False,
) as dag:

    # Task: Run and monitor the pipeline
    run_pipeline_task = PythonOperator(
        task_id="run_vertex_pipeline",
        python_callable=run_pipeline,
    )
    #     python_callable=run_and_monitor_pipeline,
    #     op_args=[],
    #     op_kwargs={
    #         "SERVICE_ACCOUNT_KEY_PATH": GCS_SERVICE_ACCOUNT_KEY,
    #         "GCP_PROJECT": GCS_PROJECT_ID,
    #         "BUCKET_NAME": GCS_BUCKET_NAME,
    #         "GCP_REGION": GCS_REGION
    #     },
    # )


    # Set dependencies
    run_pipeline_task 
