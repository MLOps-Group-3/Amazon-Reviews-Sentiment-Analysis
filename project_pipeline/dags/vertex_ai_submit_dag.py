import os
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.email import EmailOperator
from datetime import datetime, timedelta
from google.cloud import aiplatform
import sys
sys.path.append("/opt/airflow/dags/model_utils")  # Path to the pipeline.py file
from model_utils.pipeline import run_and_monitor_pipeline  # Import the pipeline function

# Fetch environment variables
SMTP_USER = os.getenv("SMTP_USER", "mlopsgrp3@gmail.com")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
SMTP_MAIL_FROM = os.getenv("SMTP_MAIL_FROM", SMTP_USER)
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
GCS_SERVICE_ACCOUNT_KEY = os.getenv("GCS_SERVICE_ACCOUNT_KEY", "/opt/airflow/config/amazonreviewssentimentanalysis-8dfde6e21c1d.json")
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "model-deployment-from-airflow")
GCS_PROJECT_ID = os.getenv("GCS_PROJECT_ID", "amazonreviewssentimentanalysis")
GCS_REGION = os.getenv("GCS_REGION", "us-central1")

# Email configuration
EMAIL_RECIPIENTS = [SMTP_MAIL_FROM]

# Default DAG arguments
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


# Define the DAG
with DAG(
    dag_id="vertex_ai_pipeline_job_submission_with_run",
    default_args=default_args,
    description="Run and monitor Vertex AI pipeline using pipeline.py",
    schedule_interval=None,
    start_date=datetime(2023, 1, 1),
    catchup=False,
) as dag:

    # Task: Run and monitor the pipeline
    run_pipeline_task = PythonOperator(
        task_id="run_vertex_pipeline",
        python_callable=run_and_monitor_pipeline,
        op_args=[],
        op_kwargs={
            "SERVICE_ACCOUNT_KEY_PATH": GCS_SERVICE_ACCOUNT_KEY,
            "GCP_PROJECT": GCS_PROJECT_ID,
            "BUCKET_NAME": GCS_BUCKET_NAME,
            "GCP_REGION": GCS_REGION,
        },
    )

    # Task: Send success email
    send_success_email_task = EmailOperator(
        task_id="send_success_email",
        to=EMAIL_RECIPIENTS,
        subject="Vertex AI Pipeline Succeeded",
        html_content="""
        <p>The Vertex AI pipeline has completed successfully. Please review the results for more details.</p>
        """,
        trigger_rule="all_success",  # Send email only if the previous task succeeds
    )

    # Task: Send failure email
    send_failure_email_task = EmailOperator(
        task_id="send_failure_email",
        to=EMAIL_RECIPIENTS,
        subject="Vertex AI Pipeline Failed",
        html_content="""
        <p>The Vertex AI pipeline has failed. Please check the Airflow logs for more details.</p>
        """,
        trigger_rule="one_failed",  # Send email only if the previous task fails
    )

    # Set dependencies
    run_pipeline_task >> [send_success_email_task, send_failure_email_task]
