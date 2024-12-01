import os
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.email import EmailOperator
from datetime import datetime, timedelta
from google.cloud import aiplatform
import time
from model_utils.pipeline import run_and_monitor_pipeline

# Fetch environment variables
SMTP_USER = os.getenv("SMTP_USER", "mlopsgrp3@gmail.com")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
SMTP_MAIL_FROM = os.getenv("SMTP_MAIL_FROM", SMTP_USER)
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
GCS_SERVICE_ACCOUNT_KEY = os.getenv("GCS_SERVICE_ACCOUNT_KEY")
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
GCS_BUCKET_NAME_MODEL = os.getenv("GCS_BUCKET_NAME_MODEL")

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


def run_and_monitor_pipeline(**kwargs):
    """
    Runs the Vertex AI pipeline and monitors the execution.
    """
    try:
        # Authenticate with GCP
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GCS_SERVICE_ACCOUNT_KEY

        # Initialize Vertex AI
        aiplatform.init(
            project="amazonreviewssentimentanalysis",
            location="us-central1",
        )

        # Dynamically configure the pipeline job
        pipeline_job = aiplatform.PipelineJob(
            display_name="data-prep-and-train-pipeline",
            template_path=f"gs://{GCS_BUCKET_NAME_MODEL}/data_prep_and_train_pipeline.json",
            pipeline_root=f"gs://{GCS_BUCKET_NAME_MODEL}/pipeline_root/",
        )

        # Run the pipeline
        print("Starting pipeline execution...")
        pipeline_job.run(sync=False)  # Asynchronous execution
        print("Pipeline submitted successfully.")

        # Monitor the pipeline
        while True:
            status = pipeline_job.state  # Check the current state
            print(f"Current pipeline status: {status}")

            if status in ["PIPELINE_STATE_SUCCEEDED", "SUCCEEDED"]:
                print("Pipeline completed successfully.")
                return "Pipeline completed successfully."
            elif status in ["PIPELINE_STATE_FAILED", "FAILED"]:
                raise Exception(f"Pipeline failed with status: {status}")
            elif status in ["PIPELINE_STATE_CANCELLED", "CANCELLED"]:
                raise Exception(f"Pipeline was cancelled with status: {status}")

            time.sleep(60)  # Poll every minute

    except Exception as e:
        print(f"Error during pipeline execution: {e}")
        raise


# Define the DAG
with DAG(
    dag_id="vertex_ai_pipeline_job_submission_with_run",
    default_args=default_args,
    description="Run and monitor Vertex AI pipeline using pipeline_job.run()",
    schedule_interval=None,
    start_date=datetime(2023, 1, 1),
    catchup=False,
) as dag:

    # Task: Run and monitor the pipeline
    run_pipeline_task = PythonOperator(
        task_id="run_vertex_pipeline",
        python_callable=run_and_monitor_pipeline,
        provide_context=True,
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
