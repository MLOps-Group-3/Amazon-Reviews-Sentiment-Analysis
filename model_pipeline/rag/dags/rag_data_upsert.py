from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import os
from utils.flatten_and_save import flatten_and_save
from utils.chunk_upload_gcp import upload_to_gcs_with_service_account
from utils.embeddings import fetch_data_from_gcs, upsert_to_pinecone

# Define default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
with DAG(
    'amazon_reviews_pipeline',
    default_args=default_args,
    description='Pipeline to process Amazon reviews, generate embeddings, and upsert to Pinecone',
    schedule_interval=None,  # Run manually
    start_date=datetime(2023, 1, 1),
    catchup=False,
) as dag:

    # Paths and configurations
    RAW_DATA_PATH = "/opt/airflow/data/refined_processed_documents.json"
    CHUNKED_DATA_DIR = "/opt/airflow/data/output_chunks"
    GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
    GCS_PREFIX = "RAG/"  # Folder inside the GCS bucket
    SERVICE_ACCOUNT_JSON = os.getenv("GCS_SERVICE_ACCOUNT_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_ENV = os.getenv("PINECONE_ENVIRONMENT")
    INDEX_NAME = os.getenv("INDEX_NAME")

    # Task 1: Flatten JSON data and save locally
    def task_flatten_and_save(**kwargs):
        flatten_and_save(RAW_DATA_PATH, CHUNKED_DATA_DIR)

    flatten_task = PythonOperator(
        task_id='flatten_and_save',
        python_callable=task_flatten_and_save
    )

    # Task 2: Upload flattened data to GCS
    def task_upload_to_gcs(**kwargs):
        upload_to_gcs_with_service_account(
            local_directory=CHUNKED_DATA_DIR,
            bucket_name=GCS_BUCKET_NAME,
            service_account_path=SERVICE_ACCOUNT_JSON,
            remote_directory="RAG"
        )

    gcs_upload_task = PythonOperator(
        task_id='upload_to_gcs',
        python_callable=task_upload_to_gcs
    )

    # Task 3: Fetch data from GCS
    def task_fetch_from_gcs(**kwargs):
        records = fetch_data_from_gcs(bucket_name=GCS_BUCKET_NAME, prefix=GCS_PREFIX)
        return records  # Return data for downstream tasks

    fetch_from_gcs_task = PythonOperator(
        task_id='fetch_from_gcs',
        python_callable=task_fetch_from_gcs
    )

    # Task 4: Generate embeddings and upsert to Pinecone
    def task_embeddings_and_upsert(**kwargs):
        records = kwargs['ti'].xcom_pull(task_ids='fetch_from_gcs')
        if records:
            upsert_to_pinecone(
                records=records,
                index_name=INDEX_NAME
            )

    embeddings_task = PythonOperator(
        task_id='generate_embeddings_and_upsert',
        python_callable=task_embeddings_and_upsert,
        provide_context=True  # Enable passing XCom data
    )

    # Task dependencies
    flatten_task >> gcs_upload_task >> fetch_from_gcs_task >> embeddings_task
