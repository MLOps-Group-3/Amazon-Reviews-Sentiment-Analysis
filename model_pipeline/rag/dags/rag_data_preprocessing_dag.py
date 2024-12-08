# Import necessary libraries
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.dagrun_operator import TriggerDagRunOperator
from airflow.utils.task_group import TaskGroup
from datetime import datetime, timedelta
import os
import logging
import json
import pandas as pd
from dotenv import load_dotenv
import re

# Import your RAG preprocessing functions
from utils.review_data_processing import load_and_process_data, aggregate_data, prepare_documents, save_documents
from utils.document_processor import process_document_with_summary
from utils.config import PROCESSED_DATA_PATH, DOCUMENT_STORE_PATH
from utils.config import REFINED_PROCESSED_DATA_PATH, DOCUMENTS_FILE_PATH, PROCESSED_OUTPUT_PATH, REFINED_OUTPUT_PATH
from utils.bigquery_utils import fetch_data_from_bigquery_and_save

# Load environment variables from .env file
load_dotenv()

# Set up logging for better traceability
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

# Check if the OpenAI API key is available
if not openai_api_key:
    logger.error("OpenAI API key not found. Please set it in the .env file.")
    raise ValueError("OpenAI API key not found. Please set it in the .env file.")

# BigQuery table information
project_id = os.getenv("GCS_PROJECT_ID")
dataset_id = os.getenv("GCS_DATASET_ID")
table_id = os.getenv("GCS_TABLE_ID")

# Function to fetch data from BigQuery and save to local
def fetch_bigquery_data_task():
    logger.info("Fetching data from BigQuery and saving to local...")
    fetch_data_from_bigquery_and_save(
        project_id=project_id,
        dataset_id=dataset_id,
        table_id=table_id,
        output_path=PROCESSED_OUTPUT_PATH
    )

# RAG Data Preprocessing Task Group Functions
def load_and_process_data_task(ti):
    logger.info("Loading and processing data...")
    df = load_and_process_data(PROCESSED_DATA_PATH)
    ti.xcom_push(key='processed_df', value=df.to_json(orient="split"))  # Pass DataFrame to next tasks via XCom

def aggregate_data_task(ti):
    logger.info("Aggregating data...")
    processed_data_json = ti.xcom_pull(key='processed_df', task_ids='rag_data_preprocessing.load_and_process_data')
    df = pd.read_json(processed_data_json, orient="split")
    aggregated_df = aggregate_data(df)
    ti.xcom_push(key='aggregated_df', value=aggregated_df.to_json(orient="split"))

def prepare_documents_task(ti):
    logger.info("Preparing documents for RAG model...")
    aggregated_data_json = ti.xcom_pull(key='aggregated_df', task_ids='rag_data_preprocessing.aggregate_data')
    df = pd.read_json(aggregated_data_json, orient="split")
    documents = prepare_documents(df)
    ti.xcom_push(key='documents', value=json.dumps(documents))

def save_documents_task(ti):
    logger.info("Saving documents to JSON...")
    documents_json = ti.xcom_pull(key='documents', task_ids='rag_data_preprocessing.prepare_documents')
    documents = json.loads(documents_json)
    save_documents(documents, DOCUMENT_STORE_PATH)
    logger.info("Documents saved successfully.")

# Document Processing Task Group Functions
def load_documents_task():
    with open(DOCUMENTS_FILE_PATH, 'r') as f:
        documents = json.load(f)
    logger.info(f"Loaded {len(documents)} documents.")
    return documents

def process_documents_task(ti):
    documents = ti.xcom_pull(task_ids='document_processing.load_documents')
    processed_results = []
    
    # Pass OpenAI API key explicitly
    for idx, document in enumerate(documents):
        try:
            processed_doc = process_document_with_summary(document, api_key=openai_api_key)
            processed_results.append(processed_doc)
        except Exception as e:
            logger.error(f"Failed to process document {idx + 1}: {e}")
    return processed_results

def clean_and_validate_json_task(ti):
    documents = ti.xcom_pull(task_ids='document_processing.process_documents')
    formatted_data = []

    for item in documents:
        try:
            cleaned_analysis = re.sub(r'\\n|\\t', '', item['analysis'])
            cleaned_analysis = re.sub(r'}\s*{', '},{', cleaned_analysis)
            cleaned_analysis = re.sub(r'(?<=\})(\s*)(?=")', r',\1', cleaned_analysis)
            cleaned_analysis = re.sub(r'"Performance":\s*{([^{}]*)}\s*(?=[\]}])', r'"Performance": {\1},', cleaned_analysis)
            cleaned_analysis = re.sub(r'(?<=\{|,)(\s*)(\w+)(\s*):', r'\1"\2":', cleaned_analysis)
            cleaned_analysis = re.sub(r',\s*([\]}])', r'\1', cleaned_analysis)

            brace_diff = cleaned_analysis.count("{") - cleaned_analysis.count("}")
            if brace_diff > 0:
                cleaned_analysis += "}" * brace_diff
            elif brace_diff < 0:
                cleaned_analysis = cleaned_analysis[:brace_diff]

            item['analysis'] = json.loads(cleaned_analysis)
            formatted_data.append(item)

        except json.JSONDecodeError as e:
            logger.error(f"Error parsing `analysis`: {e}")
            continue

    return formatted_data

def save_refined_documents_task(ti):
    documents = ti.xcom_pull(task_ids='document_processing.clean_and_validate_json')
    with open(REFINED_PROCESSED_DATA_PATH, 'w') as f:
        json.dump(documents, f, indent=4)
    logger.info("Refined documents saved successfully.")

# Define the DAG
with DAG(
    dag_id='rag_data_preprocessing_dag',
    default_args={
        'owner': 'airflow',
        'start_date': datetime(2024, 10, 22),
        'retries': 1,
        'retry_delay': timedelta(minutes=5),
        'email_on_failure': True,
        'email_on_retry': False,
        'email_on_success': False,
        'email': 'vallimeenaavellaiyan@gmail.com'
    },
    schedule_interval=None,
    catchup=False,
    description='DAG for data preprocessing for the RAG model pipeline',
) as dag:

    # Fetch BigQuery data
    fetch_bigquery_data_op = PythonOperator(
        task_id='fetch_bigquery_data',
        python_callable=fetch_data_from_bigquery_and_save,
        op_kwargs={
            'project_id': os.getenv("GCS_PROJECT_ID"),
            'dataset_id': os.getenv("GCS_DATASET_ID"),
            'table_id': os.getenv("GCS_TABLE_ID"),
            'output_path': PROCESSED_DATA_PATH,
        },
    )

    # Define the Task Group for the RAG Data Preprocessing steps
    with TaskGroup("rag_data_preprocessing", tooltip="RAG Data Preprocessing Steps") as rag_data_preprocessing_group:
        
        load_and_process_data_task_op = PythonOperator(
            task_id='load_and_process_data',
            python_callable=load_and_process_data_task,
        )

        aggregate_data_task_op = PythonOperator(
            task_id='aggregate_data',
            python_callable=aggregate_data_task,
        )

        prepare_documents_task_op = PythonOperator(
            task_id='prepare_documents',
            python_callable=prepare_documents_task,
        )

        save_documents_task_op = PythonOperator(
            task_id='save_documents',
            python_callable=save_documents_task,
        )

        load_and_process_data_task_op >> aggregate_data_task_op >> prepare_documents_task_op >> save_documents_task_op

    # Define the Task Group for Document Processing steps
    with TaskGroup("document_processing", tooltip="Document Processing Steps") as document_processing_group:

        load_documents_task_op = PythonOperator(
            task_id='load_documents',
            python_callable=load_documents_task,
        )

        process_documents_task_op = PythonOperator(
            task_id='process_documents',
            python_callable=process_documents_task,
        )

        clean_and_validate_json_task_op = PythonOperator(
            task_id='clean_and_validate_json',
            python_callable=clean_and_validate_json_task,
        )

        save_refined_documents_task_op = PythonOperator(
            task_id='save_refined_documents',
            python_callable=save_refined_documents_task,
        )

        load_documents_task_op >> process_documents_task_op >> clean_and_validate_json_task_op >> save_refined_documents_task_op

    # Set dependencies for the two task groups
    fetch_bigquery_data_op >> rag_data_preprocessing_group >> document_processing_group

    # Trigger the second DAG after all tasks in the first DAG are complete
    save_refined_documents_task_op >> TriggerDagRunOperator(
        task_id='trigger_second_dag',
        trigger_dag_id='flatten_and_embeddings_dag',  # The ID of the second DAG
        conf={"key": "value"},  
        wait_for_completion=False,  # Wait for the second DAG to finish before continuing
        dag=dag
    )
