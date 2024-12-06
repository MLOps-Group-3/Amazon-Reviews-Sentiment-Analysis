"""
This script orchestrates the training, evaluation, and deployment of a machine learning model 
using a Kubeflow pipeline, Google Cloud Storage (GCS), and Vertex AI.

### Key Components:
1. **Pipeline Definition:** 
    - The pipeline is defined in `dsl_pipeline.py` using reusable components from `dsl_components.py`.
    - Handles data preparation, hyperparameter optimization, model training, evaluation, bias detection, and deployment.

2. **GCS Integration:**
    - Uploads local folders to GCS for pipeline execution using `upload_folder_to_gcs`.

3. **Vertex AI Integration:**
    - Compiles and submits the pipeline to Vertex AI for execution using `submit_vertex_ai_pipeline`.

### Main Functions:
1. **`upload_folder_to_gcs`:**
    Uploads a local folder to GCS, preserving its structure and preparing it for pipeline execution.

2. **`submit_vertex_ai_pipeline`:**
    Compiles the pipeline into a JSON file and submits it to Vertex AI for execution.

3. **`main`:**
    - Uploads the source code folder to GCS.
    - Submits the pipeline for execution on Vertex AI.

4. **`run_pipeline`:**
    - Authenticates using a service account.
    - Uploads the Airflow DAGs and other source files to GCS.
    - Submits the pipeline to Vertex AI.

"""

import os
from typing import NamedTuple

import kfp
from kfp.v2 import dsl
from kfp.v2.dsl import component, Input, Output, Dataset, Artifact
from google.cloud import storage
import os
from google.cloud import aiplatform
from google.oauth2 import service_account
import logging
import sys
import json
from kfp.v2.compiler import Compiler
from google.cloud import aiplatform

from model_utils.dsl_components import *
from model_utils.dsl_pipeline import *

from model_utils.model_config import *

def upload_folder_to_gcs(local_folder, bucket, destination_folder):
    """
    Uploads a local folder to a Google Cloud Storage (GCS) bucket.

    Args:
        local_folder (str): Path to the local folder to upload.
        bucket (google.cloud.storage.bucket.Bucket): GCS bucket object.
        destination_folder (str): Destination folder in the GCS bucket, as a GCS path or relative path.
    """    
    # Strip the `gs://<bucket_name>/` prefix from the destination path
    if destination_folder.startswith(f"gs://{bucket.name}/"):
        destination_folder = destination_folder[len(f"gs://{bucket.name}/"):]
    # logging.info("HERE")
    for root, _, files in os.walk(local_folder):
        for file in files:

            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, local_folder)
            logging.info(local_path,relative_path)

            gcs_path = os.path.join(destination_folder,"src",relative_path).replace("\\", "/")
            blob = bucket.blob(gcs_path)
            blob.upload_from_filename(local_path)
            logging.info(f"Uploaded {local_path} to gs://{bucket.name}/{gcs_path}")

def submit_vertex_ai_pipeline(model_pipeline, GCP_PROJECT, GCP_REGION, BUCKET_NAME, pipeline_file_path="model_pipeline.json"):
    """
    Function to compile and submit a pipeline to Vertex AI.
    
    Args:
        model_pipeline: The pipeline function to compile and submit.
        GCP_PROJECT: Google Cloud project ID.
        GCP_REGION: Google Cloud region for Vertex AI.
        BUCKET_NAME: Google Cloud Storage bucket name for pipeline root.
        pipeline_file_path: Optional path to save the compiled pipeline JSON file (default: 'model_pipeline.json').
    """
    # Compile the pipeline
    logging.info("Compiling pipeline...")
    Compiler().compile(pipeline_func=model_pipeline, package_path=pipeline_file_path)

    # Initialize Vertex AI
    logging.info("Initializing Vertex AI...")
    aiplatform.init(project=GCP_PROJECT, location=GCP_REGION)

    # Submit the pipeline job to Vertex AI
    logging.info("Submitting pipeline job to Vertex AI...")
    pipeline_job = aiplatform.PipelineJob(
        display_name="model-pipeline",
        template_path=pipeline_file_path,
        pipeline_root=f"gs://{BUCKET_NAME}/pipeline_root/",
    )

    # Run the pipeline
    logging.info("Running the pipeline...")
    pipeline_job.run(sync=True)
    logging.info("Pipeline job submitted.")

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    client = storage.Client(project=GCP_PROJECT)
    bucket = client.bucket(BUCKET_NAME)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)  

    logging.info(f"CWD:{os.getcwd()}")

    # Define the local folder and GCS destination path
    local_folder = "src/"  # The local folder to upload
    CODE_BUCKET_PATH = f"gs://{BUCKET_NAME}/code"  # GCS destination folder

    # Step 1: Upload the folder to GCS
    logging.info(f"Uploading folder to GCS...,\n local_folder: {local_folder}, CODE_BUCKET_PATH:{CODE_BUCKET_PATH}")
    upload_folder_to_gcs(local_folder, bucket, CODE_BUCKET_PATH)
    logging.info("Uploaded to GCS")

    logging.info("Submitting pipeline to Vertex AI...")
    submit_vertex_ai_pipeline(model_pipeline, GCP_PROJECT, GCP_REGION, BUCKET_NAME)
    logging.info("Pipeline submission completed")


def run_pipeline():
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Authenticate using service account key
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GCS_SERVICE_ACCOUNT_KEY
    # Initialize credentials
    credentials = service_account.Credentials.from_service_account_file(
        GCS_SERVICE_ACCOUNT_KEY
    )

    # Initialize GCS client
    client = storage.Client(project=GCP_PROJECT,credentials=credentials)
    bucket = client.bucket(BUCKET_NAME)

    # Step 1: Upload folder to GCS
    logging.info(f"Uploading folder to GCS...,\n local_folder: {AIRFLOW_LOCAL}, CODE_BUCKET_PATH:{CODE_BUCKET_PATH}")
    upload_folder_to_gcs(AIRFLOW_LOCAL, bucket, CODE_BUCKET_PATH)
    logging.info("Uploaded to GCS")

    # Step 2: Submit the pipeline to Vertex AI
    logging.info("Submitting pipeline to Vertex AI...")
    submit_vertex_ai_pipeline(model_pipeline, GCP_PROJECT, GCP_REGION, BUCKET_NAME)
    logging.info("Pipeline submission completed.")

if __name__ == '__main__':
    main()
