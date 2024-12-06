import kfp
from kfp.v2 import dsl
from kfp.v2.dsl import component, Input, Output, Dataset, Artifact
from google.cloud import storage
import os
from google.cloud import aiplatform
import logging
from datetime import datetime
import time
from kfp.v2.compiler import Compiler
from google.cloud import aiplatform

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Environment Variables
GCP_PROJECT = "amazonreviewssentimentanalysis"
GCP_REGION = "us-central1"
BUCKET_NAME = "arsa_model_deployment_uscentral_v2"
DATA_PATH = f"gs://{BUCKET_NAME}/input/labeled_data_1perc.csv"
OUTPUT_DIR = f"gs://{BUCKET_NAME}/output/data/"
CODE_BUCKET_PATH = f"gs://{BUCKET_NAME}/code"
SOURCE_CODE = f"gs://{BUCKET_NAME}/code/src"
SLICE_METRIC_PATH = f"gs://{BUCKET_NAME}/output/metrics"
MODEL_SAVE_PATH = f"gs://{BUCKET_NAME}/output/models/final_model.pth"
VERSION = 1
APP_NAME = "review_sentiment_bert_model"
MODEL_DISPLAY_NAME = f"{APP_NAME}-v{VERSION}"
MODEL_DESCRIPTION = "PyTorch serve deploymend model for amazon reviews classification"
health_route = "/ping"
predict_route = f"/predictions/{APP_NAME}"
serving_container_ports = [7080]
PROJECT_ID = "amazonreviewssentimentanalysis"
APP_NAME = "review_sentiment_bert_model"
DOCKER_IMAGE_NAME = "pytorch_predict_{APP_NAME}"
CUSTOM_PREDICTOR_IMAGE_URI = f"gcr.io/{PROJECT_ID}/pytorch_predict_{APP_NAME}"

# Initialize Google Cloud Storage client
client = storage.Client(project=GCP_PROJECT)
bucket = client.bucket(BUCKET_NAME)

# Function to upload folder to GCS
def upload_folder_to_gcs(local_folder, bucket, destination_folder):
    logger.info(f"Starting upload of folder {local_folder} to GCS")
    start_time = time.time()
    
    try:
        if destination_folder.startswith(f"gs://{bucket.name}/"):
            destination_folder = destination_folder[len(f"gs://{bucket.name}/"):]

        for root, _, files in os.walk(local_folder):
            for file in files:
                local_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_path, local_folder)
                logger.info(f"Uploading file: {local_path}")
                gcs_path = os.path.join(destination_folder, local_path).replace("\\", "/")
                blob = bucket.blob(gcs_path)
                blob.upload_from_filename(local_path)
                logger.info(f"Uploaded {local_path} to gs://{bucket.name}/{gcs_path}")
        
        end_time = time.time()
        logger.info(f"Folder upload completed. Time taken: {end_time - start_time:.2f} seconds")
    except Exception as e:
        logger.error(f"Error occurred during folder upload: {str(e)}")
        raise

upload_folder_to_gcs("src", bucket, CODE_BUCKET_PATH)

from kfp.v2.dsl import (
    Input,
    Output,
    Artifact,
    Dataset,
    Model,
    Metrics,
    component,
    pipeline,
)
from typing import NamedTuple

@component(
    packages_to_install=["pandas", "scikit-learn", "google-cloud-storage", "torch", "gcsfs"],
)
def data_prep_stage(
    code_bucket_path: str,
    input_path: str,
    output_dir: str,
    train_data: Output[Dataset],
    val_data: Output[Dataset],
    test_data: Output[Dataset],
):
    import os
    import sys
    import importlib.util
    import pandas as pd
    from google.cloud import storage
    import logging
    import time

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Starting data preparation stage")
    start_time = time.time()

    try:
        client = storage.Client()
        bucket = client.bucket(code_bucket_path.split('/')[2])
        prefix = '/'.join(code_bucket_path.split('/')[3:])
        blobs = client.list_blobs(bucket, prefix=prefix)
        code_dir = "/tmp/code"
        os.makedirs(code_dir, exist_ok=True)
        ALLOWED_EXTENSIONS = {".py", ".json", ".yaml", ".csv", ".pkl"}

        logger.info("Downloading code from GCS")
        for blob in blobs:
            if any(blob.name.endswith(ext) for ext in ALLOWED_EXTENSIONS):
                relative_path = blob.name[len(prefix):].lstrip("/")
                file_path = os.path.join(code_dir, relative_path)
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                blob.download_to_filename(file_path)
                logger.info(f"Downloaded {blob.name} to {file_path}")

        logger.info(f"Files in {code_dir}: {os.listdir(code_dir)}")
        sys.path.insert(0, code_dir)

        def load_module_from_file(file_path):
            module_name = os.path.splitext(os.path.basename(file_path))[0]
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module

        logger.info("Loading prepare_data module")
        prepare_data_module = load_module_from_file(f"{code_dir}/prepare_data.py")

        logger.info("Splitting and saving data")
        train_df, val_df, test_df, label_encoder = prepare_data_module.split_and_save_data(input_path, output_dir)

        logger.info("Saving data to pickle files")
        train_df.to_pickle(train_data.path)
        val_df.to_pickle(val_data.path)
        test_df.to_pickle(test_data.path)

        logger.info("Artifacts for train, dev, and test data created successfully.")
        
        end_time = time.time()
        logger.info(f"Data preparation stage completed. Time taken: {end_time - start_time:.2f} seconds")
    except Exception as e:
        logger.error(f"Error occurred during data preparation: {str(e)}")
        raise

@component(
    packages_to_install=[
        "pandas",
        "torch==1.12.1",
        "transformers==4.21.0",
        "scikit-learn",
        "accelerate==0.12.0",
        "google-cloud-storage",
        "PyYAML>=6.0",
        "tensorboard",
    ],
    base_image="us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-12.py310:latest",
)
def train_save_stage(
    code_bucket_path: str,
    data_path: str,
    model_save_path: str,
    train_data: Input[Dataset],
    val_data: Input[Dataset],
    model: Output[Model],
    model_metrics: Output[Metrics],
):
    import os
    import sys
    import logging
    from google.cloud import storage
    import importlib.util
    from accelerate import Accelerator
    import time

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Starting train and save stage")
    start_time = time.time()

    try:
        accelerator = Accelerator()
        logger.info(f"Using device: {accelerator.device}")

        client = storage.Client()
        bucket = client.bucket(code_bucket_path.split('/')[2])
        prefix = '/'.join(code_bucket_path.split('/')[3:])
        blobs = client.list_blobs(bucket, prefix=prefix)
        code_dir = "/tmp/code"
        os.makedirs(code_dir, exist_ok=True)
        ALLOWED_EXTENSIONS = {".py", ".json", ".yaml", ".csv", ".pkl"}

        logger.info("Downloading code from GCS")
        for blob in blobs:
            if any(blob.name.endswith(ext) for ext in ALLOWED_EXTENSIONS):
                relative_path = blob.name[len(prefix):].lstrip("/")
                file_path = os.path.join(code_dir, relative_path)
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                blob.download_to_filename(file_path)
                logger.info(f"Downloaded {blob.name} to {file_path}")

        logger.info(f"Files in {code_dir}: {os.listdir(code_dir)}")
        sys.path.insert(0, code_dir)

        def load_module_from_file(file_path):
            module_name = os.path.splitext(os.path.basename(file_path))[0]
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module

        logger.info("Loading train_save module")
        train_save_module = load_module_from_file(f"{code_dir}/train_save.py")

        hyperparameters_path = os.path.join(code_dir, "best_hyperparameters.json")
        
        logger.info("Starting model training and saving")
        returned_model_path, epoch_metrics = train_save_module.train_and_save_final_model(
            hyperparameters=train_save_module.load_hyperparameters(hyperparameters_path),
            data_path=data_path,
            train_data=train_data,
            val_data=val_data,
            model_save_path=model_save_path,
        )

        model.metadata["gcs_path"] = returned_model_path
        logger.info(f"Model artifact metadata updated with GCS path: {returned_model_path}")

        logger.info(f"epoch_metrics: {epoch_metrics}")

        logger.info("Logging metrics to Vertex AI UI")
        for epoch, metric in enumerate(epoch_metrics, start=1):
            model_metrics.log_metric(f"epoch_{epoch}_accuracy", metric["eval_accuracy"])
            model_metrics.log_metric(f"epoch_{epoch}_loss", metric["eval_loss"])
            model_metrics.log_metric(f"epoch_{epoch}_precision", metric["eval_precision"])
            model_metrics.log_metric(f"epoch_{epoch}_recall", metric["eval_recall"])
            model_metrics.log_metric(f"epoch_{epoch}_f1", metric["eval_f1"])
            logger.info(f"Logged metrics for epoch {epoch}: {metric}")

        end_time = time.time()
        logger.info(f"Train and save stage completed. Time taken: {end_time - start_time:.2f} seconds")
    except Exception as e:
        logger.error(f"Error occurred during train and save stage: {str(e)}")
        raise

@component(
    packages_to_install=[
        "torch==1.12.1",
        "transformers==4.21.0",
        "pandas",
        "scikit-learn",
        "google-cloud-storage",
        "gcsfs",
    ],
    base_image="python:3.9",
)
def evaluate_model_component(
    code_bucket_path: str,
    model_gcs_path: Input[Model],
    test_data: Input[Dataset],
    eval_metrics: Output[Metrics],
    f1_threshold: float = 0.6,
) -> NamedTuple("output", [("eval_pass", str)]):
    import logging
    import json
    import importlib.util
    from google.cloud import storage
    import os
    import sys
    import time

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Starting model evaluation")
    start_time = time.time()

    try:
        client = storage.Client()
        bucket = client.bucket(code_bucket_path.split('/')[2])
        prefix = '/'.join(code_bucket_path.split('/')[3:])
        blobs = client.list_blobs(bucket, prefix=prefix)
        code_dir = "/tmp/code"
        os.makedirs(code_dir, exist_ok=True)
        ALLOWED_EXTENSIONS = {".py", ".json", ".yaml", ".csv", ".pkl"}

        logger.info("Downloading code from GCS")
        for blob in blobs:
            if any(blob.name.endswith(ext) for ext in ALLOWED_EXTENSIONS):
                relative_path = blob.name[len(prefix):].lstrip("/")
                file_path = os.path.join(code_dir, relative_path)
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                blob.download_to_filename(file_path)
                logger.info(f"Downloaded {blob.name} to {file_path}")

        logger.info(f"Files in {code_dir}: {os.listdir(code_dir)}")
        sys.path.insert(0, code_dir)

        def load_module_from_file(file_path):
            module_name = os.path.splitext(os.path.basename(file_path))[0]
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module

        evaluate_script_path = os.path.join(code_dir, "evaluate_model.py")
        if not os.path.exists(evaluate_script_path):
            raise FileNotFoundError(f"`evaluate_model.py` not found in {code_dir}")

        logger.info("Loading evaluate_model module")
        evaluate_module = load_module_from_file(evaluate_script_path)

        logger.info(f"model_gcs_path : {model_gcs_path}, model_gcs_path.uri {model_gcs_path.uri}, metadata {model_gcs_path.metadata['gcs_path']}")

        logger.info("Starting model evaluation")
        accuracy, precision, recall, f1 = evaluate_module.gcp_eval(
            test_df=test_data,
            model_path=model_gcs_path.metadata["gcs_path"],
        )

        logger.info("Logging metrics to Vertex AI")
        eval_metrics.log_metric("accuracy", accuracy)
        eval_metrics.log_metric("precision", precision)
        eval_metrics.log_metric("recall", recall)
        eval_metrics.log_metric("f1", f1)

        if f1 >= f1_threshold:
            logger.info(f"Model passed the F1 threshold: {f1:.4f} >= {f1_threshold}")
            eval_pass = "true"
        else:
            logger.info(f"Model failed to meet the F1 threshold: {f1:.4f} < {f1_threshold}")
            eval_pass = "false"

        logger.info(f"Evaluation metrics: Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

        end_time = time.time()
        logger.info(f"Model evaluation completed. Time taken: {end_time - start_time:.2f} seconds")

        from collections import namedtuple
        output = namedtuple("output", ["eval_pass"])
        return output(eval_pass)

    except Exception as e:
        logger.error(f"Error occurred during model evaluation: {str(e)}")
        raise

@component(
    packages_to_install=[
        "google-cloud-aiplatform",
        "google-cloud-storage",
    ],
    base_image="python:3.9",
)
def deploy_model(
    project: str,
    region: str,
    model_display_name: str,
    model_description: str,
    artifact_uri: str,
    serving_container_image_uri: str,
    serving_container_predict_route: str,
    serving_container_health_route: str,
    serving_container_ports: list,
) -> NamedTuple("Outputs", [("model", Artifact), ("endpoint", Artifact)]):
    import logging
    from google.cloud import aiplatform
    import time

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Starting model deployment")
    start_time = time.time()

    try:
        aiplatform.init(project=project, location=region)

        logger.info(f"Creating model with display name: {model_display_name}")
        model = aiplatform.Model.upload(
            display_name=model_display_name,
            description=model_description,
            artifact_uri=artifact_uri,
            serving_container_image_uri=serving_container_image_uri,
            serving_container_predict_route=serving_container_predict_route,
            serving_container_health_route=serving_container_health_route,
            serving_container_ports=serving_container_ports,
        )
        logger.info(f"Model created: {model.resource_name}")

        logger.info("Deploying model to endpoint")
        endpoint = model.deploy(
            machine_type="n1-standard-4",
            min_replica_count=1,
            max_replica_count=1,
            accelerator_type=None,
            accelerator_count=None,
        )
        logger.info(f"Model deployed to endpoint: {endpoint.resource_name}")

        end_time = time.time()
        logger.info(f"Model deployment completed. Time taken: {end_time - start_time:.2f} seconds")

        from collections import namedtuple

        output = namedtuple("Outputs", ["model", "endpoint"])
        return output(model.resource_name, endpoint.resource_name)

    except Exception as e:
        logger.error(f"Error occurred during model deployment: {str(e)}")
        raise

@pipeline(name="sentiment-analysis-pipeline", description="Sentiment Analysis Pipeline")
def sentiment_analysis_pipeline(
    project_id: str,
    region: str,
    data_path: str,
    output_dir: str,
    code_bucket_path: str,
    model_save_path: str,
    model_display_name: str,
    model_description: str,
    serving_container_image_uri: str,
    serving_container_predict_route: str,
    serving_container_health_route: str,
    serving_container_ports: list,
):
    data_prep_task = data_prep_stage(
        code_bucket_path=code_bucket_path,
        input_path=data_path,
        output_dir=output_dir,
    )

    train_save_task = train_save_stage(
        code_bucket_path=code_bucket_path,
        data_path=output_dir,
        model_save_path=model_save_path,
        train_data=data_prep_task.outputs["train_data"],
        val_data=data_prep_task.outputs["val_data"],
    )

    evaluate_task = evaluate_model_component(
        code_bucket_path=code_bucket_path,
        model_gcs_path=train_save_task.outputs["model"],
        test_data=data_prep_task.outputs["test_data"],
    )

    with dsl.Condition(evaluate_task.outputs["eval_pass"] == "true", name="deploy_model"):
        deploy_task = deploy_model(
            project=project_id,
            region=region,
            model_display_name=model_display_name,
            model_description=model_description,
            artifact_uri=model_save_path,
            serving_container_image_uri=serving_container_image_uri,
            serving_container_predict_route=serving_container_predict_route,
            serving_container_health_route=serving_container_health_route,
            serving_container_ports=serving_container_ports,
        )

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Starting pipeline compilation")
    start_time = time.time()

    try:
        Compiler().compile(
            pipeline_func=sentiment_analysis_pipeline,
            package_path="sentiment_analysis_pipeline.json",
        )

        end_time = time.time()
        logger.info(f"Pipeline compilation completed. Time taken: {end_time - start_time:.2f} seconds")

        logger.info("Starting pipeline job")
        job_start_time = time.time()

        job = aiplatform.PipelineJob(
            display_name="sentiment-analysis-pipeline",
            template_path="sentiment_analysis_pipeline.json",
            pipeline_root=OUTPUT_DIR,
            parameter_values={
                "project_id": GCP_PROJECT,
                "region": GCP_REGION,
                "data_path": DATA_PATH,
                "output_dir": OUTPUT_DIR,
                "code_bucket_path": CODE_BUCKET_PATH,
                "model_save_path": MODEL_SAVE_PATH,
                "model_display_name": MODEL_DISPLAY_NAME,
                "model_description": MODEL_DESCRIPTION,
                "serving_container_image_uri": CUSTOM_PREDICTOR_IMAGE_URI,
                "serving_container_predict_route": predict_route,
                "serving_container_health_route": health_route,
                "serving_container_ports": serving_container_ports,
            },
        )

        # job.submit()
        job.run(sync=True)

        job_end_time = time.time()
        logger.info(f"Pipeline job submitted. Time taken: {job_end_time - job_start_time:.2f} seconds")

    except Exception as e:
        logger.error(f"Error occurred during pipeline execution: {str(e)}")
        raise

