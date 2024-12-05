import os

def run_and_monitor_pipeline(
    SERVICE_ACCOUNT_KEY_PATH,
    GCP_PROJECT="amazonreviewssentimentanalysis",
    BUCKET_NAME="model-deployment-from-airflow",
    GCP_REGION="us-central1",
):
    import kfp
    from kfp.v2 import dsl
    from kfp.v2.dsl import component, Input, Output, Dataset, Artifact
    from google.cloud import storage
    import os
    from google.cloud import aiplatform
    from google.oauth2 import service_account
    import logging

    # # Set Service Account Key Path
    # SERVICE_ACCOUNT_KEY_PATH = "path/to/your-service-account-key.json"  # Replace with your key file path

    # Authenticate using service account key
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = SERVICE_ACCOUNT_KEY_PATH
    # Initialize credentials
    credentials = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_KEY_PATH
    )

    # Environment Variables
    # GCP_PROJECT = "amazonreviewssentimentanalysis"
    # GCP_REGION = "us-central1"
    # BUCKET_NAME = "arsa_model_deployment_uscentral_v2"
    DATA_PATH = f"gs://{BUCKET_NAME}/input/labeled_data.csv"
    OUTPUT_DIR = f"gs://{BUCKET_NAME}/output/data/"
    CODE_BUCKET_PATH = f"gs://{BUCKET_NAME}/code"
    SOURCE_CODE = f"gs://{BUCKET_NAME}/code/src"
    SLICE_METRIC_PATH = f"gs://{BUCKET_NAME}/output/metrics"
    MODEL_SAVE_PATH = f"gs://{BUCKET_NAME}/output/models/train_job/final_model.pth"
    MODEL_ARCHIVE_PATH = f"gs://{BUCKET_NAME}/output/models/archive/"

    VERSION = 1
    # APP_NAME = "review_sentiment_bert_model"
    APP_NAME="review_sentiment_bert_model"

    MODEL_DISPLAY_NAME = f"{APP_NAME}-v{VERSION}"
    MODEL_DESCRIPTION = (
        "PyTorch serve deployment model for Amazon reviews classification"
    )

    health_route = "/ping"
    predict_route = f"/predictions/{APP_NAME}"
    serving_container_ports = [7080]

    PROJECT_ID = GCP_PROJECT
    DOCKER_IMAGE_NAME = f"pytorch_predict_{APP_NAME}"
    CUSTOM_PREDICTOR_IMAGE_URI = f"gcr.io/{PROJECT_ID}/{DOCKER_IMAGE_NAME}"
    # CUSTOM_PREDICTOR_IMAGE_URI = f"gcr.io/{PROJECT_ID}/pytorch_predict_{APP_NAME}"

    # Initialize Google Cloud Storage client with credentials
    client = storage.Client(project=GCP_PROJECT, credentials=credentials)
    bucket = client.bucket(BUCKET_NAME)

    # def upload_folder_to_gcs(local_folder, bucket, destination_folder):
    #     import os
    #     from google.cloud import storage

    #     for root, _, files in os.walk(local_folder):
    #         for file in files:
    #             local_path = os.path.join(root, file)
    #             relative_path = os.path.relpath(local_path, local_folder)
    #             gcs_path = os.path.join(destination_folder, relative_path).replace("\\", "/")

    #             print(f"Uploading {local_path} to {gcs_path} in bucket {bucket.name}")
    #             blob = bucket.blob(gcs_path)
    #             blob.upload_from_filename(local_path)
    #             print(f"Uploaded {local_path} to gs://{bucket.name}/{gcs_path}")

    def upload_folder_to_gcs(local_folder, bucket, destination_folder):
        # Strip the `gs://<bucket_name>/` prefix from the destination path
        if destination_folder.startswith(f"gs://{bucket.name}/"):
            destination_folder = destination_folder[len(f"gs://{bucket.name}/"):]

        for root, _, files in os.walk(local_folder):
            for file in files:
                local_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_path, local_folder)
                print(local_path,relative_path)

                gcs_path = os.path.join(destination_folder,"src",relative_path).replace("\\", "/")
                blob = bucket.blob(gcs_path)
                blob.upload_from_filename(local_path)
                print(f"Uploaded {local_path} to gs://{bucket.name}/{gcs_path}")


    #  absolute path to the `src` folder in Airflow
    local_folder = "/opt/airflow/dags/model_utils/src/"

    logging.info(f"current working directory: {os.getcwd()}")

    upload_folder_to_gcs(local_folder, bucket, CODE_BUCKET_PATH)
    logging.info("Uploaded to GCS")

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
        packages_to_install=["pandas", "scikit-learn", "google-cloud-storage", "torch", "gcsfs","arsa-pipeline-tools"],
        # base_image="python:3.9",
    )
    def data_split(
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
        from arsa_pipeline_tools.utils import download_files_from_gcs, load_module_from_file
        # Logging setup
        import logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

        code_dir = "/tmp/code"
        os.makedirs(code_dir, exist_ok=True)

        download_files_from_gcs(code_bucket_path,code_dir)
        logger.info(f"Files in {code_dir}: {os.listdir(code_dir)}")
        sys.path.insert(0, code_dir)

        prepare_data_module = load_module_from_file(f"{code_dir}/prepare_data.py")
        train_df, val_df, test_df, label_encoder = prepare_data_module.split_and_save_data(input_path, output_dir)
        train_df.to_pickle(train_data.path)
        val_df.to_pickle(val_data.path)
        test_df.to_pickle(test_data.path)
        # label_encoder.to_pickle(label_encoder_data.path)
        logger.info("Artifacts for train, dev, and test data created successfully.")


    @component(
        packages_to_install=[
            "optuna",
            "mlflow",
            "torch",
            "transformers[torch]",  # Compatible with PyTorch 1.12
            "numpy",
            "google-cloud-storage",
            "scikit-learn",
            "arsa-pipeline-tools",
        ],
        base_image="us-docker.pkg.dev/deeplearning-platform-release/gcr.io/base-cu118.py310",  # Python 3.10 with GPU support

    )
    def run_optuna_experiment(
        code_bucket_path: str,
        data_path: str,
        train_data: Input[Dataset],
        val_data: Input[Dataset],
        test_data: Input[Dataset],
        best_hyperparams_metrics: Output[Metrics],
    ):
        import os
        import sys
        import importlib.util
        import logging
        from google.cloud import storage
        from arsa_pipeline_tools.utils import download_files_from_gcs, load_module_from_file

        # Logging setup
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

        code_dir = "/tmp/code"
        os.makedirs(code_dir, exist_ok=True)
        ALLOWED_EXTENSIONS = {".py", ".json", ".yaml"}

        download_files_from_gcs(code_bucket_path,code_dir,ALLOWED_EXTENSIONS)

        logger.info(f"Files in {code_dir}: {os.listdir(code_dir)}")
        sys.path.insert(0, code_dir)

        # Ensure `experiment_runner_optuna.py` exists
        script_path = os.path.join(code_dir, "experiment_runner_optuna.py")
        if not os.path.exists(script_path):
            raise FileNotFoundError(f"`experiment_runner_optuna.py` not found in {code_dir}")

        # Load and execute the experiment
        experiment_module = load_module_from_file(script_path)

        # Run the Optuna experiment
        best_hyperparameters = experiment_module.find_best_hyperparameters(data_path)
        logger.info(best_hyperparameters)
        # Save the best hyperparameters to the output artifact
        # Log hyperparameters to Metrics artifact
        for key, value in best_hyperparameters.items():
            best_hyperparams_metrics.log_metric(key, value)

    @component(
        packages_to_install=[
            "torch",
            "transformers[torch]",  
            "scikit-learn",
            "numpy",        
            "pandas",        
            # "accelerate==0.12.0",  
            "google-cloud-storage",
            # "kfp==2.0.0", 
            "PyYAML>=6.0",  
            "tensorboard",
            "arsa-pipeline-tools",
        ],
        base_image="us-docker.pkg.dev/deeplearning-platform-release/gcr.io/base-cu118.py310",  # Python 3.10 with GPU support

    )
    def train_save_stage(
        code_bucket_path: str,
        data_path: str,
        model_save_path: str,
        train_data: Input[Dataset],
        val_data: Input[Dataset],
        best_hyperparams_metrics: Input[Metrics],
        model: Output[Model],
        model_metrics: Output[Metrics],
        

    ):
        import os
        import sys
        import logging
        from google.cloud import storage
        import importlib.util
        from accelerate import Accelerator
        from arsa_pipeline_tools.utils import download_files_from_gcs, load_module_from_file


        # Logging setup
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        # Initialize Accelerator
        accelerator = Accelerator()
        
        # Check available device
        logger.info(f"Using device: {accelerator.device}")


        code_dir = "/tmp/code"
        os.makedirs(code_dir, exist_ok=True)
        ALLOWED_EXTENSIONS = {".py", ".json", ".yaml", ".csv", ".pkl"}

        download_files_from_gcs(code_bucket_path,code_dir,ALLOWED_EXTENSIONS)

        logger.info(f"Files in {code_dir}: {os.listdir(code_dir)}")
        sys.path.insert(0, code_dir)

        train_save_module = load_module_from_file(f"{code_dir}/train_save.py")
        # hyperparameters_path = os.path.join(code_dir, "best_hyperparameters.json")

        best_hyperparams = {key: value for key, value in best_hyperparams_metrics.metadata.items()}
        logger.info(f"Read best hyperparameters from metrics: {best_hyperparams}")

        returned_model_path, epoch_metrics = train_save_module.train_and_save_final_model(
            hyperparameters=best_hyperparams, 
            data_path=data_path,
            train_data = train_data,
            val_data = val_data, 
            model_save_path=model_save_path,
        )


        model.metadata["gcs_path"] = returned_model_path
        logger.info(f"Model artifact metadata updated with GCS path: {returned_model_path}")

        print(epoch_metrics)
        logger.info(f"epoch_metrics: {epoch_metrics}")
        # Log metrics to the Vertex AI UI
        for epoch, metric in enumerate(epoch_metrics, start=1):
            # Log accuracy and loss (ensure keys match)
            model_metrics.log_metric(f"epoch_{epoch}_accuracy", metric["eval_accuracy"])
            model_metrics.log_metric(f"epoch_{epoch}_loss", metric["eval_loss"])
            model_metrics.log_metric(f"epoch_{epoch}_precision", metric["eval_precision"])
            model_metrics.log_metric(f"epoch_{epoch}_recall", metric["eval_recall"])
            model_metrics.log_metric(f"epoch_{epoch}_f1", metric["eval_f1"])
            # Log to standard output
            logger.info(f"Logged metrics for epoch {epoch}: {metric}")

    @component(
        packages_to_install=[
            "google-cloud-storage",
            "gcsfs",
        ],
        base_image="python:3.9",  # Python 3.10 with GPU support
    )
    def load_latest_model(
        model_archive_path: str,
        archive_model: Output[Model]

    ):
        from google.cloud import storage
        import logging
        from urllib.parse import urlparse

        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

        # Parse the GCS path
        if not model_archive_path.startswith("gs://"):
            raise ValueError(f"Invalid GCS path: {model_archive_path}")

        parsed_url = urlparse(model_archive_path)
        bucket_name = parsed_url.netloc  # Extract bucket name
        prefix = parsed_url.path.lstrip("/")  # Extract the prefix, stripping leading "/"

        # Initialize the GCS client
        client = storage.Client()
        bucket = client.bucket(bucket_name)

        # List all the blobs (files) in the specified GCS folder (model_archive_path)
        blobs = list(bucket.list_blobs(prefix=prefix))

        if not blobs:
            logger.error(f"No models found in the specified GCS path: {model_archive_path}")
            archive_model.metadata["gcs_path"] = None
        else:
                
            # Find the latest model by comparing the 'updated' timestamp of each blob
            latest_blob = max(blobs, key=lambda blob: blob.updated)

            # Return the path to the latest model
            latest_model_path = f"gs://{bucket_name}/{latest_blob.name}"

            logger.info(latest_model_path)
            archive_model.metadata["gcs_path"] = latest_model_path


    @component(
        packages_to_install=[
            "torch",  # PyTorch version 1.12.1, verified to work with transformers and accelerate
            "transformers[torch]",  # Compatible with PyTorch 1.12
            "pandas",
            "numpy",
            "scikit-learn",
            "google-cloud-storage",
            "gcsfs",
            "arsa-pipeline-tools",
        ],
        # base_image="python:3.9",  # Python 3.10 with GPU support
        base_image="us-docker.pkg.dev/deeplearning-platform-release/gcr.io/base-cu118.py310",  # Python 3.10 with GPU support

    )
    def evaluate_model_component(
        code_bucket_path: str,
        model_gcs_path: Input[Model],
        test_data: Input[Dataset],
        eval_metrics: Output[Metrics],
        # f1_score: Output[float],
        f1_threshold: float = 0.6,
    )-> NamedTuple("output", [("eval_pass", str),("f1_score", float)]):
        import logging
        import json
        import importlib.util
        from google.cloud import storage
        import os
        import sys
        from arsa_pipeline_tools.utils import download_files_from_gcs, load_module_from_file


        # Logging setup
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)


        code_dir = "/tmp/code"
        os.makedirs(code_dir, exist_ok=True)
        ALLOWED_EXTENSIONS = {".py", ".json", ".yaml", ".csv", ".pkl"}
        download_files_from_gcs(code_bucket_path,code_dir,ALLOWED_EXTENSIONS)


        logger.info(f"Files in {code_dir}: {os.listdir(code_dir)}")
        sys.path.insert(0, code_dir)

        # Ensure `evaluate_model.py` exists
        evaluate_script_path = os.path.join(code_dir, "evaluate_model.py")
        if not os.path.exists(evaluate_script_path):
            raise FileNotFoundError(f"`evaluate_model.py` not found in {code_dir}")

        # Load `evaluate_model.py` dynamically
        evaluate_module = load_module_from_file(evaluate_script_path)

        logger.info(f"model_gcs_path : {model_gcs_path},\t model_gcs_path.uri {model_gcs_path.uri}, metadata {model_gcs_path.metadata['gcs_path']}")
        # Call `gcp_eval` method from the module
        if model_gcs_path.metadata["gcs_path"]==None:
            eval_pass = "false"
            f1 = 0.0
            return (eval_pass,f1)
        accuracy, precision, recall, f1 = evaluate_module.gcp_eval(
            test_df=test_data,
            model_path=model_gcs_path.metadata["gcs_path"],
        )

        # Log metrics to Vertex AI
        eval_metrics.log_metric("accuracy", accuracy)
        eval_metrics.log_metric("precision", precision)
        eval_metrics.log_metric("recall", recall)
        eval_metrics.log_metric("f1", f1)
        # Conditional check
        if f1 >= f1_threshold:
            logger.info(f"Model passed the F1 threshold: {f1:.4f} >= {f1_threshold}")
            eval_pass = "true"
            return (eval_pass,f1)
        else:
            logger.error(f"Model failed to meet the F1 threshold: {f1:.4f} < {f1_threshold}")
            eval_pass = "false"
            return (eval_pass,f1)

    @component(
        packages_to_install=[
            "torch",  # PyTorch version 1.12.1, verified to work with transformers and accelerate
            "transformers[torch]",  # Compatible with PyTorch 1.12
            "pandas",
            "numpy",
            "scikit-learn",
            "google-cloud-storage",
            "gcsfs",
            "arsa-pipeline-tools",
        ],
        # base_image="python:3.9",  # Python 3.10 with GPU support
        base_image="us-docker.pkg.dev/deeplearning-platform-release/gcr.io/base-cu118.py310",  # Python 3.10 with GPU support

    )
    def evaluate_slices_component(
        code_bucket_path: str,
        model_gcs_path: Input[Model],
        test_data: Input[Dataset],
        eval_slices_metrics: Output[Metrics],
        gcs_artifact_path: str,
        f1_threshold: float = 0.6,
    ):
        import logging
        import json
        import importlib.util
        from google.cloud import storage
        import os
        import sys
        from arsa_pipeline_tools.utils import download_files_from_gcs, load_module_from_file


        # Logging setup
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)


        code_dir = "/tmp/code"
        os.makedirs(code_dir, exist_ok=True)
        ALLOWED_EXTENSIONS = {".py", ".json", ".yaml", ".csv", ".pkl"}

        download_files_from_gcs(code_bucket_path,code_dir,ALLOWED_EXTENSIONS)
        logger.info(f"Files in {code_dir}: {os.listdir(code_dir)}")
        sys.path.insert(0, code_dir)

        # Ensure `evaluate_module_slices.py` exists
        evaluate_script_path = os.path.join(code_dir, "evaluate_model_slices.py")
        if not os.path.exists(evaluate_script_path):
            raise FileNotFoundError(f"`evaluate_module_slices.py` not found in {code_dir}")

        # Load `evaluate_module_slices.py` dynamically
        evaluate_module_slices = load_module_from_file(evaluate_script_path)

        logger.info(f"model_gcs_path : {model_gcs_path},\t model_gcs_path.uri {model_gcs_path.uri}, metadata {model_gcs_path.metadata['gcs_path']}")
        # Call `gcp_eval` method from the module
        metrics_df = evaluate_module_slices.gcp_eval_slices(
            test_df=test_data,
            model_path=model_gcs_path.metadata["gcs_path"],
        )
        logger.info(metrics_df)

        gcs_bucket_name = gcs_artifact_path.split('/')[2]
        gcs_blob_path = '/'.join(gcs_artifact_path.split('/')[3:])
        csv_filename = f"{gcs_blob_path}/slice_metrics.csv"
        json_filename = f"{gcs_blob_path}/slice_metrics.json"
        
        client = storage.Client()
        bucket = client.bucket(gcs_bucket_name)

        # Save as CSV
        csv_blob = bucket.blob(csv_filename)
        csv_blob.upload_from_string(metrics_df.to_csv(index=False), content_type="text/csv")
        logger.info(f"Slice metrics saved to GCS as CSV at gs://{gcs_bucket_name}/{csv_filename}")

        # Save as JSON
        json_blob = bucket.blob(json_filename)
        json_blob.upload_from_string(metrics_df.to_json(orient="records"), content_type="application/json")
        logger.info(f"Slice metrics saved to GCS as JSON at gs://{gcs_bucket_name}/{json_filename}")

        # Log paths of the artifacts in metrics
        eval_slices_metrics.metadata["slice_metrics_csv"] = f"gs://{gcs_bucket_name}/{csv_filename}"
        eval_slices_metrics.metadata["slice_metrics_json"] = f"gs://{gcs_bucket_name}/{json_filename}"

    @component(
        packages_to_install=[
            "torch",  # PyTorch version 1.12.1, verified to work with transformers and accelerate
            "transformers[torch]",  # Compatible with PyTorch 1.12
            "pandas",
            "scikit-learn",
            "google-cloud-storage",
            "gcsfs",
            "arsa-pipeline-tools",
        ],
        base_image="python:3.9",
    )
    def bias_detect_component(
        code_bucket_path: str,
        metrics: Input[Metrics],
        gcs_artifact_path: str,
    )-> NamedTuple("output", [("bias_detect", str)]):
        import logging
        import json
        import importlib.util
        from google.cloud import storage
        import os
        import sys
        from arsa_pipeline_tools.utils import download_files_from_gcs, load_module_from_file



        # Logging setup
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

        # Download code from GCS

        code_dir = "/tmp/code"
        os.makedirs(code_dir, exist_ok=True)
        ALLOWED_EXTENSIONS = {".py", ".json", ".yaml", ".csv", ".pkl"}
        
        download_files_from_gcs(code_bucket_path,code_dir,ALLOWED_EXTENSIONS)
        logger.info(f"Files in {code_dir}: {os.listdir(code_dir)}")
        sys.path.insert(0, code_dir)

        # Ensure `evaluate_module_slices.py` exists
        evaluate_script_path = os.path.join(code_dir, "bias_detect.py")
        if not os.path.exists(evaluate_script_path):
            raise FileNotFoundError(f"`evaluate_module_slices.py` not found in {code_dir}")

        # Load `evaluate_module_slices.py` dynamically
        bias_detect = load_module_from_file(evaluate_script_path)


        # Call `gcp_eval` method from the module
        biased_rows, f1_threshold = bias_detect.detect_bias(
            slice_metrics_path=metrics.metadata["slice_metrics_csv"],
        )
        bias_report = {}

        # Log results
        if not biased_rows.empty:
            bias_report["bias_detected"] = True
            bias_report["details"] = biased_rows.to_dict(orient="records")

            logger.warning("Potential bias detected in the following slices:")
            for _, row in biased_rows.iterrows():
                logger.error(
                    f"Slice Column: {row['Slice Column']}, Slice Value: {row['Slice Value']}, "
                    f"Samples: {row['Samples']}, F1 Score: {row['F1 Score']:.4f} (Threshold: {f1_threshold:.4f})"
                )
            logger.error("Potential bias detected. Check bias_detection.log for details.")
        else:
            bias_report["bias_detected"] = False
            bias_report["details"] = []
            logger.info("No significant bias detected.")
            # print("No significant bias detected.")

        # Save bias report as JSON to GCS
        gcs_bucket_name = gcs_artifact_path.split('/')[2]
        gcs_blob_path = '/'.join(gcs_artifact_path.split('/')[3:])
        bias_json_path = f"{gcs_blob_path}/bias.json"

        try:
            client = storage.Client()
            bucket = client.bucket(gcs_bucket_name)
            blob = bucket.blob(bias_json_path)
            blob.upload_from_string(json.dumps(bias_report, indent=4), content_type="application/json")
            logging.info(f"Bias report saved to GCS at gs://{gcs_bucket_name}/{bias_json_path}")
        except Exception as e:
            logging.error(f"Failed to save bias report to GCS: {e}")
            raise

        # bias_metrics.log_metric("bias_detected",)
        
        # Raise an error if bias is detected to stop the pipeline
        if bias_report["bias_detected"]:
            # bias_metrics.log_metric("bias_detected",True)
            bias_detect = "true"
            return (bias_detect,)
            # raise RuntimeError("Bias detected in slice metrics. Stopping the pipeline.")
        else:
            bias_detect = "false"
            return (bias_detect,)
            # bias_metrics.log_metric("bias_detected",False)


    @component(
        packages_to_install=["google-cloud-storage", "google-cloud-build"],
        base_image="python:3.9",
    )
    def build_and_push_torchserve_image(
        code_bucket_path: str,
        gcp_project: str, 
        gcp_region: str, 
        bucket_name: str, 
        docker_image_name: str,
        model_gcs_path: Input[Model]
    ):
        # Import inside the component
        from google.cloud.devtools import cloudbuild_v1 as cloudbuild
        from google.cloud import storage
        import logging
        import os
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

        # Define environment variables
        TORCH_SERVE_PATH = f"gs://{bucket_name}/code/predictor/"
        CUSTOM_PREDICTOR_IMAGE_URI = f"gcr.io/{gcp_project}/{docker_image_name}"

        # Set up the CloudBuild client
        client = cloudbuild.CloudBuildClient()

        # Log the environment variables for debugging
        logger.info(f"GCP Project: {gcp_project}")
        logger.info(f"GCP Region: {gcp_region}")
        logger.info(f"Bucket Name: {bucket_name}")
        logger.info(f"TorchServe Path: {TORCH_SERVE_PATH}")
        logger.info(f"Docker Image Name: {docker_image_name}")
        logger.info(f"Custom Docker Image URI: {CUSTOM_PREDICTOR_IMAGE_URI}")
            
        model_gcs_path = model_gcs_path.metadata["gcs_path"]
        logger.info(model_gcs_path)
        model_gcs_path = f"gs://{bucket_name}/output/models/train_job/"
        # Create Cloud Build configuration (cloudbuild.yaml)
        cloudbuild_config = {
            'steps': [
                # Step 1: Download code files from GCS
                {
                    "name": "gcr.io/cloud-builders/gsutil",
                    "args": [
                        'cp',
                        '-r',  # Recursive copy
                        f'{code_bucket_path}/*',  # Copy all contents from the code folder
                        '.'  # Copy to the current working directory
                    ],
                },
                # Step 2: Create the destination directory for model files
                {
                    "name": "ubuntu",
                    "args": [
                        "mkdir",
                        "-p",  # Create parent directories as needed
                        "./bert-sent-model"
                    ],
                },
                # Step 3: Download model files from GCS
                {
                    "name": "gcr.io/cloud-builders/gsutil",
                    "args": [
                        'cp',
                        '-r',  # Recursive copy
                        f'{model_gcs_path}*',  # Add wildcard to include all files in the folder
                        './bert-sent-model/'  # Ensure the trailing slash
                    ],
                },
                # Step 3: List files in the current working directory
                {
                    "name": "ubuntu",
                    "args": [
                        "ls",
                        "-R",  # Recursive listing
                        "."    # Current working directory
                    ],
                },
                # Step 4: Build the Docker image
                {
                    'name': 'gcr.io/cloud-builders/docker',
                    'args': [
                        'build',
                        '-t',
                        CUSTOM_PREDICTOR_IMAGE_URI,
                        '.'
                    ],
                },
                # Step 5: Push the Docker image to the container registry
                {
                    'name': 'gcr.io/cloud-builders/docker',
                    'args': [
                        'push',
                        CUSTOM_PREDICTOR_IMAGE_URI
                    ],
                },
            ],
            'images': [CUSTOM_PREDICTOR_IMAGE_URI],
        }

        # Create a Cloud Build build request
        build = cloudbuild.Build(
            steps=cloudbuild_config['steps'],
            images=cloudbuild_config['images'],
        )

        # Trigger Cloud Build job
        build_response = client.create_build(project_id=gcp_project, build=build)

        logging.info("IN PROGRESS:")
        logging.info(build_response.metadata)

        # get build status
        result = build_response.result()
        logging.info("RESULT:", result.status)

    @component(
        packages_to_install=["google-cloud-aiplatform", "google-auth","google-cloud-storage","gcsfs"],
        base_image="python:3.9",
    )
    def upload_model_to_registry(
        project_id: str,
        region: str,
        bucket_name: str,
        model_display_name: str,
        docker_image_uri: str,
        model_description: str,
        app_name: str,
        model_save_path: str,
        model_archive_path: str,
        health_route: str = "/ping",
        predict_route: str = "/predictions/",
        serving_container_ports: list = [7080],
    ) -> NamedTuple("Outputs", [("model_display_name", str), ("model_resource_name", str), ("model_version", str)]):
        """Uploads the model to the AI platform and ensures versioning."""
        from google.cloud import aiplatform
        from google.cloud import storage
        import logging
        import time
        import os
        import sys
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

        # Initialize the AI Platform
        aiplatform.init(project=project_id, location=region, staging_bucket=bucket_name)

        # Check if the model with the same display name exists
        existing_models = aiplatform.Model.list(filter=f"display_name={model_display_name}")
        
        if existing_models:
            # Model exists, register as a new version
            model_resource_name = existing_models[0].resource_name
            print(f"Model with display name '{model_display_name}' exists. Registering as a new version.")
            model_version = f"v{len(existing_models) + 1}"  # Increment version number
        else:
            # Model does not exist, create a new one
            model_resource_name = None
            print(f"Model with display name '{model_display_name}' does not exist. Creating a new model.")
            model_version = "v1"

        # Upload the model
        model = aiplatform.Model.upload(
            display_name=model_display_name,
            description=model_description,
            serving_container_image_uri=docker_image_uri,
            serving_container_predict_route=predict_route + app_name,
            serving_container_health_route=health_route,
            serving_container_ports=serving_container_ports,
            parent_model=model_resource_name,  # Register under an existing model if applicable
        )

        model.wait()

        logger.info(f"model.display_name: {model.display_name}, model.resource_name: {model.resource_name}, model_version: {model_version}")
        # Return output information

        try:
            # Initialize GCS client
            client = storage.Client()

            # Extract bucket and blob paths from the GCS paths
            model_bucket_name, model_blob_path = model_save_path.replace("gs://", "").split("/", 1)
            archive_bucket_name, archive_blob_path = model_archive_path.replace("gs://", "").split("/", 1)

            logger.info(f"Source model path: {model_save_path}")
            logger.info(f"Destination archive path: {model_archive_path}")

            # Get the source bucket and blob objects
            model_bucket = client.get_bucket(model_bucket_name)
            model_blob = model_bucket.blob(model_blob_path)

            # Add timestamp to the model file name for versioning
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            new_blob_name = f"{os.path.splitext(os.path.basename(model_blob_path))[0]}_{timestamp}{os.path.splitext(model_blob_path)[1]}"
            archive_bucket = client.get_bucket(archive_bucket_name)
            archive_blob_path = archive_blob_path.rstrip('/')  # Remove trailing slash        
            archive_blob = archive_bucket.blob(f"{archive_blob_path}/{new_blob_name}")

            logger.info(f"Renaming model to: {new_blob_name}")

            # Perform the copy operation
            model_bucket.copy_blob(model_blob, archive_bucket, f"{archive_blob_path}/{new_blob_name}")

            logger.info(f"Model successfully copied and renamed to {new_blob_name}")
            
            # Set the output artifact with the new path
            # model_output.uri = f"gs://{archive_bucket_name}/{archive_blob_path}/{new_blob_name}"

        except Exception as e:
            logger.error(f"An error occurred while copying the model: {str(e)}")
            # raise

        return (model.display_name, model.resource_name, model_version)

    @dsl.pipeline(
        name="model-pipeline",
        pipeline_root=f"gs://{BUCKET_NAME}/pipeline_root/",
    )
    def model_pipeline():
        # Step 1: Data Preparation
        data_prep_task = data_split(
            code_bucket_path=SOURCE_CODE,
            input_path=DATA_PATH,
            output_dir=OUTPUT_DIR,
        )
        
        optuna_experiment_task = run_optuna_experiment(
            code_bucket_path=SOURCE_CODE,
            data_path=DATA_PATH,
            train_data=data_prep_task.outputs["train_data"],
            val_data=data_prep_task.outputs["val_data"],
            test_data=data_prep_task.outputs["test_data"],
        ).set_cpu_limit("8") \
        .set_memory_limit("32G") \
        .set_gpu_limit(1) \
        .set_accelerator_type("NVIDIA_TESLA_T4")


        # Step 2: Training and Saving Model
        train_save_task = train_save_stage(
            code_bucket_path=SOURCE_CODE,
            data_path=OUTPUT_DIR,
            model_save_path=MODEL_SAVE_PATH,
            train_data=data_prep_task.outputs["train_data"],
            val_data=data_prep_task.outputs["val_data"],
            best_hyperparams_metrics=optuna_experiment_task.outputs["best_hyperparams_metrics"],
        ).set_cpu_limit("8") \
        .set_memory_limit("32G") \
        .set_gpu_limit(1) \
        .set_accelerator_type("NVIDIA_TESLA_T4")


        # Step 3: Model Evaluation
        evaluate_task = evaluate_model_component(
            code_bucket_path=SOURCE_CODE,
            model_gcs_path=train_save_task.outputs["model"],  # Pass Model artifact
            test_data=data_prep_task.outputs["test_data"],  # Pass Test Data artifact
            f1_threshold=0.6,
        ).set_cpu_limit("8") \
        .set_gpu_limit(1) \
        .set_accelerator_type("NVIDIA_TESLA_T4")

        load_latest_model_task = load_latest_model(
            model_archive_path=MODEL_ARCHIVE_PATH
        )

        evaluate_archive_task = evaluate_model_component(
            code_bucket_path=SOURCE_CODE,
            model_gcs_path=load_latest_model_task.outputs["archive_model"],  # Pass Model artifact
            test_data=data_prep_task.outputs["test_data"],  # Pass Test Data artifact
            f1_threshold=0.6,
        ).set_cpu_limit("8") \
        .set_gpu_limit(1) \
        .set_accelerator_type("NVIDIA_TESLA_T4")

        optuna_experiment_task.after(data_prep_task)
        train_save_task.after(optuna_experiment_task)
        # data_prep_task.set_caching_options(False)
        load_latest_model_task.set_caching_options(False)

        # Conditional Logic: Check if eval passed
        with dsl.If(evaluate_task.outputs["f1_score"] >= evaluate_archive_task.outputs["f1_score"], name="conditional-validation-check"): #evaluate_archive_task.outputs["f1_score"]
            # Step 4: Evaluate Slices
            evaluate_slices_task = evaluate_slices_component(
                code_bucket_path=SOURCE_CODE,
                model_gcs_path=train_save_task.outputs["model"], 
                test_data=data_prep_task.outputs["test_data"],  
                gcs_artifact_path=SLICE_METRIC_PATH,
                f1_threshold=0.6,
            ).set_cpu_limit("8") \
            .set_gpu_limit(1) \
            .set_accelerator_type("NVIDIA_TESLA_T4")
            
            # Step 5: Bias Detection
            bias_detect_task = bias_detect_component(
                code_bucket_path=SOURCE_CODE,
                metrics=evaluate_slices_task.outputs["eval_slices_metrics"],
                gcs_artifact_path=SLICE_METRIC_PATH,
            )
            evaluate_slices_task.after(evaluate_task)
            bias_detect_task.after(evaluate_slices_task)

            with dsl.If(bias_detect_task.outputs["bias_detect"] == "false", name="bias-check-condtional-deploy"):

                # Step 6: Build and Push TorchServe Image
                build_and_push_torchserve_image_op = build_and_push_torchserve_image(
                    code_bucket_path=SOURCE_CODE, 
                    gcp_project=GCP_PROJECT,
                    gcp_region=GCP_REGION,
                    bucket_name=BUCKET_NAME,
                    docker_image_name="pytorch_predict_review_sentiment_bert_model",
                    model_gcs_path=train_save_task.outputs["model"],
                )

                upload_model_task = upload_model_to_registry(
                    project_id=PROJECT_ID,
                    region=GCP_REGION,
                    bucket_name=BUCKET_NAME,
                    model_display_name=MODEL_DISPLAY_NAME,
                    docker_image_uri=CUSTOM_PREDICTOR_IMAGE_URI,
                    model_description=MODEL_DESCRIPTION,
                    app_name=APP_NAME,
                    model_save_path=MODEL_SAVE_PATH,
                    model_archive_path=MODEL_ARCHIVE_PATH,
                )
                upload_model_task.set_caching_options(False)
                build_and_push_torchserve_image_op.set_caching_options(False)
                build_and_push_torchserve_image_op.after(bias_detect_task)
                upload_model_task.after(build_and_push_torchserve_image_op)


    from kfp.v2.compiler import Compiler
    from google.cloud import aiplatform

    # Define the pipeline file path
    pipeline_file_path = "model_pipeline.json"

    # Compile the pipeline
    Compiler().compile(pipeline_func=model_pipeline, package_path=pipeline_file_path)

    # Initialize Vertex AI
    aiplatform.init(project=GCP_PROJECT, location=GCP_REGION)

    # Submit the pipeline to Vertex AI
    pipeline_job = aiplatform.PipelineJob(
        display_name="model-pipeline",
        template_path=pipeline_file_path,
        pipeline_root=f"gs://{BUCKET_NAME}/pipeline_root/",
    )



    # pipeline_job.submit()
    pipeline_job.run(sync=True)


if __name__ == "__main__":
    run_and_monitor_pipeline(
        SERVICE_ACCOUNT_KEY_PATH="/home/ssd/Desktop/Project/Amazon-Reviews-Sentiment-Analysis/project_pipeline/config/amazonreviewssentimentanalysis-8dfde6e21c1d.json"
    )
