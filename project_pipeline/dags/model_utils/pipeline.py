def run_and_monitor_pipeline(SERVICE_ACCOUNT_KEY_PATH,GCP_PROJECT="amazonreviewssentimentanalysis",BUCKET_NAME="model-deployment-from-airflow",GCP_REGION="us-central1"): 
    import kfp
    from kfp.v2 import dsl
    from kfp.v2.dsl import component, Input, Output, Dataset, Artifact
    from google.cloud import storage
    import os
    from google.cloud import aiplatform
    from google.oauth2 import service_account

    # # Set Service Account Key Path
    # SERVICE_ACCOUNT_KEY_PATH = "path/to/your-service-account-key.json"  # Replace with your key file path
    
    # Authenticate using service account key
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = SERVICE_ACCOUNT_KEY_PATH
    # Initialize credentials
    credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_KEY_PATH)

    # Environment Variables
    # GCP_PROJECT = "amazonreviewssentimentanalysis"
    # GCP_REGION = "us-central1"
    # BUCKET_NAME = "arsa_model_deployment_uscentral_v2"
    DATA_PATH = f"gs://{BUCKET_NAME}/input/labeled_data_1perc.csv"
    OUTPUT_DIR = f"gs://{BUCKET_NAME}/output/data/"
    CODE_BUCKET_PATH = f"gs://{BUCKET_NAME}/code"
    SOURCE_CODE = f"gs://{BUCKET_NAME}/code/src"
    SLICE_METRIC_PATH = f"gs://{BUCKET_NAME}/output/metrics"
    MODEL_SAVE_PATH = f"gs://{BUCKET_NAME}/output/models/final_model.pth"
    VERSION = 1
    APP_NAME = "review_sentiment_bert_model"

    MODEL_DISPLAY_NAME = f"{APP_NAME}-v{VERSION}"
    MODEL_DESCRIPTION = "PyTorch serve deployment model for Amazon reviews classification"

    health_route = "/ping"
    predict_route = f"/predictions/{APP_NAME}"
    serving_container_ports = [7080]

    PROJECT_ID = "amazonreviewssentimentanalysis" 
    APP_NAME = "review_sentiment_bert_model"
    DOCKER_IMAGE_NAME = "pytorch_predict_{APP_NAME}"
    CUSTOM_PREDICTOR_IMAGE_URI = f"gcr.io/{PROJECT_ID}/pytorch_predict_{APP_NAME}"

    # Initialize Google Cloud Storage client with credentials
    client = storage.Client(project=GCP_PROJECT, credentials=credentials)
    bucket = client.bucket(BUCKET_NAME)

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

        # Logging setup
        import logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

        # Download code from GCS
        client = storage.Client()
        bucket = client.bucket(code_bucket_path.split('/')[2])
        prefix = '/'.join(code_bucket_path.split('/')[3:])
        blobs = client.list_blobs(bucket, prefix=prefix)

        code_dir = "/tmp/code"
        os.makedirs(code_dir, exist_ok=True)
        ALLOWED_EXTENSIONS = {".py", ".json", ".yaml", ".csv", ".pkl"}

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

        prepare_data_module = load_module_from_file(f"{code_dir}/prepare_data.py")
        train_df, val_df, test_df, label_encoder = prepare_data_module.split_and_save_data(input_path, output_dir)
        train_df.to_pickle(train_data.path)
        val_df.to_pickle(val_data.path)
        test_df.to_pickle(test_data.path)
        # label_encoder.to_pickle(label_encoder_data.path)
        logger.info("Artifacts for train, dev, and test data created successfully.")

    @component(
        # packages_to_install=["torch", "google-cloud-storage", "transformers", "pandas", "scikit-learn", "gcsfs","accelerate"],
        packages_to_install=[
            "pandas",
            "torch==1.12.1",  # PyTorch version 1.12.1, verified to work with transformers and accelerate
            "transformers==4.21.0",  # Compatible with PyTorch 1.12
            "scikit-learn",
            "accelerate==0.12.0",  # Compatible with PyTorch 1.12 and transformers
            "google-cloud-storage",
            # "kfp==2.0.0",  # Compatible version of kfp
            "PyYAML>=6.0",  # A stable version compatible with the other libraries
            "tensorboard",
        ],
        base_image="us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-12.py310:latest",  # Python 3.10 with GPU support
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


        # Logging setup
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        # Initialize Accelerator
        accelerator = Accelerator()

        # Check available device
        logger.info(f"Using device: {accelerator.device}")

        # Download code from GCS
        client = storage.Client()
        bucket = client.bucket(code_bucket_path.split('/')[2])
        prefix = '/'.join(code_bucket_path.split('/')[3:])
        blobs = client.list_blobs(bucket, prefix=prefix)

        code_dir = "/tmp/code"
        os.makedirs(code_dir, exist_ok=True)
        ALLOWED_EXTENSIONS = {".py", ".json", ".yaml", ".csv", ".pkl"}

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

        train_save_module = load_module_from_file(f"{code_dir}/train_save.py")
        hyperparameters_path = os.path.join(code_dir, "best_hyperparameters.json")

        returned_model_path, epoch_metrics = train_save_module.train_and_save_final_model(
            hyperparameters=train_save_module.load_hyperparameters(hyperparameters_path),
            data_path=data_path,
            train_data = train_data,
            val_data = val_data, 
            model_save_path=model_save_path,
        )

        # model_file
        # = os.listdir(model_file)
        # for file_name in model_file:
        #     local_path = os.path.join(model_save_path, file_name)
        #     blob_path = f"output/models/{file_name}"
        #     blob = bucket.blob(blob_path)
        #     blob.upload_from_filename(local_path)
        #     logger.info(f"Uploaded {local_path} to gs://{bucket.name}/{blob_path}")

        model.metadata["gcs_path"] = returned_model_path
        logger.info(f"Model artifact metadata updated with GCS path: {returned_model_path}")

        print(epoch_metrics)
        logger.info(f"epoch_metrics: {epoch_metrics}")
        # Log metrics to the Vertex AI UI
        # Corrected logging for Vertex AI
        for epoch, metric in enumerate(epoch_metrics, start=1):
            # Log accuracy and loss (ensure keys match)
            model_metrics.log_metric(f"epoch_{epoch}_accuracy", metric["eval_accuracy"])
            model_metrics.log_metric(f"epoch_{epoch}_loss", metric["eval_loss"])
            model_metrics.log_metric(f"epoch_{epoch}_precision", metric["eval_precision"])
            model_metrics.log_metric(f"epoch_{epoch}_recall", metric["eval_recall"])
            model_metrics.log_metric(f"epoch_{epoch}_f1", metric["eval_f1"])

            # metrics.log_metric(f"epoch_{epoch}_runtime", metric["eval_runtime"])
            # metrics.log_metric(f"epoch_{epoch}_samples_per_second", metric["eval_samples_per_second"])
            # metrics.log_metric(f"epoch_{epoch}_steps_per_second", metric["eval_steps_per_second"])

            # Log to standard output
            logger.info(f"Logged metrics for epoch {epoch}: {metric}")

    @component(
        packages_to_install=[
            "torch==1.12.1",  # PyTorch version 1.12.1, verified to work with transformers and accelerate
            "transformers==4.21.0",  # Compatible with PyTorch 1.12
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
        # f1_score: Output[float],
        f1_threshold: float = 0.6,
    )-> NamedTuple("output", [("eval_pass", str)]):
        import logging
        import json
        import importlib.util
        from google.cloud import storage
        import os
        import sys


        # Logging setup
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

        # Download code from GCS
        client = storage.Client()
        bucket = client.bucket(code_bucket_path.split('/')[2])
        prefix = '/'.join(code_bucket_path.split('/')[3:])
        blobs = client.list_blobs(bucket, prefix=prefix)

        code_dir = "/tmp/code"
        os.makedirs(code_dir, exist_ok=True)
        ALLOWED_EXTENSIONS = {".py", ".json", ".yaml", ".csv", ".pkl"}

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

        # Ensure `evaluate_model.py` exists
        evaluate_script_path = os.path.join(code_dir, "evaluate_model.py")
        if not os.path.exists(evaluate_script_path):
            raise FileNotFoundError(f"`evaluate_model.py` not found in {code_dir}")

        # Load `evaluate_model.py` dynamically
        evaluate_module = load_module_from_file(evaluate_script_path)

        logger.info(f"model_gcs_path : {model_gcs_path},\t model_gcs_path.uri {model_gcs_path.uri}, metadata {model_gcs_path.metadata['gcs_path']}")
        # Call `gcp_eval` method from the module
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
            return (eval_pass,)
        else:
            logger.error(f"Model failed to meet the F1 threshold: {f1:.4f} < {f1_threshold}")
            eval_pass = "false"
            return (eval_pass,)

            # raise ValueError(f"F1 score {f1:.4f} is below the threshold {f1_threshold}")

    @component(
        packages_to_install=[
            "torch==1.12.1",  # PyTorch version 1.12.1, verified to work with transformers and accelerate
            "transformers==4.21.0",  # Compatible with PyTorch 1.12
            "pandas",
            "scikit-learn",
            "google-cloud-storage",
            "gcsfs",
        ],
        base_image="python:3.9",
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


        # Logging setup
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

        # Download code from GCS
        client = storage.Client()
        bucket = client.bucket(code_bucket_path.split('/')[2])
        prefix = '/'.join(code_bucket_path.split('/')[3:])
        blobs = client.list_blobs(bucket, prefix=prefix)

        code_dir = "/tmp/code"
        os.makedirs(code_dir, exist_ok=True)
        ALLOWED_EXTENSIONS = {".py", ".json", ".yaml", ".csv", ".pkl"}

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
        # # Log metrics to Vertex AI
        # metrics.log_metric("accuracy", accuracy)
        # metrics.log_metric("precision", precision)
        # metrics.log_metric("recall", recall)
        # metrics.log_metric("f1", f1)

        # # Conditional check
        # if f1 >= f1_threshold:
        #     logger.info(f"Model passed the F1 threshold: {f1:.4f} >= {f1_threshold}")
        # else:
        #     logger.error(f"Model failed to meet the F1 threshold: {f1:.4f} < {f1_threshold}")
        #     raise ValueError(f"F1 score {f1:.4f} is below the threshold {f1_threshold}")

    @component(
        packages_to_install=[
            "torch==1.12.1",  # PyTorch version 1.12.1, verified to work with transformers and accelerate
            "transformers==4.21.0",  # Compatible with PyTorch 1.12
            "pandas",
            "scikit-learn",
            "google-cloud-storage",
            "gcsfs",
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


        # Logging setup
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

        # Download code from GCS
        client = storage.Client()
        bucket = client.bucket(code_bucket_path.split('/')[2])
        prefix = '/'.join(code_bucket_path.split('/')[3:])
        blobs = client.list_blobs(bucket, prefix=prefix)

        code_dir = "/tmp/code"
        os.makedirs(code_dir, exist_ok=True)
        ALLOWED_EXTENSIONS = {".py", ".json", ".yaml", ".csv", ".pkl"}

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
        code_bucket_path: str,  # This is the missing input parameter
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
        model_gcs_path = f"gs://arsa_model_deployment_uscentral_v2/output/models/"
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
        packages_to_install=["google-cloud-aiplatform", "google-auth"],
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
        health_route: str = "/ping",
        predict_route: str = "/predictions/",
        serving_container_ports: list = [7080],
    ) -> NamedTuple("Outputs", [("model_display_name", str), ("model_resource_name", str), ("model_version", str)]):
        """Uploads the model to the AI platform and ensures versioning."""
        from google.cloud import aiplatform

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

        # Return output information
        return (model.display_name, model.resource_name, model_version)

    @dsl.pipeline(
        name="data-prep-and-train",
        pipeline_root=f"gs://{BUCKET_NAME}/pipeline_root/",
    )
    def data_prep_and_train_pipeline():
        # Step 1: Data Preparation
        data_prep_task = data_prep_stage(
            code_bucket_path=SOURCE_CODE,
            input_path=DATA_PATH,
            output_dir=OUTPUT_DIR,
        )

        # Step 2: Training and Saving Model
        train_save_task = train_save_stage(
            code_bucket_path=SOURCE_CODE,
            data_path=OUTPUT_DIR,
            model_save_path=MODEL_SAVE_PATH,
            train_data=data_prep_task.outputs["train_data"],
            val_data=data_prep_task.outputs["val_data"],
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
        )

        # Conditional Logic: Check if eval passed
        with dsl.If(evaluate_task.outputs["eval_pass"] == "true", name="conditional-validation-check"):
            # Step 4: Evaluate Slices
            evaluate_slices_task = evaluate_slices_component(
                code_bucket_path=SOURCE_CODE,
                model_gcs_path=train_save_task.outputs["model"], 
                test_data=data_prep_task.outputs["test_data"],  
                gcs_artifact_path=SLICE_METRIC_PATH,
                f1_threshold=0.6,
            )

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
            # Task dependencies within the successful branch


                upload_model_task = upload_model_to_registry(
                    project_id=PROJECT_ID,
                    region=GCP_REGION,
                    bucket_name=BUCKET_NAME,
                    model_display_name=MODEL_DISPLAY_NAME,
                    docker_image_uri=CUSTOM_PREDICTOR_IMAGE_URI,
                    model_description=MODEL_DESCRIPTION,
                    app_name=APP_NAME,
                )
                build_and_push_torchserve_image_op.after(bias_detect_task)
                upload_model_task.after(build_and_push_torchserve_image_op)


        # Loop Back: If F1 < 0.6
        # with dsl.If(evaluate_task.outputs["f1_score"]  <= 0.6):
        #     train_save_task.after(evaluate_task)

    from kfp.v2.compiler import Compiler
    from google.cloud import aiplatform

    # Define the pipeline file path
    pipeline_file_path = "data_prep_and_train_pipeline.json"

    # Compile the pipeline
    Compiler().compile(pipeline_func=data_prep_and_train_pipeline, package_path=pipeline_file_path)

    # Initialize Vertex AI
    aiplatform.init(project=GCP_PROJECT, location=GCP_REGION)

    # Submit the pipeline to Vertex AI
    pipeline_job = aiplatform.PipelineJob(
        display_name="data-prep-and-train-pipeline",
        template_path=pipeline_file_path,
        pipeline_root=f"gs://{BUCKET_NAME}/pipeline_root/",
    )

    # pipeline_job.submit()
    pipeline_job.run(sync=True)



if __name__ == '__main__':
    run_and_monitor_pipeline(SERVICE_ACCOUNT_KEY_PATH='/home/ssd/Desktop/Project/Amazon-Reviews-Sentiment-Analysis/project_pipeline/config/amazonreviewssentimentanalysis-8dfde6e21c1d.json')