import kfp
from kfp.v2 import dsl
from typing import NamedTuple

# Importing necessary components and types from kfp.v2.dsl
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

@component(
    packages_to_install=["pandas", "scikit-learn", "google-cloud-storage", "torch", "gcsfs","arsa-pipeline-tools"],
    base_image="python:3.7",
)
def data_split(
    code_bucket_path: str,
    input_path: str,
    output_dir: str,
    train_data: Output[Dataset],
    val_data: Output[Dataset],
    test_data: Output[Dataset],

):
    """
    A component to split input data into training, validation, and test datasets,
    and save them as artifacts in a pipeline-friendly format.

    Args:
        code_bucket_path (str): Path to the GCS bucket containing the necessary code files.
        input_path (str): Path to the input data file to be split.
        output_dir (str): Directory path to save the output files locally.
        train_data (Output[Dataset]): Output artifact for the training dataset.
        val_data (Output[Dataset]): Output artifact for the validation dataset.
        test_data (Output[Dataset]): Output artifact for the test dataset.

    Returns:
        None: The function generates and saves the train, validation, and test datasets
        as pipeline artifacts.
    """
    import os
    import sys
    import importlib.util
    import pandas as pd
    from google.cloud import storage
    from arsa_pipeline_tools.utils import download_files_from_gcs, load_module_from_file
    import logging

    # Logging setup
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Create a temporary directory for the code
    code_dir = "/tmp/code"
    os.makedirs(code_dir, exist_ok=True)

    # Download files from the specified GCS bucket
    download_files_from_gcs(code_bucket_path,code_dir)
    logger.info(f"Files in {code_dir}: {os.listdir(code_dir)}")

    # Add the code directory to the system path
    sys.path.insert(0, code_dir)

    # Load the custom data preparation module
    prepare_data_module = load_module_from_file(f"{code_dir}/prepare_data.py")

    # Split and save the data using the module's method    
    train_df, val_df, test_df, label_encoder = prepare_data_module.split_and_save_data(input_path, output_dir)
    
    # Save the train, validation, and test datasets to their respective paths
    train_df.to_pickle(train_data.path)
    val_df.to_pickle(val_data.path)
    test_df.to_pickle(test_data.path)

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
    """
    A component to run an Optuna hyperparameter optimization experiment on the provided data.

    Args:
        code_bucket_path (str): Path to the GCS bucket containing the required experiment scripts.
        data_path (str): Path to the data used for running the experiment.
        train_data (Input[Dataset]): Input artifact containing the training dataset.
        val_data (Input[Dataset]): Input artifact containing the validation dataset.
        test_data (Input[Dataset]): Input artifact containing the test dataset.
        best_hyperparams_metrics (Output[Metrics]): Output artifact to log the best hyperparameters
                                                    and their corresponding metrics.

    Returns:
        None: The function runs the Optuna experiment and logs the best hyperparameters as a pipeline artifact.
        
    Workflow:
        1. Downloads the necessary code files from the specified GCS bucket.
        2. Ensures that the `experiment_runner_optuna.py` script is available.
        3. Loads the custom experiment module from the downloaded files.
        4. Executes the experiment to find the best hyperparameters using Optuna.
        5. Logs the best hyperparameters and their associated metrics to the output artifact.
    """
    
    import os
    import sys
    import importlib.util
    import logging
    from google.cloud import storage
    from arsa_pipeline_tools.utils import download_files_from_gcs, load_module_from_file

    # Logging setup
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Create a temporary directory for the code
    code_dir = "/tmp/code"
    os.makedirs(code_dir, exist_ok=True)
    ALLOWED_EXTENSIONS = {".py", ".json", ".yaml"}

    # Download files from the GCS bucket
    download_files_from_gcs(code_bucket_path,code_dir,ALLOWED_EXTENSIONS)

    # Add the code directory to the system path
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
    """
    A component to train a model using the provided data, save the trained model, and log its performance metrics.

    Args:
        code_bucket_path (str): Path to the GCS bucket containing the code files needed for training.
        data_path (str): Path to the dataset to be used for training and validation.
        model_save_path (str): Path where the trained model will be saved.
        train_data (Input[Dataset]): Input artifact containing the training dataset.
        val_data (Input[Dataset]): Input artifact containing the validation dataset.
        best_hyperparams_metrics (Input[Metrics]): Input artifact with the best hyperparameters for training.
        model (Output[Model]): Output artifact for saving the trained model metadata.
        model_metrics (Output[Metrics]): Output artifact to log the model's performance metrics.

    Returns:
        None: The function trains the model, saves it to the specified path, and logs its metrics.
    
    Workflow:
        1. Downloads code files from the specified GCS bucket and ensures required extensions are present.
        2. Initializes the accelerator to leverage GPU or CPU resources.
        3. Loads a custom training module from the downloaded files.
        4. Reads hyperparameters from the input metrics artifact.
        5. Trains the model and saves it to the specified GCS path.
        6. Logs the model's metadata and epoch-wise performance metrics (accuracy, loss, precision, recall, F1 score).
    """

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

    # Extract best hyperparameters from metrics artifact
    best_hyperparams = {key: value for key, value in best_hyperparams_metrics.metadata.items()}
    logger.info(f"Read best hyperparameters from metrics: {best_hyperparams}")

    # Train model and save it
    returned_model_path, epoch_metrics = train_save_module.train_and_save_final_model(
        hyperparameters=best_hyperparams, 
        data_path=data_path,
        train_data = train_data,
        val_data = val_data, 
        model_save_path=model_save_path,
    )

    # Update model metadata with GCS path
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
    """
    A component to retrieve the latest model artifact from a specified Google Cloud Storage (GCS) path.

    Args:
        model_archive_path (str): The GCS path to the directory containing the archived models.
                                  Must be a valid path starting with "gs://".
        archive_model (Output[Model]): Output artifact to store the metadata of the latest model.

    Returns:
        None: Updates the metadata of the `archive_model` output artifact with the GCS path of the latest model.
    
    Workflow:
        1. Parses the provided GCS path to extract the bucket name and prefix.
        2. Lists all files (blobs) in the specified GCS directory.
        3. Identifies the most recently updated model file.
        4. Stores the GCS path of the latest model in the metadata of the output artifact.
    """

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

        # Update output artifact metadata
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
    
    """
    A component to evaluate a trained model on a test dataset and log evaluation metrics.
    The evaluation checks if the model's F1 score meets a specified threshold.

    Args:
        code_bucket_path (str): Path to the GCS bucket containing the evaluation script (`evaluate_model.py`).
        model_gcs_path (Input[Model]): Input artifact containing the GCS path to the trained model.
        test_data (Input[Dataset]): Input artifact containing the test dataset for evaluation.
        eval_metrics (Output[Metrics]): Output artifact to log evaluation metrics (accuracy, precision, recall, F1 score).
        f1_threshold (float): Threshold for the F1 score to determine if the model passes evaluation. Default is 0.6.

    Returns:
        NamedTuple: A tuple with:
            - eval_pass (str): "true" if the model passes the F1 threshold, "false" otherwise.
            - f1_score (float): The evaluated F1 score of the model.

    Workflow:
        1. Downloads the evaluation script (`evaluate_model.py`) from the specified GCS bucket.
        2. Dynamically loads the script to execute the evaluation logic.
        3. Evaluates the model on the test dataset using the `gcp_eval` method from the script.
        4. Logs evaluation metrics (accuracy, precision, recall, F1 score) to the output artifact.
        5. Compares the F1 score with the threshold and determines the evaluation result.
        6. Returns whether the model passed the evaluation and the F1 score.

    Raises:
        ValueError: If the GCS path of the model is invalid.
    """

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

    """
    A component to evaluate model performance on specific data slices and log the results as metrics.
    The metrics are saved to GCS in both CSV and JSON formats.

    Args:
        code_bucket_path (str): GCS path to the bucket containing the `evaluate_model_slices.py` script.
        model_gcs_path (Input[Model]): Input artifact containing the GCS path of the trained model.
        test_data (Input[Dataset]): Input artifact containing the test dataset.
        eval_slices_metrics (Output[Metrics]): Output artifact to log the slice evaluation metrics.
        gcs_artifact_path (str): GCS path where the slice evaluation artifacts (CSV, JSON) will be saved.
        f1_threshold (float): Threshold for F1 score to evaluate performance (not used directly in this function).

    Returns:
        None: Outputs are logged as GCS artifacts and evaluation metrics.

    Workflow:
        1. Downloads the required evaluation script (`evaluate_model_slices.py`) from the specified GCS bucket.
        2. Dynamically loads the evaluation script to execute the slice evaluation logic.
        3. Evaluates the model on test data slices using the loaded script and generates metrics as a DataFrame.
        4. Saves the slice metrics to GCS in both CSV and JSON formats.
        5. Logs the GCS paths of the artifacts in the `eval_slices_metrics` output.

    Raises:
        ValueError: If the model GCS path is invalid or missing.
    """
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
    """
    A component to detect potential bias in data slices based on metrics.
    Generates a bias report and saves it to Google Cloud Storage (GCS).

    Args:
        code_bucket_path (str): GCS path containing the `bias_detect.py` script.
        metrics (Input[Metrics]): Input artifact containing metrics data for slices, including a path to slice metrics.
        gcs_artifact_path (str): GCS path where the bias detection report will be saved.

    Returns:
        NamedTuple: A tuple with:
            - bias_detect (str): "true" if bias is detected, "false" otherwise.

    Workflow:
        1. Downloads the `bias_detect.py` script from the provided GCS path.
        2. Dynamically loads the `bias_detect.py` script to execute bias detection logic.
        3. Evaluates the slice metrics to identify potential biases.
        4. Logs detailed information about detected biases, including affected slices and their metrics.
        5. Saves the bias detection report as a JSON file to GCS.
        6. Returns "true" if bias is detected, otherwise "false".

    Raises:
        Exception: If the bias detection report cannot be saved to GCS.
    """    
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
    """
    A component to build and push a TorchServe Docker image using Google Cloud Build.

    Args:
        code_bucket_path (str): GCS path to the folder containing the TorchServe predictor code.
        gcp_project (str): Google Cloud project ID.
        gcp_region (str): Google Cloud region for the project.
        bucket_name (str): Name of the GCS bucket to store the model and related files.
        docker_image_name (str): Name for the Docker image to be built and pushed.
        model_gcs_path (Input[Model]): Input artifact containing the GCS path of the trained model.

    Returns:
        None: The function builds and pushes the Docker image to the specified container registry.

    Workflow:
        1. Sets up environment variables and paths for TorchServe and Docker.
        2. Configures a Cloud Build YAML equivalent as Python dictionary for the build steps:
            - Downloads predictor code from GCS.
            - Creates a directory for the model.
            - Downloads the model files from GCS into the directory.
            - Lists the downloaded files for verification.
            - Builds the Docker image using the downloaded files.
            - Pushes the Docker image to the Google Container Registry (GCR).
        3. Triggers the Cloud Build process using the defined configuration.
        4. Monitors the build status and logs progress.

    Raises:
        Exception: Logs any issues encountered during the build or push process.
    """    
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
    """
    Uploads a model to Google Cloud AI Platform, supports versioning, and archives the previous model.

    Args:
        project_id (str): Google Cloud project ID.
        region (str): Google Cloud region for the AI platform.
        bucket_name (str): GCS bucket for staging and AI platform uploads.
        model_display_name (str): Display name of the model in the AI platform.
        docker_image_uri (str): URI of the Docker image containing the serving container.
        model_description (str): Description of the model for documentation purposes.
        app_name (str): Name of the application used in the prediction route.
        model_save_path (str): GCS path where the trained model is saved.
        model_archive_path (str): GCS path where previous models will be archived.
        health_route (str, optional): Health check endpoint of the serving container. Defaults to "/ping".
        predict_route (str, optional): Prediction endpoint of the serving container. Defaults to "/predictions/".
        serving_container_ports (list, optional): List of ports exposed by the serving container. Defaults to [7080].

    Returns:
        NamedTuple: Contains:
            - model_display_name (str): The display name of the uploaded model.
            - model_resource_name (str): The resource name of the uploaded model.
            - model_version (str): The version of the model registered in AI Platform.

    Workflow:
        1. Initializes AI Platform with the specified project, region, and bucket.
        2. Checks if a model with the given display name already exists:
            - If yes, registers the model as a new version under the existing model.
            - If no, creates a new model in AI Platform.
        3. Uploads the model to AI Platform with the provided container configuration.
        4. Archives the previous model version in the specified GCS archive path.
        5. Returns the display name, resource name, and version of the uploaded model.

    Raises:
        ValueError: If an error occurs during the model archiving process.
        Exception: For general issues during AI Platform operations or GCS interactions.

    """    
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
        raise ValueError(f"An error occurred while copying the model: {str(e)}")
        # raise

    return (model.display_name, model.resource_name, model_version)
