"""
This module defines a Kubeflow pipeline for training, evaluating, and deploying a machine learning model.

The pipeline performs the following steps:
1. **Data Preparation:** Splits the input dataset into training, validation, and test datasets.
2. **Hyperparameter Optimization:** Runs an Optuna experiment to find the best hyperparameters for model training.
3. **Model Training:** Trains a model using the best hyperparameters and saves the resulting model.
4. **Model Evaluation:** Evaluates the trained model and compares it with the latest archived model.
5. **Slice Evaluation:** Analyzes model performance across data slices to detect any performance inconsistencies.
6. **Bias Detection:** Identifies potential biases in the model using the slice evaluation metrics.
7. **Containerization:** Builds and pushes a TorchServe-compatible Docker image for model serving.
8. **Model Deployment:** Uploads the model to Google Cloud AI Platform, ensuring proper versioning and archiving.

Pipeline Highlights:
- **Modular Design:** The pipeline is built using reusable components (`dsl_components`) to promote modularity and reusability.
- **Conditional Logic:** Conditional checks are included to handle dynamic flows based on model evaluation and bias detection results.
- **Integration with GCP:** Integrates with Google Cloud services like AI Platform, Cloud Build, and Cloud Storage for seamless operations.
- **Resource Optimization:** Supports configuration of compute resources (CPU, memory, GPU) for each pipeline step.

Environment Variables:
- `SOURCE_CODE`: Path to the source code bucket for the pipeline components.
- `DATA_PATH`: Path to the input dataset.
- `OUTPUT_DIR`: Directory to store intermediate outputs.
- `MODEL_SAVE_PATH`: Path to save the trained model.
- `MODEL_ARCHIVE_PATH`: Path to archive older model versions.
- `SLICE_METRIC_PATH`: Path to store slice evaluation metrics.
- `GCP_PROJECT`: Google Cloud project ID.
- `GCP_REGION`: Region for AI Platform and other GCP services.
- `BUCKET_NAME`: Google Cloud Storage bucket name for pipeline artifacts.
- `MODEL_DISPLAY_NAME`: Display name for the model on AI Platform.
- `CUSTOM_PREDICTOR_IMAGE_URI`: URI of the custom Docker image for model serving.
- `MODEL_DESCRIPTION`: Description of the model for documentation purposes.
- `APP_NAME`: Application name for routing in the serving container.

Key Functions:
- **`data_split`:** Splits the dataset into training, validation, and test subsets.
- **`run_optuna_experiment`:** Optimizes hyperparameters using Optuna.
- **`train_save_stage`:** Trains the model and saves the resulting artifact.
- **`evaluate_model_component`:** Evaluates the model against the test dataset.
- **`load_latest_model`:** Loads the latest archived model for comparison.
- **`evaluate_slices_component`:** Evaluates model performance across data slices.
- **`bias_detect_component`:** Detects potential biases in the model based on slice metrics.
- **`build_and_push_torchserve_image`:** Builds and pushes a Docker image compatible with TorchServe.
- **`upload_model_to_registry`:** Uploads the model to AI Platform and manages versioning.

Usage:
1. Update the environment variables and GCS paths in the `model_config` module.
2. Ensure all required components are defined in the `dsl_components` module.
3. Compile the pipeline and run it using the Kubeflow Pipelines UI or CLI.

This pipeline automates the ML lifecycle, including training, evaluation, bias detection, and deployment, with robust integration with Google Cloud services.
"""
import kfp
from kfp.v2 import dsl
from google.cloud import storage
import os
from google.cloud import aiplatform
from google.oauth2 import service_account
from model_utils.model_config import *
from model_utils.dsl_components import (
    data_split,
    run_optuna_experiment,
    train_save_stage,
    evaluate_model_component,
    load_latest_model,
    evaluate_slices_component,
    bias_detect_component,
    build_and_push_torchserve_image,
    upload_model_to_registry
)


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
    ).set_cpu_limit("8") 
    # \
    # .set_gpu_limit(1) \
    # .set_accelerator_type("NVIDIA_TESLA_T4")

    optuna_experiment_task.after(data_prep_task)
    train_save_task.after(optuna_experiment_task)
    data_prep_task.set_caching_options(False)
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

