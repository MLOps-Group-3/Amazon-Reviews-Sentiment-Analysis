import kfp
from kfp.v2 import dsl
from google.cloud import storage
import os
from google.cloud import aiplatform
from google.oauth2 import service_account

from dsl_components import (
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


GCP_PROJECT="amazonreviewssentimentanalysis"
BUCKET_NAME="model-deployment-from-airflow"
GCP_REGION="us-central1"

DATA_PATH = f"gs://{BUCKET_NAME}/input/labeled_data_1perc.csv"
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

