import os
from google.cloud import aiplatform
from google.cloud.aiplatform import pipeline_jobs
from kfp import dsl
from kfp.dsl import component, Input, Output, Dataset, Model, Metrics, ClassificationMetrics
import mlflow

# Set up Google Cloud project details
PROJECT_ID = "amazonreviewssentimentanalysis"
REGION = "us-east1"
BUCKET_NAME = "amazon-reviews-sentiment-analysis"
PIPELINE_ROOT = f"gs://{BUCKET_NAME}/pipeline/vertex-ai"
DATASET_ID = "5951559684128243712"  # Your dataset ID

# Initialize Vertex AI
aiplatform.init(project=PROJECT_ID, location=REGION)

# Set up MLflow
os.environ["GOOGLE_CLOUD_PROJECT"] = PROJECT_ID
mlflow.set_tracking_uri(f"{PIPELINE_ROOT}/mlflow")

@component(
    packages_to_install=[
        "pandas",
        "scikit-learn",
        "google-cloud-aiplatform",
        "gcsfs",  # Enables reading directly from GCS
        "accelerate>=0.26.0"
    ],
    base_image="python:3.9",
)
def prepare_data(
    project_id: str,
    region: str,
    dataset_id: str,
    train_data: Output[Dataset],
    val_data: Output[Dataset],
    test_data: Output[Dataset],
    class_labels: Output[Dataset],
) -> None:
    # Code remains the same
    pass

@component(
    packages_to_install=[
        "pandas",
        "scikit-learn",
        "torch",
        "transformers",
        "mlflow",
        "accelerate>=0.26.0"
    ],
    base_image="gcr.io/deeplearning-platform-release/pytorch-gpu",  # GPU-compatible image
)
def train_model(
    train_data: Input[Dataset],
    val_data: Input[Dataset],
    class_labels: Input[Dataset],
    model_name: str,
    learning_rate: float,
    batch_size: int,
    num_epochs: int,
    weight_decay: float,
    dropout_rate: float,
    trained_model: Output[Model],
    metrics: Output[Metrics],
) -> None:
    # Training code here
    pass

@component(
    packages_to_install=[
        "pandas",
        "scikit-learn",
        "torch",
        "transformers",
        "mlflow",
        "accelerate>=0.26.0"
    ],
    base_image="gcr.io/deeplearning-platform-release/pytorch-gpu",  # GPU-compatible image
)
def evaluate_model(
    model_path: Input[Model],
    test_data: Input[Dataset],
    class_labels: Input[Dataset],
    metrics: Output[ClassificationMetrics],
) -> None:
    # Evaluation code here
    pass

@dsl.pipeline(
    name="sentiment-analysis-pipeline",
    description="A pipeline for sentiment analysis using BERT",
)
def sentiment_analysis_pipeline(
    project_id: str,
    region: str,
    dataset_id: str,
    model_name: str,
    learning_rate: float,
    batch_size: int,
    num_epochs: int,
    weight_decay: float,
    dropout_rate: float,
):
    prepare_data_task = prepare_data(
        project_id=project_id,
        region=region,
        dataset_id=dataset_id
    )

    train_model_task = train_model(
        train_data=prepare_data_task.outputs["train_data"],
        val_data=prepare_data_task.outputs["val_data"],
        class_labels=prepare_data_task.outputs["class_labels"],
        model_name=model_name,
        learning_rate=learning_rate,
        batch_size=batch_size,
        num_epochs=num_epochs,
        weight_decay=weight_decay,
        dropout_rate=dropout_rate,
    )

    evaluate_model_task = evaluate_model(
        model_path=train_model_task.outputs["trained_model"],
        test_data=prepare_data_task.outputs["test_data"],
        class_labels=prepare_data_task.outputs["class_labels"],
    )

# Compile the pipeline
from kfp import compiler

compiler.Compiler().compile(
    pipeline_func=sentiment_analysis_pipeline,
    package_path='sentiment_analysis_pipeline.json'
)

# Create and submit the pipeline job
pipeline_job = pipeline_jobs.PipelineJob(
    display_name="sentiment-analysis-job",
    template_path="sentiment_analysis_pipeline.json",
    pipeline_root=PIPELINE_ROOT,
    parameter_values={
        "project_id": PROJECT_ID,
        "region": REGION,
        "dataset_id": DATASET_ID,
        "model_name": "BERT",
        "learning_rate": 2e-5,
        "batch_size": 32,
        "num_epochs": 3,
        "weight_decay": 0.01,
        "dropout_rate": 0.1,
    }
)

# Submit the pipeline job
pipeline_job.submit()