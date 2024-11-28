import os
from google.cloud import aiplatform
from kfp import dsl
from kfp.dsl import component, Input, Output, Dataset, Model, Metrics
import mlflow

# Set up Google Cloud project details
PROJECT_ID = "amazonreviewssentimentanalysis"
REGION = "us-central1"
BUCKET_NAME = "amazon-reviews-sentiment-analysis"
PIPELINE_ROOT = f"gs://{BUCKET_NAME}/pipeline/vertex-ai"
DATASET_ID = "5951559684128243712"
STAGING_BUCKET = f"gs://{BUCKET_NAME}/staging"

# Initialize Vertex AI with staging bucket
aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=STAGING_BUCKET)

# Set up MLflow
os.environ["GOOGLE_CLOUD_PROJECT"] = PROJECT_ID
mlflow.set_tracking_uri(f"{PIPELINE_ROOT}/mlflow")

@component(
    packages_to_install=[
        "pandas",
        "scikit-learn",
        "google-cloud-aiplatform",
        "gcsfs"
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
    from google.cloud import aiplatform
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder

    # Initialize Vertex AI
    aiplatform.init(project=project_id, location=region)

    dataset = aiplatform.TabularDataset(dataset_id)
    gcs_uri = dataset._gca_resource.metadata["inputConfig"]["gcsSource"]["uri"][0]

    # Load the data from GCS URI as a pandas DataFrame
    df = pd.read_csv(gcs_uri)

    df['text'] = df['text'].fillna('')
    df['title'] = df['title'].fillna('')
    df['price'] = pd.to_numeric(df['price'].replace("unknown", None), errors='coerce')
    df['price_missing'] = df['price'].isna().astype(int)
    df['price'] = df['price'].fillna(0).astype(float)
    df['helpful_vote'] = df['helpful_vote'].fillna(0).astype(int)
    df['verified_purchase'] = df['verified_purchase'].apply(lambda x: 1 if x else 0)

    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['sentiment_label'])

    df['review_date_timestamp'] = pd.to_datetime(df['review_date_timestamp'])
    df = df.sort_values(by='review_date_timestamp').reset_index(drop=True)
    
    train_end = int(len(df) * 0.8)
    val_end = int(len(df) * 0.9)
    
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    
    train_df.to_csv(train_data.path, index=False)
    val_df.to_csv(val_data.path, index=False)
    test_df.to_csv(test_data.path, index=False)
    pd.Series(label_encoder.classes_).to_csv(class_labels.path, index=False, header=False)

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

    # Use CustomPythonPackageTrainingJob for GPU setup
    training_job = aiplatform.CustomPythonPackageTrainingJob(
        display_name="sentiment-analysis-train-job",
        python_package_gcs_uri=f"{PIPELINE_ROOT}/trainer.tar.gz",
        python_module_name="trainer.task",
        container_uri="us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-9:latest",
    )

    # Allow Vertex AI to auto-select compatible GPU (default)
    training_job.run(
        replica_count=1,
        machine_type="g2-standard-8",
        accelerator_type="NVIDIA_L4",
        accelerator_count=1,
        args=[
            f"--project_id={project_id}",
            f"--region={region}",
            f"--dataset_id={dataset_id}",
            f"--model_name={model_name}",
            f"--learning_rate={learning_rate}",
            f"--batch_size={batch_size}",
            f"--num_epochs={num_epochs}",
            f"--weight_decay={weight_decay}",
            f"--dropout_rate={dropout_rate}",
        ],
        sync=True,
    )

# Compile the pipeline
from kfp import compiler

compiler.Compiler().compile(
    pipeline_func=sentiment_analysis_pipeline,
    package_path='sentiment_analysis_pipeline.json'
)

# Create and submit the pipeline job
pipeline_job = aiplatform.PipelineJob(
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