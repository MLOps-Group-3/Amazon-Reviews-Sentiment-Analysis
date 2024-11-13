import os
from google.cloud import aiplatform
from google.cloud.aiplatform import pipeline_jobs
from kfp import dsl
from kfp.dsl import component, Input, Output, Dataset, Model, Metrics, ClassificationMetrics
import mlflow
from google.oauth2 import service_account

# Set up Google Cloud project details
PROJECT_ID = "amazonreviewssentimentanalysis"
REGION = "us-east1"
BUCKET_NAME = "amazon-reviews-sentiment-analysis"
PIPELINE_ROOT = f"gs://{BUCKET_NAME}/pipeline/vertex-ai"
SERVICE_ACCOUNT_KEY_PATH = "/home/niresh-s/Downloads/amazonreviewssentimentanalysis-092e2b7bf64c.json"

# Create credentials using the service account key file
credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_KEY_PATH,
    scopes=["https://www.googleapis.com/auth/cloud-platform"]
)

# Initialize Vertex AI with the service account credentials
aiplatform.init(
    project=PROJECT_ID,
    location=REGION,
    credentials=credentials
)

# Set up MLflow
os.environ["GOOGLE_CLOUD_PROJECT"] = PROJECT_ID
mlflow.set_tracking_uri(f"{PIPELINE_ROOT}/mlflow")

@component(
    packages_to_install=["pandas", "scikit-learn"],
    base_image="python:3.9",
)
def prepare_data(
    data_path: str,
    train_data: Output[Dataset],
    val_data: Output[Dataset],
    test_data: Output[Dataset],
    class_labels: Output[Dataset],
) -> None:
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    from utils.data_loader import load_and_process_data, split_data_by_timestamp

    df, label_encoder = load_and_process_data(data_path)
    train_df, val_df, test_df = split_data_by_timestamp(df)
    
    train_df.to_csv(train_data.path, index=False)
    val_df.to_csv(val_data.path, index=False)
    test_df.to_csv(test_data.path, index=False)
    pd.Series(label_encoder.classes_).to_csv(class_labels.path, index=False, header=False)

@component(
    packages_to_install=["pandas", "scikit-learn", "torch", "transformers", "mlflow"],
    base_image="python:3.9",
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
    import pandas as pd
    import torch
    import mlflow
    from transformers import BertTokenizer, RobertaTokenizer
    from utils.data_loader import SentimentDataset
    from utils.bert_model import initialize_bert_model, train_bert_model
    from utils.roberta_model import initialize_roberta_model, train_roberta_model

    # Load datasets
    train_df = pd.read_csv(train_data.path)
    val_df = pd.read_csv(val_data.path)
    class_labels = pd.read_csv(class_labels.path, header=None)[0].tolist()

    # Initialize model and tokenizer
    if model_name == "BERT":
        model_init = initialize_bert_model
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        train_func = train_bert_model
    elif model_name == "RoBERTa":
        model_init = initialize_roberta_model
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        train_func = train_roberta_model
    else:
        raise ValueError(f"Model {model_name} is not supported.")

    # Create datasets
    train_dataset = SentimentDataset(train_df['text'].tolist(), tokenizer)
    val_dataset = SentimentDataset(val_df['text'].tolist(), tokenizer)

    # Initialize model
    model = model_init(num_labels=len(class_labels))

    # Train model
    training_args = {
        "learning_rate": learning_rate,
        "per_device_train_batch_size": batch_size,
        "num_train_epochs": num_epochs,
        "weight_decay": weight_decay,
        "dropout_rate": dropout_rate,
    }
    
    eval_results, trainer = train_func(model, train_dataset, val_dataset, **training_args)

@component(
  packages_to_install=["torch", "transformers", "scikit-learn", "mlflow"],
  base_image="python:3.9",
)
def evaluate_model(
  model_path: Input[Model],
  test_data: Input[Dataset],
  class_labels: Input[Dataset],
  metrics: Output[ClassificationMetrics],
) -> None:
  import torch
  import pandas as pd
  import numpy as np
  from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score

  # Load model and data for evaluation here...
  
@dsl.pipeline(
  name="sentiment-analysis-pipeline",
  description="A pipeline for sentiment analysis using BERT or RoBERTa",
)
def sentiment_analysis_pipeline(
  data_path: str,
  model_name: str,
  learning_rate: float,
  batch_size: int,
  num_epochs: int,
  weight_decay: float,
  dropout_rate: float,
):
  prepare_data_task = prepare_data(data_path=data_path)

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
      "data_path": f"gs://{BUCKET_NAME}/pipeline/data/labeled/labeled_data.csv",
      "model_name": "BERT",
      "learning_rate": 2e-5,
      "batch_size": 32,
      "num_epochs": 3,
      "weight_decay": 0.01,
      "dropout_rate": 0.1,
  }
)

pipeline_job.submit()
