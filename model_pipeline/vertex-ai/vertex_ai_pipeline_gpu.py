import os
import kfp.compiler as compiler
from google.cloud import aiplatform
from google.cloud.aiplatform import pipeline_jobs
from kfp import dsl
from kfp.dsl import component, Input, Output, Dataset, Model, Metrics

# Set up Google Cloud project details
PROJECT_ID = "amazonreviewssentimentanalysis"
REGION = "us-central1"
BUCKET_NAME = "amazon-reviews-sentiment-analysis-vertex-ai"
PIPELINE_ROOT = f"gs://{BUCKET_NAME}/pipeline/vertex-ai"
DATASET_ID = "2110090906806779904"  # 1% Dataset
# DATASET_ID = "7683013970701058048" # Actual Sampled and Labeled Data

# Initialize Vertex AI
aiplatform.init(project=PROJECT_ID, location=REGION)


@component(
    packages_to_install=[
        "pandas",
        "scikit-learn",
        "google-cloud-aiplatform",
        "gcsfs",
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

    # Load the dataset and retrieve the GCS URI
    dataset = aiplatform.TabularDataset(dataset_id)
    gcs_uri = dataset._gca_resource.metadata["inputConfig"]["gcsSource"]["uri"][0]

    # Load the data from GCS URI as a pandas DataFrame
    df = pd.read_csv(gcs_uri)

    # Data processing code
    df['text'] = df['text'].fillna('')
    df['title'] = df['title'].fillna('')
    df['price'] = pd.to_numeric(df['price'].replace("unknown", None), errors='coerce')
    df['price_missing'] = df['price'].isna().astype(int)
    df['price'] = df['price'].fillna(0).astype(float)
    df['helpful_vote'] = df['helpful_vote'].fillna(0).astype(int)
    df['verified_purchase'] = df['verified_purchase'].apply(lambda x: 1 if x else 0)

    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['sentiment_label'])

    # Split the data
    train_end = int(len(df) * 0.8)
    val_end = int(len(df) * 0.9)

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    train_df.to_csv(train_data.path, index=False)
    val_df.to_csv(val_data.path, index=False)
    test_df.to_csv(test_data.path, index=False)
    pd.Series(label_encoder.classes_).to_csv(class_labels.path, index=False, header=False)


@component(
    packages_to_install=[
        "pandas",
        "torch",
        "transformers",
        "scikit-learn",
        "accelerate>=0.26.0",
        "google-cloud-storage",
    ],
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
    from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score
    from google.cloud import storage
    import json
    import os

    class SentimentDataset(torch.utils.data.Dataset):
        def __init__(self, texts, labels, tokenizer, max_length=128):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            text = self.texts[idx]
            label = self.labels[idx]

            encoding = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            return {
                "input_ids": encoding["input_ids"].flatten(),
                "attention_mask": encoding["attention_mask"].flatten(),
                "labels": torch.tensor(label, dtype=torch.long),
            }

    # Load datasets
    train_df = pd.read_csv(train_data.path)
    val_df = pd.read_csv(val_data.path)
    class_labels = pd.read_csv(class_labels.path, header=None).squeeze().tolist()

    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(
        model_name, num_labels=len(class_labels)
    )

    train_dataset = SentimentDataset(train_df["text"], train_df["label"], tokenizer)
    val_dataset = SentimentDataset(val_df["text"], val_df["label"], tokenizer)

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        weight_decay=weight_decay,
        load_best_model_at_end=True,
    )

    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
        accuracy = accuracy_score(labels, preds)
        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    tokenizer.save_pretrained(trained_model.path) 
    trainer.save_model(trained_model.path)

    # Ensure the output directory exists
    os.makedirs(metrics.path, exist_ok=True)

    # Compute final metrics
    eval_metrics = trainer.evaluate()
    metrics_path = os.path.join(metrics.path, "train_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(eval_metrics, f)

    # Upload metrics to GCS
    storage_client = storage.Client()
    bucket_name = metrics.path.split("/")[2]
    destination_blob_name = "/".join(metrics.path.split("/")[3:]) + "/train_metrics.json"
    blob = storage_client.bucket(bucket_name).blob(destination_blob_name)
    blob.upload_from_filename(metrics_path)


@component(
    packages_to_install=[
        "pandas",
        "torch",
        "transformers",
        "scikit-learn",
        "google-cloud-storage",
    ],
    base_image="python:3.9",
)
def evaluate_model(
    model_path: Input[Model],
    test_data: Input[Dataset],
    class_labels: Input[Dataset],
    metrics: Output[Metrics],
) -> None:
    import pandas as pd
    import torch
    from transformers import BertTokenizer, BertForSequenceClassification
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    from google.cloud import storage
    import json
    import os

    class SentimentDataset(torch.utils.data.Dataset):
        def __init__(self, texts, labels, tokenizer, max_length=128):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            text = self.texts[idx]
            label = self.labels[idx]

            encoding = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            return {
                "input_ids": encoding["input_ids"].flatten(),
                "attention_mask": encoding["attention_mask"].flatten(),
                "labels": torch.tensor(label, dtype=torch.long),
            }

    # Load test data and class labels
    test_df = pd.read_csv(test_data.path)
    class_labels = pd.read_csv(class_labels.path, header=None).squeeze().tolist()

    # Load the model
    tokenizer = BertTokenizer.from_pretrained(model_path.path)
    model = BertForSequenceClassification.from_pretrained(model_path.path)

    test_dataset = SentimentDataset(test_df["text"], test_df["label"], tokenizer)
    model.eval()

    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in torch.utils.data.DataLoader(test_dataset, batch_size=16):
            inputs = {
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
            }
            outputs = model(**inputs)
            preds = outputs.logits.argmax(dim=1).cpu().numpy()
            predictions.extend(preds)
            true_labels.extend(batch["labels"].cpu().numpy())

    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average="weighted")
    accuracy = accuracy_score(true_labels, predictions)

    eval_results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

    # Ensure the output directory exists
    os.makedirs(metrics.path, exist_ok=True)

    # Save evaluation metrics
    metrics_path = os.path.join(metrics.path, "eval_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(eval_results, f)

    # Upload metrics to GCS
    storage_client = storage.Client()
    bucket_name = metrics.path.split("/")[2]
    destination_blob_name = "/".join(metrics.path.split("/")[3:]) + "/eval_metrics.json"
    blob = storage_client.bucket(bucket_name).blob(destination_blob_name)
    blob.upload_from_filename(metrics_path)


# Define the pipeline
@dsl.pipeline(
    name="Sentiment Analysis Pipeline",
    description="A pipeline for sentiment analysis using a transformer model"
)
def sentiment_analysis_pipeline(
    project_id: str,
    region: str,
    dataset_id: str,
    model_name: str = "bert-base-uncased",
    learning_rate: float = 2e-5,
    batch_size: int = 32,
    num_epochs: int = 3,
    weight_decay: float = 0.01,
    dropout_rate: float = 0.1,
):
    # Prepare data
    prepare_data_task = prepare_data(
        project_id=project_id,
        region=region,
        dataset_id=dataset_id,
    )

    # Train model with specific resources
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
    ).set_cpu_limit("8") \
     .set_memory_limit("32G") \
     .set_gpu_limit(1) \
     .set_accelerator_type("NVIDIA_TESLA_T4")


    # Evaluate model
    evaluate_model_task = evaluate_model(
        model_path=train_model_task.outputs["trained_model"],
        test_data=prepare_data_task.outputs["test_data"],
        class_labels=prepare_data_task.outputs["class_labels"],
    ).set_cpu_limit("4") \
     .set_memory_limit("16G") \
     .set_gpu_limit(1) \
     .set_accelerator_type("NVIDIA_TESLA_T4")

# Compile the pipeline into a YAML file
compiled_pipeline_path = "sentiment_analysis_pipeline.yaml"
compiler.Compiler().compile(sentiment_analysis_pipeline, compiled_pipeline_path)

# Define parameter values for the pipeline
parameter_values = {
    "project_id": PROJECT_ID,
    "region": REGION,
    "dataset_id": DATASET_ID,
}

# Create the pipeline job
pipeline_job = aiplatform.PipelineJob(
    display_name="sentiment-analysis-pipeline",
    template_path=compiled_pipeline_path,
    pipeline_root=PIPELINE_ROOT,
    location=REGION,
    parameter_values=parameter_values,  # Pass pipeline parameters here
)

# Run the pipeline job
pipeline_job.run(sync=True)

