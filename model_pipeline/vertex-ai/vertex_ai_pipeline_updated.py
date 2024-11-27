import os
from google.cloud import aiplatform
from google.cloud.aiplatform import pipeline_jobs
from kfp import dsl
from kfp.dsl import component, Input, Output, Dataset, Model, Metrics, ClassificationMetrics

# Set up Google Cloud project details
PROJECT_ID = "amazonreviewssentimentanalysis"
REGION = "us-central1"
BUCKET_NAME = "amazon-reviews-sentiment-analysis-vertex-ai"
PIPELINE_ROOT = f"gs://{BUCKET_NAME}/pipeline/vertex-ai"
DATASET_ID = "2110090906806779904"  # Your dataset ID

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
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

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

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=len(class_labels)
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
    trainer.save_model(trained_model.path)


@component(
    packages_to_install=[
        "pandas",
        "scikit-learn",
        "torch",
        "transformers",
        "google-cloud-storage",
    ],
    base_image="python:3.9",
)
def evaluate_model(
    model_path: Input[Model],
    test_data: Input[Dataset],
    class_labels: Input[Dataset],
    metrics: Output[Dataset],
) -> None:
    import pandas as pd
    import torch
    import os
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
        classification_report,
    )
    from transformers import BertTokenizer, BertForSequenceClassification
    from torch.utils.data import DataLoader, TensorDataset
    from google.cloud import storage
    import json

    # Load the model
    model = BertForSequenceClassification.from_pretrained(model_path.path)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Load test data and class labels
    test_df = pd.read_csv(test_data.path)
    class_labels = pd.read_csv(class_labels.path, header=None)[0].tolist()

    # Tokenize and create tensors
    encoded_data = tokenizer.batch_encode_plus(
        test_df["text"].tolist(),
        add_special_tokens=True,
        return_attention_mask=True,
        pad_to_max_length=True,
        max_length=128,
        return_tensors="pt",
    )

    input_ids = encoded_data["input_ids"]
    attention_masks = encoded_data["attention_mask"]
    true_labels = torch.tensor(test_df["label"].tolist())

    # Create DataLoader for test data
    dataset = TensorDataset(input_ids, attention_masks, true_labels)
    dataloader = DataLoader(dataset, batch_size=32)

    # Evaluate
    model.eval()
    predictions = []
    true_labels_list = []

    for batch in dataloader:
        batch_input_ids, batch_attention_masks, batch_labels = tuple(t for t in batch)

        with torch.no_grad():
            outputs = model(batch_input_ids, attention_mask=batch_attention_masks)

        logits = outputs.logits
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        predictions.extend(preds)
        true_labels_list.extend(batch_labels.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(true_labels_list, predictions)
    precision = precision_score(true_labels_list, predictions, average="weighted")
    recall = recall_score(true_labels_list, predictions, average="weighted")
    f1 = f1_score(true_labels_list, predictions, average="weighted")
    cm = confusion_matrix(true_labels_list, predictions)

    # Create metrics dictionary
    metrics_data = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            true_labels_list, predictions, target_names=class_labels, output_dict=True
        ),
    }

    # Ensure the output directory exists
    os.makedirs(metrics.path, exist_ok=True)

    # Save metrics to a JSON file
    metrics_path = os.path.join(metrics.path, "evaluation_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=4)

    print(f"Evaluation metrics saved to: {metrics_path}")

    # Upload metrics JSON file to the specified bucket
    storage_client = storage.Client()
    bucket_name = metrics.path.split("/")[2]  # Extract bucket name from path
    bucket = storage_client.bucket(bucket_name)
    destination_blob_name = "/".join(metrics.path.split("/")[3:]) + "/evaluation_metrics.json"
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(metrics_path)
    print(f"Evaluation metrics uploaded to: gs://{bucket_name}/{destination_blob_name}")

@dsl.pipeline(name="sentiment-analysis-pipeline")
def sentiment_analysis_pipeline():
    prepare_step = prepare_data(project_id=PROJECT_ID, region=REGION, dataset_id=DATASET_ID).set_caching_options(enable_caching=False)
    train_step = train_model(
        train_data=prepare_step.outputs["train_data"],
        val_data=prepare_step.outputs["val_data"],
        class_labels=prepare_step.outputs["class_labels"],
        model_name="bert-base-uncased",
        learning_rate=5e-5,
        batch_size=16,
        num_epochs=1,
        weight_decay=0.01,
        dropout_rate=0.1,
    ).set_caching_options(enable_caching=False)
    evaluate_model(
        model_path=train_step.outputs["trained_model"],
        test_data=prepare_step.outputs["test_data"],
        class_labels=prepare_step.outputs["class_labels"],
    ).set_caching_options(enable_caching=False)


# Compile and submit pipeline
from kfp.compiler import Compiler

Compiler().compile(
    pipeline_func=sentiment_analysis_pipeline, package_path="sentiment_analysis_pipeline.json"
)

pipeline_job = pipeline_jobs.PipelineJob(
    display_name="sentiment-analysis-pipeline",
    template_path="sentiment_analysis_pipeline.json",
    pipeline_root=PIPELINE_ROOT,
)
pipeline_job.submit()
