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
        "gcsfs"  # Enables reading directly from GCS
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

@component(
    packages_to_install=[
        "pandas",
        "scikit-learn",
        "torch",
        "transformers",
        "mlflow",
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
    import mlflow
    from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
    from sklearn.preprocessing import LabelEncoder
    from torch.utils.data import Dataset

    class SentimentDataset(Dataset):
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

            encoding = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_length,
                return_token_type_ids=False,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt',
            )

            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(label, dtype=torch.long)
            }

    # Load datasets
    train_df = pd.read_csv(train_data.path)
    val_df = pd.read_csv(val_data.path)
    class_labels = pd.read_csv(class_labels.path, header=None)[0].tolist()

    # Initialize tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(class_labels))

    # Create datasets
    train_dataset = SentimentDataset(train_df['text'].tolist(), train_df['label'].tolist(), tokenizer)
    val_dataset = SentimentDataset(val_df['text'].tolist(), val_df['label'].tolist(), tokenizer)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=500,
        weight_decay=weight_decay,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # Train the model
    trainer.train()

    # Evaluate the model
    eval_results = trainer.evaluate()

    # Log metrics
    metrics.log_metric("val_loss", eval_results["eval_loss"])
    metrics.log_metric("val_accuracy", eval_results["eval_accuracy"])

    # Save the model
    mlflow.pytorch.save_model(model, trained_model.path)

@component(
    packages_to_install=[
        "pandas",
        "scikit-learn",
        "torch",
        "transformers",
        "mlflow",
    ],
    base_image="python:3.9",
)
def evaluate_model(
    model_path: Input[Model],
    test_data: Input[Dataset],
    class_labels: Input[Dataset],
    metrics: Output[ClassificationMetrics],
) -> None:
    import pandas as pd
    import torch
    import numpy as np
    from transformers import BertTokenizer, BertForSequenceClassification
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    import mlflow

    # Load the model
    model = mlflow.pytorch.load_model(model_path.path)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Load test data and class labels
    test_df = pd.read_csv(test_data.path)
    class_labels = pd.read_csv(class_labels.path, header=None)[0].tolist()

    # Tokenize and create tensors
    encoded_data = tokenizer.batch_encode_plus(
        test_df['text'].tolist(),
        add_special_tokens=True,
        return_attention_mask=True,
        pad_to_max_length=True,
        max_length=128,
        return_tensors='pt'
    )

    input_ids = encoded_data['input_ids']
    attention_masks = encoded_data['attention_mask']
    labels = torch.tensor(test_df['label'].tolist())

    # Create DataLoader for test data
    from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

    dataset = TensorDataset(input_ids, attention_masks, labels)
    dataloader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=32)

    # Evaluate
    model.eval()
    predictions = []
    true_labels = []

    for batch in dataloader:
        batch_input_ids, batch_attention_masks, batch_labels = tuple(t for t in batch)
        
        with torch.no_grad():
            outputs = model(batch_input_ids, attention_mask=batch_attention_masks)
        
        logits = outputs.logits
        predictions.extend(torch.argmax(logits, dim=1).tolist())
        true_labels.extend(batch_labels.tolist())

    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='weighted')

    # Log metrics
    metrics.log_accuracy(true_labels, predictions)
    metrics.log_confusion_matrix(class_labels, true_labels, predictions)
    metrics.log_roc_curve(true_labels, predictions)

    # Log additional metrics
    metrics.log_metric("test_accuracy", accuracy)
    metrics.log_metric("test_precision", precision)
    metrics.log_metric("test_recall", recall)
    metrics.log_metric("test_f1", f1)

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
