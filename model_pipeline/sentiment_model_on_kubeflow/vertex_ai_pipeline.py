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
DATASET_ID = "1119760233913122816"  # Your dataset ID

# Initialize Vertex AI
aiplatform.init(project=PROJECT_ID, location=REGION)

# Set up MLflow
os.environ["GOOGLE_CLOUD_PROJECT"] = PROJECT_ID
mlflow.set_tracking_uri(f"{PIPELINE_ROOT}/mlflow")

@component(
    packages_to_install=[
        "pandas", "scikit-learn", "google-cloud-aiplatform", "gcsfs", "accelerate>=0.26.0"
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

    aiplatform.init(project=project_id, location=region)
    dataset = aiplatform.TabularDataset(dataset_id)
    gcs_uri = dataset._gca_resource.metadata["inputConfig"]["gcsSource"]["uri"][0]
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

@component(
    packages_to_install=[
        "pandas", "scikit-learn", "torch", "transformers", "mlflow", "optuna", "accelerate>=0.26.0"
    ],
    base_image="python:3.9",
)
def run_experiment(
    train_data: Input[Dataset],
    val_data: Input[Dataset],
    class_labels: Input[Dataset],
    best_hyperparameters: Output[Dataset],
    best_model: Output[str],
) -> None:
    import pandas as pd
    import torch
    import mlflow
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
    from sklearn.metrics import f1_score
    import json
    import optuna
    import logging

    logging.basicConfig(level=logging.INFO)

    try:
        train_df = pd.read_csv(train_data.path)
        val_df = pd.read_csv(val_data.path)
        class_labels = pd.read_csv(class_labels.path, header=None)[0].tolist()
        if not class_labels:
            raise ValueError("class_labels is empty")
        logging.info(f"Loaded {len(train_df)} training samples, {len(val_df)} validation samples, and {len(class_labels)} class labels")
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        raise

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

    def objective(trial):
        model_name = trial.suggest_categorical("model_name", ["BERT", "RoBERTa"])
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
        weight_decay = trial.suggest_loguniform("weight_decay", 1e-5, 1e-1)
        dropout_rate = trial.suggest_uniform("dropout_rate", 0.1, 0.5)

        if model_name == "BERT":
            model_checkpoint = 'bert-base-uncased'
        elif model_name == "RoBERTa":
            model_checkpoint = 'roberta-base'

        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=len(class_labels))

        train_dataset = SentimentDataset(train_df['text'].tolist(), train_df['label'].tolist(), tokenizer)
        val_dataset = SentimentDataset(val_df['text'].tolist(), val_df['label'].tolist(), tokenizer)

        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=1,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            warmup_steps=500,
            weight_decay=weight_decay,
            logging_dir='./logs',
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )

        trainer.train()
        eval_results = trainer.evaluate()
        predictions = trainer.predict(val_dataset)
        preds = predictions.predictions.argmax(-1)
        f1 = f1_score(val_dataset.labels, preds, average='weighted')

        return f1

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)

    best_params = study.best_params
    best_params['batch_size'] = 16
    best_params['num_epochs'] = 1
    logging.info(f"Best hyperparameters: {best_params}")

    with open(best_hyperparameters.path, 'w') as f:
        json.dump(best_params, f)

    with open(best_model.path, 'w') as f:
        f.write(best_params['model_name'])

@component(
    packages_to_install=[
        "pandas", "scikit-learn", "torch", "transformers", "mlflow", "accelerate>=0.26.0"
    ],
    base_image="python:3.9",
)
def train_model(
    train_data: Input[Dataset],
    val_data: Input[Dataset],
    class_labels: Input[Dataset],
    best_hyperparameters: Input[Dataset],
    model_name: str,
    trained_model: Output[Model],
    metrics: Output[Metrics],
) -> None:
    import pandas as pd
    import torch
    import mlflow
    import json
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
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

    train_df = pd.read_csv(train_data.path)
    val_df = pd.read_csv(val_data.path)
    class_labels = pd.read_csv(class_labels.path, header=None)[0].tolist()

    with open(best_hyperparameters.path, 'r') as f:
        best_params = json.load(f)

    if model_name == "BERT":
        model_checkpoint = 'bert-base-uncased'
    elif model_name == "RoBERTa":
        model_checkpoint = 'roberta-base'
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=len(class_labels))

    train_dataset = SentimentDataset(train_df['text'].tolist(), train_df['label'].tolist(), tokenizer)
    val_dataset = SentimentDataset(val_df['text'].tolist(), val_df['label'].tolist(), tokenizer)

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=best_params['num_epochs'],
        per_device_train_batch_size=best_params['batch_size'],
        per_device_eval_batch_size=best_params['batch_size'],
        warmup_steps=500,
        weight_decay=best_params['weight_decay'],
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()
    eval_results = trainer.evaluate()

    metrics.log_metric("val_loss", eval_results["eval_loss"])
    metrics.log_metric("val_accuracy", eval_results["eval_accuracy"])

    mlflow.pytorch.save_model(model, trained_model.path)

@component(
    packages_to_install=[
        "pandas", "scikit-learn", "torch", "transformers", "mlflow", "accelerate>=0.26.0",
    ],
    base_image="python:3.9",
)
def evaluate_model(
    model_path: Input[Model],
    test_data: Input[Dataset],
    class_labels: Input[Dataset],
    best_hyperparameters: Input[Dataset],
    model_name: str,
    metrics: Output[ClassificationMetrics],
) -> None:
    import pandas as pd
    import torch
    import numpy as np
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    import mlflow
    import json
    from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

    model = mlflow.pytorch.load_model(model_path.path)

    with open(best_hyperparameters.path, 'r') as f:
        best_params = json.load(f)

    if model_name == "BERT":
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    elif model_name == "RoBERTa":
        tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    test_df = pd.read_csv(test_data.path)
    class_labels = pd.read_csv(class_labels.path, header=None)[0].tolist()

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

    dataset = TensorDataset(input_ids, attention_masks, labels)
    dataloader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=32)

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

    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='weighted')

    metrics.log_accuracy(true_labels, predictions)
    metrics.log_confusion_matrix(class_labels, true_labels, predictions)
    metrics.log_roc_curve(true_labels, predictions)
    metrics.log_metric("test_accuracy", accuracy)
    metrics.log_metric("test_precision", precision)
    metrics.log_metric("test_recall", recall)
    metrics.log_metric("test_f1", f1)

@component(
    packages_to_install=[
        "pandas", "scikit-learn", "torch", "transformers", "mlflow", "accelerate>=0.26.0",
    ],
    base_image="python:3.9",
)
def evaluate_model_slices(
    model_path: Input[Model],
    test_data: Input[Dataset],
    class_labels: Input[Dataset],
    best_hyperparameters: Input[Dataset],
    model_name: str,
    slice_metrics: Output[Dataset],
) -> None:
    import pandas as pd
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    import mlflow
    import json

    def evaluate_slice(data, model, tokenizer):
        encoded_data = tokenizer.batch_encode_plus(
            data['text'].tolist(),
            add_special_tokens=True,
            return_attention_mask=True,
            pad_to_max_length=True,
            max_length=128,
            return_tensors='pt'
        )
        input_ids = encoded_data['input_ids']
        attention_masks = encoded_data['attention_mask']
        labels = torch.tensor(data['label'].tolist())

        model.eval()
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_masks)
            predictions = torch.argmax(outputs.logits, dim=1)

        accuracy = accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions, average='weighted')
        recall = recall_score(labels, predictions, average='weighted')
        f1 = f1_score(labels, predictions, average='weighted')

        return accuracy, precision, recall, f1

    model = mlflow.pytorch.load_model(model_path.path)
    test_df = pd.read_csv(test_data.path)
    
    if model_name == "BERT":
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    elif model_name == "RoBERTa":
        tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    slice_columns = ['year', 'main_category']
    metrics_list = []

    # Evaluate full dataset
    full_accuracy, full_precision, full_recall, full_f1 = evaluate_slice(test_df, model, tokenizer)
    metrics_list.append({
        'Slice Column': 'Full Dataset',
        'Slice Value': 'All',
        'Samples': len(test_df),
        'Accuracy': full_accuracy,
        'Precision': full_precision,
        'Recall': full_recall,
        'F1 Score': full_f1
    })

    for column in slice_columns:
        for value in test_df[column].unique():
            slice_data = test_df[test_df[column] == value]
            if len(slice_data) > 0:
                accuracy, precision, recall, f1 = evaluate_slice(slice_data, model, tokenizer)
                metrics_list.append({
                    'Slice Column': column,
                    'Slice Value': value,
                    'Samples': len(slice_data),
                    'Accuracy': accuracy,
                    'Precision': precision,
                    'Recall': recall,
                    'F1 Score': f1
                })

    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.to_csv(slice_metrics.path, index=False)

@component(
    packages_to_install=[
        "pandas",
    ],
    base_image="python:3.9",
)
def detect_bias(
    slice_metrics: Input[Dataset],
    bias_report: Output[Dataset],
) -> None:
    import pandas as pd
    import logging

    logging.basicConfig(level=logging.INFO)

    metrics_df = pd.read_csv(slice_metrics.path)
    
    full_dataset_row = metrics_df[metrics_df["Slice Column"] == "Full Dataset"]
    if full_dataset_row.empty:
        logging.error("Full Dataset row not found in metrics. Bias detection cannot proceed.")
        return

    full_samples = int(full_dataset_row["Samples"].iloc[0])
    full_f1 = float(full_dataset_row["F1 Score"].iloc[0])

    min_samples_threshold = 0.1 * full_samples
    f1_threshold = full_f1 * 0.9

    biased_rows = metrics_df[
        (metrics_df["Samples"] >= min_samples_threshold) &
        (metrics_df["F1 Score"] < f1_threshold)
    ]

    bias_report_data = []
    if not biased_rows.empty:
        for _, row in biased_rows.iterrows():
            bias_report_data.append({
                "Slice Column": row['Slice Column'],
                "Slice Value": row['Slice Value'],
                "Samples": row['Samples'],
                "F1 Score": row['F1 Score'],
                "F1 Threshold": f1_threshold
            })
        logging.warning(f"Potential bias detected in {len(biased_rows)} slices.")
    else:
        logging.info("No significant bias detected.")

    bias_report_df = pd.DataFrame(bias_report_data)
    bias_report_df.to_csv(bias_report.path, index=False)

@dsl.pipeline(
    name="sentiment-analysis-pipeline",
    description="A pipeline for sentiment analysis using the best model with hyperparameter tuning and bias detection",
)
def sentiment_analysis_pipeline(
    project_id: str,
    region: str,
    dataset_id: str,
):
    prepare_data_task = prepare_data(
        project_id=project_id,
        region=region,
        dataset_id=dataset_id
    )

    experiment_task = run_experiment(
        train_data=prepare_data_task.outputs["train_data"],
        val_data=prepare_data_task.outputs["val_data"],
        class_labels=prepare_data_task.outputs["class_labels"],
    )

    train_best_model_task = train_model(
        train_data=prepare_data_task.outputs["train_data"],
        val_data=prepare_data_task.outputs["val_data"],
        class_labels=prepare_data_task.outputs["class_labels"],
        best_hyperparameters=experiment_task.outputs["best_hyperparameters"],
        model_name=experiment_task.outputs["best_model"],
    ).set_display_name("train-best-model")

    evaluate_best_model_task = evaluate_model(
        model_path=train_best_model_task.outputs["trained_model"],
        test_data=prepare_data_task.outputs["test_data"],
        class_labels=prepare_data_task.outputs["class_labels"],
        best_hyperparameters=experiment_task.outputs["best_hyperparameters"],
        model_name=experiment_task.outputs["best_model"],
    ).set_display_name("evaluate-best-model")

    evaluate_model_slices_task = evaluate_model_slices(
        model_path=train_best_model_task.outputs["trained_model"],
        test_data=prepare_data_task.outputs["test_data"],
        class_labels=prepare_data_task.outputs["class_labels"],
        best_hyperparameters=experiment_task.outputs["best_hyperparameters"],
        model_name=experiment_task.outputs["best_model"],
    ).set_display_name("evaluate-model-slices")

    detect_bias_task = detect_bias(
        slice_metrics=evaluate_model_slices_task.outputs["slice_metrics"],
    ).set_display_name("detect-and-handle-bias")

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
    }
)

# Submit the pipeline job
pipeline_job.submit()
