import mlflow
import itertools
import torch
import numpy as np
from utils.bert_model import initialize_bert_model, train_bert_model, evaluate_bert_model
from utils.roberta_model import initialize_roberta_model, train_roberta_model, evaluate_roberta_model
from utils.data_loader import load_and_process_data, split_data_by_timestamp, SentimentDataset
from transformers import BertTokenizer, RobertaTokenizer

# Configuration for data path and output
DATA_PATH = "labeled_data.csv"

# Define hyperparameter grid with additional parameters
param_grid = {
    "learning_rate": [2e-5],          # 1 option
    "batch_size": [64],               # 1 option
    "num_epochs": [3],                # 1 option
    "weight_decay": [0.01, 0.001],    # 2 options
    "dropout_rate": [0.1, 0.3]        # 2 options
}

# Generate all combinations of hyperparameters
param_combinations = list(itertools.product(*param_grid.values()))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Prepare the data and get class labels
df, label_encoder = load_and_process_data(DATA_PATH)
train_df, val_df, test_df = split_data_by_timestamp(df)
class_labels = label_encoder.classes_

# Log label mapping in MLflow for reference
label_mapping = {i: label for i, label in enumerate(class_labels)}
with mlflow.start_run(run_name="Label Mapping"):
    mlflow.log_param("label_mapping", label_mapping)

# Loop through each model type
for model_name, initialize_model, tokenizer_class, train_model, eval_model in [
    ("BERT", initialize_bert_model, BertTokenizer, train_bert_model, evaluate_bert_model),
    ("RoBERTa", initialize_roberta_model, RobertaTokenizer, train_roberta_model, evaluate_roberta_model)
]:

    # Initialize tokenizer
    tokenizer = tokenizer_class.from_pretrained(f"{model_name.lower()}-base-uncased" if model_name == "BERT" else "roberta-base")
    
    # Create datasets
    train_dataset = SentimentDataset(
        train_df['text'].tolist(),
        train_df['title'].tolist(),
        train_df['price'].tolist(),
        train_df['price_missing'].tolist(),
        train_df['helpful_vote'].tolist(),
        train_df['verified_purchase'].tolist(),
        train_df['label'].tolist(),
        tokenizer
    )
    val_dataset = SentimentDataset(
        val_df['text'].tolist(),
        val_df['title'].tolist(),
        val_df['price'].tolist(),
        val_df['price_missing'].tolist(),
        val_df['helpful_vote'].tolist(),
        val_df['verified_purchase'].tolist(),
        val_df['label'].tolist(),
        tokenizer
    )
    test_dataset = SentimentDataset(
        test_df['text'].tolist(),
        test_df['title'].tolist(),
        test_df['price'].tolist(),
        test_df['price_missing'].tolist(),
        test_df['helpful_vote'].tolist(),
        test_df['verified_purchase'].tolist(),
        test_df['label'].tolist(),
        tokenizer
    )

    # Experiment with each combination of hyperparameters
    for params in param_combinations:
        learning_rate, batch_size, num_epochs, weight_decay, dropout_rate = params

        # Start an MLflow run
        with mlflow.start_run(run_name=f"{model_name}_experiment"):
            # Log model type and hyperparameters
            mlflow.log_param("model", model_name)
            mlflow.log_param("learning_rate", learning_rate)
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("num_epochs", num_epochs)
            mlflow.log_param("weight_decay", weight_decay)
            mlflow.log_param("dropout_rate", dropout_rate)

            # Log the label encoding mapping for this experiment
            mlflow.log_param("label_mapping", label_mapping)

            # Initialize model with custom dropout rate
            model = initialize_model(num_labels=len(label_encoder.classes_))
            model.to(DEVICE)

            # Train the model with custom training arguments
            training_args = {
                "learning_rate": learning_rate,
                "per_device_train_batch_size": batch_size,
                "num_train_epochs": num_epochs,
                "weight_decay": weight_decay
            }
            eval_results, trainer = train_model(
                model=model,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                output_dir=f"./{model_name.lower()}_output",
                **training_args
            )

            # Log evaluation results
            for metric, value in eval_results.items():
                print(f"Logging validation metric: val_{metric} = {value}")
                mlflow.log_metric(f"val_{metric}", value)

            # Evaluate the model on the test set
            test_results = eval_model(trainer, test_dataset)
            for metric, value in test_results.items():
                print(f"Logging test metric: test_{metric} = {value}")  # Print metric name and value before logging
                # If value is an array (e.g., per-class precision/recall), log each element separately with class labels
                if isinstance(value, (list, tuple, np.ndarray)):
                    for i, class_value in enumerate(value):
                        class_label = class_labels[i]  # Get class label
                        print(f"Logging test metric for {class_label}: test_{metric}_{class_label} = {class_value}")
                        mlflow.log_metric(f"test_{metric}_{class_label}", float(class_value))  # Convert to float if needed
                else:
                    # Log scalar metrics directly
                    mlflow.log_metric(f"test_{metric}", float(value))  # Convert to float if needed

            # Log the trained model as an artifact
            mlflow.pytorch.log_model(model, f"{model_name}_model")

            print(f"Completed experiment for {model_name} with learning rate {learning_rate}, batch size {batch_size}, num epochs {num_epochs}")

        # Release VRAM memory after each run
        del model
        del trainer
        torch.cuda.empty_cache()
