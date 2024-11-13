import mlflow
import itertools
import torch
import numpy as np
import logging
from utils.bert_model import initialize_bert_model, train_bert_model, evaluate_bert_model
from utils.roberta_model import initialize_roberta_model, train_roberta_model, evaluate_roberta_model
from utils.data_loader import load_and_process_data, split_data_by_timestamp, SentimentDataset
from transformers import BertTokenizer, RobertaTokenizer

# Configuration for data path and device
DATA_PATH = "labeled_data.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Define hyperparameter grid with additional parameters
param_grid = {
    "learning_rate": [2e-5],
    "batch_size": [64],
    "num_epochs": [3],
    "weight_decay": [0.01, 0.001],
    "dropout_rate": [0.1, 0.3]
}

# 1. Data Preparation
def prepare_data(data_path):
    df, label_encoder = load_and_process_data(data_path)
    train_df, val_df, test_df = split_data_by_timestamp(df)
    logger.info("Data loaded and split into train, validation, and test sets.")
    return train_df, val_df, test_df, label_encoder.classes_

# 2. Model Initialization and Tokenizer
def initialize_model_and_tokenizer(model_name):
    if model_name == "BERT":
        model_init = initialize_bert_model
        tokenizer_class = BertTokenizer
        base_model = "bert-base-uncased"
    elif model_name == "RoBERTa":
        model_init = initialize_roberta_model
        tokenizer_class = RobertaTokenizer
        base_model = "roberta-base"
    
    tokenizer = tokenizer_class.from_pretrained(base_model)
    logger.info(f"{model_name} model and tokenizer initialized.")
    return model_init, tokenizer

# 3. Experiment Runner
def run_experiment(model_name, initialize_model, tokenizer, train_df, val_df, test_df, class_labels):
    # Initialize datasets
    train_dataset = SentimentDataset(
        train_df['text'].tolist(), train_df['title'].tolist(), train_df['price'].tolist(),
        train_df['price_missing'].tolist(), train_df['helpful_vote'].tolist(),
        train_df['verified_purchase'].tolist(), train_df['label'].tolist(), tokenizer
    )
    val_dataset = SentimentDataset(
        val_df['text'].tolist(), val_df['title'].tolist(), val_df['price'].tolist(),
        val_df['price_missing'].tolist(), val_df['helpful_vote'].tolist(),
        val_df['verified_purchase'].tolist(), val_df['label'].tolist(), tokenizer
    )
    test_dataset = SentimentDataset(
        test_df['text'].tolist(), test_df['title'].tolist(), test_df['price'].tolist(),
        test_df['price_missing'].tolist(), test_df['helpful_vote'].tolist(),
        test_df['verified_purchase'].tolist(), test_df['label'].tolist(), tokenizer
    )

    # Run experiments for all hyperparameter combinations
    param_combinations = list(itertools.product(*param_grid.values()))
    for params in param_combinations:
        train_and_log(model_name, initialize_model, train_dataset, val_dataset, test_dataset, params, class_labels)

# 4. Training and Logging
def train_and_log(model_name, initialize_model, train_dataset, val_dataset, test_dataset, params, class_labels):
    learning_rate, batch_size, num_epochs, weight_decay, dropout_rate = params
    with mlflow.start_run(run_name=f"{model_name}_experiment"):
        # Log hyperparameters
        mlflow.log_params({
            "model": model_name,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "weight_decay": weight_decay,
            "dropout_rate": dropout_rate
        })

        # Initialize model
        model = initialize_model(num_labels=len(class_labels)).to(DEVICE)
        training_args = {
            "learning_rate": learning_rate,
            "per_device_train_batch_size": batch_size,
            "num_train_epochs": num_epochs,
            "weight_decay": weight_decay
        }

        # Train and log metrics
        eval_results, trainer = train_bert_model(model, train_dataset, val_dataset, output_dir=f"./{model_name.lower()}_output", **training_args)
        log_metrics("val", eval_results)

        # Evaluate on test set and log metrics
        test_results = evaluate_bert_model(trainer, test_dataset)
        log_metrics("test", test_results, class_labels)

        # Log model as artifact
        mlflow.pytorch.log_model(model, f"{model_name}_model")
        logger.info(f"Completed experiment for {model_name} with params: {params}")

        # Release memory
        del model
        del trainer
        torch.cuda.empty_cache()

# 5. Logging Metrics
def log_metrics(prefix, metrics, class_labels=None):
    for metric, value in metrics.items():
        if isinstance(value, (list, tuple, np.ndarray)):
            for i, class_value in enumerate(value):
                class_label = class_labels[i] if class_labels else f"class_{i}"
                mlflow.log_metric(f"{prefix}_{metric}_{class_label}", float(class_value))
                logger.info(f"Logged metric {prefix}_{metric}_{class_label} = {class_value}")
        else:
            mlflow.log_metric(f"{prefix}_{metric}", float(value))
            logger.info(f"Logged metric {prefix}_{metric} = {value}")

# 6. Retrieve Top Models by Metric
def get_top_models(metric_name="test_f1", top_n=2):
    client = mlflow.tracking.MlflowClient()
    experiment_id = client.get_experiment_by_name("YourExperimentName").experiment_id
    top_runs = client.search_runs(experiment_id, order_by=[f"metrics.{metric_name} DESC"], max_results=top_n)
    for i, run in enumerate(top_runs, start=1):
        logger.info(f"\nTop {i} Model:")
        logger.info(f"Run ID: {run.info.run_id}")
        logger.info(f"Model: {run.data.params['model']}")
        logger.info(f"Learning Rate: {run.data.params['learning_rate']}")
        logger.info(f"Batch Size: {run.data.params['batch_size']}")
        logger.info(f"Num Epochs: {run.data.params['num_epochs']}")
        logger.info(f"Weight Decay: {run.data.params['weight_decay']}")
        logger.info(f"Dropout Rate: {run.data.params['dropout_rate']}")
        logger.info(f"{metric_name}: {run.data.metrics[metric_name]}")

# Main Execution
if __name__ == "__main__":
    # Prepare data
    train_df, val_df, test_df, class_labels = prepare_data(DATA_PATH)
    
    # Loop through model types and run experiments
    for model_name in ["BERT", "RoBERTa"]:
        initialize_model, tokenizer = initialize_model_and_tokenizer(model_name)
        run_experiment(model_name, initialize_model, tokenizer, train_df, val_df, test_df, class_labels)
    
    # Retrieve and display top 2 models based on F1 score
    get_top_models(metric_name="test_f1", top_n=2)
