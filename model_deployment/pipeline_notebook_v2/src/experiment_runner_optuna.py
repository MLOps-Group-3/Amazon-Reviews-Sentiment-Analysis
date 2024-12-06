import optuna
import mlflow
import torch
import numpy as np
import logging
import copy
import json
from utils.bert_model import initialize_bert_model, train_bert_model, evaluate_bert_model
from utils.roberta_model import initialize_roberta_model, train_roberta_model, evaluate_roberta_model
from utils.data_loader import load_and_process_data, split_data_by_timestamp, SentimentDataset
from transformers import BertTokenizer, RobertaTokenizer
from sklearn.metrics import f1_score
from datetime import datetime
import os
from config import DATA_PATH

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, DATA_PATH.lstrip("/"))
# BUCKET_NAME = "arsa_model_deployment_uscentral_v2"
# DATA_PATH = f"gs://{BUCKET_NAME}/input/labeled_data_1perc.csv"
# print(DATA_PATH)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Experiment setup
timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
experiment_name = f"Review_Sentiment_{timestamp}"
mlflow.set_experiment(experiment_name)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Global variables to track the best model
best_model_state = None
best_model_params = None
best_score = float('-inf')

# 1. Data Preparation (with sampling)
def prepare_data(data_path, frac=1.0):
    df, label_encoder = load_and_process_data(data_path)
    # Use a subset of the data if `frac` < 1.0
    if frac < 1.0:
        df = df.sample(frac=frac, random_state=42)
    train_df, val_df, test_df = split_data_by_timestamp(df)
    return train_df, val_df, test_df, label_encoder.classes_

# Model Initialization and Tokenizer
def initialize_model_and_tokenizer(model_name):
    if model_name == "BERT":
        model_init = initialize_bert_model
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    elif model_name == "RoBERTa":
        model_init = initialize_roberta_model
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    else:
        raise ValueError(f"Model {model_name} is not supported.")
    return model_init, tokenizer

# 3. Objective Function for Hyperparameter Tuning
def create_objective_function(datapath):

    def objective(trial):
        global best_model_state, best_model_params, best_score
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        # Sample hyperparameters
        model_name = trial.suggest_categorical("model_name", ["BERT"])
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
        batch_size = trial.suggest_categorical("batch_size", [64])
        num_epochs = trial.suggest_int("num_epochs", 1, 1)
        weight_decay = trial.suggest_loguniform("weight_decay", 1e-4, 0.1)
        dropout_rate = trial.suggest_uniform("dropout_rate", 0.1, 0.5)

        initialize_model_func, tokenizer = initialize_model_and_tokenizer(model_name)
        
        # Load a subset of data (e.g., 10%)

        logger.info(f"Input Data path:{datapath}")
        train_df, val_df, test_df, class_labels = prepare_data(datapath, frac=0.1)
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

        # Format the run name based on model type and hyperparameters
        run_name = f"{model_name}_lr_{learning_rate:.0e}_wd_{weight_decay:.0e}_do_{dropout_rate:.1f}"

        with mlflow.start_run(run_name=run_name) as run:
            # Log hyperparameters
            mlflow.log_params({
                "model_name": model_name,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "num_epochs": num_epochs,
                "weight_decay": weight_decay,
                "dropout_rate": dropout_rate
            })

            model = initialize_model_func(num_labels=len(class_labels)).to(DEVICE)
            training_args = {
                "learning_rate": learning_rate,
                "per_device_train_batch_size": batch_size,
                "num_train_epochs": num_epochs,
                "weight_decay": weight_decay
            }

            # Train and evaluate the model
            if model_name == "BERT":
                eval_results, trainer = train_bert_model(model, train_dataset, val_dataset,
                                                        output_dir=f"./{model_name.lower()}_output", **training_args)
            elif model_name == "RoBERTa":
                eval_results, trainer = train_roberta_model(model, train_dataset, val_dataset,
                                                            output_dir=f"./{model_name.lower()}_output", **training_args)

            # Evaluate on test set
            test_results, test_f1_score = evaluate_and_log_test_metrics(trainer, test_dataset, class_labels)
            mlflow.log_metric("test_f1", test_f1_score)

            # Track the best model
            if test_f1_score > best_score:
                best_score = test_f1_score
                best_model_state = copy.deepcopy(model.state_dict())
                best_model_params = {
                    "model_name": model_name,
                    "learning_rate": learning_rate,
                    "batch_size": batch_size,
                    "num_epochs": num_epochs,
                    "weight_decay": weight_decay,
                    "dropout_rate": dropout_rate
                }
                mlflow.log_params(best_model_params)
                logger.info(f"New best model found: {test_f1_score}")

            del model
            del trainer
            torch.cuda.empty_cache()

        return test_f1_score
    return objective

# 4. Evaluate and Log Test Metrics
def evaluate_and_log_test_metrics(trainer, test_dataset, class_labels):
    predictions_output = trainer.predict(test_dataset)
    predictions = predictions_output.predictions
    labels = predictions_output.label_ids
    predictions = np.argmax(predictions, axis=1)
    overall_f1_score = f1_score(labels, predictions, average='weighted')
    return {"f1": f1_score(labels, predictions, average=None)}, overall_f1_score

# Function to save the best hyperparameters to a JSON file
def save_hyperparameters(hyperparameters, path="best_hyperparameters.json"):
    with open(path, "w") as f:
        json.dump(hyperparameters, f, indent=4)
    logger.info(f"Best hyperparameters saved to {path}")

# Function to retrieve the best hyperparameters after optimization
def find_best_hyperparameters(datapath):
    study = optuna.create_study(direction="maximize")
    objective = create_objective_function(datapath)
    study.optimize(objective, n_trials=8)
    

    # Retrieve the best model name and parameters for later use
    best_trial = study.best_trial
    logger.info(f"Best trial F1 score: {best_trial.value}")
    logger.info(f"Best hyperparameters: {best_trial.params}")
    
    # Save best parameters to JSON
    save_hyperparameters(best_model_params)
    return best_model_params

if __name__ == "__main__":
    # Run the tuning on a subset of data and log the best params
    best_params = find_best_hyperparameters(DATA_PATH)
    print("Best Model Hyperparameters:", best_params)
