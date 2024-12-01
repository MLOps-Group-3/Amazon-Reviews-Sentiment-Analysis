import torch
import logging
import json
import os
import pickle
from utils.bert_model import initialize_bert_model, train_bert_model
from utils.roberta_model import initialize_roberta_model, train_roberta_model
from utils.data_loader import SentimentDataset
from transformers import BertTokenizer, RobertaTokenizer
from google.cloud import storage
import gcsfs
import pandas as pd
import re
# from config import DATA_SAVE_PATH, MODEL_SAVE_PATH
# Disable MLflow autologging
# import mlflow
# mlflow.autolog(disable=True)
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths and Device Configuration
# DATA_PATH =  DATA_PATH # Directory where train/val/test data is saved as .pkl
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"DEVICE:{DEVICE}")
# Clear GPU memory at the start
if DEVICE == "cuda":
    torch.cuda.empty_cache()
    logger.info("Cleared GPU memory before starting.")

# Load hyperparameters from the JSON file
def load_hyperparameters(path="best_hyperparameters.json"):
    try:
        with open(path, "r") as f:
            best_hyperparameters = json.load(f)
        logger.info(f"Loaded hyperparameters: {best_hyperparameters}")
        return best_hyperparameters
    except FileNotFoundError:
        logger.error(f"Hyperparameter file not found at {path}")
        raise

# Load train and validation data from pickled files
# def load_data(data_save_path):
#     try:
#         train_path = os.path.join(data_save_path, "train.pkl")
#         val_path = os.path.join(data_save_path, "val.pkl")

#         with open(train_path, "rb") as f:
#             train_df = pickle.load(f)
#         with open(val_path, "rb") as f:
#             val_df = pickle.load(f)

#         logger.info("Train and validation data loaded successfully.")
#         return train_df, val_df
#     except Exception as e:
#         logger.error(f"Error loading pickled data: {e}")
#         raise

def load_data(data_save_path):
    try:
        # Initialize the GCS filesystem
        fs = gcsfs.GCSFileSystem()
        data_save_path = data_save_path.rstrip('/')

        train_path = f"{data_save_path}/train.pkl"
        val_path = f"{data_save_path}/val.pkl"

        # Open GCS files using gcsfs
        with fs.open(train_path, "rb") as f:
            train_df = pickle.load(f)
        with fs.open(val_path, "rb") as f:
            val_df = pickle.load(f)

        logger.info("Train and validation data loaded successfully.")
        return train_df, val_df
    except Exception as e:
        logger.error(f"Error loading pickled data from GCS: {e}")
        raise

# Initialize model and tokenizer based on the model name
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

# Train and save the model
def train_and_save_final_model(hyperparameters,data_path,train_data=None,val_data=None,model_save_path=None):
    model_name = hyperparameters["model_name"]
    learning_rate = hyperparameters["learning_rate"]
    batch_size = hyperparameters["batch_size"]
    num_epochs = hyperparameters["num_epochs"]
    weight_decay = hyperparameters["weight_decay"]

    if train_data==None or val_data==None:
        # Load train and validation data
        train_df, val_df = load_data(data_path)
    else:
        train_df = pd.read_pickle(train_data.path)
        val_df = pd.read_pickle(val_data.path)
        logger.info("Loaded train and validation df from Artifact")

    # Initialize model and tokenizer
    initialize_model_func, tokenizer = initialize_model_and_tokenizer(model_name)
    model = initialize_model_func(num_labels=len(train_df['label'].unique())).to(DEVICE)

    # Convert dataframes to datasets
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

    # Set up training arguments
    training_args = {
        "learning_rate": learning_rate,
        "per_device_train_batch_size": batch_size,
        "num_train_epochs": num_epochs,
        "weight_decay": weight_decay
    }
    # Initialize metrics dictionary
    epoch_metrics = []
    # Train the model
    logger.info(f"Training {model_name} model with final hyperparameters")
    if model_name == "BERT":
        _, trainer = train_bert_model(
            model,
            train_dataset,
            val_dataset,
            output_dir=f"./{model_name.lower()}_output",
            metrics_logger=epoch_metrics.append,
            **training_args
        )
    elif model_name == "RoBERTa":
        _, trainer = train_roberta_model(
            model,
            train_dataset,
            val_dataset,
            output_dir=f"./{model_name.lower()}_output",
            metrics_logger=epoch_metrics.append,
            **training_args
        )

    # Save the model's state dictionary
    # model_save_path = f"final_model.pth"
    local_model_path = "/tmp/final_model.pth"
    torch.save(model.state_dict(), local_model_path)
    logger.info(f"Model state dictionary saved at {local_model_path}")

    # Upload model to GCP Cloud Storage
    # gcs_bucket_name = "model_storage_arsa"  # Replace with your bucket name
    # gcs_model_path = f"models/{model_save_path}"
    upload_to_gcs(model_save_path,local_model_path)

    return model_save_path, epoch_metrics

# Upload the model to Google Cloud Storage
def upload_to_gcs(model_save_path, source_file_name):
    try:
        logger.info(f"Model save path:{model_save_path}")
        # Parse bucket name and blob path from model_save_path
        match = re.match(r"gs://([^/]+)/(.+)", model_save_path)
        if not match:
            raise ValueError(f"Invalid GCS path format: {model_save_path}")

        bucket_name, blob_path = match.groups()

        # Initialize GCS client and upload file
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)

        logger.info(f"Uploading to bucket: {bucket_name}, blob path: {blob_path}")
        blob.upload_from_filename(source_file_name)
        logger.info(f"Uploaded {source_file_name} to {model_save_path}")

        # Confirm the file exists in GCS after upload
        if blob.exists():
            logger.info(f"File successfully uploaded to {model_save_path}")
        else:
            logger.error(f"File not found in GCS after upload: {model_save_path}")

    except Exception as e:
        logger.error(f"Error uploading to GCS: {e}")
        raise

if __name__ == "__main__":
    # Load best hyperparameters
    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # hyper_param_path = os.path.join(script_dir, "best_hyperparameters.json".lstrip("/"))
    # data_path = os.path.join(script_dir, DATA_SAVE_PATH.lstrip("/"))
    # model_path = os.path.join(script_dir,MODEL_SAVE_PATH.lstrip("/"))
    # best_hyperparameters = load_hyperparameters(hyper_param_path)

    # # Train and save final model
    # train_and_save_final_model(best_hyperparameters,data_path,model_path)
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True, help="Path to the training data directory.")
    parser.add_argument("--model_save_path", required=True, help="Path to save the trained model.")
    args = parser.parse_args()

    hyper_param_path = "best_hyperparameters.json"
    hyperparameters = load_hyperparameters(hyper_param_path)
    train_and_save_final_model(hyperparameters, args.data_path, args.model_save_path)
