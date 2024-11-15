import torch
import logging
import json
from utils.bert_model import initialize_bert_model, train_bert_model
from utils.roberta_model import initialize_roberta_model, train_roberta_model
from utils.data_loader import load_and_process_data, split_data_by_timestamp, SentimentDataset
from transformers import BertTokenizer, RobertaTokenizer
from google.cloud import storage
import mlflow

# Disable MLflow autologging
mlflow.autolog(disable=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths and Device Configuration
DATA_PATH = "labeled_data.csv"  # Full dataset path
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Clear GPU memory at the start
if DEVICE == "cuda":
    torch.cuda.empty_cache()
    logger.info("Cleared GPU memory before starting.")

# Load the best hyperparameters from the tuning phase
def load_hyperparameters(path="best_hyperparameters.json"):
    try:
        with open(path, "r") as f:
            best_hyperparameters = json.load(f)
        logger.info(f"Loaded hyperparameters: {best_hyperparameters}")
        return best_hyperparameters
    except FileNotFoundError:
        logger.error(f"Hyperparameter file not found at {path}")
        raise

# Prepare full dataset for training and validation
def prepare_data(data_path):
    try:
        df, label_encoder = load_and_process_data(data_path)
        train_df, val_df, _ = split_data_by_timestamp(df)  # Only use train and validation
        return train_df, val_df, label_encoder.classes_
    except Exception as e:
        logger.error(f"Error loading or processing data: {e}")
        raise

# Model Initialization and Tokenizer Selection
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

# Train and Save the Final Model
def train_and_save_final_model(hyperparameters):
    model_name = hyperparameters["model_name"]
    learning_rate = hyperparameters["learning_rate"]
    batch_size = hyperparameters["batch_size"]
    num_epochs = hyperparameters["num_epochs"]
    weight_decay = hyperparameters["weight_decay"]

    # Load train and validation data
    train_df, val_df, class_labels = prepare_data(DATA_PATH)

    # Initialize model and tokenizer
    initialize_model_func, tokenizer = initialize_model_and_tokenizer(model_name)
    model = initialize_model_func(num_labels=len(class_labels)).to(DEVICE)

    # Convert dataframes to datasets
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

    # Set up training arguments
    training_args = {
        "learning_rate": learning_rate,
        "per_device_train_batch_size": batch_size,
        "num_train_epochs": num_epochs,
        "weight_decay": weight_decay
    }

    # Train the model
    logger.info(f"Training {model_name} model with final hyperparameters")
    if model_name == "BERT":
        _, trainer = train_bert_model(model, train_dataset, val_dataset,
                                      output_dir=f"./{model_name.lower()}_output", **training_args)
    elif model_name == "RoBERTa":
        _, trainer = train_roberta_model(model, train_dataset, val_dataset,
                                         output_dir=f"./{model_name.lower()}_output", **training_args)

    # Prepare dummy inputs for scripting
    sample_input_ids = torch.ones((1, 512), dtype=torch.long).to(DEVICE)
    sample_attention_mask = torch.ones((1, 512), dtype=torch.long).to(DEVICE)
    num_additional_features = 4  # Replace with the actual number of additional features
    sample_additional_features = torch.zeros((1, num_additional_features)).to(DEVICE)

    # Script the model
    model.eval()
    scripted_model = torch.jit.script(
        model
    )
    torchscript_path = f"{model_name}_final_model.pt"
    scripted_model.save(torchscript_path)
    logger.info(f"Saved final model as TorchScript at {torchscript_path}")

    # Upload model to GCP Cloud Storage
    gcs_bucket_name = "model_storage_arsa"  # Replace with your bucket name
    gcs_model_path = f"models/{model_name}_final_model.pt"
    upload_to_gcs(gcs_bucket_name, torchscript_path, gcs_model_path)

def upload_to_gcs(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to Google Cloud Storage."""
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(source_file_name)
        logger.info(f"Uploaded {source_file_name} to gs://{bucket_name}/{destination_blob_name}")
    except Exception as e:
        logger.error(f"Error uploading to GCS: {e}")
        raise

if __name__ == "__main__":
    # Load best hyperparameters
    best_hyperparameters = load_hyperparameters("best_hyperparameters.json")

    # Train and save final model
    train_and_save_final_model(best_hyperparameters)
