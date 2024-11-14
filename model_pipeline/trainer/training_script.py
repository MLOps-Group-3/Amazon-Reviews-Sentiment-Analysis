import logging
import torch
from transformers import RobertaTokenizer
from google.cloud import storage  # Add this import for GCS upload
from roberta_model import initialize_roberta_model, train_roberta_model, evaluate_roberta_model
from data_loader import load_and_process_data, split_data_by_timestamp, SentimentDataset
import os 

if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/hrs/Documents/GCP_keys/key.json"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define GCS bucket URI (Replace with your GCS bucket path)
GCS_BUCKET = "gs://model_storage_arsa/model/"  # Update this to your GCS bucket path

# Hyperparameters and configurations
DATA_PATH = "labeled_data.csv"  # Path to the data file
OUTPUT_DIR = "./roberta_output"  # Directory to save the trained model and logs
EPOCHS = 1  # Number of epochs to train
BATCH_SIZE = 64  # Batch size for training
LEARNING_RATE = 2e-5  # Learning rate for training
WEIGHT_DECAY = 0.01  # Weight decay for regularization
DROPOUT_RATE = 0.3  # Dropout rate for the model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Function to upload files to GCS
def upload_to_gcs(local_path, bucket_name, destination_path):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_path)
    blob.upload_from_filename(local_path)
    logger.info(f"Uploaded {local_path} to {destination_path} in bucket {bucket_name}")

# Main training and evaluation function
def main():
    logger.info("Loading and processing data...")
    df, label_encoder = load_and_process_data(DATA_PATH)
    
    # Split data into train, validation, and test sets
    train_df, val_df, test_df = split_data_by_timestamp(df)

    # Initialize the tokenizer and datasets
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
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

    # Initialize and configure the model
    model = initialize_roberta_model(num_labels=len(label_encoder.classes_), model_name="roberta-base")
    model.to(DEVICE)

    # Define training parameters
    training_args = {
        "learning_rate": LEARNING_RATE,
        "per_device_train_batch_size": BATCH_SIZE,
        "per_device_eval_batch_size": BATCH_SIZE,
        "num_train_epochs": EPOCHS,
        "weight_decay": WEIGHT_DECAY,
        "evaluation_strategy": "epoch",
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "logging_dir": OUTPUT_DIR + '/logs'
    }

    # Train the model
    logger.info("Starting training...")
    eval_results, trainer = train_roberta_model(model, train_dataset, val_dataset, OUTPUT_DIR, **training_args)

    # Evaluate the model on the test dataset
    logger.info("Evaluating the model on the test dataset...")
    test_results = evaluate_roberta_model(trainer, test_dataset)
    logger.info(f"Test Results: {test_results}")

    # Save the model locally
    # Path to save the model
    model_save_path = f"{OUTPUT_DIR}/best_model"
    os.makedirs(model_save_path, exist_ok=True)

    # Save the entire model (both pretrained and custom layers)
    torch.save(model, os.path.join(model_save_path, "complete_model.pt"))
    tokenizer.save_pretrained(model_save_path)
    logger.info(f"Complete model and tokenizer saved locally at {model_save_path}")

    # Upload to GCS
    bucket_name = GCS_BUCKET.split("/")[2]  # Extract bucket name from GCS_BUCKET
    destination_path = "/".join(GCS_BUCKET.split("/")[3:]) + "best_model"  # Define the destination path in GCS

    upload_to_gcs(os.path.join(model_save_path, "complete_model.pt"), bucket_name, f"{destination_path}/complete_model.pt")
    upload_to_gcs(os.path.join(model_save_path, "tokenizer_config.json"), bucket_name, f"{destination_path}/tokenizer_config.json")
    upload_to_gcs(os.path.join(model_save_path, "vocab.json"), bucket_name, f"{destination_path}/vocab.json")
    logger.info("Complete model and tokenizer uploaded to GCS successfully.")

if __name__ == "__main__":
    main()
