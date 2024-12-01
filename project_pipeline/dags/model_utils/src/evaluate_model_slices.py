import torch
import logging
import pickle
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import BertTokenizer, RobertaTokenizer
from config import DATA_SAVE_PATH, MODEL_SAVE_PATH
from utils.data_loader import SentimentDataset
from utils.bert_model import initialize_bert_model
from utils.roberta_model import initialize_roberta_model
import json
from google.cloud import storage


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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

# Load hyperparameters
def load_hyperparameters(file_path):
    try:
        with open(file_path, "r") as f:
            hyperparams = json.load(f)
        logger.info("Hyperparameters loaded successfully.")
        return hyperparams
    except Exception as e:
        logger.error(f"Error loading hyperparameters: {e}")
        raise

# Evaluate performance on slices and the full dataset
def evaluate_slices(data, model, tokenizer, data_path):
    # Define slices based on `year`, `main_category`
    slice_columns = ["year", "main_category"]
    metrics_by_slice = []

    # Evaluate slices
    for column in slice_columns:
        unique_values = data[column].unique()
        logger.info(f"Evaluating slices for column: {column}")

        for value in unique_values:
            slice_data = data[data[column] == value]
            logger.info(f"Evaluating slice: {column} = {value} ({len(slice_data)} samples)")

            metrics = evaluate_dataset(slice_data, model, tokenizer)
            metrics.update({"Slice Column": column, "Slice Value": value, "Samples": len(slice_data)})
            metrics_by_slice.append(metrics)

    # Evaluate full dataset
    logger.info("Evaluating the full dataset.")
    full_metrics = evaluate_dataset(data, model, tokenizer)
    full_metrics.update({"Slice Column": "Full Dataset", "Slice Value": "All", "Samples": len(data)})
    metrics_by_slice.append(full_metrics)

    # Create a DataFrame for metrics
    metrics_df = pd.DataFrame(metrics_by_slice)
    metrics_df.to_csv(f"{data_path}/slice_metrics.csv", index=False)
    logger.info("Metrics by slice saved to slice_metrics.csv.")
    return metrics_df

# Helper function to evaluate a dataset
def evaluate_dataset(data, model, tokenizer):
    # Convert data to dataset
    dataset = SentimentDataset(
        data["text"].tolist(),
        data["title"].tolist(),
        data["price"].tolist(),
        data["price_missing"].tolist(),
        data["helpful_vote"].tolist(),
        data["verified_purchase"].tolist(),
        data["label"].tolist(),
        tokenizer
    )

    # Perform inference
    model.eval()
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]
            input_ids = sample["input_ids"].unsqueeze(0).to(DEVICE)
            attention_mask = sample["attention_mask"].unsqueeze(0).to(DEVICE)
            additional_features = sample["additional_features"].to(DEVICE)
            label = sample["labels"].item()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, additional_features=additional_features)
            prediction = torch.argmax(outputs["logits"], dim=1).item()

            all_labels.append(label)
            all_predictions.append(prediction)

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average="weighted")
    recall = recall_score(all_labels, all_predictions, average="weighted")
    f1 = f1_score(all_labels, all_predictions, average="weighted")

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }

def gcp_eval_slices(test_df=None,model_path=None):

    script_dir = os.path.dirname(os.path.abspath(__file__))
    hyperparams_path = os.path.join(script_dir, "best_hyperparameters.json")
    hyperparams = load_hyperparameters(hyperparams_path)

    # Get model name from hyperparameters
    model_name = hyperparams.get("model_name", "BERT")  # Default to BERT if not specified
    data_path = os.path.join(script_dir, DATA_SAVE_PATH.lstrip("/"))

    # Load paths
    if test_df==None:
        data_path = os.path.join(script_dir, DATA_SAVE_PATH.lstrip("/"))
        test_df = load_test_data(data_path)
    else:
        test_df = pd.read_pickle(test_df.path)
        logger.info("Loaded test_df from dataprep artifact")

    if model_path==None:
        model_path = os.path.join(script_dir, MODEL_SAVE_PATH.lstrip("/"))
    else:
        # model_path = model_path.uri
        if model_path.startswith("gs://"):
            local_model_path = "/tmp/final_model.pth"  # Temporary path to save the downloaded model
            client = storage.Client()
            bucket_name = model_path.split("/")[2]
            blob_path = "/".join(model_path.split("/")[3:])
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_path)
            
            logger.info(f"Downloading model from {model_path} to {local_model_path}")
            blob.download_to_filename(local_model_path)
            model_path = local_model_path
        else:
            raise ValueError(f"Invalid model_path format. Expected GCS URI starting with 'gs://', got: {model_path}")

    from evaluate_model import load_test_data, load_label_encoder
    label_encoder = load_label_encoder(data_path)

    if not os.path.exists(model_path):
        logger.error(f"Trained model not found at {model_path}")
        return

    # Initialize model and tokenizer
    initialize_model_func, tokenizer = initialize_model_and_tokenizer(model_name)
    model = initialize_model_func(num_labels=len(label_encoder.classes_)).to(DEVICE)

    # Load trained model
    if not os.path.exists(model_path):
        logger.error(f"Trained model not found at {model_path}")
        return
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    logger.info("Trained model loaded successfully.")

    # Evaluate slices and the full dataset
    metrics_df = evaluate_slices(test_df, model, tokenizer, data_path)
    logger.info(f"Metrics:\n{metrics_df}")

    return metrics_df


def main():
    # Load hyperparameters
    script_dir = os.path.dirname(os.path.abspath(__file__))
    hyperparams_path = os.path.join(script_dir, "best_hyperparameters.json")
    hyperparams = load_hyperparameters(hyperparams_path)

    # Get model name from hyperparameters
    model_name = hyperparams.get("model_name", "BERT")  # Default to BERT if not specified

    # Load paths
    data_path = os.path.join(script_dir, DATA_SAVE_PATH.lstrip("/"))
    model_path = os.path.join(script_dir, MODEL_SAVE_PATH.lstrip("/"))

    # Load test data and label encoder
    from evaluate_model import load_test_data, load_label_encoder  # Import your functions
    test_df = load_test_data(data_path)
    label_encoder = load_label_encoder(data_path)

    # Initialize model and tokenizer
    initialize_model_func, tokenizer = initialize_model_and_tokenizer(model_name)
    model = initialize_model_func(num_labels=len(label_encoder.classes_)).to(DEVICE)

    # Load trained model
    if not os.path.exists(model_path):
        logger.error(f"Trained model not found at {model_path}")
        return
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    logger.info("Trained model loaded successfully.")

    # Evaluate slices and the full dataset
    metrics_df = evaluate_slices(test_df, model, tokenizer, data_path)
    logger.info(f"Metrics:\n{metrics_df}")

if __name__ == "__main__":
    main()
