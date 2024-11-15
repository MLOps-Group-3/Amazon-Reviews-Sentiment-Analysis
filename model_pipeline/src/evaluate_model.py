import torch
import logging
import pickle
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from transformers import BertTokenizer, RobertaTokenizer
from config import DATA_SAVE_PATH, MODEL_SAVE_PATH
from utils.data_loader import SentimentDataset
from utils.bert_model import initialize_bert_model
from utils.roberta_model import initialize_roberta_model

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Device Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load the test data
def load_test_data(data_save_path):
    try:
        test_path = os.path.join(data_save_path, "test.pkl")
        with open(test_path, "rb") as f:
            test_df = pickle.load(f)
        
        # Verify the 'label' column exists
        if 'label' not in test_df.columns:
            raise KeyError("'label' column is missing in the test dataset.")
        
        logger.info("Test data loaded successfully.")
        return test_df
    except Exception as e:
        logger.error(f"Error loading test data: {e}")
        raise

# Load the label encoder
def load_label_encoder(data_save_path):
    try:
        label_encoder_path = os.path.join(data_save_path, "label_encoder.pkl")
        with open(label_encoder_path, "rb") as f:
            label_encoder = pickle.load(f)
        logger.info("Label encoder loaded successfully.")
        return label_encoder
    except Exception as e:
        logger.error(f"Error loading label encoder: {e}")
        raise

# Initialize model and tokenizer
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

# Evaluate the model on the test dataset
def evaluate_model(model, test_dataset, label_encoder):
    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for i in range(len(test_dataset)):
            sample = test_dataset[i]
            input_ids = sample["input_ids"].unsqueeze(0).to(DEVICE)
            attention_mask = sample["attention_mask"].unsqueeze(0).to(DEVICE)
            additional_features = sample["additional_features"].to(DEVICE)
            label = sample["labels"].item()
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, additional_features=additional_features)
            prediction = torch.argmax(outputs["logits"], dim=1).item()

            all_labels.append(label)
            all_predictions.append(prediction)

    # Map labels back to class names
    class_names = label_encoder.classes_
    report = classification_report(all_labels, all_predictions, target_names=class_names)
    logger.info(f"Classification Report:\n{report}")

    # Calculate overall metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average="weighted")
    recall = recall_score(all_labels, all_predictions, average="weighted")
    f1 = f1_score(all_labels, all_predictions, average="weighted")

    return accuracy, precision, recall, f1

def main():
    # Model details
    model_name = "BERT"  # Change to "RoBERTa" if needed

    # Load paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, DATA_SAVE_PATH.lstrip("/"))
    model_path = os.path.join(script_dir, MODEL_SAVE_PATH.lstrip("/"))

    # Load test data and label encoder
    test_df = load_test_data(data_path)
    label_encoder = load_label_encoder(data_path)

    # Initialize model and tokenizer
    initialize_model_func, tokenizer = initialize_model_and_tokenizer(model_name)
    model = initialize_model_func(num_labels=len(label_encoder.classes_)).to(DEVICE)

    # Load the trained model's state dictionary
    if not os.path.exists(model_path):
        logger.error(f"Trained model not found at {model_path}")
        return

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    logger.info("Trained model loaded successfully.")

    # Convert test dataframe to dataset
    test_dataset = SentimentDataset(
        test_df["text"].tolist(),
        test_df["title"].tolist(),
        test_df["price"].tolist(),
        test_df["price_missing"].tolist(),
        test_df["helpful_vote"].tolist(),
        test_df["verified_purchase"].tolist(),
        test_df["label"].tolist(),
        tokenizer
    )

    # Evaluate the model
    accuracy, precision, recall, f1 = evaluate_model(model, test_dataset, label_encoder)
    logger.info(f"Overall Metrics:\n"
                f"Accuracy: {accuracy:.4f}\n"
                f"Precision: {precision:.4f}\n"
                f"Recall: {recall:.4f}\n"
                f"F1 Score: {f1:.4f}")

if __name__ == "__main__":
    main()
