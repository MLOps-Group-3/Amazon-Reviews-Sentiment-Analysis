import os
import torch
from flask import Flask, request, jsonify
from transformers import BertTokenizer
from utils.bert_model import initialize_bert_model

app = Flask(__name__)

# Environment variables
# MODEL_NAME = os.getenv("MODEL_NAME", "BERT")
MODEL_FILE = os.getenv("MODEL_FILE", f"final_model.pth")  # Path to the local model file
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Load the model from the local file system
def load_model():
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(f"Model file not found at {MODEL_FILE}")

    model = initialize_bert_model(num_labels=3).to(DEVICE)  # Adjust `num_labels` as needed
    model.load_state_dict(torch.load(MODEL_FILE, map_location=DEVICE))
    model.eval()
    print(f"Model loaded successfully from {MODEL_FILE}")
    return model

model = load_model()

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print(f"Received data: {data}")

        texts = data.get("texts", [])
        if not texts:
            return jsonify({"error": "No texts provided"}), 400

        # Tokenize inputs
        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(DEVICE)

        # Prepare additional features if your model uses them
        additional_features = data.get("additional_features", None)
        if additional_features is not None:
            additional_features = torch.tensor(additional_features, dtype=torch.float32).to(DEVICE)
        else:
            # Use dummy additional features if not provided
            num_additional_features = 4  # Adjust based on your model
            additional_features = torch.zeros((len(texts), num_additional_features)).to(DEVICE)

        # Make predictions
        with torch.no_grad():
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                additional_features=additional_features
            )
            predictions = torch.argmax(outputs, dim=1).cpu().numpy().tolist()

        return jsonify({"predictions": predictions})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Health check endpoint
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
    app.debug(True)