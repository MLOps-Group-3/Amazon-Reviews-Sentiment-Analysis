import torch
import pandas as pd
from transformers import RobertaTokenizer
from roberta_model import CustomRoBERTaModel, initialize_roberta_model

# Constants
MODEL_PATH = "./roberta_output/best_model/complete_model.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_LABELS = 3  # Adjust this based on your actual number of labels


# Load the model
def load_model(model_path):
    model = torch.load(model_path, map_location=DEVICE)
    model.to(DEVICE)
    model.eval()
    return model


model = load_model(MODEL_PATH)
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')


# Preprocess input
def preprocess_input(text, title, price, price_missing, helpful_vote, verified_purchase):
    inputs = tokenizer(f"{title} {text}", padding=True, truncation=True, return_tensors="pt")
    additional_features = torch.tensor([[price, price_missing, helpful_vote, verified_purchase]], dtype=torch.float)
    return inputs, additional_features


# Inference function
def predict(model, text, title, price, price_missing, helpful_vote, verified_purchase):
    inputs, additional_features = preprocess_input(text, title, price, price_missing, helpful_vote, verified_purchase)

    with torch.no_grad():
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        additional_features = additional_features.to(DEVICE)
        outputs = model(input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        additional_features=additional_features)

    logits = outputs['logits']
    probabilities = torch.softmax(logits, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()

    return predicted_class, probabilities.squeeze().tolist()


# Single prediction function
def single_prediction(text, title, price, price_missing, helpful_vote, verified_purchase):
    predicted_class, probabilities = predict(model, text, title, price, price_missing, helpful_vote, verified_purchase)
    return predicted_class, probabilities


# Batch prediction function
def batch_prediction(input_file, output_file):
    df = pd.read_csv(input_file)

    results = []
    for _, row in df.iterrows():
        predicted_class, probabilities = predict(
            model,
            row['text'],
            row['title'],
            row['price'],
            row['price_missing'],
            row['helpful_vote'],
            row['verified_purchase']
        )
        results.append({
            'id': row.get('id', ''),
            'predicted_class': predicted_class,
            'probabilities': probabilities
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    print(f"Batch inference completed. Results saved to {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RoBERTa Model Inference")
    parser.add_argument("--mode", choices=["single", "batch"], default="single", help="Inference mode: single or batch")
    parser.add_argument("--input", help="Input CSV file for batch prediction")
    parser.add_argument("--output", help="Output CSV file for batch prediction results")

    args = parser.parse_args()

    if args.mode == "single":
        # Example input for single prediction
        text = "This product is amazing! I love it."
        title = "Great purchase"
        price = 29.99
        price_missing = 0
        helpful_vote = 5
        verified_purchase = 1

        predicted_class, probabilities = single_prediction(text, title, price, price_missing, helpful_vote,
                                                           verified_purchase)
        print(f"Predicted class: {predicted_class}")
        print(f"Class probabilities: {probabilities}")
    elif args.mode == "batch":
        if not args.input or not args.output:
            print("Please provide input and output file paths for batch prediction.")
        else:
            batch_prediction(args.input, args.output)


# To use this script for batch prediction:
# Prepare an input CSV file with columns: 'text', 'title', 'price', 'price_missing', 'helpful_vote', 'verified_purchase', and optionally 'id'.
# Run the script with the following command:
# python inference.py --mode batch --input input_file.csv --output output_file.csv