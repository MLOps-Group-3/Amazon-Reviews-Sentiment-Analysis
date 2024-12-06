
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json
import logging
import torch
from transformers import BertTokenizer
from ts.torch_handler.base_handler import BaseHandler
from bert_model_torch_serve import initialize_bert_model

logger = logging.getLogger(__name__)

class TransformersClassifierHandler(BaseHandler):
    """
    A custom handler for PyTorch Serve to handle BERT-based models
    with additional layers and `.pth`-saved state_dict.
    """
    def __init__(self):
        super(TransformersClassifierHandler, self).__init__()
        self.initialized = False

    def initialize(self, ctx):
        """
        Initialize the model, tokenizer, and label mapping for inference.
        """
        self.manifest = ctx.manifest
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load the model architecture
        num_labels = 3  # Adjust based on your dataset
        self.model = initialize_bert_model(num_labels=num_labels).to(self.device)

        # Load the model state_dict
        serialized_file = self.manifest["model"]["serializedFile"]
        model_path = os.path.join(model_dir, serialized_file)
        if not os.path.isfile(model_path):
            raise RuntimeError(f"Missing the model file: {model_path}")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        logger.info(f"Model loaded from {model_path}")

        # Load the tokenizer
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        # Load the label mapping
        mapping_file_path = os.path.join(model_dir, "index_to_name.json")
        if os.path.isfile(mapping_file_path):
            with open(mapping_file_path, "r") as f:
                self.mapping = json.load(f)
        else:
            logger.warning("index_to_name.json not found. Using default label mapping.")
            self.mapping = {"0": "Negative", "1": "Neutral", "2": "Positive"}

        self.initialized = True

    def preprocess(self, data):
        """
        Preprocessing input request by tokenizing and extracting additional features.
        Extend with your own preprocessing steps as needed.
        """
        try:
            # Log the received data
            logger.info(f"Received input data: {data}")
            logger.info(f"Data type: {type(data)}")


            # Extract instances
            instances = data#.keys()
            logger.info(f"Parsed instances: {instances}")

            texts = []
            additional_features = []
            
            # Extract and validate data from each instance
            for instance in instances:
                if isinstance(instance,list):
                    logger.info(instance)
                    instance_data = instance#.get("data")
                    # if not all(key in instance for key in ["text", "price", "price_missing", "helpful_vote", "verified_purchase"]):
                    #     raise ValueError(f"Invalid instance format: {instance}")
                    
                    texts.append(instance[0])
                    additional_features.append([
                        float(instance_data[1]),
                        float(instance_data[2]),
                        float(instance_data[3]),
                        float(instance_data[4])
                    ])

                else:
                    instance_data = instance#.get("data")
                    # if not all(key in instance for key in ["text", "price", "price_missing", "helpful_vote", "verified_purchase"]):
                    #     raise ValueError(f"Invalid instance format: {instance}")
                    
                    texts.append(instance_data["text"])
                    additional_features.append([
                        float(instance_data["price"]),
                        float(instance_data["price_missing"]),
                        float(instance_data["helpful_vote"]),
                        float(instance_data["verified_purchase"])
                    ])
            # Log extracted inputs
            logger.info(f"Extracted texts: {texts}")
            logger.info(f"Extracted additional features: {additional_features}")

            # Tokenize text inputs
            tokenized = self.tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=128,
                return_tensors="pt"
            )
            logger.info(f"Tokenized input_ids: {tokenized['input_ids']}")
            logger.info(f"Tokenized attention_mask: {tokenized['attention_mask']}")

            # Return the processed inputs
            return {
                "input_ids": tokenized["input_ids"].to(self.device),
                "attention_mask": tokenized["attention_mask"].to(self.device),
                "additional_features": torch.tensor(additional_features, dtype=torch.float32).to(self.device)
            }
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            raise ValueError(f"Error during preprocessing: {e}")

    def inference(self, inputs):
        """
        Perform inference using the loaded model and processed inputs.
        """
        try:
            logger.info(f"Inference inputs: {inputs}")
            with torch.no_grad():
                outputs = self.model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    additional_features=inputs["additional_features"]
                )
                logits = outputs["logits"]
                predictions = torch.argmax(logits, dim=1).tolist()
                probs = torch.nn.functional.softmax(logits, dim=-1)

            logger.info(f"Inference outputs (logits): {logits}")
            logger.info(f"Inference probs: {probs}")

            logger.info(f"Predicted labels: {predictions}")
            logger.info(self.mapping,[(self.mapping.get(str(pred), "Unknown"),pred) for pred in predictions])
            confidence_scores = []
            for i, prob in enumerate(probs):
                class_confidence = {self.mapping.get(str(idx), "Unknown")[:3]: round(prob[idx].item(),2) for idx in range(len(prob))}
                confidence_scores.append(class_confidence)
    
        #     predicted_labels = [self.mapping.get(str(pred), "Unknown") for pred in predictions]
        #     result = {
        #     "predictions": predicted_labels,
        #     "probabilities": confidence_scores
        # }

            # return [self.mapping.get(str(pred), "Unknown") for pred in predictions]
            return [(self.mapping.get(str(predictions[i]), "Unknown"),confidence_scores[i]) for i in range(len(predictions))]

        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise ValueError(f"Error during inference: {e}")

    def postprocess(self, inference_output):
        """
        Post-processes the model output for returning to the client.
        """
        logger.info(f"Postprocessing output: {inference_output}")
        # return {"predictions": inference_output}
        return inference_output

