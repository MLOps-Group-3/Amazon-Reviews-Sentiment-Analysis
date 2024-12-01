import logging
import os
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, RobertaConfig, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from .data_loader import load_and_process_data, split_data_by_timestamp, SentimentDataset

# Configuration for training
DATA_PATH = "labeled_data.csv"
OUTPUT_DIR = "./roberta_output"
EPOCHS = 1
BATCH_SIZE = 128
LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomRoBERTaModel(torch.nn.Module):
    def __init__(self, base_model, num_labels):
        super(CustomRoBERTaModel, self).__init__()
        self.base_model = base_model
        self.dropout = torch.nn.Dropout(0.3)
        self.additional_layer = torch.nn.Linear(4, 32)  # Processing additional features
        self.classifier = torch.nn.Linear(self.base_model.config.hidden_size + 32, num_labels)

    def forward(self, input_ids, attention_mask, additional_features, labels=None):
        base_output = self.base_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        pooled_output = base_output.hidden_states[-1][:, 0, :]
        additional_output = torch.relu(self.additional_layer(additional_features)).squeeze(1)
        combined_output = torch.cat((pooled_output, additional_output), dim=1)
        logits = self.classifier(self.dropout(combined_output))
        
        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        
        return {'loss': loss, 'logits': logits}


def initialize_roberta_model(num_labels, model_name="roberta-base"):
    config = RobertaConfig.from_pretrained(model_name)
    base_model = RobertaForSequenceClassification.from_pretrained(model_name, config=config)
    model = CustomRoBERTaModel(base_model, num_labels=num_labels)
    model.to(DEVICE)
    return model


def train_roberta_model(model, train_dataset, val_dataset, output_dir, **training_args):
    default_args = {
        "learning_rate": 2e-5,
        "per_device_train_batch_size": 16,
        "per_device_eval_batch_size": 16,
        "num_train_epochs": 3,
        "evaluation_strategy": "epoch",
        "save_strategy": "no",
        "load_best_model_at_end": False,
        "logging_dir": os.path.join(output_dir, 'logs')
    }
    default_args.update(training_args)
    args = TrainingArguments(output_dir=output_dir, **default_args)
    
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )
    
    trainer.train()
    eval_results = trainer.evaluate()
    # trainer.save_model(os.path.join(output_dir, "best_model"))
    return eval_results, trainer


def evaluate_roberta_model(trainer, test_dataset):
    predictions_output = trainer.predict(test_dataset)
    logits = predictions_output.predictions
    labels = predictions_output.label_ids
    predictions = logits.argmax(axis=1)
    
    precision, recall, f1, support = precision_recall_fscore_support(labels, predictions, average=None)
    accuracy = accuracy_score(labels, predictions)
    
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    accuracy = accuracy_score(labels, predictions)
    return {'accuracy': accuracy, 'f1': f1, 'precision': precision, 'recall': recall}


if __name__ == "__main__":
    # Load and process data
    df, label_encoder = load_and_process_data(DATA_PATH)
    train_df, val_df, test_df = split_data_by_timestamp(df)

    # Initialize tokenizer and datasets
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
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

    # Initialize RoBERTa model and move it to the appropriate device
    model = initialize_roberta_model(num_labels=len(label_encoder.classes_))

    # Training parameters
    training_args = {
        "num_train_epochs": EPOCHS,
        "per_device_train_batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
    }

    # Train the model
    eval_results, trainer = train_roberta_model(model, train_dataset, val_dataset, OUTPUT_DIR, **training_args)

    # Evaluate the model
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
    test_results = evaluate_roberta_model(trainer, test_dataset)
    print("Test Results:", test_results)
