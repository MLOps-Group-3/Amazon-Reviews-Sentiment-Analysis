import logging
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
import torch
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
torch.cuda.empty_cache()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_process_data(data_path):
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    df['text'] = df['text'].fillna('')
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['sentiment_label'])
    logger.info("Data loaded and processed successfully")
    return df, label_encoder

def split_data_by_timestamp(df, train_size=0.8, val_size=0.1):
    logger.info("Splitting data by timestamp")
    df['review_date_timestamp'] = pd.to_datetime(df['review_date_timestamp'])
    df = df.sort_values(by='review_date_timestamp').reset_index(drop=True)
    
    train_end = int(len(df) * train_size)
    val_end = int(len(df) * (train_size + val_size))
    
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    logger.info("Data split into train, validation, and test sets")
    return train_df, val_df, test_df

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def initialize_tokenizer():
    logger.info("Initializing tokenizer")
    return BertTokenizer.from_pretrained('bert-base-uncased')

def initialize_model(num_labels):
    logger.info("Initializing BERT model")
    config = BertConfig.from_pretrained("bert-base-uncased")
    config.num_labels = num_labels  # Set the number of labels here
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", config=config)
    return model

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    accuracy = accuracy_score(labels, predictions)
    return {'accuracy': accuracy, 'f1': f1, 'precision': precision, 'recall': recall}

def train_model(model, train_dataset, val_dataset, output_dir='./saved_model'):
    logger.info("Setting up training arguments and trainer")
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=256,
        per_device_eval_batch_size=128,
        num_train_epochs=1,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_dir=os.path.join(output_dir, 'logs')
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )
    logger.info("Starting model training")
    trainer.train()
    eval_results = trainer.evaluate()
    logger.info(f"Validation set evaluation results: {eval_results}")
    
    # Save the best model
    model_save_path = os.path.join(output_dir, "best_model")
    logger.info(f"Saving best model to {model_save_path}")
    trainer.save_model(model_save_path)
    return trainer

def evaluate_on_test_set(trainer, test_dataset, label_encoder):
    logger.info("Evaluating model on test dataset")
    predictions_output = trainer.predict(test_dataset)
    logits = predictions_output.predictions
    labels = predictions_output.label_ids
    predictions = logits.argmax(axis=1)
    
    precision, recall, f1, support = precision_recall_fscore_support(labels, predictions, average=None)
    accuracy = accuracy_score(labels, predictions)
    label_classes = label_encoder.classes_

    logger.info(f"Test set accuracy: {accuracy}")
    for i, (p, r, f, s) in enumerate(zip(precision, recall, f1, support)):
        label_name = label_classes[i]
        logger.info(f"Class '{label_name}': Precision: {p:.4f}, Recall: {r:.4f}, F1 Score: {f:.4f}, Support: {s}")

if __name__ == "__main__":
    # Define paths and settings
    data_path = "labeled_data.csv"
    model_output_dir = "./saved_bert_text_model"

    # Load and process data
    df, label_encoder = load_and_process_data(data_path)

    # Split data
    train_df, val_df, test_df = split_data_by_timestamp(df)

    # Initialize tokenizer
    tokenizer = initialize_tokenizer()

    # Prepare datasets
    train_dataset = SentimentDataset(train_df['text'].tolist(), train_df['label'].tolist(), tokenizer)
    val_dataset = SentimentDataset(val_df['text'].tolist(), val_df['label'].tolist(), tokenizer)
    test_dataset = SentimentDataset(test_df['text'].tolist(), test_df['label'].tolist(), tokenizer)

    # Initialize model
    model = initialize_model(num_labels=len(label_encoder.classes_))

    # Train model
    trainer = train_model(model, train_dataset, val_dataset, output_dir=model_output_dir)

    # Evaluate on test set
    evaluate_on_test_set(trainer, test_dataset, label_encoder)
