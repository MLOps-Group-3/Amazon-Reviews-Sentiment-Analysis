import logging
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
import torch
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_process_data(data_path):
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    df['text'] = df['text'].fillna('')
    df['title'] = df['title'].fillna('')
    df['price'] = pd.to_numeric(df['price'].replace("unknown", None), errors='coerce')
    df['price_missing'] = df['price'].isna().astype(int)
    df['price'] = df['price'].fillna(0).astype(float)
    df['helpful_vote'] = df['helpful_vote'].fillna(0).astype(int)
    df['verified_purchase'] = df['verified_purchase'].apply(lambda x: 1 if x else 0)
    
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
    def __init__(self, texts, titles, prices, price_missing_col, helpful_votes, verified_purchases, labels, tokenizer, max_length=128):
        self.texts = texts
        self.titles = titles
        self.prices = prices
        self.price_missing = price_missing_col
        self.helpful_votes = helpful_votes
        self.verified_purchases = verified_purchases
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        combined_text = f"{self.titles[idx]} {self.texts[idx]}"
        encoding = self.tokenizer(
            combined_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        additional_features = torch.tensor([self.prices[idx], self.price_missing[idx], self.helpful_votes[idx], self.verified_purchases[idx]], dtype=torch.float).unsqueeze(0)
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'additional_features': additional_features,
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def initialize_tokenizer():
    logger.info("Initializing tokenizer")
    return BertTokenizer.from_pretrained('bert-base-uncased')

class CustomBERTModel(torch.nn.Module):
    def __init__(self, base_model, num_labels):
        super(CustomBERTModel, self).__init__()
        self.base_model = base_model
        self.dropout = torch.nn.Dropout(0.3)
        self.additional_layer = torch.nn.Linear(4, 32)
        self.classifier = torch.nn.Linear(self.base_model.config.hidden_size + 32, num_labels)

    def forward(self, input_ids, attention_mask, additional_features, labels=None):
        base_output = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = base_output.hidden_states[-1][:, 0, :]
        additional_output = torch.relu(self.additional_layer(additional_features))
        additional_output = additional_output.squeeze(1)  # Ensure additional_output is 2D
        
        combined_output = torch.cat((pooled_output, additional_output), dim=1)
        logits = self.classifier(self.dropout(combined_output))
        
        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        
        return {'loss': loss, 'logits': logits}

def initialize_model(num_labels):
    logger.info("Initializing BERT model")
    config = BertConfig.from_pretrained("bert-base-uncased", output_hidden_states=True)
    base_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", config=config)
    return CustomBERTModel(base_model, num_labels=num_labels)

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
    model_output_dir = "./saved_model"

    # Load and process data
    df, label_encoder = load_and_process_data(data_path)

    # Split data
    train_df, val_df, test_df = split_data_by_timestamp(df)

    # Initialize tokenizer
    tokenizer = initialize_tokenizer()

    # Prepare datasets
    train_dataset = SentimentDataset(train_df['text'].tolist(), train_df['title'].tolist(), train_df['price'].tolist(),
                                     train_df['price_missing'].tolist(), train_df['helpful_vote'].tolist(),
                                     train_df['verified_purchase'].tolist(), train_df['label'].tolist(), tokenizer)
    val_dataset = SentimentDataset(val_df['text'].tolist(), val_df['title'].tolist(), val_df['price'].tolist(),
                                   val_df['price_missing'].tolist(), val_df['helpful_vote'].tolist(),
                                   val_df['verified_purchase'].tolist(), val_df['label'].tolist(), tokenizer)
    test_dataset = SentimentDataset(test_df['text'].tolist(), test_df['title'].tolist(), test_df['price'].tolist(),
                                    test_df['price_missing'].tolist(), test_df['helpful_vote'].tolist(),
                                    test_df['verified_purchase'].tolist(), test_df['label'].tolist(), tokenizer)

    # Initialize model
    model = initialize_model(num_labels=len(label_encoder.classes_))

    # Train model
    trainer = train_model(model, train_dataset, val_dataset, output_dir=model_output_dir)

    # Evaluate on test set
    evaluate_on_test_set(trainer, test_dataset, label_encoder)
