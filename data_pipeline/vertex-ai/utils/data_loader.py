import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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