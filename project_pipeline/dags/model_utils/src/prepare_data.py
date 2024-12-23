import logging
import os
import pickle
from utils.data_loader import split_data_by_timestamp, load_and_process_data  
from config import DATA_PATH, DATA_SAVE_PATH
import pandas as pd
from sklearn.utils import resample

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def balance_data(df, column="main_category", min_samples=50):
    """
    Balances the data by oversampling underrepresented slices.
    
    Args:
        df (pd.DataFrame): Input dataframe.
        column (str): Column used for slicing (e.g., "main_category").
        min_samples (int): Minimum samples required for each slice.
    
    Returns:
        pd.DataFrame: Balanced dataframe.
    """
    unique_values = df[column].unique()
    balanced_data = []
    for value in unique_values:
        slice_data = df[df[column] == value]
        if len(slice_data) < min_samples:
            oversampled_slice = resample(slice_data, replace=True, n_samples=min_samples, random_state=42)
            balanced_data.append(oversampled_slice)
        else:
            balanced_data.append(slice_data)
    balanced_df = pd.concat(balanced_data, ignore_index=True)
    logger.info(f"Balanced data using column '{column}' with minimum {min_samples} samples per slice.")
    return balanced_df

def split_and_save_data(data_path, output_dir):
    """
    Splits the data using the `split_data_by_timestamp` function and saves it as pickled objects.
    
    Args:
        data_path (str): Path to the input dataset (e.g., CSV file).
        output_dir (str): Directory to save the split data.
    """
    # Split data using the provided function
    df, label_encoder = load_and_process_data(data_path)
    train_df, val_df, test_df = split_data_by_timestamp(df)
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save splits as pickled objects
    for split_name, split_data in zip(["train", "val", "test"], [train_df, val_df, test_df]):
        file_path = os.path.join(output_dir, f"{split_name}.pkl")
        with open(file_path, "wb") as f:
            pickle.dump(split_data, f)
        logger.info(f"{split_name.capitalize()} data saved to {file_path}.")

    label_encoder_path = os.path.join(output_dir, "label_encoder.pkl")
    with open(label_encoder_path, "wb") as f:
        pickle.dump(label_encoder, f)
    logger.info(f"Label encoder saved to {label_encoder_path}.")

    return train_df, val_df, test_df, label_encoder


def main():
    """
    Main function to test the split and save functionality.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))

    data_path = os.path.join(script_dir, DATA_PATH.lstrip("/")) 
    output_dir = os.path.join(script_dir, DATA_SAVE_PATH.lstrip("/"))
    
    split_and_save_data(data_path, output_dir)
    logger.info("Data splitting and saving completed successfully.")

if __name__ == "__main__":
    main()
