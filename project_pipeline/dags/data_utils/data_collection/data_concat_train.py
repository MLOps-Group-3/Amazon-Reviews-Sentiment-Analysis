import os
import pandas as pd
from data_utils.data_collection.dynamic_month_train import get_next_training_period
from data_utils.config import (
    SAMPLED_TRAINING_DIRECTORY, 
    DEFAULT_TRAINING_START_YEAR, 
    DEFAULT_TRAINING_START_MONTH,
    CATEGORIES
)

def concatenate_and_save_csv_files(input_dir, output_file):
    """
    Concatenate all sampled training data CSV files in the specified directory
    and save them to a single output file.

    Args:
        input_dir (str): The directory containing sampled CSV files.
        output_file (str): The path where the concatenated file will be saved.
    """
    try:
        # List to store DataFrames
        dfs = []

        # Loop through the files in the input directory
        for filename in os.listdir(input_dir):
            if filename.endswith('.csv') and filename.startswith('sampled_data_'):
                file_path = os.path.join(input_dir, filename)
                df = pd.read_csv(file_path)
                dfs.append(df)

        # Concatenate all DataFrames
        if dfs:
            concatenated_df = pd.concat(dfs, ignore_index=True)
            concatenated_df.to_csv(output_file, index=False)
            print(f"Saved concatenated training data to {output_file}")
        else:
            print(f"No files found in {input_dir} for concatenation.")
    except Exception as e:
        print(f"An error occurred during concatenation: {e}")

if __name__ == "__main__":
    # Dynamically determine the training period for the first category
    # (assuming all categories have the same period)
    start_date, end_date = get_next_training_period(
        SAMPLED_TRAINING_DIRECTORY,
        CATEGORIES[0],
        default_start_year=DEFAULT_TRAINING_START_YEAR,
        default_start_month=DEFAULT_TRAINING_START_MONTH
    )

    # Define input and output paths
    input_dir = SAMPLED_TRAINING_DIRECTORY
    output_file = os.path.join(input_dir, f'concatenated_training_data_{start_date}_to_{end_date}.csv')

    # Perform concatenation
    concatenate_and_save_csv_files(input_dir, output_file)
