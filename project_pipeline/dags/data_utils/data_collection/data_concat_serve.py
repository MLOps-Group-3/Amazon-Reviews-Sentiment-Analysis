import os
import pandas as pd
from data_utils.data_collection.dynamic_month import get_next_serving_month
from data_utils.config import SAMPLED_SERVING_DIRECTORY, DEFAULT_SERVING_YEAR, DEFAULT_SERVING_MONTH


def concatenate_and_save_csv_files(input_dir, output_file):
    """
    Concatenate all sampled serving data CSV files in the specified directory
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
            if filename.startswith("sampled_data_") and filename.endswith('.csv'):
                file_path = os.path.join(input_dir, filename)
                df = pd.read_csv(file_path)
                dfs.append(df)

        # Concatenate all DataFrames
        if dfs:
            concatenated_df = pd.concat(dfs, ignore_index=True)
            concatenated_df.to_csv(output_file, index=False)
            print(f"Saved concatenated serving data to {output_file}")
        else:
            print(f"No files found in {input_dir} for concatenation.")
    except Exception as e:
        print(f"An error occurred during concatenation: {e}")


if __name__ == "__main__":
    # Dynamically determine the year and month based on existing concatenated or sampled data files
    year, month = get_next_serving_month(SAMPLED_SERVING_DIRECTORY, '', DEFAULT_SERVING_YEAR, DEFAULT_SERVING_MONTH)

    # Define input and output paths
    input_dir = SAMPLED_SERVING_DIRECTORY
    output_file = os.path.join(input_dir, f'concatenated_serving_data_{year}_{str(month).zfill(2)}.csv')

    # Perform concatenation
    concatenate_and_save_csv_files(input_dir, output_file)
