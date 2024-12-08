import os
import pandas as pd
from ..config import TARGET_DIRECTORY_SAMPLED


def concatenate_and_save_csv_files():
    # Lists to store DataFrames
    dfs_2018_2019 = []
    dfs_2020 = []

    # Loop through the files in the target directory
    for filename in os.listdir(TARGET_DIRECTORY_SAMPLED):
        if filename.endswith('.csv'):
            file_path = os.path.join(TARGET_DIRECTORY_SAMPLED, filename)
            if 'sampled_data_2018_2019_' in filename:
                df = pd.read_csv(file_path)
                dfs_2018_2019.append(df)
            elif 'sampled_data_2021_' in filename:
                df = pd.read_csv(file_path)
                dfs_2020.append(df)

    # Concatenate and save DataFrames for 2018_2019
    if dfs_2018_2019:
        concatenated_2018_2019 = pd.concat(dfs_2018_2019, ignore_index=True)
        output_path_2018_2019 = os.path.join(TARGET_DIRECTORY_SAMPLED, 'sampled_data_2018_2019.csv')
        concatenated_2018_2019.to_csv(output_path_2018_2019, index=False)
        print(f"Saved concatenated 2018-2019 data to {output_path_2018_2019}")
    else:
        print("No 2018-2019 files found to concatenate")

    # Concatenate and save DataFrames for 2020
    if dfs_2020:
        concatenated_2020 = pd.concat(dfs_2020, ignore_index=True)
        output_path_2020 = os.path.join(TARGET_DIRECTORY_SAMPLED, 'sampled_data_2021.csv')
        concatenated_2020.to_csv(output_path_2020, index=False)
        print(f"Saved concatenated 2020 data to {output_path_2020}")
    else:
        print("No 2020 files found to concatenate")


if __name__ == "__main__":
    concatenate_and_save_csv_files()