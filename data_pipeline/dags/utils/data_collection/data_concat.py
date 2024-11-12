import os
import dask.dataframe as dd
from ..config import TARGET_DIRECTORY_SAMPLED
import logging
from distributed import Client

logger = logging.getLogger(__name__)

def concatenate_and_save_csv_files():
    client = Client('tcp://dask-scheduler:8786')
    logger.info(f"Using Dask client with {len(client.scheduler_info()['workers'])} workers")

    try:
        # Lists to store Dask DataFrames
        dfs_2018_2019 = []
        dfs_2020 = []

        # Loop through the files in the target directory
        for filename in os.listdir(TARGET_DIRECTORY_SAMPLED):
            if filename.endswith('.csv'):
                file_path = os.path.join(TARGET_DIRECTORY_SAMPLED, filename)
                if 'sampled_data_2018_2019_' in filename:
                    df = dd.read_csv(file_path)
                    dfs_2018_2019.append(df)
                elif 'sampled_data_2020_' in filename:
                    df = dd.read_csv(file_path)
                    dfs_2020.append(df)

        # Concatenate and save DataFrames for 2018_2019
        if dfs_2018_2019:
            concatenated_2018_2019 = dd.concat(dfs_2018_2019)
            output_path_2018_2019 = os.path.join(TARGET_DIRECTORY_SAMPLED, 'sampled_data_2018_2019.csv')
            concatenated_2018_2019.compute().to_csv(output_path_2018_2019, index=False)
            logger.info(f"Saved concatenated 2018-2019 data to {output_path_2018_2019}")
        else:
            logger.info("No 2018-2019 files found to concatenate")

        # Concatenate and save DataFrames for 2020
        if dfs_2020:
            concatenated_2020 = dd.concat(dfs_2020)
            output_path_2020 = os.path.join(TARGET_DIRECTORY_SAMPLED, 'sampled_data_2020.csv')
            concatenated_2020.compute().to_csv(output_path_2020, index=False)
            logger.info(f"Saved concatenated 2020 data to {output_path_2020}")
        else:
            logger.info("No 2020 files found to concatenate")

    except Exception as e:
        logger.error(f"An error occurred during data concatenation: {str(e)}")
        raise
    finally:
        client.close()

if __name__ == "__main__":
    # This block is for testing purposes only
    concatenate_and_save_csv_files()
