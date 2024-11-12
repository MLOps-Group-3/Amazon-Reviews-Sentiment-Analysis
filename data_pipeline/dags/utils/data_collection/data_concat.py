import dask.dataframe as dd
import gcsfs
import logging
from distributed import Client
from utils.config import GCS_SAMPLED_DATA_PATH

logger = logging.getLogger(__name__)


def concatenate_and_save_csv_files():
    client = Client('tcp://dask-scheduler:8786')
    logger.info(f"Using Dask client with {len(client.scheduler_info()['workers'])} workers")

    try:
        fs = gcsfs.GCSFileSystem()

        dfs_2018_2019 = []
        dfs_2020 = []

        for file in fs.glob(f"{GCS_SAMPLED_DATA_PATH}/sampled_data_*.csv"):
            if 'sampled_data_2018_2019_' in file:
                df = dd.read_csv(file, storage_options={'fs': fs})
                dfs_2018_2019.append(df)
            elif 'sampled_data_2020_' in file:
                df = dd.read_csv(file, storage_options={'fs': fs})
                dfs_2020.append(df)

        if dfs_2018_2019:
            concatenated_2018_2019 = dd.concat(dfs_2018_2019)
            output_path_2018_2019 = f"{GCS_SAMPLED_DATA_PATH}/sampled_data_2018_2019.csv"
            concatenated_2018_2019.compute().to_csv(output_path_2018_2019, index=False, storage_options={'fs': fs})
            logger.info(f"Saved concatenated 2018-2019 data to {output_path_2018_2019}")
        else:
            logger.info("No 2018-2019 files found to concatenate")

        if dfs_2020:
            concatenated_2020 = dd.concat(dfs_2020)
            output_path_2020 = f"{GCS_SAMPLED_DATA_PATH}/sampled_data_2020.csv"
            concatenated_2020.compute().to_csv(output_path_2020, index=False, storage_options={'fs': fs})
            logger.info(f"Saved concatenated 2020 data to {output_path_2020}")
        else:
            logger.info("No 2020 files found to concatenate")

    except Exception as e:
        logger.error(f"An error occurred during data concatenation: {str(e)}")
        raise
    finally:
        client.close()


if __name__ == "__main__":
    concatenate_and_save_csv_files()