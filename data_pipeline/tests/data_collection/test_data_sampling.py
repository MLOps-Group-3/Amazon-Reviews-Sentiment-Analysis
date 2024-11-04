import pytest
import pandas as pd
import os
from unittest.mock import MagicMock, patch

# Mock the airflow module
mock_airflow = MagicMock()
mock_variable = MagicMock()
mock_airflow.models.Variable = mock_variable
import sys
sys.modules['airflow'] = mock_airflow
sys.modules['airflow.models'] = MagicMock()

# Get the path to the data_pipeline directory
data_pipeline_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, data_pipeline_dir)

from dags.utils.data_collection import sampling

# Ignore warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Mock the config values
sampling.SAMPLING_FRACTION = 0.5
sampling.TARGET_DIRECTORY = "/tmp/mock_target_directory"
sampling.TARGET_DIRECTORY_SAMPLED = "/tmp/mock_sampled_directory"

@pytest.fixture
def sample_data_fixture():
    return pd.DataFrame({
        'timestamp': [1514764800000, 1546300800000, 1577836800000],
        'images': ['img1', 'img2', 'img3'],
        'parent_asin': ['A1', 'A2', 'A3'],
        'rating': [4, 5, 3]
    })

@pytest.fixture
def sample_metadata_fixture():
    return pd.DataFrame({
        'parent_asin': ['A1', 'A2', 'A3'],
        'title': ['Product1', 'Product2', 'Product3'],
        'main_category': ['Cat1', 'Cat2', 'Cat3'],
        'categories': [['Cat1'], ['Cat2'], ['Cat3']],
        'price': [10.0, 20.0, 30.0],
        'average_rating': [4.5, 4.0, 3.5],
        'rating_number': [100, 200, 300]
    })

def test_process_reviews_df(sample_data_fixture):
    """
    Test processing of reviews DataFrame.
    Checks if:
    1. 'review_date_timestamp' column is added
    2. 'images' column is removed
    3. The processed DataFrame has the correct number of rows
    """
    processed_df = sampling.process_reviews_df(sample_data_fixture)
    assert 'review_date_timestamp' in processed_df.columns
    assert 'images' not in processed_df.columns
    assert len(processed_df) == 3

def test_join_dataframes(sample_data_fixture, sample_metadata_fixture):
    """
    Test joining of reviews and metadata DataFrames.
    Checks if:
    1. 'product_name' column is present in the joined DataFrame
    2. The joined DataFrame has the correct number of rows
    """
    processed_reviews = sampling.process_reviews_df(sample_data_fixture)
    joined_df = sampling.join_dataframes(processed_reviews, sample_metadata_fixture)
    assert 'product_name' in joined_df.columns
    assert len(joined_df) == 3

def test_sample_data(sample_data_fixture, sample_metadata_fixture):
    """
    Test data sampling functionality.
    Checks if:
    1. The sampled DataFrame has fewer or equal rows compared to the input
    """
    processed_reviews = sampling.process_reviews_df(sample_data_fixture)
    joined_df = sampling.join_dataframes(processed_reviews, sample_metadata_fixture)
    sampled_df = sampling.sample_data(joined_df)
    assert len(sampled_df) <= len(joined_df)

def test_process_sampled_data(sample_data_fixture, sample_metadata_fixture):
    """
    Test processing of sampled data.
    Checks if:
    1. The sum of rows in split DataFrames equals the input DataFrame
    2. All rows in the 2020 DataFrame have the year 2020
    """
    processed_reviews = sampling.process_reviews_df(sample_data_fixture)
    joined_df = sampling.join_dataframes(processed_reviews, sample_metadata_fixture)
    sampled_df = sampling.sample_data(joined_df)
    df_2018_2019, df_2020 = sampling.process_sampled_data(sampled_df)
    assert len(df_2018_2019) + len(df_2020) == len(sampled_df)
    assert all(df_2020['year'] == 2020)

@patch('dags.utils.data_collection.sampling.load_jsonl_gz')
@patch('dags.utils.data_collection.sampling.process_reviews_df')
@patch('dags.utils.data_collection.sampling.join_dataframes')
@patch('dags.utils.data_collection.sampling.sample_data')
@patch('dags.utils.data_collection.sampling.process_sampled_data')
@patch('pandas.DataFrame.to_csv')
@patch('os.makedirs')
def test_sample_category(mock_makedirs, mock_to_csv, mock_process_sampled, mock_sample, mock_join, mock_process_reviews, mock_load):
    """
    Test the entire sampling process for a category.
    Checks if:
    1. All expected functions are called
    2. CSV files are created (mock_to_csv is called twice)
    3. Directory is created (mock_makedirs is called)
    """
    mock_load.side_effect = [
        pd.DataFrame({'timestamp': [1514764800000], 'text': ['review']}),
        pd.DataFrame({'parent_asin': ['A1'], 'title': ['Product1']})
    ]
    mock_process_reviews.return_value = pd.DataFrame({'parent_asin': ['A1'], 'review_text': ['Good']})
    mock_join.return_value = pd.DataFrame({'parent_asin': ['A1'], 'review_text': ['Good'], 'product_name': ['Product1']})
    mock_sample.return_value = pd.DataFrame({'parent_asin': ['A1'], 'review_text': ['Good'], 'product_name': ['Product1']})
    mock_process_sampled.return_value = (
        pd.DataFrame({'year': [2018], 'review_text': ['Good']}),
        pd.DataFrame({'year': [2020], 'review_text': ['Good']})
    )

    sampling.sample_category('TestCategory')

    mock_load.assert_called()
    mock_process_reviews.assert_called()
    mock_join.assert_called()
    mock_sample.assert_called()
    mock_process_sampled.assert_called()
    assert mock_to_csv.call_count == 2
    mock_makedirs.assert_called()

# Setup for mocking Variable.get
@pytest.fixture(autouse=True)
def mock_variable_get():
    mock_variable.get.return_value = "/tmp/mock_log_directory"

