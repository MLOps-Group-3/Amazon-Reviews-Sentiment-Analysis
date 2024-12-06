import pytest
import sys
import os
import requests
from unittest.mock import MagicMock, patch, call

# Mock the airflow module
mock_airflow = MagicMock()
mock_variable = MagicMock()
mock_airflow.models.Variable = mock_variable
sys.modules['airflow'] = mock_airflow
sys.modules['airflow.models'] = MagicMock()

# Get the path to the data_pipeline directory
data_pipeline_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Add the data_pipeline directory to sys.path
sys.path.insert(0, data_pipeline_dir)

# Now you can import from dags.utils.data_collection
from dags.data_utils.data_collection import data_acquisition

# Mock the config values
data_acquisition.REVIEW_BASE_URL = "http://mock-review-url/"
data_acquisition.META_BASE_URL = "http://mock-meta-url/"
data_acquisition.CATEGORIES = ["MockCategory1", "MockCategory2"]
data_acquisition.TARGET_DIRECTORY = "/mock/target/directory"
data_acquisition.MAX_WORKERS = 2

@pytest.fixture
def mock_requests_get():
    with patch('requests.get') as mock_get:
        yield mock_get

@pytest.fixture
def mock_os_makedirs():
    with patch('os.makedirs') as mock_makedirs:
        yield mock_makedirs

@pytest.fixture
def mock_open():
    with patch('builtins.open', create=True) as mock_open:
        yield mock_open

@pytest.fixture
def mock_threadpoolexecutor():
    with patch('concurrent.futures.ThreadPoolExecutor') as mock_executor:
        yield mock_executor

@patch('dags.utils.data_collection.data_acquisition.tqdm')
def test_download_file_success(mock_tqdm, mock_requests_get, mock_open):
    """
    Test successful file download scenario.
    """
    mock_response = MagicMock()
    mock_response.headers.get.return_value = '1000'
    mock_response.iter_content.return_value = [b'data'] * 10
    mock_requests_get.return_value = mock_response

    # Mock tqdm to simply return the file object
    mock_tqdm.return_value.__enter__.return_value = mock_open.return_value

    # Mock the file object's write method
    mock_file = MagicMock()
    mock_open.return_value.__enter__.return_value = mock_file

    data_acquisition.download_file('http://example.com/file.gz', 'file.gz')

    mock_requests_get.assert_called_once_with('http://example.com/file.gz', stream=True)
    mock_open.assert_called_once_with('file.gz', 'wb')
    
    # Assert that write was called 10 times with b'data'
    assert mock_file.write.call_count == 10
    mock_file.write.assert_has_calls([call(b'data')] * 10)

def test_download_file_http_error(mock_requests_get):
    """
    Test handling of HTTP errors during file download.
    """
    mock_requests_get.side_effect = requests.exceptions.HTTPError('404 Client Error')

    data_acquisition.download_file('http://example.com/file.gz', 'file.gz')
    # The function should log the error and not raise an exception

@patch('dags.utils.data_collection.data_acquisition.download_file')
def test_download_category(mock_download_file):
    """
    Test downloading of both review and meta files for a category.
    """
    data_acquisition.download_category('MockCategory1', data_acquisition.REVIEW_BASE_URL, data_acquisition.META_BASE_URL, data_acquisition.TARGET_DIRECTORY)

    assert mock_download_file.call_count == 2
    mock_download_file.assert_any_call(
        f"{data_acquisition.REVIEW_BASE_URL}MockCategory1.jsonl.gz",
        os.path.join(data_acquisition.TARGET_DIRECTORY, "MockCategory1_reviews.jsonl.gz")
    )
    mock_download_file.assert_any_call(
        f"{data_acquisition.META_BASE_URL}meta_MockCategory1.jsonl.gz",
        os.path.join(data_acquisition.TARGET_DIRECTORY, "MockCategory1_meta.jsonl.gz")
    )

@patch('dags.utils.data_collection.data_acquisition.ThreadPoolExecutor')
@patch('os.makedirs')
def test_acquire_data(mock_makedirs, mock_executor):
    """
    Test the main acquire_data function to ensure it sets up the directory
    and uses ThreadPoolExecutor correctly.
    """
    mock_executor_instance = MagicMock()
    mock_executor.return_value.__enter__.return_value = mock_executor_instance

    data_acquisition.acquire_data()

    mock_makedirs.assert_called_once_with(data_acquisition.TARGET_DIRECTORY, exist_ok=True)
    mock_executor_instance.map.assert_called_once()

@pytest.mark.parametrize('category', data_acquisition.CATEGORIES)
@patch('dags.utils.data_collection.data_acquisition.download_file')
def test_download_category_for_each_category(mock_download_file, category):
    """
    Test downloading files for each category to ensure all categories are processed.
    """
    data_acquisition.download_category(category, data_acquisition.REVIEW_BASE_URL, data_acquisition.META_BASE_URL, data_acquisition.TARGET_DIRECTORY)

    assert mock_download_file.call_count == 2
    mock_download_file.assert_any_call(
        f"{data_acquisition.REVIEW_BASE_URL}{category}.jsonl.gz",
        os.path.join(data_acquisition.TARGET_DIRECTORY, f"{category}_reviews.jsonl.gz")
    )
    mock_download_file.assert_any_call(
        f"{data_acquisition.META_BASE_URL}meta_{category}.jsonl.gz",
        os.path.join(data_acquisition.TARGET_DIRECTORY, f"{category}_meta.jsonl.gz")
    )

# Setup for mocking Variable.get
@pytest.fixture(autouse=True)
def mock_variable_get():
    mock_variable.get.return_value = "/mock/log/directory"
