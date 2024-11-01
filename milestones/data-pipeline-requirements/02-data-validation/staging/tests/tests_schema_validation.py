"""
FILE : tests/test_schema_validation.py
"""

import os
import sys
import pandas as pd
import pytest

# Add the utils directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../dags/utils')))

from schema_validation import validate_schema  # Import after setting up the path

# Fixture for a valid DataFrame sample that matches the EXPECTED_SCHEMA
@pytest.fixture
def valid_data():
    return pd.DataFrame({
        "review_month": [1, 2],
        "rating": [4.5, 3.0],
        "parent_asin": ["B000123", "B000124"],
        "asin": ["A0001", "A0002"],
        "helpful_vote": [10, 15],
        "text": ["Good product", "Bad product"],
        "timestamp": [1610000000, 1620000000],
        "title": ["Great", "Not Good"],
        "user_id": ["user1", "user2"],
        "verified_purchase": [True, False],
        "review_date_timestamp": ["2021-01-01", "2021-02-01"],
        "main_category": ["Electronics", "Home"],
        "product_name": ["Product 1", "Product 2"],
        "categories": ["Category 1", "Category 2"],
        "price": [100.0, 200.0],
        "average_rating": [4.2, 3.5],
        "rating_number": [100, 200],
        "year": [2021, 2022]
    })

# Test cases

def test_valid_schema(valid_data):
    """Test if validate_schema returns True for a valid DataFrame"""
    assert validate_schema(valid_data) is True

def test_missing_column(valid_data):
    """Test if validate_schema returns False for a DataFrame with missing columns"""
    data = valid_data.drop(columns=["rating"])
    assert validate_schema(data) is False

def test_extra_column(valid_data):
    """Test if validate_schema logs an extra column but returns True"""
    data = valid_data.copy()
    data["extra_column"] = [1, 2]
    assert validate_schema(data) is True

def test_incorrect_dtype(valid_data):
    """Test if validate_schema returns False for a DataFrame with incorrect data types"""
    data = valid_data.copy()
    data["rating"] = ["4.5", "3.0"]  # Incorrect type: should be float64
    assert validate_schema(data) is False

def test_empty_dataframe():
    """Test if validate_schema returns False for an empty DataFrame"""
    data = pd.DataFrame()
    assert validate_schema(data) is False
