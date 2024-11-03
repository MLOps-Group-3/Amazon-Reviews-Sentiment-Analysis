# test_data_cleaning.py
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../dags/utils')))

import sys
import os
import unittest
import pandas as pd
from data_preprocessing.data_cleaning_pandas import clean_amazon_reviews

class TestDataCleaning(unittest.TestCase):

    def setUp(self):
        """Set up a simple DataFrame for testing."""
        # Define test data with relevant columns, matching your sample data structure
        self.test_data = [
            (2, 1.0, "B001DTE5Q2", "B001DTE5Q2", 0, "Very flimsy hooks.🤣 Purses fall off constantly. Not worth money.", 
             1581880140993, "Waste of money", "AFAHC6E2UT4DJT4E2GBGNE5C2LPQ", True, 1581880140993, 
             "Amazon Home", "Perfect Curve Purse Rack HPC", "Home & Kitchen,Storage & Organization,Clothing & Closet Storage", 
             None, 4.1, 1973, 2020),
            (2, 5.0, "B003Z3URR0", "B001G3ZP8W", 1, "", 1582731248091, None, 
             "AHB5U3ZMDUW7SKKDGVIKAOVCFY5Q", False, 1582731248091, "Amazon Home", "Hem Precious Musk Fragrance", 
             "Home & Kitchen,Home Décor Products", 6.98, 4.6, 1229, 2020),  # Empty text, null title
            (2, None, "B00PILE4SK", "B00FZ1JR6M", 0, "Good brand, but received a used model", 
             1582822571741, "Used model received", "AGIEN7AW4DV6OMUK25PCPFZAU6KQ", True, 1582822571741, 
             "Amazon Home", "KitchenAid Blender", "Home & Kitchen,Kitchen & Dining", None, 4.5, 3905, 2020),  # Null rating and price
            # Add additional rows if necessary...
        ]
        
        # Define columns for the DataFrame
        self.columns = [
            'review_month', 'rating', 'parent_asin', 'asin', 'helpful_vote', 'text', 'timestamp', 
            'title', 'user_id', 'verified_purchase', 'review_date_timestamp', 'main_category', 
            'product_name', 'categories', 'price', 'average_rating', 'rating_number', 'year'
        ]

        # Create a DataFrame from the test data
        self.df_raw = pd.DataFrame(self.test_data, columns=self.columns)

    def test_clean_amazon_reviews(self):
        """Test the data cleaning function."""
        cleaned_df = clean_amazon_reviews(self.df_raw,[0])

        # Check if null values in 'price' are replaced with 'unknown'
        self.assertEqual((cleaned_df["price"] == "unknown").sum(), 1)  # Adjust count as needed

        # Check for duplicates (assuming duplicates are removed based on `asin` and `user_id` or other criteria)
        # Adjust count to match expected rows after removing duplicates
        self.assertEqual(len(cleaned_df.drop_duplicates()), len(self.test_data) - 1) 

        # Check for nulls in 'text' and 'rating'
        self.assertEqual(cleaned_df["text"].isnull().sum(), 0)  # Expecting no null 'text'
        self.assertEqual(cleaned_df["rating"].isnull().sum(), 0)  # Expecting no null 'rating'

        # Check if whitespaces are removed in 'title' and 'text'
        self.assertNotIn("   ", cleaned_df["text"].iloc[0])  # Adjust index if needed
        self.assertNotIn("   ", cleaned_df["title"].iloc[0])  # Adjust index if needed

        # Additional checks for this dataset
        # Check if 'review_date_timestamp' is not null or invalid
        self.assertEqual(cleaned_df["review_date_timestamp"].isnull().sum(), 0)

        # Check if 'verified_purchase' values are standardized (assuming True/False only)
        self.assertTrue(cleaned_df["verified_purchase"].isin([True, False]).all())

if __name__ == "__main__":
    unittest.main()
