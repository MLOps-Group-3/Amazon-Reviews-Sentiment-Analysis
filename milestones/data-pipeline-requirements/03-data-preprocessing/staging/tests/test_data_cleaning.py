import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../dags/utils')))

import unittest
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

from data_cleaning import clean_amazon_reviews  

class TestDataCleaning(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up a Spark session for the tests."""
        cls.spark = SparkSession.builder \
            .appName("TestAmazonReviewDataCleaning") \
            .master("local[*]") \
            .getOrCreate()
    
    @classmethod
    def tearDownClass(cls):
        """Stop the Spark session after tests."""
        cls.spark.stop()

    def setUp(self):
        """Set up a simple DataFrame for testing."""
        self.test_data = [
            (5.0, "Title 1", "This is a review.", "2018-01-01 00:00:00", 1, 1, "[Category1]", "20.99", "ASIN123"),
            (None, "Title 2", "This is another review.", "2018-01-02 00:00:00", 0, 1, None, None, "ASIN124"),
            (4.0, "Title 3", "   ", "2018-01-03 00:00:00", 2, 1, "[Category2]", "15.00", "ASIN125"),
            (3.0, "Title 4", "This is another review too", "2018-01-02 00:00:00", 0, 1, None, None, "ASIN124"),

        ]
        
        columns = ['rating', 'title_x', 'text', 'timestamp', 'helpful_vote', 'verified_purchase', 'categories', 'price', 'parent_asin']
        self.df_raw = self.spark.createDataFrame(self.test_data, columns)

    def test_clean_amazon_reviews(self):
        """Test the data cleaning function."""
        # self.df_raw.show()
        cleaned_df = clean_amazon_reviews(self.spark, self.df_raw)
        # cleaned_df.show()
        # Check if null values in 'price' are replaced with 'unknown'
        self.assertEqual(cleaned_df.filter(col("price") == "unknown").count(), 1)

        # Check for duplicates
        self.assertEqual(cleaned_df.dropDuplicates().count(), 3)

        # Check for nulls in 'text' and 'rating'
        self.assertEqual(cleaned_df.filter(col("text").isNull()).count(), 0)
        self.assertEqual(cleaned_df.filter(col("rating").isNull()).count(), 0)

        # Check if whitespaces are removed in 'title_x' and 'text'
        self.assertNotIn("   ", cleaned_df.select("text").first()[0])
        self.assertNotIn("   ", cleaned_df.select("title_x").first()[0])

if __name__ == "__main__":
    unittest.main()
