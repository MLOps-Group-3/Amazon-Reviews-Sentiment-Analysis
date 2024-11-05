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
        # self.test_data = [
        #     (5.0, "Title 1", "This is a review.", "2018-01-01 00:00:00", 1, 1, "[Category1]", "20.99", "ASIN123"),
        #     (None, "Title 2", "This is another review.", "2018-01-02 00:00:00", 0, 1, None, None, "ASIN124"),
        #     (4.0, "Title 3", "   ", "2018-01-03 00:00:00", 2, 1, "[Category2]", "15.00", "ASIN125"),
        #     (3.0, "Title 4", "This is another review too", "2018-01-02 00:00:00", 0, 1, None, None, "ASIN124"),

        # ]
        self.test_data = [
            (2, 1.0, "B001DTE5Q2", "B001DTE5Q2", 0, "Very flimsy hooks. Purses fall off constantly. Not worth money.", 1581880140993, "Waste of money", "AFAHC6E2UT4DJT4E2GBGNE5C2LPQ", True, 1581880140993, "Amazon Home", "Perfect Curve Purse Rack HPC", "Home & Kitchen,Storage & Organization,Clothing & Closet Storage", None, 4.1, 1973, 2020),
            (2, 5.0, "B003Z3URR0", "B001G3ZP8W", 1, "", 1582731248091, None, "AHB5U3ZMDUW7SKKDGVIKAOVCFY5Q", False, 1582731248091, "Amazon Home", "Hem Precious Musk Fragrance", "Home & Kitchen,Home Décor Products", 6.98, 4.6, 1229, 2020),  # Empty text, null title
            (2, None, "B00PILE4SK", "B00FZ1JR6M", 0, "Good brand, but received a used model", 1582822571741, "Used model received", "AGIEN7AW4DV6OMUK25PCPFZAU6KQ", True, 1582822571741, "Amazon Home", "KitchenAid Blender", "Home & Kitchen,Kitchen & Dining", None, 4.5, 3905, 2020),  # Null rating and price
            (1, 3.0, "B00U8QGU32", "B00Q7OFEM2", 1, "Happy until I washed it; filling bunched up", None, "Careful when washing", "AGEWDSIDG4LO3ZM26G74F7MVKNHQ", True, 1580683343441, "Amazon Home", "Microfiber Comforter", "Home & Kitchen,Bedding", 36.95, 4.5, 40403, 2020),  # Missing timestamp
            (2, 4.0, "B078MPFN55", "B078MPFN55", 0, "Piece of junk. It made a horrible noise.", 1580923330717, "Get A Roomba instead.", "AHHKVMSOCT6J6Q3US2WDMSXK3TKQ", True, 1580923330717, "Amazon Home", "Neato Robotics Botvac", "Home & Kitchen,Vacuums & Floor Care", 395.51, 4.0, 2821, 2020),  # Text with special characters, high price
            (2, 4.0, "B078MPFN55", "B078MPFN55", 0, "Piece of junk. It made a horrible noise.", 1580923330717, "Get A Roomba instead.", "AHHKVMSOCT6J6Q3US2WDMSXK3TKQ", True, 1580923330717, "Amazon Home", "Neato Robotics Botvac", "Home & Kitchen,Vacuums & Floor Care", 395.51, 4.0, 2821, 2020),  # Text with special characters, high price
            (2, 1.0, "B0018P53K2", "B0018P53K2", None, None, 1582731248091, "Title only review", "AUSERID124", False, 1582731248091, None, "Product X", "Electronics,Accessories", 0.99, None, 50, 2020),  # Missing review text and helpful_vote
        ]
        
        # columns = ['rating', 'title_x', 'text', 'timestamp', 'helpful_vote', 'verified_purchase', 'categories', 'price', 'parent_asin']
        columns = [
            'review_month', 'rating', 'parent_asin', 'asin', 'helpful_vote', 'text', 'timestamp', 
            'title', 'user_id', 'verified_purchase', 'review_date_timestamp', 'main_category', 
            'product_name', 'categories', 'price', 'average_rating', 'rating_number', 'year'
        ]

        self.df_raw = self.spark.createDataFrame(self.test_data, columns)

    def test_clean_amazon_reviews(self):
        """Test the data cleaning function."""
        # self.df_raw.show()
        cleaned_df = clean_amazon_reviews(self.spark, self.df_raw)
        # cleaned_df.show()
        # Check if null values in 'price' are replaced with 'unknown'
        # self.assertEqual(cleaned_df.filter(col("price") == "unknown").count(), 1)

        # # Check for duplicates
        # self.assertEqual(cleaned_df.dropDuplicates().count(), 3)

        # # Check for nulls in 'text' and 'rating'
        # self.assertEqual(cleaned_df.filter(col("text").isNull()).count(), 0)
        # self.assertEqual(cleaned_df.filter(col("rating").isNull()).count(), 0)

        # # Check if whitespaces are removed in 'title_x' and 'text'
        # self.assertNotIn("   ", cleaned_df.select("text").first()[0])
        # self.assertNotIn("   ", cleaned_df.select("title_x").first()[0])

        # Check if null values in 'price' are replaced with 'unknown'
        self.assertEqual(cleaned_df.filter(col("price") == "unknown").count(), 1)  # adjust count as needed

        # Check for duplicates (assuming duplicates are removed based on `asin` and `user_id` or other criteria)
        self.assertEqual(cleaned_df.dropDuplicates().count(), len(self.test_data) - 3) 

        # Check for nulls in 'text' and 'rating'
        self.assertEqual(cleaned_df.filter(col("text").isNull()).count(), 0)  # expecting no null 'text'
        self.assertEqual(cleaned_df.filter(col("rating").isNull()).count(), 0)  # expecting no null 'rating'

        # Check if whitespaces are removed in 'title' and 'text'
        self.assertNotIn("   ", cleaned_df.select("text").first()[0])
        self.assertNotIn("   ", cleaned_df.select("title").first()[0])

        # Additional checks for this dataset
        # Check if 'review_date_timestamp' has been converted or cleaned properly, for example, is not null or invalid
        self.assertEqual(cleaned_df.filter(col("review_date_timestamp").isNull()).count(), 0)

        # Check if 'verified_purchase' values are cleaned or standardized (assuming you standardize to True/False)
        self.assertTrue(cleaned_df.select("verified_purchase").distinct().count() <= 2)  # Expecting True/False only


if __name__ == "__main__":
    unittest.main()
