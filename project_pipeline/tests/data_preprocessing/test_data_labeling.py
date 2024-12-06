import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

import unittest
import pandas as pd
from data_pipeline.dags.utils.data_preprocessing.data_labeling import apply_labelling

class TestDataLabeling(unittest.TestCase):

    def setUp(self):
        """Set up a simple DataFrame for testing."""
        # Sample test data based on the provided CSV structure
        self.test_data = [
            {
                'rating': 1,
                'text': "Horrible don’t buy",
                'asin': "B017FYTGHQ",
                'user_id': "AERILB555X2XGT2JW43ZYJ45NESA",
                'timestamp': "2019-01-22 01:54:43.330",
                'helpful_vote': 0,
                'verified_purchase': True
            },
            {
                'rating': 4,
                'text': "Excellent product, very satisfied!",
                'asin': "B07KQ4XNDV",
                'user_id': "AFULCCKP3Z677YSIVX6BW7J3PY6Q",
                'timestamp': "2019-01-15 13:14:56.979",
                'helpful_vote': 0,
                'verified_purchase': True
            },
            {
                'rating': 2,
                'text': "Okayish product has some good, some bad",
                'asin': "B079K7DK8X",
                'user_id': "AFB3EPFSQKO53U3FIXZ2IHMD5MGQ",
                'timestamp': "2019-01-18 19:26:02.216",
                'helpful_vote': 0,
                'verified_purchase': False
            }
        ]

        self.df_raw = pd.DataFrame(self.test_data)

    def test_apply_labelling(self):
        """Test the labeling function."""
        labeled_df = apply_labelling(self.df_raw)

        # Check if the sentiment label column is added
        self.assertIn('sentiment_label', labeled_df.columns)

        # Verify that the labels are correctly assigned, checking only for POSITIVE & NEGATIVE
        self.assertEqual(labeled_df['sentiment_label'].iloc[0], 'NEGATIVE')  
        self.assertEqual(labeled_df['sentiment_label'].iloc[1], 'POSITIVE')  


if __name__ == "__main__":
    unittest.main()
