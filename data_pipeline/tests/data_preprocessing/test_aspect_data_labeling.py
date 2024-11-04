import unittest
import pandas as pd
from data_pipeline.dags.utils.data_preprocessing.aspect_data_labeling import apply_vader_labeling

class TestAspectDataLabeling(unittest.TestCase):

    def setUp(self):
        """Set up a simple DataFrame for testing."""
        # Sample test data based on the expected DataFrame structure
        self.test_data = [
            {
                'relevant_sentences': "The delivery was fast and the product quality is great.",
                'user_id': "USER1",
                'asin': "B0001"
            },
            {
                'relevant_sentences': "I had a terrible experience with customer service.",
                'user_id': "USER2",
                'asin': "B0002"
            },
            {
                'relevant_sentences': "The design is stylish but the price is too high.",
                'user_id': "USER3",
                'asin': "B0003"
            }
        ]

        # Create a DataFrame from the test data
        self.df_raw = pd.DataFrame(self.test_data)

    def test_apply_vader_labeling(self):
        """Test the VADER labeling function to ensure sentiment labels are added."""
        labeled_df = apply_vader_labeling(self.df_raw)
     
        # Check if the 'sentiment_label' column is added
        self.assertIn('sentiment_label', labeled_df.columns)
    
        # Check if the sentiment labels are correctly assigned
        self.assertTrue(labeled_df['sentiment_label'].notnull().all())  # Ensure no null values in 'sentiment_label'


if __name__ == "__main__":
    unittest.main()
