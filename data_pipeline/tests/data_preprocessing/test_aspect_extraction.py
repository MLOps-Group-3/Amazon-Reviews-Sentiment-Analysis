import unittest
import pandas as pd
from data_pipeline.dags.utils.data_preprocessing.aspect_extraction import tag_and_expand_aspects

class TestAspectExtraction(unittest.TestCase):

    def setUp(self):
        """Set up a simple DataFrame for testing."""
        # Sample test data based on the expected DataFrame structure
        self.test_data = [
            {
                'text': "The delivery was fast and the product quality is great.",
                'user_id': "USER1",
                'asin': "B0001"
            },
            {
                'text': "I had a terrible experience with customer service.",
                'user_id': "USER2",
                'asin': "B0002"
            },
            {
                'text': "The design is stylish but the price is too high.",
                'user_id': "USER3",
                'asin': "B0003"
            }
        ]

        # Create a DataFrame from the test data
        self.df_raw = pd.DataFrame(self.test_data)

    def test_tag_and_expand_aspects(self):
        """Test the aspect tagging and expansion function."""
        # Call the function to test
        tagged_df = tag_and_expand_aspects(self.df_raw, aspects={  # Pass the relevant aspects
            "delivery": {"delivery", "fast", "shipping"},
            "quality": {"quality", "great", "good"},
            "customer_service": {"service", "helpful", "customer"},
            "product_design": {"design", "stylish"},
            "cost": {"price", "expensive", "cheap"}
        })

        # Check if the 'aspect' column is added
        self.assertIn('aspect', tagged_df.columns)
        # print(tagged_df[['aspect','relevant_sentences']])
        # Check if relevant aspects are correctly tagged
        # First review has 2 aspects, each get captured:
        self.assertEqual(tagged_df['aspect'].iloc[0], 'delivery')  
        self.assertEqual(tagged_df['aspect'].iloc[1], 'quality')  

        # Second review has 1 aspect
        self.assertEqual(tagged_df['aspect'].iloc[2], 'customer_service')  

        # Third review has 2 ascpects
        self.assertEqual(tagged_df['aspect'].iloc[3], 'product_design')  
        self.assertEqual(tagged_df['aspect'].iloc[4], 'cost')  

if __name__ == "__main__":
    unittest.main()
