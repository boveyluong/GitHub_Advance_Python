import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from modules.signal_preprocessor import SignalPreprocessor   # adjust the import path as needed

class TestSignalPreprocessor(unittest.TestCase):
    
    def setUp(self):
        # Initialize SignalPreprocessor with a sample window size
        self.preprocessor = SignalPreprocessor(window_size=5)
        # Create a simple synthetic signal for testing
        self.synthetic_signal = pd.Series(np.random.rand(100))
        # Sample DataFrame including 'measurement' column for testing duplicate removal
        self.synthetic_data = pd.DataFrame({
            'measurement': ['m1']*50 + ['m2']*50,
            'data': np.random.rand(100)
        })

    def test_reduce_noise(self):
        # Test the noise reduction method
        reduced_noise_signal = self.preprocessor.reduce_noise(self.synthetic_signal)
        self.assertEqual(len(reduced_noise_signal), len(self.synthetic_signal))  # Length should be unchanged

    def test_normalize_signal(self):
        # Test the signal normalization method
        normalized_signal = self.preprocessor.normalize_signal(self.synthetic_signal)
        self.assertTrue((normalized_signal >= 0).all() and (normalized_signal <= 1).all())  # All values should be in [0, 1]

    def test_detrend_signal(self):
        # Test the detrending method
        detrended_signal = self.preprocessor.detrend_signal(self.synthetic_signal)
        self.assertEqual(len(detrended_signal), len(self.synthetic_signal))  # Length should be unchanged

    def test_estimate_trend(self):
        # Test the trend estimation method
        estimated_trend = self.preprocessor.estimate_trend(self.synthetic_signal)
        self.assertEqual(len(estimated_trend), len(self.synthetic_signal))  # Length should be unchanged

    def test_remove_measurement_duplicates(self):
        # Test duplicate removal
        unique_data = self.preprocessor.remove_measurement_duplicates(self.synthetic_data)
        self.assertTrue(unique_data['measurement'].value_counts().max() == 50)  # Assuming no duplicates, counts should remain

    def test_preprocess(self):
        # Test the entire preprocessing workflow
        preprocessed_data = self.preprocessor.preprocess(self.synthetic_data)
        # Assertions based on expected behavior of preprocessing
        self.assertIn('trend', preprocessed_data.columns)  # Check if 'trend' column is present

# This allows the test to be run from the command line
if __name__ == '__main__':
    unittest.main()
