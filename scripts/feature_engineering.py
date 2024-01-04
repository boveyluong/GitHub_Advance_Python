# Import necessary libraries and classes
import pandas as pd
from pathlib import Path
import sys
import os
# Add the path to the DataLoader script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from modules.data_loader import DataLoader

from modules.signal_preprocessor import SignalPreprocessor  
from modules.feature_extractor import FeatureExtractor  

def main():
    # Load the data
    data_loader = DataLoader('config.json')
    raw_data = data_loader.load_experiment_data('experiment1')

    # Print column names to debug
    print("Column names in raw_data:", raw_data.columns)

    # Ensure 'data' column exists or adjust to the correct column name
    if 'data' not in raw_data.columns:
        print("The column 'data' was not found. Please check the raw DataFrame.")
        return  # Exit if the required column is not found

    # Segment data into windows
    windows = segment_into_windows(raw_data['data'])

    # Initialize the preprocessor and preprocess the data
    preprocessor = SignalPreprocessor(window_size=5)

    # Assuming 'data' is the name of the column containing the signal
    preprocessed_data = preprocessor.preprocess(raw_data)

    # Initialize the feature extractor and extract features
    feature_extractor = FeatureExtractor()
    features_df = feature_extractor.extract_all_features(preprocessed_data['data_detrended'])

    # Save the features to a new file for training and evaluation
    features_df.to_csv('features_for_model.csv', index=False)

    print("Feature engineering completed. Features saved to 'features_for_model.csv'.")

if __name__ == "__main__":
    main()
