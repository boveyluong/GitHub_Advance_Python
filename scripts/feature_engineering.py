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

    # Ensure the DataFrame contains 'data' column
    if 'data' not in raw_data.columns:
        print("The column 'data' was not found. Please check the raw DataFrame.")
        return  # Exit if the required column is not found

    # Initialize the preprocessor and preprocess the data
    preprocessor = SignalPreprocessor(window_size=5)

    # Preprocess the data (including duplicate removal, windowing, noise reduction, normalization, and detrending)
    preprocessed_windows = preprocessor.preprocess(raw_data)

    # Initialize the feature extractor
    feature_extractor = FeatureExtractor()

    # Apply feature extraction to each preprocessed window and collect results
    all_features = []
    for window_data in preprocessed_windows:
        # Ensure the 'data_detrended' column is present after preprocessing
        if 'data_detrended' not in window_data.columns:
            print("The column 'data_detrended' was not found in preprocessed data.")
            continue  # Skip this window if the required column is not found

        features = feature_extractor.extract_all_features(window_data['data_detrended'])
        all_features.append(features)

    # Combine all features into a single DataFrame
    features_df = pd.concat(all_features, ignore_index=True)

    # Save the features to a new file for training and evaluation
    features_df.to_csv('features_for_model.csv', index=False)

    print("Feature engineering completed. Features saved to 'features_for_model.csv'.")

if __name__ == "__main__":
    main()
