import pandas as pd
import numpy as np
import json
import os
import logging
from typing import Any, Dict, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataLoader:
    """
    DataLoader class to load and union datasets for different experiments.
    """
    def __init__(self, config_path: str):
        """
        Initialize the DataLoader with a configuration file.
        """
        try:
            with open(config_path, 'r') as config_file:
                self.config = json.load(config_file)
            self.validate_config()
            logging.info("Configuration loaded and validated successfully.")
        except FileNotFoundError:
            logging.error(f"Configuration file not found at {config_path}")
            raise
        except json.JSONDecodeError as e:
            logging.error(f"Invalid JSON format in configuration: {e}")
            raise
    
    def validate_config(self):
        """
        Validates the loaded configuration for necessary fields and formats.
        """
        required_fields = ['experiments']
        for field in required_fields:
            if field not in self.config:
                raise ValueError(f"Missing required field in config: {field}")

    def load_experiment_data(self, experiment_name: Union[str, None] = None) -> pd.DataFrame:
        """
        Load datasets for a given experiment or all experiments.
        """
        if experiment_name and experiment_name not in self.config['experiments']:
            raise ValueError(f"Experiment {experiment_name} not found in configuration.")

        experiment_data = []
        experiments_to_load = self.config['experiments'] if not experiment_name else {experiment_name: self.config['experiments'][experiment_name]}

        for experiment, files in experiments_to_load.items():
            for file_info in files:
                file_path = file_info['path']
                file_type = file_info['type']
                try:
                    data = self.load_file(file_path, file_type, experiment)
                    file_name = os.path.basename(file_path).split('.')[0]
                    
                    # Add 'experiment' and 'measurement' columns
                    data['experiment'] = experiment
                    data['measurement'] = file_name

                    # Keep 'time', 'experiment', and 'measurement' as regular columns
                    experiment_data.append(data)
                except Exception as e:
                    logging.error(f"Error loading file {file_path}: {e}")
                    continue

        if not experiment_data:
            logging.warning(f"No data loaded for experiment(s).")
            return pd.DataFrame()

        # Concatenate data as is without changing index
        return pd.concat(experiment_data, ignore_index=True)

    def load_file(self, file_path: str, file_type: str, experiment_name: str) -> pd.DataFrame:
        """
        Load data from a file based on its type.
        Replace commas with dots in numerical values for consistency.
        """
        if file_type == 'csv':
            data = pd.read_csv(file_path, converters={'RawData': lambda x: x.replace(',', '.')})
        elif file_type == 'tsv':
            data = pd.read_csv(file_path, sep='\t', converters={'RawData': lambda x: x.replace(',', '.')})
        elif file_type == 'pkl':
            data = pd.read_pickle(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

        if data.empty:
            logging.warning(f"No data found in file: {file_path}")

        # For experiment 4, ensure the 'RawData' column is loaded as 'data'
        if 'RawData' in data.columns and experiment_name == 'experiment4':
            data.rename(columns={'RawData': 'data'}, inplace=True)
        elif 'data' not in data.columns:
            # Assume first column is the data column
            first_column = data.columns[0]
            data.rename(columns={first_column: 'data'}, inplace=True)

        # Ensure all numerical values use a dot as the decimal separator
        data['data'] = data['data'].astype(str).str.replace(',', '.').astype(float)

        # Generate time index in seconds
        actual_number_of_samples = len(data)
        sampling_rate = 10e3  # Update this if different for Experiment 4
        time_in_seconds = np.arange(0, actual_number_of_samples) / sampling_rate
        data['time'] = time_in_seconds  # Add time column but don't set as index here


        return data

# Example usage
try:
    data_loader = DataLoader('config.json')

    # Load a specific experiment
    experiment_data = data_loader.load_experiment_data('experiment4')
    print(experiment_data.head())
    print(experiment_data.count())


except Exception as e:
    logging.error(f"An error occurred: {e}")
