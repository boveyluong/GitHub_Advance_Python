import pandas as pd
import json
import os

class DataLoader:
    def __init__(self, config_path):
        with open(config_path, 'r') as config_file:
            self.config = json.load(config_file)

    def load_experiment_data(self, experiment_name):
        """
        Load and union all datasets for a given experiment.
        :param experiment_name: Name of the experiment.
        :return: Unified DataFrame for the experiment.
        """
        experiment_data = []
        for file_info in self.config['experiments'][experiment_name]:
            file_path = file_info['path']
            file_type = file_info['type']
            data = self.load_file(file_path, file_type)
            data['experiment'] = os.path.basename(file_path).split('.')[0]
            experiment_data.append(data)

        return pd.concat(experiment_data, ignore_index=True)

    def load_file(self, file_path, file_type):
        """
        Load data from a file based on its type and rename the first column to 'data'.
        :param file_path: Path to the data file.
        :param file_type: Type of the data file (csv, tsv, pkl).
        :return: DataFrame with the data, with the first column renamed to 'data'.
        """
        if file_type == 'csv':
            data = pd.read_csv(file_path)
        elif file_type == 'tsv':
            data = pd.read_csv(file_path, sep='\t')
        elif file_type == 'pkl':
            data = pd.read_pickle(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

        # Rename the first column to 'data'
        data.columns = ['data'] + data.columns.tolist()[1:]
        return data

# Example usage
data_loader = DataLoader('../config.json')
experiment_data = data_loader.load_experiment_data('experiment1')
print(experiment_data.head())
