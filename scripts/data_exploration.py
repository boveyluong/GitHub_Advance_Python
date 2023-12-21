import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path # needed for dataloader import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent)) # needed for dataloader import

from modules.data_loader import DataLoader
import numpy as np
import os

# Set plot style
sns.set_style("darkgrid")

# Load data
data_loader = DataLoader('../config.json')
experiment1_data = data_loader.load_experiment_data('experiment2')

# Basic Data Overview
print(experiment1_data.describe())
# Check for missing values
print(experiment1_data.isnull().sum())

# Time Series Visualization
plt.figure(figsize=(12, 6))
plt.plot(experiment1_data['data'])
plt.title('Time Series - Experiment 1')
plt.xlabel('Time')
plt.ylabel('Signal')

# Create the 'plots' directory if it doesn't exist
os.makedirs('plots', exist_ok=True)
plt.savefig('plots/experiment1_timeseries.png')
