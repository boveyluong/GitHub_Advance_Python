import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.seasonal import STL

class SignalPreprocessor:
    """
    Preprocess signals for feature extraction and deep learning.
    """
    
    def __init__(self, window_size: int = 5):
        """
        Initialize the preprocessor.
        :param window_size: The window size for the rolling average for noise reduction.
        """
        self.window_size = window_size
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
    def reduce_noise(self, data: pd.Series) -> pd.Series:
        """
        Apply a rolling average to reduce noise in the signal.
        :param data: Pandas Series with signal data.
        :return: Pandas Series with reduced noise.
        """
        return data.rolling(window=self.window_size, center=True).mean()
    
    def normalize_signal(self, data: pd.Series) -> pd.Series:
        """
        Normalize the signal data to a range between 0 and 1.
        :param data: Pandas Series with signal data.
        :return: Pandas Series with normalized data.
        """
        scaled_data = self.scaler.fit_transform(data.values.reshape(-1, 1))
        return pd.Series(scaled_data.flatten(), index=data.index)
    
    def detrend_signal(self, data: pd.Series) -> pd.Series:
        """
        Detrend the signal using a differencing method.
        :param data: Pandas Series with signal data.
        :return: Pandas Series with detrended data.
        """
        return data.diff().bfill()
    
    def estimate_trend(self, data: pd.Series, window_size: int = None) -> pd.Series:
        """
        Estimate the trend component of the signal using a rolling window mean.
        :param data: Pandas Series with signal data.
        :param window_size: The size of the rolling window to use for trend estimation.
        :return: Pandas Series representing the estimated trend.
        """
        if window_size is None:
            # Default to a reasonable value if none provided
            window_size = int(len(data) / 10)
        trend = data.rolling(window=window_size, center=True, min_periods=1).mean()
        return trend

    def remove_measurement_duplicates(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicates within each measurement of the experiment data.
        :param data: Pandas DataFrame with signal and measurement data.
        :return: DataFrame with duplicates removed.
        """
        # Count and print duplicates for each measurement
        measurement_duplicates = data.groupby('measurement').apply(lambda x: x.duplicated().sum())
        print(f"Duplicates per measurement:")
        print(measurement_duplicates)

        # Remove duplicates for each measurement
        data = data.groupby('measurement').apply(lambda x: x.drop_duplicates()).reset_index(drop=True)

        return data

def segment_into_windows(data, window_size_points=1000):
    """
    Segments the data into windows of the given size.
    
    :param data: The complete signal data (Pandas Series).
    :param window_size_points: The size of each window in terms of the number of data points.
    :return: A list of windows, each is a Pandas Series.
    """
    windows = [data[i:i+window_size_points] for i in range(0, len(data), window_size_points)]
    return windows
    
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all preprocessing steps to the data.
        :param data: Pandas DataFrame with signal data and possibly other metadata like 'measurement'.
        :return: DataFrame with preprocessed data.
        """
       # Check if 'measurement' column exists, then remove duplicates
        if 'measurement' in data.columns:
            data = self.remove_measurement_duplicates(data)

        # Continue with other preprocessing steps...
        # Ensure 'data' is a series for the following operations
        if isinstance(data, pd.DataFrame) and 'data' in data.columns:
            signal_series = data['data']
        else:
            raise ValueError("Data for preprocessing must include 'data' column")

        # Select only the numeric data for noise reduction and other operations
        if 'data' in data.columns:
            signal_series = data['data']
        else:
            raise ValueError("Data for preprocessing must include 'data' column")

        # Apply noise reduction, normalization, and detrending only on numeric data
        signal_noised_reduced = self.reduce_noise(signal_series)
        signal_normalized = self.normalize_signal(signal_noised_reduced)
        signal_detrended = self.detrend_signal(signal_normalized)

        # Estimate the trend using the instance's window_size
        trend = self.estimate_trend(signal_normalized, window_size=self.window_size)

        # Store results in the DataFrame
        preprocessed_data = pd.DataFrame({
            'data_noised_reduced': signal_noised_reduced,
            'data_normalized': signal_normalized,
            'data_detrended': signal_detrended,
            'trend': trend
        })
        
        return preprocessed_data