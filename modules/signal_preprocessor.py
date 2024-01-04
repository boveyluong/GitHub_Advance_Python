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
        return data.diff().fillna(method='bfill')
    
    def decompose_signal(self, data: pd.Series, period: int = None) -> dict:
        """
        Decompose the signal into trend, seasonal, and residual components.
        :param data: Pandas Series with signal data.
        :param period: The period of the seasonal component.
        :return: Dictionary with trend, seasonal, and residual components.
        """
        stl = STL(data, seasonal=period)
        result = stl.fit()
        return {
            'trend': result.trend,
            'seasonal': result.seasonal,
            'residual': result.resid
        }
    
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all preprocessing steps to the data.
        :param data: Pandas DataFrame with signal data.
        :return: Pandas DataFrame with preprocessed data.
        """
        signal = data['data']
        signal_noised_reduced = self.reduce_noise(signal)
        signal_normalized = self.normalize_signal(signal_noised_reduced)
        signal_detrended = self.detrend_signal(signal_normalized)
        decomposed = self.decompose_signal(signal_detrended)

        # Store results in the DataFrame
        data['data_noised_reduced'] = signal_noised_reduced
        data['data_normalized'] = signal_normalized
        data['data_detrended'] = signal_detrended
        data['trend'] = decomposed['trend']
        data['seasonal'] = decomposed['seasonal']
        data['residual'] = decomposed['residual']
        
        return data
