import pandas as pd
import numpy as np
from scipy.fftpack import fft
from scipy.signal import welch

class FeatureExtractor:
    """
    Extracts features from a signal time series for machine learning.
    """

    def __init__(self):
        pass

    def extract_basic_features(self, data: pd.Series) -> dict:
        """
        Extract basic statistical features from the time series data.
        :param data: Pandas Series with signal data.
        :return: Dictionary with basic features.
        """
        features = {
            'mean': data.mean(),
            'std': data.std(),
            'max': data.max(),
            'min': data.min(),
            'range': data.max() - data.min(),
            'skewness': data.skew(),
            'kurtosis': data.kurt()
        }
        return features

    def extract_fft_features(self, data: pd.Series) -> dict:
        """
        Extract Fast Fourier Transform features from the time series data.
        :param data: Pandas Series with signal data.
        :return: Dictionary with FFT features.
        """
        # Ensure data is in the correct format (NumPy array)
        if isinstance(data, pd.Series):
            data = data.values

        # Compute FFT and frequencies
        signal_fft = fft(data)
        magnitude = np.abs(signal_fft)
        angle = np.angle(signal_fft)

        # Only use the first half of the FFT as it is symmetrical for real signals
        half = len(data) // 2
        features = {
            'fft_magnitude_mean': np.mean(magnitude[:half]),
            'fft_magnitude_std': np.std(magnitude[:half]),
            'fft_angle_mean': np.mean(angle[:half]),
            'fft_angle_std': np.std(angle[:half]),
        }
        return features

    def extract_power_spectral_density_features(self, data: pd.Series) -> dict:
        """
        Extract Power Spectral Density features from the time series data.
        :param data: Pandas Series with signal data.
        :return: Dictionary with PSD features.
        """
        freqs, psd = welch(data)
        features = {
            'psd_mean': np.mean(psd),
            'psd_std': np.std(psd),
            'psd_max': np.max(psd),
            'psd_min': np.min(psd),
        }
        return features
    
    def extract_all_features(self, data: pd.Series) -> pd.DataFrame:
        """
        Extract all features and return as a DataFrame for machine learning.
        :param data: Pandas Series with signal data.
        :return: DataFrame with all features.
        """
        basic_features = self.extract_basic_features(data)
        fft_features = self.extract_fft_features(data)
        psd_features = self.extract_power_spectral_density_features(data)

        # Combine all features into a single dictionary
        all_features = {**basic_features, **fft_features, **psd_features}

        # Convert to DataFrame
        features_df = pd.DataFrame([all_features])
        return features_df
