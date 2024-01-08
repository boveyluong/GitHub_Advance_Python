import pandas as pd
import numpy as np
from scipy.fftpack import fft
from scipy.signal import welch
from statsmodels.tsa.stattools import adfuller

class FeatureExtractor:
    """
    Extract features from a univariate time series for binary outcome prediction.
    """

    def extract_statistical_features(self, data: pd.Series) -> dict:
        """
        Extract statistical features from the data.
        :param data: Pandas Series with signal data.
        :return: Dictionary with statistical features.
        """
        return {
            'mean': data.mean(),
            'std': data.std(),
            'max': data.max(),
            'min': data.min(),
            'median': data.median(),
            'skew': data.skew(),
            'kurtosis': data.kurt(),
            'quantile1': data.quantile(0.25),
            'quantile3': data.quantile(0.75),
            'iqr': data.quantile(0.75) - data.quantile(0.25),
        }

    def extract_fft_features(self, data: pd.Series) -> dict:
        """
        Extract Fast Fourier Transform features from the data.
        :param data: Pandas Series with signal data.
        :return: Dictionary with FFT features.
        """
        # Convert the Pandas Series to a NumPy array
        data_array = data.to_numpy()

        # Compute the fast Fourier transform
        fft_spectrum = fft(data_array)
        fft_magnitude = np.abs(fft_spectrum)
        
        # Extract basic features from FFT coefficients
        return {
            'fft_mean': fft_magnitude.mean(),
            'fft_std': fft_magnitude.std(),
        }


    def extract_power_spectral_density_features(self, data: pd.Series) -> dict:
        """
        Extract Power Spectral Density features from the data.
        :param data: Pandas Series with signal data.
        :return: Dictionary with PSD features.
        """
        freqs, psd = welch(data)
        return {
            'psd_mean': psd.mean(),
            'psd_std': psd.std(),
        }

    def extract_time_features(self, data: pd.Series) -> dict:
        """
        Extract time-based features from the data.
        :param data: Pandas Series with signal data.
        :return: Dictionary with time-based features.
        """
        # Drop NaNs or replace them with a suitable value
        cleaned_data = data.dropna()

        # Check if the cleaned data is not empty
        if cleaned_data.empty:
            raise ValueError("Data contains only NaNs.")

        # Replace any infinite values with NaN, then drop them
        cleaned_data.replace([np.inf, -np.inf], np.nan, inplace=True)
        cleaned_data.dropna(inplace=True)

        # Check if the cleaned data is not empty again after dropping infinities
        if cleaned_data.empty:
            raise ValueError("Data contains only infinities or NaNs after cleaning.")

        # Check for stationarity
        adf_result = adfuller(cleaned_data)
        return {
            'adf_statistic': adf_result[0],
            'adf_pvalue': adf_result[1],
        }

    def extract_all_features(self, data: pd.Series) -> pd.DataFrame:
        """
        Extract all features and return as a DataFrame for machine learning.
        :param data: Pandas Series with signal data.
        :return: DataFrame with all features.
        """
        all_features = {}
        all_features.update(self.extract_statistical_features(data))
        all_features.update(self.extract_fft_features(data))
        all_features.update(self.extract_power_spectral_density_features(data))
        all_features.update(self.extract_time_features(data))

        # Convert to DataFrame
        features_df = pd.DataFrame([all_features])
        return features_df
