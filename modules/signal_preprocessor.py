import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class SignalPreprocessor:
    """
    Preprocess signals for feature extraction and deep learning.
    """

    def __init__(self, window_size: int = 5, window_size_points: int = 1000):
        """
        Initialize the preprocessor.
        :param window_size: The window size for the rolling average for noise reduction.
        :param window_size_points: The number of data points in each window for segmentation.
        """
        self.window_size = window_size
        self.window_size_points = window_size_points
        self.scaler = MinMaxScaler(feature_range=(0, 1))

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

    def segment_into_windows(self, data: pd.Series) -> [pd.Series]:
        """
        Segments the data into windows of the given size.
        :param data: The complete signal data (Pandas Series).
        :return: A list of windows, each is a Pandas Series.
        """
        return [data[i:i+self.window_size_points] for i in range(0, len(data), self.window_size_points)]

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

    def preprocess(self, data: pd.DataFrame) -> [pd.DataFrame]:
        """
        Apply all preprocessing steps to the data.
        :param data: Pandas DataFrame with signal data and possibly other metadata like 'measurement'.
        :return: List of DataFrames, each representing preprocessed data for a window.
        """

        # Ensure 'data' column is present for further processing
        if 'data' not in data.columns:
            raise ValueError("Data for preprocessing must include 'data' column")

        # Extract 'data' series for further processing
        signal_series = data['data']

        # Step 2: Segment data into windows
        windows = self.segment_into_windows(signal_series)

        # Apply further preprocessing to each window
        preprocessed_windows = []
        for window in windows:
            # Step 3: Noise Reduction
            noised_reduced = self.reduce_noise(window)

            # Step 4: Normalization
            normalized = self.normalize_signal(noised_reduced)

            # Step 5: Detrending
            detrended = self.detrend_signal(normalized)

            # Combine preprocessed data into a DataFrame for this window
            window_df = pd.DataFrame({
                'data_noised_reduced': noised_reduced,
                'data_normalized': normalized,
                'data_detrended': detrended,
            })

            preprocessed_windows.append(window_df)

        return preprocessed_windows