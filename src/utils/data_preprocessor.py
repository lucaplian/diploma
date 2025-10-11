"""
Data Preprocessing Utilities for Smart Home Sensor Data
"""

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Tuple, Optional


class DataPreprocessor:
    """
    Preprocesses sensor data for model input.
    Handles normalization, sequence creation, and feature engineering.
    """
    
    def __init__(
        self,
        sequence_length: int = 50,
        scaling_method: str = 'standard'
    ):
        """
        Initialize the data preprocessor.
        
        Args:
            sequence_length: Length of sequences to create
            scaling_method: 'standard' or 'minmax' for data scaling
        """
        self.sequence_length = sequence_length
        self.scaling_method = scaling_method
        self.scaler = None
        
        if scaling_method == 'standard':
            self.scaler = StandardScaler()
        elif scaling_method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError("scaling_method must be 'standard' or 'minmax'")
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Fit the scaler and transform the data.
        
        Args:
            data: Input data (samples, features)
            
        Returns:
            Scaled data
        """
        return self.scaler.fit_transform(data)
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted scaler.
        
        Args:
            data: Input data (samples, features)
            
        Returns:
            Scaled data
        """
        if self.scaler is None:
            raise ValueError("Scaler must be fitted before transform")
        return self.scaler.transform(data)
    
    def create_sequences(
        self,
        data: np.ndarray,
        labels: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Create sequences from time-series data.
        
        Args:
            data: Input data (timesteps, features)
            labels: Optional labels for each timestep
            
        Returns:
            Tuple of (sequences, sequence_labels)
        """
        sequences = []
        sequence_labels = [] if labels is not None else None
        
        for i in range(len(data) - self.sequence_length):
            seq = data[i:i + self.sequence_length]
            sequences.append(seq)
            
            if labels is not None:
                # Use the label at the end of the sequence
                sequence_labels.append(labels[i + self.sequence_length])
        
        sequences = np.array(sequences)
        
        if sequence_labels is not None:
            sequence_labels = np.array(sequence_labels)
        
        return sequences, sequence_labels
    
    def add_temporal_features(self, data: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
        """
        Add temporal features to the data (hour, day of week, etc.).
        
        Args:
            data: Input data (samples, features)
            timestamps: Unix timestamps for each sample
            
        Returns:
            Data with additional temporal features
        """
        # Extract temporal features
        hours = (timestamps % 86400) / 3600  # Hour of day (0-23)
        day_of_week = ((timestamps // 86400) % 7)  # Day of week (0-6)
        
        # Normalize temporal features
        hours_norm = hours / 23.0
        day_norm = day_of_week / 6.0
        
        # Add as new features
        temporal_features = np.column_stack([hours_norm, day_norm])
        augmented_data = np.concatenate([data, temporal_features], axis=1)
        
        return augmented_data
    
    def sliding_window_split(
        self,
        sequences: np.ndarray,
        labels: np.ndarray,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split sequences into train, validation, and test sets.
        
        Args:
            sequences: Input sequences (samples, sequence_length, features)
            labels: Labels for each sequence
            train_ratio: Ratio of data for training
            val_ratio: Ratio of data for validation
            
        Returns:
            Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        n_samples = len(sequences)
        train_size = int(n_samples * train_ratio)
        val_size = int(n_samples * val_ratio)
        
        X_train = sequences[:train_size]
        y_train = labels[:train_size]
        
        X_val = sequences[train_size:train_size + val_size]
        y_val = labels[train_size:train_size + val_size]
        
        X_test = sequences[train_size + val_size:]
        y_test = labels[train_size + val_size:]
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def handle_missing_values(self, data: np.ndarray, method: str = 'interpolate') -> np.ndarray:
        """
        Handle missing values in sensor data.
        
        Args:
            data: Input data with potential missing values (NaN)
            method: 'interpolate', 'forward_fill', or 'mean'
            
        Returns:
            Data with missing values handled
        """
        if method == 'interpolate':
            # Linear interpolation
            for col in range(data.shape[1]):
                mask = np.isnan(data[:, col])
                if mask.any():
                    valid_indices = np.where(~mask)[0]
                    if len(valid_indices) > 0:
                        data[mask, col] = np.interp(
                            np.where(mask)[0],
                            valid_indices,
                            data[valid_indices, col]
                        )
        elif method == 'forward_fill':
            # Forward fill
            for col in range(data.shape[1]):
                mask = np.isnan(data[:, col])
                if mask.any():
                    idx = np.where(~mask, np.arange(len(mask)), 0)
                    np.maximum.accumulate(idx, out=idx)
                    data[:, col] = data[idx, col]
        elif method == 'mean':
            # Fill with column mean
            for col in range(data.shape[1]):
                col_mean = np.nanmean(data[:, col])
                data[np.isnan(data[:, col]), col] = col_mean
        else:
            raise ValueError("method must be 'interpolate', 'forward_fill', or 'mean'")
        
        return data
    
    def extract_statistical_features(self, sequences: np.ndarray) -> np.ndarray:
        """
        Extract statistical features from sequences for enhanced representation.
        
        Args:
            sequences: Input sequences (samples, sequence_length, features)
            
        Returns:
            Statistical features (samples, num_stats * features)
        """
        # Calculate statistics along the time axis
        mean_features = np.mean(sequences, axis=1)
        std_features = np.std(sequences, axis=1)
        min_features = np.min(sequences, axis=1)
        max_features = np.max(sequences, axis=1)
        
        # Concatenate all statistical features
        stat_features = np.concatenate([
            mean_features,
            std_features,
            min_features,
            max_features
        ], axis=1)
        
        return stat_features
