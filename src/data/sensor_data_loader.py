"""
Sensor Data Loader for Smart Home Environment
Loads and manages sensor data from various sources
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict
import os


class SensorDataLoader:
    """
    Loads sensor data from smart home environment.
    Supports multiple data formats and sensor types.
    """
    
    # Standard sensor types for AAL, HAR, and HA domains
    SENSOR_TYPES = {
        'motion': 'Motion detection sensor',
        'door': 'Door/window sensor',
        'temperature': 'Temperature sensor',
        'humidity': 'Humidity sensor',
        'light': 'Light level sensor',
        'energy': 'Energy consumption sensor',
        'presence': 'Presence detection sensor',
        'pressure': 'Pressure mat sensor',
        'appliance': 'Appliance usage sensor',
        'location': 'Location tracking sensor'
    }
    
    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize the sensor data loader.
        
        Args:
            data_dir: Directory containing sensor data files
        """
        self.data_dir = data_dir
        self.sensor_data = None
        self.labels = None
        self.metadata = {}
    
    def load_from_csv(
        self,
        filepath: str,
        timestamp_col: str = 'timestamp',
        label_col: Optional[str] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Load sensor data from CSV file.
        
        Args:
            filepath: Path to CSV file
            timestamp_col: Name of timestamp column
            label_col: Name of label column (optional)
            
        Returns:
            Tuple of (sensor_data, labels)
        """
        df = pd.read_csv(filepath)
        
        # Store metadata
        self.metadata['source'] = filepath
        self.metadata['timestamp_column'] = timestamp_col
        self.metadata['num_samples'] = len(df)
        
        # Extract labels if present
        labels = None
        if label_col and label_col in df.columns:
            labels = df[label_col].values
            df = df.drop(columns=[label_col])
        
        # Remove timestamp column from features
        if timestamp_col in df.columns:
            self.metadata['timestamps'] = df[timestamp_col].values
            df = df.drop(columns=[timestamp_col])
        
        self.metadata['feature_names'] = df.columns.tolist()
        self.metadata['num_features'] = len(df.columns)
        
        sensor_data = df.values
        
        self.sensor_data = sensor_data
        self.labels = labels
        
        return sensor_data, labels
    
    def load_from_numpy(
        self,
        data_path: str,
        labels_path: Optional[str] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Load sensor data from NumPy files.
        
        Args:
            data_path: Path to .npy file with sensor data
            labels_path: Path to .npy file with labels (optional)
            
        Returns:
            Tuple of (sensor_data, labels)
        """
        sensor_data = np.load(data_path)
        
        labels = None
        if labels_path and os.path.exists(labels_path):
            labels = np.load(labels_path)
        
        self.metadata['source'] = data_path
        self.metadata['num_samples'] = sensor_data.shape[0]
        self.metadata['num_features'] = sensor_data.shape[1] if len(sensor_data.shape) > 1 else 1
        
        self.sensor_data = sensor_data
        self.labels = labels
        
        return sensor_data, labels
    
    def generate_synthetic_data(
        self,
        num_samples: int = 10000,
        num_features: int = 10,
        noise_level: float = 0.1,
        behaviour_distribution: Tuple[float, float, float] = (0.6, 0.2, 0.2)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic sensor data for testing and demonstration.
        
        Args:
            num_samples: Number of samples to generate
            num_features: Number of sensor features
            noise_level: Level of random noise to add
            behaviour_distribution: Distribution of (normal, automated, optimising)
            
        Returns:
            Tuple of (sensor_data, labels)
        """
        # Generate base patterns for different behaviours
        data = []
        labels = []
        
        # Calculate number of samples for each behaviour type
        n_normal = int(num_samples * behaviour_distribution[0])
        n_automated = int(num_samples * behaviour_distribution[1])
        n_optimising = num_samples - n_normal - n_automated
        
        # Generate normal behaviour (random walk)
        for i in range(n_normal):
            sample = np.random.randn(num_features) * 0.5 + np.random.uniform(-1, 1, num_features)
            data.append(sample)
            labels.append(0)  # Normal
        
        # Generate automated behaviour (periodic patterns)
        for i in range(n_automated):
            t = i / n_automated * 4 * np.pi
            sample = np.array([
                np.sin(t + j * np.pi / num_features) + np.random.randn() * noise_level
                for j in range(num_features)
            ])
            data.append(sample)
            labels.append(1)  # Automated
        
        # Generate optimising behaviour (trend patterns)
        for i in range(n_optimising):
            trend = i / n_optimising
            sample = np.array([
                trend * (j + 1) / num_features + np.random.randn() * noise_level
                for j in range(num_features)
            ])
            data.append(sample)
            labels.append(2)  # Optimising
        
        # Shuffle data
        indices = np.random.permutation(num_samples)
        sensor_data = np.array(data)[indices]
        labels = np.array(labels)[indices]
        
        self.metadata['source'] = 'synthetic'
        self.metadata['num_samples'] = num_samples
        self.metadata['num_features'] = num_features
        self.metadata['behaviour_distribution'] = behaviour_distribution
        
        self.sensor_data = sensor_data
        self.labels = labels
        
        return sensor_data, labels
    
    def get_sensor_info(self) -> Dict:
        """
        Get information about loaded sensor data.
        
        Returns:
            Dictionary with sensor data information
        """
        info = {
            'metadata': self.metadata,
            'data_shape': self.sensor_data.shape if self.sensor_data is not None else None,
            'labels_shape': self.labels.shape if self.labels is not None else None,
            'has_labels': self.labels is not None,
            'sensor_types': list(self.SENSOR_TYPES.keys())
        }
        
        if self.labels is not None:
            unique_labels, counts = np.unique(self.labels, return_counts=True)
            info['label_distribution'] = dict(zip(unique_labels.tolist(), counts.tolist()))
        
        return info
    
    def save_data(
        self,
        data_path: str,
        labels_path: Optional[str] = None
    ):
        """
        Save sensor data to NumPy files.
        
        Args:
            data_path: Path to save sensor data
            labels_path: Path to save labels (optional)
        """
        if self.sensor_data is None:
            raise ValueError("No sensor data to save")
        
        np.save(data_path, self.sensor_data)
        
        if labels_path and self.labels is not None:
            np.save(labels_path, self.labels)
    
    def get_data(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Get the loaded sensor data and labels.
        
        Returns:
            Tuple of (sensor_data, labels)
        """
        return self.sensor_data, self.labels
