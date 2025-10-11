"""
Smart Home AI-Powered Analysis System
Main orchestrator for behaviour detection and automation/optimization
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

from models.bilstm_detector import BehaviourDetector
from models.lstm_optimizer import AutomationOptimizer
from utils.data_preprocessor import DataPreprocessor
from utils.action_handler import ActionHandler
from data.sensor_data_loader import SensorDataLoader


class SmartHomeSystem:
    """
    Main system for AI-powered smart home analysis.
    Integrates BI-LSTM detection and LSTM optimization/automation.
    
    Domains: 
    - Ambient Assisted Living (AAL)
    - Human Activity Recognition (HAR)
    - Smart Home Automation (HA)
    """
    
    # Default action mappings for smart home control
    DEFAULT_ACTIONS = {
        0: "Adjust lighting based on occupancy",
        1: "Optimize HVAC temperature settings",
        2: "Manage appliance power consumption",
        3: "Activate security protocols",
        4: "Schedule device operations"
    }
    
    def __init__(
        self,
        sequence_length: int = 50,
        n_features: int = 10,
        n_actions: int = 5,
        action_map: Optional[Dict[int, str]] = None
    ):
        """
        Initialize the Smart Home System.
        
        Args:
            sequence_length: Length of input sequences
            n_features: Number of sensor features
            n_actions: Number of possible actions
            action_map: Custom mapping of action IDs to descriptions
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.n_actions = n_actions
        self.action_map = action_map or self.DEFAULT_ACTIONS
        
        # Initialize components
        self.detector = BehaviourDetector(
            sequence_length=sequence_length,
            n_features=n_features
        )
        
        self.automation_optimizer = AutomationOptimizer(
            sequence_length=sequence_length,
            n_features=n_features,
            n_actions=n_actions,
            mode='automation'
        )
        
        self.optimisation_optimizer = AutomationOptimizer(
            sequence_length=sequence_length,
            n_features=n_features,
            n_actions=n_actions,
            mode='optimisation'
        )
        
        self.preprocessor = DataPreprocessor(sequence_length=sequence_length)
        self.action_handler = ActionHandler(log_actions=True)
        self.data_loader = SensorDataLoader()
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        self.is_trained = False
    
    def train_system(
        self,
        sensor_data: np.ndarray,
        behaviour_labels: np.ndarray,
        action_labels: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        validation_split: float = 0.15
    ):
        """
        Train the complete system (detector and optimizers).
        
        Args:
            sensor_data: Raw sensor data (samples, features)
            behaviour_labels: Labels for behaviour type (0=normal, 1=automated, 2=optimising)
            action_labels: Labels for optimal actions
            epochs: Training epochs
            batch_size: Batch size
            validation_split: Validation data split ratio
        """
        self.logger.info("Starting system training...")
        
        # Preprocess data
        self.logger.info("Preprocessing data...")
        scaled_data = self.preprocessor.fit_transform(sensor_data)
        sequences, seq_labels = self.preprocessor.create_sequences(
            scaled_data,
            behaviour_labels
        )
        _, action_seq_labels = self.preprocessor.create_sequences(
            scaled_data,
            action_labels
        )
        
        # Split data
        X_train, y_train, X_val, y_val, X_test, y_test = \
            self.preprocessor.sliding_window_split(sequences, seq_labels)
        
        _, y_action_train, _, y_action_val, _, y_action_test = \
            self.preprocessor.sliding_window_split(sequences, action_seq_labels)
        
        self.logger.info(f"Training data shape: {X_train.shape}")
        self.logger.info(f"Validation data shape: {X_val.shape}")
        self.logger.info(f"Test data shape: {X_test.shape}")
        
        # Train behaviour detector
        self.logger.info("Training BI-LSTM Behaviour Detector...")
        self.detector.build_model()
        self.detector.train(
            X_train, y_train,
            X_val, y_val,
            epochs=epochs,
            batch_size=batch_size
        )
        
        # Evaluate detector
        detector_metrics = self.detector.evaluate(X_test, y_test)
        self.logger.info(f"Detector performance: {detector_metrics}")
        
        # Filter data for automation training (only automated behaviour)
        automated_mask = y_train == 1
        if automated_mask.any():
            self.logger.info("Training LSTM Automation Optimizer...")
            X_auto_train = X_train[automated_mask]
            y_auto_train = y_action_train[automated_mask]
            
            automated_mask_val = y_val == 1
            X_auto_val = X_val[automated_mask_val] if automated_mask_val.any() else None
            y_auto_val = y_action_val[automated_mask_val] if automated_mask_val.any() else None
            
            self.automation_optimizer.build_model()
            self.automation_optimizer.train(
                X_auto_train, y_auto_train,
                X_auto_val, y_auto_val,
                epochs=epochs,
                batch_size=batch_size
            )
        
        # Filter data for optimisation training (only optimising behaviour)
        optimising_mask = y_train == 2
        if optimising_mask.any():
            self.logger.info("Training LSTM Optimisation Optimizer...")
            X_opt_train = X_train[optimising_mask]
            y_opt_train = y_action_train[optimising_mask]
            
            optimising_mask_val = y_val == 2
            X_opt_val = X_val[optimising_mask_val] if optimising_mask_val.any() else None
            y_opt_val = y_action_val[optimising_mask_val] if optimising_mask_val.any() else None
            
            self.optimisation_optimizer.build_model()
            self.optimisation_optimizer.train(
                X_opt_train, y_opt_train,
                X_opt_val, y_opt_val,
                epochs=epochs,
                batch_size=batch_size
            )
        
        self.is_trained = True
        self.logger.info("System training completed successfully!")
    
    def analyze_and_act(
        self,
        sensor_data: np.ndarray,
        execute_actions: bool = True
    ) -> Dict:
        """
        Analyze sensor data and take appropriate actions.
        
        Args:
            sensor_data: Real-time sensor data (samples, features)
            execute_actions: Whether to execute recommended actions
            
        Returns:
            Dictionary with analysis results and actions taken
        """
        if not self.is_trained:
            raise ValueError("System must be trained before analysis")
        
        # Preprocess incoming data
        scaled_data = self.preprocessor.transform(sensor_data)
        sequences, _ = self.preprocessor.create_sequences(scaled_data)
        
        if len(sequences) == 0:
            return {
                'status': 'insufficient_data',
                'message': 'Not enough data to form sequences'
            }
        
        # Detect behaviour type
        behaviour_predictions = self.detector.predict(sequences)
        behaviour_probs = self.detector.predict_proba(sequences)
        
        # Get the most recent prediction
        latest_behaviour = behaviour_predictions[-1]
        latest_probs = behaviour_probs[-1]
        behaviour_type = self.detector.get_behaviour_type(latest_behaviour)
        
        self.logger.info(f"Detected behaviour: {behaviour_type} (confidence: {latest_probs[latest_behaviour]:.2f})")
        
        result = {
            'behaviour': behaviour_type,
            'behaviour_id': int(latest_behaviour),
            'confidence': float(latest_probs[latest_behaviour]),
            'all_probabilities': {
                'normal': float(latest_probs[0]),
                'automated': float(latest_probs[1]),
                'optimising': float(latest_probs[2])
            },
            'actions_taken': []
        }
        
        # Take action based on behaviour type
        if latest_behaviour == 1:  # Automated behaviour
            self.logger.info("Handling automated behaviour...")
            action_result = self.automation_optimizer.optimize_for_behaviour(
                sequences[-1:],
                'automated',
                self.action_map
            )
            result['optimization_result'] = action_result
            
            if execute_actions and action_result['recommendations']:
                top_action = action_result['recommendations'][0]
                action_executed = self.action_handler.execute_action(
                    action_id=top_action['action_id'],
                    action_name=top_action['action_name'],
                    behaviour_type='automated',
                    confidence=top_action['confidence']
                )
                result['actions_taken'].append(action_executed)
        
        elif latest_behaviour == 2:  # Optimising behaviour
            self.logger.info("Handling optimising behaviour...")
            action_result = self.optimisation_optimizer.optimize_for_behaviour(
                sequences[-1:],
                'optimising',
                self.action_map
            )
            result['optimization_result'] = action_result
            
            if execute_actions and action_result['recommendations']:
                top_action = action_result['recommendations'][0]
                action_executed = self.action_handler.execute_action(
                    action_id=top_action['action_id'],
                    action_name=top_action['action_name'],
                    behaviour_type='optimising',
                    confidence=top_action['confidence']
                )
                result['actions_taken'].append(action_executed)
        
        else:  # Normal behaviour
            self.logger.info("Normal behaviour detected - no action required")
            result['message'] = 'Normal behaviour - no optimization needed'
        
        return result
    
    def get_system_statistics(self) -> Dict:
        """
        Get overall system statistics.
        
        Returns:
            Dictionary with system statistics
        """
        action_stats = self.action_handler.get_action_statistics()
        
        stats = {
            'is_trained': self.is_trained,
            'sequence_length': self.sequence_length,
            'n_features': self.n_features,
            'n_actions': self.n_actions,
            'action_statistics': action_stats
        }
        
        return stats
    
    def save_models(self, base_path: str = 'models'):
        """
        Save all trained models.
        
        Args:
            base_path: Base directory for saving models
        """
        import os
        os.makedirs(base_path, exist_ok=True)
        
        self.detector.save_model(f'{base_path}/bilstm_detector.h5')
        self.automation_optimizer.save_model(f'{base_path}/lstm_automation.h5')
        self.optimisation_optimizer.save_model(f'{base_path}/lstm_optimisation.h5')
        
        self.logger.info(f"Models saved to {base_path}/")
    
    def load_models(self, base_path: str = 'models'):
        """
        Load trained models from disk.
        
        Args:
            base_path: Base directory containing saved models
        """
        self.detector.load_model(f'{base_path}/bilstm_detector.h5')
        self.automation_optimizer.load_model(f'{base_path}/lstm_automation.h5')
        self.optimisation_optimizer.load_model(f'{base_path}/lstm_optimisation.h5')
        
        self.is_trained = True
        self.logger.info(f"Models loaded from {base_path}/")
