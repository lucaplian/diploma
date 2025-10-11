"""
LSTM Model for Automation and Optimization
Handles automated behaviour patterns and optimisation strategies
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from typing import Tuple, Optional, Dict


class AutomationOptimizer:
    """
    LSTM model for automation and optimization in smart home environments.
    Predicts optimal actions based on detected automated or optimising behaviour.
    """
    
    def __init__(
        self,
        sequence_length: int = 50,
        n_features: int = 10,
        n_actions: int = 5,
        hidden_units: int = 128,
        dropout_rate: float = 0.3,
        mode: str = 'automation'
    ):
        """
        Initialize the LSTM Automation Optimizer.
        
        Args:
            sequence_length: Length of input sequences
            n_features: Number of sensor features
            n_actions: Number of possible actions
            hidden_units: Number of LSTM hidden units
            dropout_rate: Dropout rate for regularization
            mode: 'automation' or 'optimisation'
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.n_actions = n_actions
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.mode = mode
        self.model = None
        self.history = None
        
    def build_model(self) -> Model:
        """
        Build the LSTM architecture for automation/optimization.
        
        Returns:
            Compiled Keras model
        """
        inputs = layers.Input(shape=(self.sequence_length, self.n_features))
        
        # First LSTM layer with return sequences
        x = layers.LSTM(
            self.hidden_units,
            return_sequences=True,
            kernel_regularizer=keras.regularizers.l2(0.01)
        )(inputs)
        x = layers.Dropout(self.dropout_rate)(x)
        
        # Second LSTM layer
        x = layers.LSTM(
            self.hidden_units // 2,
            return_sequences=False,
            kernel_regularizer=keras.regularizers.l2(0.01)
        )(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        # Dense layers for action prediction
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(self.dropout_rate / 2)(x)
        x = layers.Dense(32, activation='relu')(x)
        
        # Output layer - softmax for action selection
        outputs = layers.Dense(self.n_actions, activation='softmax')(x)
        
        model = Model(
            inputs=inputs,
            outputs=outputs,
            name=f'lstm_{self.mode}_optimizer'
        )
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 50,
        batch_size: int = 32,
        patience: int = 10
    ) -> dict:
        """
        Train the LSTM model.
        
        Args:
            X_train: Training data (samples, sequence_length, features)
            y_train: Training action labels
            X_val: Validation data
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size for training
            patience: Early stopping patience
            
        Returns:
            Training history
        """
        if self.model is None:
            self.build_model()
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                f'best_lstm_{self.mode}.h5',
                monitor='val_loss' if X_val is not None else 'loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None
        
        self.history = self.model.fit(
            X_train,
            y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history.history
    
    def predict_action(self, X: np.ndarray) -> np.ndarray:
        """
        Predict optimal actions for input sequences.
        
        Args:
            X: Input data (samples, sequence_length, features)
            
        Returns:
            Predicted action indices
        """
        if self.model is None:
            raise ValueError("Model must be built and trained before prediction")
        
        predictions = self.model.predict(X)
        return np.argmax(predictions, axis=1)
    
    def predict_action_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get probability distributions for action predictions.
        
        Args:
            X: Input data (samples, sequence_length, features)
            
        Returns:
            Probability distributions for each action
        """
        if self.model is None:
            raise ValueError("Model must be built and trained before prediction")
        
        return self.model.predict(X)
    
    def get_top_k_actions(self, X: np.ndarray, k: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get top K recommended actions with their probabilities.
        
        Args:
            X: Input data (samples, sequence_length, features)
            k: Number of top actions to return
            
        Returns:
            Tuple of (action indices, probabilities)
        """
        if self.model is None:
            raise ValueError("Model must be built and trained before prediction")
        
        predictions = self.model.predict(X)
        top_k_indices = np.argsort(predictions, axis=1)[:, -k:][:, ::-1]
        top_k_probs = np.take_along_axis(predictions, top_k_indices, axis=1)
        
        return top_k_indices, top_k_probs
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Evaluate model performance on test data.
        
        Args:
            X_test: Test data
            y_test: Test action labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model must be built and trained before evaluation")
        
        results = self.model.evaluate(X_test, y_test, verbose=1)
        metrics = {
            'loss': results[0],
            'accuracy': results[1]
        }
        
        return metrics
    
    def save_model(self, filepath: str):
        """Save the trained model to disk."""
        if self.model is None:
            raise ValueError("Model must be built before saving")
        self.model.save(filepath)
    
    def load_model(self, filepath: str):
        """Load a trained model from disk."""
        self.model = keras.models.load_model(filepath)
    
    def optimize_for_behaviour(
        self,
        X: np.ndarray,
        behaviour_type: str,
        action_map: Dict[int, str]
    ) -> Dict:
        """
        Generate optimized actions based on behaviour type.
        
        Args:
            X: Input sensor data
            behaviour_type: Type of detected behaviour ('automated' or 'optimising')
            action_map: Mapping from action indices to action descriptions
            
        Returns:
            Dictionary with recommended actions and their details
        """
        if behaviour_type not in ['automated', 'optimising']:
            raise ValueError("behaviour_type must be 'automated' or 'optimising'")
        
        # Get top 3 recommended actions
        top_actions, top_probs = self.get_top_k_actions(X, k=3)
        
        recommendations = []
        for i in range(len(top_actions)):
            action_idx = top_actions[i][0]
            prob = top_probs[i][0]
            action_name = action_map.get(action_idx, f"Action_{action_idx}")
            
            recommendations.append({
                'action_id': int(action_idx),
                'action_name': action_name,
                'confidence': float(prob),
                'behaviour_type': behaviour_type
            })
        
        return {
            'behaviour_type': behaviour_type,
            'mode': self.mode,
            'recommendations': recommendations
        }
