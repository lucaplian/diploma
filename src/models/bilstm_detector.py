"""
Bidirectional LSTM Model for Behaviour Detection
Determines whether behaviour is automated or optimising
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from typing import Tuple, Optional


class BehaviourDetector:
    """
    BI-LSTM model for detecting behaviour patterns in smart home environments.
    Classifies behaviour as:
    - 0: Normal behaviour
    - 1: Automated behaviour
    - 2: Optimising behaviour
    """
    
    def __init__(
        self,
        sequence_length: int = 50,
        n_features: int = 10,
        hidden_units: int = 128,
        dropout_rate: float = 0.3,
        num_classes: int = 3
    ):
        """
        Initialize the BI-LSTM Behaviour Detector.
        
        Args:
            sequence_length: Length of input sequences
            n_features: Number of sensor features
            hidden_units: Number of LSTM hidden units
            dropout_rate: Dropout rate for regularization
            num_classes: Number of behaviour classes
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.model = None
        self.history = None
        
    def build_model(self) -> Model:
        """
        Build the BI-LSTM architecture for behaviour detection.
        
        Returns:
            Compiled Keras model
        """
        inputs = layers.Input(shape=(self.sequence_length, self.n_features))
        
        # First BI-LSTM layer
        x = layers.Bidirectional(
            layers.LSTM(
                self.hidden_units,
                return_sequences=True,
                kernel_regularizer=keras.regularizers.l2(0.01)
            )
        )(inputs)
        x = layers.Dropout(self.dropout_rate)(x)
        
        # Second BI-LSTM layer
        x = layers.Bidirectional(
            layers.LSTM(
                self.hidden_units // 2,
                return_sequences=False,
                kernel_regularizer=keras.regularizers.l2(0.01)
            )
        )(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        # Dense layers for classification
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(self.dropout_rate / 2)(x)
        x = layers.Dense(32, activation='relu')(x)
        
        # Output layer with softmax for multi-class classification
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs=inputs, outputs=outputs, name='bilstm_behaviour_detector')
        
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
        Train the BI-LSTM model.
        
        Args:
            X_train: Training data (samples, sequence_length, features)
            y_train: Training labels
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
                'best_bilstm_detector.h5',
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
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict behaviour type for input sequences.
        
        Args:
            X: Input data (samples, sequence_length, features)
            
        Returns:
            Predicted class labels
        """
        if self.model is None:
            raise ValueError("Model must be built and trained before prediction")
        
        predictions = self.model.predict(X)
        return np.argmax(predictions, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get probability distributions for behaviour predictions.
        
        Args:
            X: Input data (samples, sequence_length, features)
            
        Returns:
            Probability distributions for each class
        """
        if self.model is None:
            raise ValueError("Model must be built and trained before prediction")
        
        return self.model.predict(X)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Evaluate model performance on test data.
        
        Args:
            X_test: Test data
            y_test: Test labels
            
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
    
    def get_behaviour_type(self, prediction: int) -> str:
        """
        Convert prediction to behaviour type string.
        
        Args:
            prediction: Predicted class (0, 1, or 2)
            
        Returns:
            Behaviour type string
        """
        behaviour_map = {
            0: "normal",
            1: "automated",
            2: "optimising"
        }
        return behaviour_map.get(prediction, "unknown")
