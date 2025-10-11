# System Architecture

## Overview

The Smart Home AI-Powered Analysis System is designed with a modular architecture that separates concerns and allows for easy extension and maintenance.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    Smart Home Environment                       │
│                  (Sensors: Motion, Door, Temp, etc.)            │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             │ Sensor Data Stream
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SensorDataLoader                             │
│  - Load from CSV, NumPy, or generate synthetic data             │
│  - Support for multiple sensor types (AAL, HAR, HA)             │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             │ Raw Sensor Data
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   DataPreprocessor                              │
│  - Scaling (StandardScaler/MinMaxScaler)                        │
│  - Sequence creation (sliding window)                           │
│  - Missing value handling                                       │
│  - Feature engineering                                          │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             │ Preprocessed Sequences
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                SmartHomeSystem (Orchestrator)                   │
│  ┌────────────────────────────────────────────────────────┐    │
│  │           BI-LSTM Behaviour Detector                   │    │
│  │  ┌──────────────────────────────────────────────────┐  │    │
│  │  │  Input: (batch, sequence_length, n_features)     │  │    │
│  │  │  ↓                                                │  │    │
│  │  │  Bidirectional LSTM (128 units)                  │  │    │
│  │  │  ↓                                                │  │    │
│  │  │  Dropout (0.3)                                    │  │    │
│  │  │  ↓                                                │  │    │
│  │  │  Bidirectional LSTM (64 units)                   │  │    │
│  │  │  ↓                                                │  │    │
│  │  │  Dense Layers (64, 32)                           │  │    │
│  │  │  ↓                                                │  │    │
│  │  │  Output: Behaviour Class (3 classes)             │  │    │
│  │  │  - 0: Normal                                     │  │    │
│  │  │  - 1: Automated                                  │  │    │
│  │  │  - 2: Optimising                                 │  │    │
│  │  └──────────────────────────────────────────────────┘  │    │
│  └────────────────────────────────────────────────────────┘    │
│                             │                                   │
│                             │ Detected Behaviour                │
│                             ▼                                   │
│  ┌────────────────────────────────────────────────────────┐    │
│  │     LSTM Automation/Optimisation Optimizer             │    │
│  │  ┌──────────────────────────────────────────────────┐  │    │
│  │  │  Mode: Automation (for automated behaviour)      │  │    │
│  │  │  Mode: Optimisation (for optimising behaviour)   │  │    │
│  │  │  ↓                                                │  │    │
│  │  │  Input: (batch, sequence_length, n_features)     │  │    │
│  │  │  ↓                                                │  │    │
│  │  │  LSTM (128 units)                                │  │    │
│  │  │  ↓                                                │  │    │
│  │  │  Dropout (0.3)                                    │  │    │
│  │  │  ↓                                                │  │    │
│  │  │  LSTM (64 units)                                 │  │    │
│  │  │  ↓                                                │  │    │
│  │  │  Dense Layers (64, 32)                           │  │    │
│  │  │  ↓                                                │  │    │
│  │  │  Output: Recommended Action (n_actions classes)  │  │    │
│  │  └──────────────────────────────────────────────────┘  │    │
│  └────────────────────────────────────────────────────────┘    │
│                             │                                   │
│                             │ Recommended Actions               │
│                             ▼                                   │
│  ┌────────────────────────────────────────────────────────┐    │
│  │              ActionHandler                             │    │
│  │  - Execute actions based on behaviour type             │    │
│  │  - Log all actions and decisions                       │    │
│  │  - Track statistics and history                        │    │
│  └────────────────────────────────────────────────────────┘    │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             │ Control Signals
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Smart Home Devices                             │
│  - Lighting Systems                                             │
│  - HVAC Systems                                                 │
│  - Appliances                                                   │
│  - Security Systems                                             │
└─────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. SensorDataLoader
**File**: `src/data/sensor_data_loader.py`

**Responsibilities**:
- Load sensor data from various sources (CSV, NumPy files)
- Generate synthetic data for testing and development
- Manage metadata about sensors and data

**Key Methods**:
- `load_from_csv()`: Load data from CSV files
- `load_from_numpy()`: Load data from NumPy arrays
- `generate_synthetic_data()`: Create synthetic sensor data

### 2. DataPreprocessor
**File**: `src/utils/data_preprocessor.py`

**Responsibilities**:
- Scale and normalize sensor data
- Create time-series sequences
- Handle missing values
- Extract temporal and statistical features

**Key Methods**:
- `fit_transform()`: Fit scaler and transform data
- `create_sequences()`: Create sliding window sequences
- `handle_missing_values()`: Deal with missing sensor readings
- `add_temporal_features()`: Add time-based features

### 3. BehaviourDetector (BI-LSTM)
**File**: `src/models/bilstm_detector.py`

**Responsibilities**:
- Detect behaviour patterns in sensor data
- Classify behaviour as normal, automated, or optimising
- Provide confidence scores for predictions

**Architecture**:
- Bidirectional LSTM layers for capturing temporal patterns
- Dropout for regularization
- Dense layers for classification
- Softmax output for multi-class prediction

**Key Methods**:
- `build_model()`: Construct the BI-LSTM architecture
- `train()`: Train the detector on labeled data
- `predict()`: Classify behaviour patterns
- `evaluate()`: Measure model performance

### 4. AutomationOptimizer (LSTM)
**File**: `src/models/lstm_optimizer.py`

**Responsibilities**:
- Recommend optimal actions based on detected behaviour
- Operate in automation or optimisation mode
- Provide top-K action recommendations with confidence scores

**Architecture**:
- LSTM layers for sequential pattern learning
- Dense layers for action prediction
- Softmax output for action selection

**Key Methods**:
- `build_model()`: Construct the LSTM architecture
- `train()`: Train the optimizer on action data
- `predict_action()`: Recommend actions
- `get_top_k_actions()`: Get multiple action recommendations
- `optimize_for_behaviour()`: Generate behaviour-specific recommendations

### 5. ActionHandler
**File**: `src/utils/action_handler.py`

**Responsibilities**:
- Execute recommended actions
- Log all actions and decisions
- Track statistics and history
- Handle different behaviour types appropriately

**Key Methods**:
- `execute_action()`: Execute a recommended action
- `get_action_history()`: Retrieve action history
- `get_action_statistics()`: Get execution statistics

### 6. SmartHomeSystem (Main Orchestrator)
**File**: `src/smart_home_system.py`

**Responsibilities**:
- Coordinate all components
- Manage the complete analysis pipeline
- Train models end-to-end
- Perform real-time analysis and action execution

**Key Methods**:
- `train_system()`: Train all models together
- `analyze_and_act()`: Analyze sensor data and take actions
- `get_system_statistics()`: Get overall system metrics
- `save_models()` / `load_models()`: Persist trained models

## Data Flow

1. **Input**: Raw sensor data from smart home devices
2. **Loading**: SensorDataLoader reads and manages the data
3. **Preprocessing**: DataPreprocessor scales, sequences, and cleans data
4. **Detection**: BI-LSTM Detector classifies behaviour patterns
5. **Decision**: Based on detected behaviour:
   - Normal → No action
   - Automated → LSTM Automation Optimizer recommends actions
   - Optimising → LSTM Optimisation Optimizer recommends actions
6. **Execution**: ActionHandler executes recommended actions
7. **Feedback**: System logs results and updates statistics

## Training Pipeline

```
Raw Data → Preprocessing → Feature Engineering
    ↓
Split (Train/Val/Test)
    ↓
┌───────────────────────────────────────┐
│  Train BI-LSTM Detector               │
│  - Input: Sensor sequences            │
│  - Output: Behaviour labels           │
└───────────────────────────────────────┘
    ↓
┌───────────────────────────────────────┐
│  Train LSTM Automation Optimizer      │
│  - Filter: Automated behaviour only   │
│  - Input: Sensor sequences            │
│  - Output: Action labels              │
└───────────────────────────────────────┘
    ↓
┌───────────────────────────────────────┐
│  Train LSTM Optimisation Optimizer    │
│  - Filter: Optimising behaviour only  │
│  - Input: Sensor sequences            │
│  - Output: Action labels              │
└───────────────────────────────────────┘
    ↓
Save Models
```

## Configuration

The system is highly configurable through `config/system_config.yaml`:

- **Model parameters**: Sequence length, features, hidden units
- **Training parameters**: Epochs, batch size, learning rate
- **Preprocessing**: Scaling method, missing value handling
- **Actions**: Custom action definitions
- **Sensors**: Supported sensor types
- **Thresholds**: Confidence thresholds for decisions

## Extensibility

The system is designed for easy extension:

1. **New Sensors**: Add to `SENSOR_TYPES` in `SensorDataLoader`
2. **New Actions**: Update action map in configuration
3. **Custom Models**: Inherit from base model classes
4. **New Behaviours**: Extend classification in detector
5. **Custom Preprocessing**: Add methods to `DataPreprocessor`

## Performance Considerations

- **Memory**: Sequences are created efficiently with sliding windows
- **Training**: Uses early stopping to prevent overfitting
- **Inference**: Models can process batches in real-time
- **Scalability**: Can handle multiple sensor types and features

## Domains Supported

### AAL (Ambient Assisted Living)
- Activity monitoring for elderly or disabled individuals
- Fall detection and emergency response
- Daily routine analysis and assistance

### HAR (Human Activity Recognition)
- Recognition of daily activities
- Pattern detection in human behaviour
- Context-aware assistance

### HA (Home Automation)
- Automated device control
- Energy optimization
- Schedule learning and prediction
- Security automation

## Future Enhancements

Potential areas for improvement:

1. **Attention Mechanisms**: Add attention layers for better feature importance
2. **Transfer Learning**: Pre-train on large datasets, fine-tune for specific homes
3. **Reinforcement Learning**: Learn optimal actions through trial and feedback
4. **Multi-modal Input**: Integrate vision, audio, and other sensor modalities
5. **Explainability**: Add SHAP or LIME for model interpretability
6. **Online Learning**: Continuously update models with new data
7. **Federated Learning**: Learn from multiple homes while preserving privacy
