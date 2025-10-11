# Smart Home AI-Powered Analysis System

An AI-powered analysis system for smart home environments that detects behaviour patterns and automatically takes appropriate actions for automation and optimization.

## Overview

This diploma project implements an intelligent system for smart home environments using deep learning models. The system operates across three key domains:

- **AAL (Ambient Assisted Living)**: Assisting residents with daily activities
- **HAR (Human Activity Recognition)**: Recognizing and understanding human activities  
- **HA (Home Automation)**: Automating and optimizing home systems

## Architecture

The system uses two specialized neural network models:

### 1. BI-LSTM Behaviour Detector
- **Purpose**: Detects and classifies behaviour patterns
- **Model**: Bidirectional Long Short-Term Memory (BI-LSTM)
- **Classification**: 
  - Normal behaviour (0)
  - Automated behaviour (1)
  - Optimising behaviour (2)

### 2. LSTM Automation/Optimization
- **Purpose**: Determines optimal actions based on detected behaviour
- **Model**: Long Short-Term Memory (LSTM)
- **Modes**:
  - Automation mode: For automated behaviour patterns
  - Optimisation mode: For optimising behaviour patterns

## Features

- **Real-time Behaviour Detection**: Continuously monitors sensor data and detects behaviour patterns
- **Intelligent Action Execution**: Automatically executes appropriate actions based on detected behaviour
- **Multi-domain Support**: Supports AAL, HAR, and HA applications
- **Comprehensive Logging**: Tracks all actions and decisions for analysis
- **Flexible Configuration**: Easily configurable through YAML files
- **Synthetic Data Generation**: Built-in data generator for testing and development

## Project Structure

```
diploma/
├── src/
│   ├── models/
│   │   ├── bilstm_detector.py      # BI-LSTM behaviour detection model
│   │   └── lstm_optimizer.py       # LSTM automation/optimization model
│   ├── utils/
│   │   ├── data_preprocessor.py    # Data preprocessing utilities
│   │   └── action_handler.py       # Action execution handler
│   ├── data/
│   │   └── sensor_data_loader.py   # Sensor data loading utilities
│   └── smart_home_system.py        # Main system orchestrator
├── config/
│   └── system_config.yaml          # System configuration
├── examples/
│   └── basic_usage.py              # Basic usage example
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/lucaplian/diploma.git
cd diploma
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Example

```python
from smart_home_system import SmartHomeSystem
from data.sensor_data_loader import SensorDataLoader

# Initialize the system
system = SmartHomeSystem(
    sequence_length=50,
    n_features=10,
    n_actions=5
)

# Load sensor data
data_loader = SensorDataLoader()
sensor_data, labels = data_loader.generate_synthetic_data(
    num_samples=5000,
    num_features=10
)

# Train the system
action_labels = np.random.randint(0, 5, size=len(labels))
system.train_system(
    sensor_data=sensor_data,
    behaviour_labels=labels,
    action_labels=action_labels,
    epochs=50
)

# Analyze real-time data
test_data, _ = data_loader.generate_synthetic_data(num_samples=100)
result = system.analyze_and_act(test_data, execute_actions=True)

print(f"Detected: {result['behaviour']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### Run the Demo

```bash
python examples/basic_usage.py
```

## Models

### BI-LSTM Behaviour Detector

The detector uses a bidirectional LSTM architecture to analyze temporal patterns in sensor data:

- **Input**: Sequences of sensor readings (sequence_length × n_features)
- **Architecture**: 
  - Bidirectional LSTM layers (128 units)
  - Dropout for regularization (0.3)
  - Dense layers with ReLU activation
  - Softmax output for classification
- **Output**: Behaviour classification (normal/automated/optimising)

### LSTM Automation Optimizer

The optimizer predicts optimal actions based on detected behaviour:

- **Input**: Sequences of sensor readings
- **Architecture**:
  - LSTM layers (128 units)
  - Dropout for regularization (0.3)
  - Dense layers with ReLU activation
  - Softmax output for action selection
- **Output**: Recommended actions with confidence scores

## Supported Actions

Default actions include:
1. Adjust lighting based on occupancy
2. Optimize HVAC temperature settings
3. Manage appliance power consumption
4. Activate security protocols
5. Schedule device operations

Custom actions can be defined in the configuration file or during initialization.

## Configuration

Edit `config/system_config.yaml` to customize:

- Model parameters (sequence length, features, hidden units)
- Training parameters (epochs, batch size, learning rate)
- Data preprocessing options
- Action definitions
- Sensor types
- Behaviour detection thresholds

## Sensor Types

The system supports various sensor types common in smart homes:

- Motion detection sensors
- Door/window sensors
- Temperature sensors
- Humidity sensors
- Light level sensors
- Energy consumption sensors
- Presence detection sensors
- Pressure mat sensors
- Appliance usage sensors
- Location tracking sensors

## Development

### Adding Custom Actions

```python
custom_actions = {
    0: "Custom action 1",
    1: "Custom action 2",
    # ... more actions
}

system = SmartHomeSystem(action_map=custom_actions)
```

### Loading Custom Data

```python
data_loader = SensorDataLoader()

# From CSV
sensor_data, labels = data_loader.load_from_csv(
    'path/to/data.csv',
    timestamp_col='timestamp',
    label_col='behaviour'
)

# From NumPy
sensor_data, labels = data_loader.load_from_numpy(
    'path/to/data.npy',
    'path/to/labels.npy'
)
```

### Saving and Loading Models

```python
# Save trained models
system.save_models(base_path='my_models')

# Load trained models
system.load_models(base_path='my_models')
```

## Performance Metrics

The system tracks:
- Detection accuracy
- Precision and recall
- Action execution statistics
- Confidence scores
- Behaviour distribution

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is part of a diploma thesis.

## Contact

For questions or feedback, please open an issue on GitHub.

## Acknowledgments

This system is designed for research and educational purposes in the domains of Ambient Assisted Living, Human Activity Recognition, and Smart Home Automation.
