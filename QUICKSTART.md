# Quick Start Guide

Get up and running with the Smart Home AI System in 5 minutes!

## Installation

```bash
# Clone the repository
git clone https://github.com/lucaplian/diploma.git
cd diploma

# Install dependencies
pip install -r requirements.txt

# Verify installation
python test_installation.py
```

## Basic Usage

### 1. Run the Demo

```bash
python examples/basic_usage.py
```

This will:
- Initialize the system
- Generate synthetic sensor data
- Train the models (BI-LSTM detector and LSTM optimizers)
- Perform real-time analysis
- Execute automated actions

### 2. Quick Code Example

```python
from smart_home_system import SmartHomeSystem
from data.sensor_data_loader import SensorDataLoader
import numpy as np

# Initialize
system = SmartHomeSystem()
loader = SensorDataLoader()

# Generate data
sensor_data, labels = loader.generate_synthetic_data(num_samples=5000)
actions = np.random.randint(0, 5, size=len(labels))

# Train
system.train_system(
    sensor_data=sensor_data,
    behaviour_labels=labels,
    action_labels=actions,
    epochs=20
)

# Analyze
test_data, _ = loader.generate_synthetic_data(num_samples=100)
result = system.analyze_and_act(test_data)

print(f"Detected: {result['behaviour']}")
print(f"Confidence: {result['confidence']:.2%}")
```

## What It Does

### 1. Behaviour Detection (BI-LSTM)
Analyzes sensor data and classifies behaviour as:
- **Normal**: Regular activity, no action needed
- **Automated**: Repetitive patterns, can be automated
- **Optimising**: Efficiency-seeking patterns, can be optimized

### 2. Action Recommendation (LSTM)
Based on detected behaviour, recommends actions like:
- Adjust lighting based on occupancy
- Optimize HVAC temperature
- Manage appliance power
- Activate security protocols
- Schedule device operations

### 3. Action Execution
Automatically executes the recommended actions and logs all activities.

## Example Output

```
======================================================================
Smart Home AI-Powered Analysis System
Domains: AAL, HAR, HA
======================================================================

Analysis Results:
   - Detected Behaviour: AUTOMATED
   - Confidence: 87.5%

   Behaviour Probabilities:
     • Normal: 8.2%
     • Automated: 87.5%
     • Optimising: 4.3%

   Recommended Actions:
     1. Optimize HVAC temperature settings
        Confidence: 92.1%

   Actions Executed:
     ✓ Optimize HVAC temperature settings
       Type: automated
       Status: executed
```

## Using Your Own Data

### CSV Format

```python
loader = SensorDataLoader()
data, labels = loader.load_from_csv(
    'your_data.csv',
    timestamp_col='timestamp',
    label_col='behaviour'
)
```

Expected CSV format:
```csv
timestamp,motion,door,temperature,humidity,light,behaviour
1633046400,1,0,22.5,45,300,0
1633046460,1,0,22.6,45,305,1
...
```

### NumPy Format

```python
data, labels = loader.load_from_numpy(
    'sensor_data.npy',
    'labels.npy'
)
```

## Configuration

Edit `config/system_config.yaml` to customize:

```yaml
model:
  sequence_length: 50    # How many timesteps to look at
  n_features: 10         # Number of sensor types
  n_actions: 5           # Number of possible actions

training:
  epochs: 50            # Training iterations
  batch_size: 32        # Samples per batch
```

## Advanced Usage

See `examples/advanced_usage.py` for:
- Custom configuration loading
- Detailed performance evaluation
- Continuous monitoring simulation
- Statistical analysis

```bash
python examples/advanced_usage.py
```

## Supported Domains

### AAL (Ambient Assisted Living)
- Monitor activities of daily living
- Detect anomalies and emergencies
- Provide adaptive assistance

### HAR (Human Activity Recognition)
- Recognize specific activities
- Learn routine patterns
- Provide context-aware responses

### HA (Home Automation)
- Automate repetitive tasks
- Optimize energy consumption
- Learn and adapt to preferences

## Next Steps

1. **Read the Docs**: Check [README.md](README.md) for detailed documentation
2. **Understand Architecture**: See [ARCHITECTURE.md](ARCHITECTURE.md) for system design
3. **Setup Details**: Review [SETUP.md](SETUP.md) for installation help
4. **Customize**: Modify `config/system_config.yaml` for your needs
5. **Integrate**: Connect to your actual smart home sensors

## Common Tasks

### Save Trained Models
```python
system.save_models(base_path='my_models')
```

### Load Trained Models
```python
system.load_models(base_path='my_models')
```

### Get Statistics
```python
stats = system.get_system_statistics()
print(f"Total actions: {stats['action_statistics']['total_actions']}")
```

### View Action History
```python
history = system.action_handler.get_action_history(limit=10)
for action in history:
    print(f"{action['timestamp']}: {action['action_name']}")
```

## Troubleshooting

### Import Errors
```bash
pip install -r requirements.txt --force-reinstall
```

### Memory Issues
Reduce batch size or sample count in configuration.

### Slow Training
This is normal on CPU. Consider:
- Reducing epochs
- Smaller batch sizes
- Fewer samples for testing

## Need Help?

1. Check [SETUP.md](SETUP.md) troubleshooting section
2. Review [ARCHITECTURE.md](ARCHITECTURE.md) for system details
3. Open an issue on GitHub with:
   - Your Python version
   - Error message
   - Steps to reproduce

## Performance Tips

- **GPU**: Use GPU for faster training (requires CUDA)
- **Batch Size**: Larger batches = faster but more memory
- **Sequence Length**: Longer sequences capture more context but slower
- **Early Stopping**: Let the model stop when it's done learning

## What's Next?

After getting familiar with the basics:

1. Train on your own smart home data
2. Customize actions for your devices
3. Adjust thresholds for your use case
4. Integrate with actual home automation systems
5. Add new sensor types as needed

Happy automating! 🏠🤖
