# Setup Guide

## Quick Start

### 1. Prerequisites

Ensure you have the following installed:
- Python 3.8 or higher
- pip (Python package installer)
- Git

Check your Python version:
```bash
python --version
```

### 2. Clone the Repository

```bash
git clone https://github.com/lucaplian/diploma.git
cd diploma
```

### 3. Create Virtual Environment (Recommended)

#### On Linux/Mac:
```bash
python -m venv venv
source venv/bin/activate
```

#### On Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- TensorFlow (Deep Learning framework)
- NumPy (Numerical computing)
- Pandas (Data manipulation)
- Scikit-learn (Machine learning utilities)
- Matplotlib (Visualization)
- Seaborn (Statistical visualization)
- PyYAML (Configuration file parsing)

### 5. Verify Installation

Run the test script to verify everything is working:

```bash
python test_installation.py
```

If successful, you should see:
```
✓ All imports successful
✓ TensorFlow available
✓ System components initialized correctly
Installation verified successfully!
```

### 6. Run Examples

#### Basic Usage:
```bash
python examples/basic_usage.py
```

#### Advanced Usage:
```bash
python examples/advanced_usage.py
```

## Directory Structure

After setup, your directory should look like:

```
diploma/
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── bilstm_detector.py
│   │   └── lstm_optimizer.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── data_preprocessor.py
│   │   └── action_handler.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── sensor_data_loader.py
│   ├── __init__.py
│   └── smart_home_system.py
├── config/
│   └── system_config.yaml
├── examples/
│   ├── basic_usage.py
│   └── advanced_usage.py
├── requirements.txt
├── README.md
├── SETUP.md
└── .gitignore
```

## Troubleshooting

### Issue: TensorFlow Installation Fails

If TensorFlow installation fails, try:

1. Update pip:
```bash
pip install --upgrade pip
```

2. Install TensorFlow specifically:
```bash
pip install tensorflow==2.12.0
```

3. For CPU-only systems:
```bash
pip install tensorflow-cpu==2.12.0
```

### Issue: Import Errors

If you get import errors when running examples:

1. Make sure you're in the project root directory
2. Check that virtual environment is activated
3. Reinstall dependencies:
```bash
pip install -r requirements.txt --force-reinstall
```

### Issue: CUDA/GPU Errors

If you don't have a GPU or CUDA installed, TensorFlow will automatically use CPU. This is normal and the system will work (just slower).

To explicitly use CPU:
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
```

### Issue: Memory Errors

If you encounter memory errors during training:

1. Reduce batch size in config:
```yaml
training:
  batch_size: 16  # or even smaller
```

2. Reduce number of samples:
```python
data_loader.generate_synthetic_data(num_samples=2000)  # instead of 5000
```

## Configuration

### Customizing Model Parameters

Edit `config/system_config.yaml`:

```yaml
model:
  sequence_length: 50      # Length of input sequences
  n_features: 10           # Number of sensor features
  n_actions: 5             # Number of possible actions
```

### Customizing Training Parameters

```yaml
training:
  epochs: 50               # Number of training epochs
  batch_size: 32           # Batch size
  patience: 10             # Early stopping patience
```

### Adding Custom Actions

```yaml
actions:
  0: "Your custom action 1"
  1: "Your custom action 2"
  # ... more actions
```

## Next Steps

1. Read the [README.md](README.md) for detailed documentation
2. Explore the example scripts in `examples/`
3. Customize the configuration in `config/system_config.yaml`
4. Start building your own smart home applications!

## Getting Help

If you encounter issues:
1. Check the Troubleshooting section above
2. Review the example scripts
3. Open an issue on GitHub with:
   - Your Python version
   - Error message
   - Steps to reproduce

## Development Mode

To modify the system:

1. Install in development mode:
```bash
pip install -e .
```

2. Make your changes to the source code

3. Test your changes:
```bash
python test_installation.py
python examples/basic_usage.py
```

## Updating

To update to the latest version:

```bash
git pull origin main
pip install -r requirements.txt --upgrade
```
