"""
Installation Test Script
Verifies that all components are properly installed and working
"""

import sys
import os

def test_imports():
    """Test that all required packages can be imported."""
    print("Testing imports...")
    
    try:
        import numpy as np
        print("  ✓ NumPy imported successfully")
    except ImportError as e:
        print(f"  ✗ NumPy import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print("  ✓ Pandas imported successfully")
    except ImportError as e:
        print(f"  ✗ Pandas import failed: {e}")
        return False
    
    try:
        import tensorflow as tf
        print(f"  ✓ TensorFlow {tf.__version__} imported successfully")
    except ImportError as e:
        print(f"  ✗ TensorFlow import failed: {e}")
        return False
    
    try:
        from sklearn.preprocessing import StandardScaler
        print("  ✓ Scikit-learn imported successfully")
    except ImportError as e:
        print(f"  ✗ Scikit-learn import failed: {e}")
        return False
    
    try:
        import yaml
        print("  ✓ PyYAML imported successfully")
    except ImportError as e:
        print(f"  ✗ PyYAML import failed: {e}")
        return False
    
    return True


def test_system_components():
    """Test that system components can be initialized."""
    print("\nTesting system components...")
    
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
    
    try:
        from models.bilstm_detector import BehaviourDetector
        detector = BehaviourDetector()
        print("  ✓ BehaviourDetector initialized")
    except Exception as e:
        print(f"  ✗ BehaviourDetector failed: {e}")
        return False
    
    try:
        from models.lstm_optimizer import AutomationOptimizer
        optimizer = AutomationOptimizer()
        print("  ✓ AutomationOptimizer initialized")
    except Exception as e:
        print(f"  ✗ AutomationOptimizer failed: {e}")
        return False
    
    try:
        from utils.data_preprocessor import DataPreprocessor
        preprocessor = DataPreprocessor()
        print("  ✓ DataPreprocessor initialized")
    except Exception as e:
        print(f"  ✗ DataPreprocessor failed: {e}")
        return False
    
    try:
        from utils.action_handler import ActionHandler
        handler = ActionHandler(log_actions=False)
        print("  ✓ ActionHandler initialized")
    except Exception as e:
        print(f"  ✗ ActionHandler failed: {e}")
        return False
    
    try:
        from data.sensor_data_loader import SensorDataLoader
        loader = SensorDataLoader()
        print("  ✓ SensorDataLoader initialized")
    except Exception as e:
        print(f"  ✗ SensorDataLoader failed: {e}")
        return False
    
    try:
        from smart_home_system import SmartHomeSystem
        system = SmartHomeSystem()
        print("  ✓ SmartHomeSystem initialized")
    except Exception as e:
        print(f"  ✗ SmartHomeSystem failed: {e}")
        return False
    
    return True


def test_data_generation():
    """Test synthetic data generation."""
    print("\nTesting data generation...")
    
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
    
    try:
        from data.sensor_data_loader import SensorDataLoader
        import numpy as np
        
        loader = SensorDataLoader()
        data, labels = loader.generate_synthetic_data(num_samples=100, num_features=5)
        
        assert data.shape == (100, 5), "Data shape mismatch"
        assert labels.shape == (100,), "Labels shape mismatch"
        assert np.all((labels >= 0) & (labels <= 2)), "Invalid label values"
        
        print(f"  ✓ Generated data shape: {data.shape}")
        print(f"  ✓ Generated labels shape: {labels.shape}")
        print(f"  ✓ Data generation working correctly")
        
        return True
    except Exception as e:
        print(f"  ✗ Data generation failed: {e}")
        return False


def test_model_building():
    """Test that models can be built."""
    print("\nTesting model building...")
    
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
    
    try:
        from models.bilstm_detector import BehaviourDetector
        
        detector = BehaviourDetector(sequence_length=10, n_features=5)
        model = detector.build_model()
        
        print(f"  ✓ BI-LSTM model built successfully")
        print(f"  ✓ Model has {model.count_params():,} parameters")
        
        return True
    except Exception as e:
        print(f"  ✗ Model building failed: {e}")
        return False


def main():
    print("=" * 70)
    print("SMART HOME AI SYSTEM - INSTALLATION TEST")
    print("=" * 70)
    print()
    
    all_passed = True
    
    # Test imports
    if not test_imports():
        all_passed = False
        print("\n⚠ Some imports failed. Please check your installation.")
    
    # Test system components
    if not test_system_components():
        all_passed = False
        print("\n⚠ Some system components failed to initialize.")
    
    # Test data generation
    if not test_data_generation():
        all_passed = False
        print("\n⚠ Data generation test failed.")
    
    # Test model building
    if not test_model_building():
        all_passed = False
        print("\n⚠ Model building test failed.")
    
    print()
    print("=" * 70)
    
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("Installation verified successfully!")
        print("\nYou can now run the examples:")
        print("  python examples/basic_usage.py")
        print("  python examples/advanced_usage.py")
    else:
        print("✗ SOME TESTS FAILED")
        print("Please check the error messages above.")
        print("\nTroubleshooting:")
        print("1. Make sure all dependencies are installed: pip install -r requirements.txt")
        print("2. Check that you're using Python 3.8 or higher")
        print("3. See SETUP.md for more detailed troubleshooting steps")
    
    print("=" * 70)
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
