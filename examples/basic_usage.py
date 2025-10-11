"""
Basic Usage Example of Smart Home AI System
Demonstrates behaviour detection and action execution
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from smart_home_system import SmartHomeSystem
from data.sensor_data_loader import SensorDataLoader


def main():
    print("=" * 70)
    print("Smart Home AI-Powered Analysis System")
    print("Domains: AAL (Ambient Assisted Living), HAR (Human Activity Recognition), HA (Home Automation)")
    print("=" * 70)
    print()
    
    # Initialize the system
    print("1. Initializing Smart Home System...")
    system = SmartHomeSystem(
        sequence_length=50,
        n_features=10,
        n_actions=5
    )
    print("   ✓ System initialized\n")
    
    # Load or generate data
    print("2. Loading sensor data...")
    data_loader = SensorDataLoader()
    
    # Generate synthetic data for demonstration
    sensor_data, behaviour_labels = data_loader.generate_synthetic_data(
        num_samples=5000,
        num_features=10,
        behaviour_distribution=(0.5, 0.3, 0.2)  # 50% normal, 30% automated, 20% optimising
    )
    
    # Generate action labels (random for demo)
    action_labels = np.random.randint(0, 5, size=len(behaviour_labels))
    
    print(f"   ✓ Generated {len(sensor_data)} samples")
    print(f"   ✓ Features: {sensor_data.shape[1]}")
    
    # Display data info
    data_info = data_loader.get_sensor_info()
    print(f"   ✓ Behaviour distribution: {data_info['label_distribution']}")
    print()
    
    # Train the system
    print("3. Training the system...")
    print("   - Training BI-LSTM Behaviour Detector")
    print("   - Training LSTM Automation Optimizer")
    print("   - Training LSTM Optimisation Optimizer")
    print()
    
    system.train_system(
        sensor_data=sensor_data,
        behaviour_labels=behaviour_labels,
        action_labels=action_labels,
        epochs=10,  # Reduced for demo
        batch_size=32
    )
    print("   ✓ Training completed\n")
    
    # Test real-time analysis
    print("4. Testing real-time analysis...")
    print("-" * 70)
    
    # Generate some test data
    test_data, _ = data_loader.generate_synthetic_data(
        num_samples=100,
        num_features=10,
        behaviour_distribution=(0.3, 0.4, 0.3)
    )
    
    # Analyze the data
    result = system.analyze_and_act(test_data, execute_actions=True)
    
    print(f"\n   Analysis Results:")
    print(f"   - Detected Behaviour: {result['behaviour'].upper()}")
    print(f"   - Confidence: {result['confidence']:.2%}")
    print(f"\n   Behaviour Probabilities:")
    for behaviour, prob in result['all_probabilities'].items():
        print(f"     • {behaviour.capitalize()}: {prob:.2%}")
    
    if 'optimization_result' in result:
        print(f"\n   Recommended Actions:")
        for i, rec in enumerate(result['optimization_result']['recommendations'][:3], 1):
            print(f"     {i}. {rec['action_name']}")
            print(f"        Confidence: {rec['confidence']:.2%}")
    
    if result['actions_taken']:
        print(f"\n   Actions Executed:")
        for action in result['actions_taken']:
            print(f"     ✓ {action['action_name']}")
            print(f"       Type: {action['behaviour_type']}")
            print(f"       Status: {action['status']}")
    
    print("\n" + "-" * 70)
    
    # Display system statistics
    print("\n5. System Statistics:")
    stats = system.get_system_statistics()
    action_stats = stats['action_statistics']
    print(f"   - Total Actions: {action_stats['total_actions']}")
    print(f"   - Automated Actions: {action_stats['automated_actions']}")
    print(f"   - Optimising Actions: {action_stats['optimising_actions']}")
    if action_stats['total_actions'] > 0:
        print(f"   - Average Confidence: {action_stats['average_confidence']:.2%}")
    print()
    
    # Save models
    print("6. Saving trained models...")
    system.save_models(base_path='saved_models')
    print("   ✓ Models saved to 'saved_models/' directory\n")
    
    print("=" * 70)
    print("Demo completed successfully!")
    print("=" * 70)


if __name__ == '__main__':
    main()
