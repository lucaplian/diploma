"""
Advanced Usage Example of Smart Home AI System
Demonstrates custom configuration, model evaluation, and detailed analysis
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import yaml
from smart_home_system import SmartHomeSystem
from data.sensor_data_loader import SensorDataLoader
from utils.data_preprocessor import DataPreprocessor


def load_config(config_path='../config/system_config.yaml'):
    """Load system configuration from YAML file."""
    config_file = os.path.join(os.path.dirname(__file__), config_path)
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config


def simulate_continuous_monitoring(system, data_loader, num_iterations=5):
    """Simulate continuous monitoring of smart home environment."""
    print("\n" + "=" * 70)
    print("CONTINUOUS MONITORING SIMULATION")
    print("=" * 70)
    
    for i in range(num_iterations):
        print(f"\nIteration {i+1}/{num_iterations}")
        print("-" * 70)
        
        # Generate new sensor readings
        sensor_data, actual_behaviour = data_loader.generate_synthetic_data(
            num_samples=100,
            num_features=10,
            behaviour_distribution=(0.3, 0.4, 0.3)
        )
        
        # Analyze the data
        result = system.analyze_and_act(sensor_data, execute_actions=True)
        
        print(f"Detected Behaviour: {result['behaviour'].upper()}")
        print(f"Confidence: {result['confidence']:.2%}")
        
        if result.get('actions_taken'):
            print(f"Action Executed: {result['actions_taken'][0]['action_name']}")
        else:
            print("No action required (normal behaviour)")
        
        print()


def detailed_analysis(system, test_data):
    """Perform detailed analysis with statistics."""
    print("\n" + "=" * 70)
    print("DETAILED ANALYSIS")
    print("=" * 70)
    
    result = system.analyze_and_act(test_data, execute_actions=False)
    
    print("\n1. Behaviour Detection Results:")
    print(f"   Primary Behaviour: {result['behaviour'].upper()}")
    print(f"   Confidence Score: {result['confidence']:.4f}")
    
    print("\n2. Full Probability Distribution:")
    for behaviour, prob in result['all_probabilities'].items():
        bar_length = int(prob * 50)
        bar = '█' * bar_length + '░' * (50 - bar_length)
        print(f"   {behaviour.capitalize():12} [{bar}] {prob:.2%}")
    
    if 'optimization_result' in result:
        print("\n3. Recommended Actions (Top 3):")
        for i, rec in enumerate(result['optimization_result']['recommendations'], 1):
            print(f"\n   Action {i}:")
            print(f"   - Name: {rec['action_name']}")
            print(f"   - ID: {rec['action_id']}")
            print(f"   - Confidence: {rec['confidence']:.2%}")
            print(f"   - Type: {rec['behaviour_type']}")
    
    print("\n" + "=" * 70)


def evaluate_system_performance(system, test_sequences, test_labels):
    """Evaluate system performance on test data."""
    print("\n" + "=" * 70)
    print("SYSTEM PERFORMANCE EVALUATION")
    print("=" * 70)
    
    # Get predictions
    predictions = system.detector.predict(test_sequences)
    probabilities = system.detector.predict_proba(test_sequences)
    
    # Calculate accuracy
    accuracy = np.mean(predictions == test_labels)
    
    # Calculate per-class metrics
    print("\n1. Overall Performance:")
    print(f"   Accuracy: {accuracy:.2%}")
    
    print("\n2. Per-Class Performance:")
    for class_id, class_name in enumerate(['Normal', 'Automated', 'Optimising']):
        class_mask = test_labels == class_id
        if class_mask.any():
            class_predictions = predictions[class_mask]
            class_accuracy = np.mean(class_predictions == class_id)
            class_count = class_mask.sum()
            print(f"\n   {class_name}:")
            print(f"   - Samples: {class_count}")
            print(f"   - Accuracy: {class_accuracy:.2%}")
            print(f"   - Avg Confidence: {probabilities[class_mask, class_id].mean():.2%}")
    
    # Confusion-like analysis
    print("\n3. Prediction Distribution:")
    unique, counts = np.unique(predictions, return_counts=True)
    total = len(predictions)
    for class_id, count in zip(unique, counts):
        class_name = ['Normal', 'Automated', 'Optimising'][class_id]
        print(f"   {class_name}: {count} ({count/total:.1%})")
    
    print("\n" + "=" * 70)


def main():
    print("=" * 70)
    print("ADVANCED SMART HOME AI SYSTEM DEMONSTRATION")
    print("=" * 70)
    
    # Load configuration
    print("\n1. Loading configuration...")
    config = load_config()
    print("   ✓ Configuration loaded")
    
    model_config = config['model']
    training_config = config['training']
    
    # Initialize system with config
    print("\n2. Initializing system with configuration...")
    action_map = {int(k): v for k, v in config['actions'].items()}
    
    system = SmartHomeSystem(
        sequence_length=model_config['sequence_length'],
        n_features=model_config['n_features'],
        n_actions=model_config['n_actions'],
        action_map=action_map
    )
    print("   ✓ System initialized")
    
    # Generate training data
    print("\n3. Generating training data...")
    data_loader = SensorDataLoader()
    sensor_data, behaviour_labels = data_loader.generate_synthetic_data(
        num_samples=8000,
        num_features=model_config['n_features'],
        behaviour_distribution=(0.4, 0.35, 0.25)
    )
    action_labels = np.random.randint(0, model_config['n_actions'], size=len(behaviour_labels))
    print(f"   ✓ Generated {len(sensor_data)} training samples")
    
    # Train system
    print("\n4. Training system...")
    system.train_system(
        sensor_data=sensor_data,
        behaviour_labels=behaviour_labels,
        action_labels=action_labels,
        epochs=training_config['epochs'],
        batch_size=training_config['batch_size']
    )
    print("   ✓ Training completed")
    
    # Prepare test data
    print("\n5. Preparing test data...")
    test_data, test_labels = data_loader.generate_synthetic_data(
        num_samples=1000,
        num_features=model_config['n_features']
    )
    
    preprocessor = DataPreprocessor(sequence_length=model_config['sequence_length'])
    scaled_test = preprocessor.fit_transform(test_data)
    test_sequences, test_seq_labels = preprocessor.create_sequences(scaled_test, test_labels)
    print(f"   ✓ Created {len(test_sequences)} test sequences")
    
    # Evaluate performance
    evaluate_system_performance(system, test_sequences, test_seq_labels)
    
    # Detailed analysis
    detailed_analysis(system, test_data[:200])
    
    # Simulate continuous monitoring
    simulate_continuous_monitoring(system, data_loader, num_iterations=5)
    
    # System statistics
    print("\n" + "=" * 70)
    print("FINAL SYSTEM STATISTICS")
    print("=" * 70)
    stats = system.get_system_statistics()
    action_stats = stats['action_statistics']
    
    print(f"\nSystem Configuration:")
    print(f"  - Sequence Length: {stats['sequence_length']}")
    print(f"  - Number of Features: {stats['n_features']}")
    print(f"  - Number of Actions: {stats['n_actions']}")
    print(f"  - Training Status: {'Trained' if stats['is_trained'] else 'Not Trained'}")
    
    print(f"\nAction Execution Statistics:")
    print(f"  - Total Actions: {action_stats['total_actions']}")
    print(f"  - Automated Actions: {action_stats['automated_actions']}")
    print(f"  - Optimising Actions: {action_stats['optimising_actions']}")
    if action_stats['total_actions'] > 0:
        print(f"  - Average Confidence: {action_stats['average_confidence']:.2%}")
        print(f"  - Automated %: {action_stats['automated_percentage']:.1f}%")
        print(f"  - Optimising %: {action_stats['optimising_percentage']:.1f}%")
    
    print("\n" + "=" * 70)
    print("ADVANCED DEMONSTRATION COMPLETED")
    print("=" * 70)


if __name__ == '__main__':
    main()
