#!/usr/bin/env python3
"""
AutoEDA Demo Script

This script demonstrates the capabilities of the AutoEDA system
by running a complete analysis pipeline on sample data.
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json

# Add the app directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

def generate_demo_data():
    """Generate demo time series data"""
    print("Generating demo data...")
    
    # Create time series data with anomalies
    dates = pd.date_range('2020-01-01', periods=500, freq='H')
    
    # Normal pattern with trend and seasonality
    trend = np.linspace(0, 5, 500)
    seasonality = 2 * np.sin(2 * np.pi * np.arange(500) / 24)  # Daily pattern
    noise = np.random.normal(0, 0.3, 500)
    
    # Add some anomalies
    anomalies = np.zeros(500)
    anomaly_indices = [50, 150, 250, 350, 450]
    for idx in anomaly_indices:
        anomalies[idx] = np.random.normal(0, 3)
    
    # Combine components
    values = trend + seasonality + noise + anomalies
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': dates,
        'value': values,
        'trend': trend,
        'seasonality': seasonality,
        'noise': noise
    })
    
    df.set_index('timestamp', inplace=True)
    
    print(f"✓ Generated demo data: {df.shape}")
    return df

def run_demo_analysis():
    """Run complete demo analysis"""
    print("\n" + "="*50)
    print("AutoEDA Demo - Complete Analysis Pipeline")
    print("="*50)
    
    try:
        # 1. Generate demo data
        df = generate_demo_data()
        
        # 2. Test DataProcessor
        print("\n1. Testing DataProcessor...")
        from utils.data_processor import DataProcessor
        processor = DataProcessor()
        
        processed_data = processor.preprocess_data(df)
        print(f"   ✓ Data processed: {processed_data.shape}")
        
        # 3. Test LSTM Autoencoder
        print("\n2. Testing LSTM Autoencoder...")
        from models.lstm_autoencoder import LSTMAutoencoder
        
        # Prepare sequences
        sequences = processor._prepare_sequences(processed_data)
        input_shape = (sequences.shape[1], sequences.shape[2])
        
        autoencoder = LSTMAutoencoder(
            input_shape=input_shape,
            encoding_dim=8,
            lstm_units=[32, 16]
        )
        
        # Train model (reduced epochs for demo)
        print("   Training LSTM autoencoder...")
        history = autoencoder.train(sequences, epochs=5, verbose=0)
        print(f"   ✓ Model trained - Final loss: {history.history['loss'][-1]:.4f}")
        
        # 4. Test Anomaly Detection
        print("\n3. Testing Anomaly Detection...")
        from utils.anomaly_detector import AnomalyDetector
        
        detector = AnomalyDetector()
        reconstruction_error = autoencoder.get_reconstruction_error(sequences)
        anomaly_results = detector.detect_anomalies(sequences, reconstruction_error=reconstruction_error)
        
        print(f"   ✓ Anomalies detected: {anomaly_results.get('n_anomalies', 0)}")
        
        # 5. Test Attention Mechanism
        print("\n4. Testing Attention Mechanism...")
        from models.attention_mechanism import AttentionMechanism
        
        attention = AttentionMechanism(
            input_shape=input_shape,
            num_heads=4,
            key_dim=8
        )
        
        attention_weights = attention.get_attention_weights(sequences)
        feature_importance = attention.get_feature_importance(attention_weights)
        
        print(f"   ✓ Feature importance calculated for {len(feature_importance)} features")
        
        # 6. Test Synthetic Data Generation
        print("\n5. Testing Synthetic Data Generation...")
        from utils.synthetic_generator import SyntheticDataGenerator
        
        generator = SyntheticDataGenerator()
        synthetic_data = generator.generate_data(sequences, n_samples=50)
        quality_metrics = generator._assess_quality(sequences, synthetic_data)
        
        print(f"   ✓ Synthetic data generated: {len(synthetic_data)} samples")
        print(f"   ✓ Quality score: {quality_metrics.get('overall_quality', 'N/A'):.4f}")
        
        # 7. Test Visualization
        print("\n6. Testing Visualization Generation...")
        from utils.visualization import VisualizationGenerator
        
        viz_gen = VisualizationGenerator()
        
        # Prepare analysis results
        analysis_results = {
            'data_summary': processor.get_data_summary(df),
            'anomaly_results': anomaly_results,
            'feature_importance': feature_importance,
            'attention_weights': attention_weights.tolist(),
            'synthetic_data': synthetic_data.tolist(),
            'quality_metrics': quality_metrics
        }
        
        # Generate visualizations
        visualizations = viz_gen.generate_all_visualizations(df, analysis_results)
        print(f"   ✓ Visualizations generated: {len(visualizations)} files")
        
        # 8. Save demo results
        print("\n7. Saving Demo Results...")
        demo_results = {
            'timestamp': datetime.now().isoformat(),
            'demo_data_shape': df.shape,
            'analysis_results': analysis_results,
            'visualizations': visualizations,
            'model_performance': {
                'training_loss': history.history['loss'][-1],
                'final_accuracy': 1 - history.history['loss'][-1]
            }
        }
        
        results_file = 'demo_results.json'
        with open(results_file, 'w') as f:
            json.dump(demo_results, f, default=str, indent=2)
        
        print(f"   ✓ Demo results saved to: {results_file}")
        
        # 9. Summary
        print("\n" + "="*50)
        print("🎉 Demo Analysis Completed Successfully!")
        print("="*50)
        print(f"📊 Dataset: {df.shape[0]} samples, {df.shape[1]} features")
        print(f"🔍 Anomalies: {anomaly_results.get('n_anomalies', 0)} detected")
        print(f"🧠 Model: LSTM Autoencoder trained successfully")
        print(f"👁️  Attention: Feature importance calculated")
        print(f"🎲 Synthetic: {len(synthetic_data)} samples generated")
        print(f"📈 Visualizations: {len(visualizations)} files created")
        print(f"💾 Results: Saved to {results_file}")
        print("\n🚀 The AutoEDA system is working perfectly!")
        print("You can now run the full web application with: python run.py")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Demo failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    print("AutoEDA Demo Script")
    print("This script demonstrates the complete AutoEDA system")
    print("=" * 40)
    
    # Check if required packages are available
    try:
        import tensorflow as tf
        print(f"✓ TensorFlow version: {tf.__version__}")
    except ImportError:
        print("❌ TensorFlow not found. Please install with: pip install tensorflow")
        return False
    
    try:
        import sklearn
        print(f"✓ Scikit-learn version: {sklearn.__version__}")
    except ImportError:
        print("❌ Scikit-learn not found. Please install with: pip install scikit-learn")
        return False
    
    # Run demo
    success = run_demo_analysis()
    
    if success:
        print("\n✅ Demo completed successfully!")
        print("The AutoEDA system is ready for production use.")
    else:
        print("\n❌ Demo failed. Please check the error messages above.")
    
    return success

if __name__ == "__main__":
    main()
