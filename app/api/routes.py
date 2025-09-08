#!/usr/bin/env python3
"""
API Routes for AutoEDA

This module defines all the API endpoints for the AutoEDA system.
"""

from flask import request, jsonify, current_app
from app.api import bp
from app.utils.data_processor import DataProcessor
from app.utils.anomaly_detector import AnomalyDetector
from app.utils.synthetic_generator import SyntheticDataGenerator
from app.utils.visualization import VisualizationGenerator
from app.models.lstm_autoencoder import LSTMAutoencoder
from app.models.attention_mechanism import AttentionMechanism
import os
import json
import pandas as pd
from datetime import datetime
import traceback

# Initialize components
data_processor = DataProcessor()
anomaly_detector = AnomalyDetector()
synthetic_generator = SyntheticDataGenerator()
visualization_generator = VisualizationGenerator()

@bp.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and initial processing"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            # Save file
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{timestamp}_{filename}"
            filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process initial data
            try:
                df = data_processor.load_data(filepath)
                data_summary = data_processor.get_data_summary(df)
                
                return jsonify({
                    'message': 'File uploaded successfully',
                    'filename': filename,
                    'data_summary': data_summary
                })
            except Exception as e:
                return jsonify({'error': f'Error processing file: {str(e)}'}), 500
        
        return jsonify({'error': 'Invalid file type'}), 400
        
    except Exception as e:
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@bp.route('/analyze', methods=['POST'])
def analyze_data():
    """Run complete EDA analysis pipeline"""
    try:
        data = request.get_json()
        filename = data.get('filename')
        
        if not filename:
            return jsonify({'error': 'No filename provided'}), 400
        
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        
        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404
        
        # Load and preprocess data
        df = data_processor.load_data(filepath)
        processed_data = data_processor.preprocess_data(df)
        
        # Prepare sequences for LSTM
        sequences = data_processor._prepare_sequences(processed_data)
        
        # Train LSTM autoencoder
        input_shape = (sequences.shape[1], sequences.shape[2])
        autoencoder = LSTMAutoencoder(
            input_shape=input_shape,
            encoding_dim=current_app.config['LSTM_ENCODING_DIM'],
            lstm_units=current_app.config['LSTM_UNITS']
        )
        
        # Train the model
        history = autoencoder.train(sequences, epochs=current_app.config['LSTM_EPOCHS'])
        
        # Get reconstruction error
        reconstruction_error = autoencoder.get_reconstruction_error(sequences)
        
        # Detect anomalies
        anomaly_results = anomaly_detector.detect_anomalies(
            sequences, 
            reconstruction_error=reconstruction_error
        )
        
        # Generate attention weights
        attention = AttentionMechanism(
            input_shape=input_shape,
            num_heads=current_app.config['ATTENTION_NUM_HEADS'],
            key_dim=current_app.config['ATTENTION_KEY_DIM']
        )
        
        attention_weights = attention.get_attention_weights(sequences)
        feature_importance = attention.get_feature_importance(attention_weights)
        
        # Generate synthetic data
        synthetic_data = synthetic_generator.generate_data(
            sequences, 
            n_samples=min(100, len(sequences))
        )
        
        # Assess synthetic data quality
        quality_metrics = synthetic_generator._assess_quality(sequences, synthetic_data)
        
        # Generate visualizations
        analysis_results = {
            'data_summary': data_processor.get_data_summary(df),
            'data_quality': {
                'missing_values': df.isnull().sum().sum(),
                'duplicates': df.duplicated().sum(),
                'completeness': ((df.size - df.isnull().sum().sum()) / df.size * 100)
            },
            'anomaly_results': anomaly_results,
            'feature_importance': feature_importance,
            'attention_weights': attention_weights.tolist(),
            'synthetic_data': synthetic_data.tolist(),
            'quality_metrics': quality_metrics,
            'model_performance': {
                'training_loss': history.history['loss'][-1],
                'training_time': sum(history.history.get('time', [0]))
            }
        }
        
        # Generate visualizations
        visualizations = visualization_generator.generate_all_visualizations(df, analysis_results)
        analysis_results['visualizations'] = visualizations
        
        # Save results
        results_file = os.path.join(current_app.config['RESULTS_FOLDER'], f"{filename}_results.json")
        with open(results_file, 'w') as f:
            json.dump(analysis_results, f, default=str)
        
        return jsonify({
            'message': 'Analysis completed successfully',
            'filename': filename,
            'results_file': f"{filename}_results.json",
            'summary': {
                'n_anomalies': anomaly_results.get('n_anomalies', 0),
                'n_features': len(feature_importance),
                'model_accuracy': 1 - history.history['loss'][-1]
            }
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@bp.route('/results/<filename>', methods=['GET'])
def get_results(filename):
    """Retrieve saved analysis results"""
    try:
        results_file = os.path.join(current_app.config['RESULTS_FOLDER'], filename)
        
        if not os.path.exists(results_file):
            return jsonify({'error': 'Results not found'}), 404
        
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({'error': f'Error retrieving results: {str(e)}'}), 500

@bp.route('/download/<filename>', methods=['GET'])
def download_results(filename):
    """Download analysis results"""
    try:
        results_file = os.path.join(current_app.config['RESULTS_FOLDER'], filename)
        
        if not os.path.exists(results_file):
            return jsonify({'error': 'Results not found'}), 404
        
        # Return file for download
        from flask import send_file
        return send_file(results_file, as_attachment=True)
        
    except Exception as e:
        return jsonify({'error': f'Error downloading results: {str(e)}'}), 500

@bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

def allowed_file(filename):
    """Check if file type is allowed"""
    ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'json', 'parquet'}
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def secure_filename(filename):
    """Secure filename for storage"""
    import re
    # Remove or replace unsafe characters
    filename = re.sub(r'[^\w\s-]', '', filename)
    filename = re.sub(r'[-\s]+', '-', filename)
    return filename.strip('-')
