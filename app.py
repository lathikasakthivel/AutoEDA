#!/usr/bin/env python3
"""
AutoEDA - Main Application Entry Point
Deep Learning-Powered Automated Exploratory Data Analysis

not all features are working if the user uploads a dataset as csv the complete EDA must be done automatically and the user must be able to view each steps happened , the features displayed in the index.html must be completely working
"""

import os
import logging
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import json
import traceback
import numpy as np

from app.models.lstm_autoencoder import LSTMAutoencoder
from app.models.attention_mechanism import AttentionMechanism
from app.utils.data_processor import DataProcessor
from app.utils.visualization import VisualizationGenerator
from app.utils.anomaly_detector import AnomalyDetector
from app.utils.synthetic_generator import SyntheticDataGenerator
from app.config.config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
# Initialize Flask app
app = Flask(__name__, template_folder="app/templates", static_folder="app/static", static_url_path="/static")
app.config.from_object(Config)

# Enable CORS
CORS(app)

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODELS_FOLDER'], exist_ok=True)

# Initialize components
data_processor = DataProcessor()
# Ensure plots are saved under app/static/plots so the browser can load them
visualization_generator = VisualizationGenerator(output_dir=os.path.join('app', 'static', 'plots'))
anomaly_detector = AnomalyDetector()
synthetic_generator = SyntheticDataGenerator()

@app.route('/')
def index():
    """Main application page"""
    return render_template("index.html")

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and initial data processing"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process uploaded data
            data_info = data_processor.process_upload(filepath)
            
            return jsonify({
                'message': 'File uploaded successfully',
                'filename': filename,
                'data_info': data_info
            })
        else:
            return jsonify({'error': 'File type not allowed'}), 400
            
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/analyze', methods=['POST'])
def analyze_data():
    """Run comprehensive EDA analysis"""
    try:
        data = request.get_json()
        filename = data.get('filename')
        analysis_config = data.get('config', {})
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Load and preprocess data
        df = data_processor.load_data(filepath)
        processed_data = data_processor.preprocess_data(df, analysis_config)
        
        # Initialize and train LSTM autoencoder
        lstm_model = LSTMAutoencoder(
            input_shape=processed_data['input_shape'],
            config=analysis_config.get('lstm_config', {})
        )
        
        # Train model
        import time as _time
        _t0 = _time.time()
        training_history = lstm_model.train(
            processed_data['train_data'],
            epochs=analysis_config.get('epochs', 50),
            batch_size=analysis_config.get('batch_size', 32)
        )
        training_duration = float(_time.time() - _t0)
        
        # Detect anomalies
        anomalies = anomaly_detector.detect_anomalies(
            lstm_model, 
            processed_data['data'],
            threshold=analysis_config.get('anomaly_threshold', 0.1)
        )
        
        # Generate attention weights
        attention_model = AttentionMechanism(lstm_model.encoder)
        attention_weights = attention_model.get_attention_weights(processed_data['data'])
        
        # Generate synthetic data if requested
        synthetic_data = None
        if analysis_config.get('generate_synthetic', False):
            synthetic_data = synthetic_generator.generate_data(
                lstm_model,
                processed_data['data'],
                n_samples=analysis_config.get('synthetic_samples', 1000)
            )
        
        # Prepare summaries aligned with UI expectations
        try:
            total_missing = int(df.isnull().sum().sum())
            total_cells = int(df.shape[0] * df.shape[1]) if df.shape[1] > 0 else 1
            completeness = 100.0 if total_cells == 0 else max(0.0, 100.0 - (total_missing / max(1, total_cells) * 100.0))
        except Exception:
            total_missing, completeness = 0, 100.0

        anomaly_results_summary = {
            'n_anomalies': anomalies.get('n_anomalies', 0),
            'anomaly_rate': anomalies.get('anomaly_ratio', 0.0),
            'avg_reconstruction_error': float(np.mean(anomalies.get('reconstruction_error', [0.0]))) if 'reconstruction_error' in anomalies else 0.0,
            'training_time': training_duration
        }

        # Bundle analysis_results for visualization generator
        analysis_results_bundle = {
            'anomaly_results': anomalies,
            'attention_weights': attention_weights.tolist() if attention_weights is not None else None,
        }
        if synthetic_data is not None:
            analysis_results_bundle['synthetic_data'] = synthetic_data.tolist()

        # Generate visualizations from original dataframe and analysis results
        visualizations = visualization_generator.generate_all_visualizations(
            df,
            analysis_results_bundle
        )
        # Convert filesystem paths to web paths
        visualizations = {k: v.replace('app/static', '/static') if isinstance(v, str) else v for k, v in visualizations.items()}
        
        # Save results
        summary_raw = data_processor.get_data_summary(df)
        data_summary_ui = {
            'rows': int(df.shape[0]),
            'columns': int(df.shape[1]),
            'memory_usage': summary_raw.get('memory_usage', None),
            'data_types': summary_raw.get('dtypes', {})
        }
        data_quality_ui = {
            'missing_values': total_missing,
            'duplicates': int(df.duplicated().sum()),
            'completeness': round(completeness, 2)
        }

        # Persist a comprehensive results blob to disk for downloads
        results_blob = {
            'filename': filename,
            'data_summary_full': summary_raw,
            'anomalies_full': anomalies,
            'attention_weights': attention_weights.tolist() if attention_weights is not None else None,
            'synthetic_data': synthetic_data.tolist() if synthetic_data is not None else None,
            'visualizations': visualizations,
            'model_performance': training_history,
            'config': analysis_config
        }
        
        results_file = os.path.join(app.config['RESULTS_FOLDER'], f"{filename}_results.json")
        with open(results_file, 'w') as f:
            json.dump(results_blob, f, default=str)
        
        # Save trained model
        model_file = os.path.join(app.config['MODELS_FOLDER'], f"{filename}_model.h5")
        lstm_model.save(model_file)
        
        # Return top-level fields aligned with front-end expectations
        return jsonify({
            'message': 'Analysis completed successfully',
            'filename': filename,
            'data_summary': data_summary_ui,
            'data_quality': data_quality_ui,
            'anomaly_results': anomaly_results_summary,
            'visualizations': visualizations,
            'model_performance': training_history,
            'model_file': model_file
        })
        
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/results/<filename>')
def get_results(filename):
    """Retrieve analysis results"""
    try:
        results_file = os.path.join(app.config['RESULTS_FOLDER'], f"{filename}_results.json")
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                results = json.load(f)
            return jsonify(results)
        else:
            return jsonify({'error': 'Results not found'}), 404
    except Exception as e:
        logger.error(f"Results retrieval error: {str(e)}")
        return jsonify({'error': f'Failed to retrieve results: {str(e)}'}), 500

@app.route('/download/<filename>')
def download_results(filename):
    """Download analysis results"""
    try:
        return send_from_directory(
            app.config['RESULTS_FOLDER'],
            f"{filename}_results.json",
            as_attachment=True
        )
    except Exception as e:
        logger.error(f"Download error: {str(e)}")
        return jsonify({'error': f'Download failed: {str(e)}'}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'service': 'AutoEDA'})

def allowed_file(filename):
    """Check if file type is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Resource not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    logger.info("Starting AutoEDA application...")
    app.run(
        host=app.config['HOST'],
        port=app.config['PORT'],
        debug=app.config['DEBUG']
    )
