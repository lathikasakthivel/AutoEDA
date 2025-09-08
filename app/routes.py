#!/usr/bin/env python3
"""
Main Routes for AutoEDA

This module defines the main web interface routes for the AutoEDA system.
"""

from flask import render_template, request, jsonify, current_app
from app import create_app
import os

app = create_app()

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload (web interface)"""
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
            
            return jsonify({
                'message': 'File uploaded successfully',
                'filename': filename
            })
        
        return jsonify({'error': 'Invalid file type'}), 400
        
    except Exception as e:
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/analyze', methods=['POST'])
def analyze_data():
    """Run EDA analysis (web interface)"""
    try:
        data = request.get_json()
        filename = data.get('filename')
        
        if not filename:
            return jsonify({'error': 'No filename provided'}), 400
        
        # This will be handled by the API
        from app.api.routes import analyze_data as api_analyze
        return api_analyze()
        
    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/results/<filename>')
def get_results(filename):
    """Get analysis results (web interface)"""
    try:
        # This will be handled by the API
        from app.api.routes import get_results as api_get_results
        return api_get_results(filename)
        
    except Exception as e:
        return jsonify({'error': f'Error retrieving results: {str(e)}'}), 500

@app.route('/download/<filename>')
def download_results(filename):
    """Download results (web interface)"""
    try:
        # This will be handled by the API
        from app.api.routes import download_results as api_download
        return api_download(filename)
        
    except Exception as e:
        return jsonify({'error': f'Error downloading results: {str(e)}'}), 500

@app.route('/health')
def health_check():
    """Health check (web interface)"""
    try:
        # This will be handled by the API
        from app.api.routes import health_check as api_health
        return api_health()
        
    except Exception as e:
        return jsonify({'error': f'Health check failed: {str(e)}'}), 500

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

# Import datetime for timestamp generation
from datetime import datetime
