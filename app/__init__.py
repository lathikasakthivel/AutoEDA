#!/usr/bin/env python3
"""
AutoEDA Flask Application Package

This package initializes the Flask application and provides the application factory.
"""

from flask import Flask
from flask_cors import CORS
import os
import logging
from logging.handlers import RotatingFileHandler

def create_app(config_name='development'):
    """
    Application factory function
    
    Args:
        config_name (str): Configuration name ('development', 'production', 'testing')
        
    Returns:
        Flask: Configured Flask application
    """
    app = Flask(__name__)
    
    # Load configuration
    if config_name == 'production':
        app.config.from_object('app.config.config.ProductionConfig')
    elif config_name == 'testing':
        app.config.from_object('app.config.config.TestingConfig')
    else:
        app.config.from_object('app.config.config.DevelopmentConfig')
    
    # Initialize CORS
    CORS(app)
    
    # Setup logging
    if not app.debug and not app.testing:
        if not os.path.exists('logs'):
            os.mkdir('logs')
        file_handler = RotatingFileHandler('logs/autoeda.log', maxBytes=10240, backupCount=10)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
        ))
        file_handler.setLevel(logging.INFO)
        app.logger.addHandler(file_handler)
        app.logger.setLevel(logging.INFO)
        app.logger.info('AutoEDA startup')
    
    # Create necessary directories
    directories = [
        'app/static/plots',
        'app/data',
        'app/models/saved',
        'app/results'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    # Import and register blueprints
    from app.api import bp as api_bp
    app.register_blueprint(api_bp, url_prefix='/api')
    
    # Import and register main routes
    from app import routes
    
    return app

# Import routes at the bottom to avoid circular imports
from app import routes
