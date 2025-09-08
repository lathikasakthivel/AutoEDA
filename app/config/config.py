"""
Configuration settings for AutoEDA application
"""

import os
from pathlib import Path

class Config:
    """Base configuration class"""
    
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'autoeda-secret-key-change-in-production'
    DEBUG = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    HOST = os.environ.get('HOST', '0.0.0.0')
    PORT = int(os.environ.get('PORT', 5000))
    
    # File upload settings
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB max file size
    UPLOAD_FOLDER = os.path.join(os.getcwd(), 'data', 'uploads')
    RESULTS_FOLDER = os.path.join(os.getcwd(), 'data', 'results')
    MODELS_FOLDER = os.path.join(os.getcwd(), 'models')
    ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls', 'json', 'parquet'}
    
    # Data processing settings
    MAX_ROWS = 1000000  # Maximum rows to process
    CHUNK_SIZE = 10000  # Process data in chunks
    MIN_SEQUENCE_LENGTH = 10  # Minimum sequence length for LSTM
    MAX_SEQUENCE_LENGTH = 1000  # Maximum sequence length for LSTM
    
    # LSTM Autoencoder settings
    LSTM_CONFIG = {
        'encoder_units': [64, 32, 16],
        'decoder_units': [16, 32, 64],
        'dropout_rate': 0.2,
        'recurrent_dropout': 0.1,
        'activation': 'tanh',
        'recurrent_activation': 'sigmoid',
        'return_sequences': True,
        'return_state': False
    }
    
    # Training settings
    DEFAULT_EPOCHS = 50
    DEFAULT_BATCH_SIZE = 32
    VALIDATION_SPLIT = 0.2
    EARLY_STOPPING_PATIENCE = 10
    LEARNING_RATE = 0.001
    
    # Anomaly detection settings
    ANOMALY_THRESHOLD = 0.1
    ANOMALY_WINDOW_SIZE = 10
    MIN_ANOMALY_SCORE = 0.05
    
    # Attention mechanism settings
    ATTENTION_HEADS = 4
    ATTENTION_DIM = 32
    
    # Synthetic data generation settings
    SYNTHETIC_SAMPLES = 1000
    SYNTHETIC_NOISE_LEVEL = 0.05
    
    # Visualization settings
    PLOT_HEIGHT = 600
    PLOT_WIDTH = 800
    PLOT_TEMPLATE = 'plotly_white'
    MAX_POINTS_PLOT = 10000  # Maximum points to show in plots
    
    # Caching settings
    CACHE_TIMEOUT = 3600  # 1 hour
    MAX_CACHE_SIZE = 1000
    
    # Security settings
    SESSION_COOKIE_SECURE = False  # Set to True in production with HTTPS
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    
    # Logging settings
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FILE = os.path.join(os.getcwd(), 'logs', 'autoeda.log')
    LOG_MAX_BYTES = 10 * 1024 * 1024  # 10MB
    LOG_BACKUP_COUNT = 5
    
    # Performance settings
    WORKER_PROCESSES = os.environ.get('WORKER_PROCESSES', 1)
    WORKER_THREADS = os.environ.get('WORKER_THREADS', 2)
    MAX_REQUESTS = 1000
    MAX_REQUESTS_JITTER = 100
    
    @staticmethod
    def init_app(app):
        """Initialize application with configuration"""
        # Create necessary directories
        for folder in [Config.UPLOAD_FOLDER, Config.RESULTS_FOLDER, Config.MODELS_FOLDER]:
            os.makedirs(folder, exist_ok=True)
        
        # Create logs directory
        log_dir = os.path.dirname(Config.LOG_FILE)
        os.makedirs(log_dir, exist_ok=True)

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    HOST = 'localhost'
    PORT = 5000
    
class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    HOST = '0.0.0.0'
    PORT = int(os.environ.get('PORT', 5000))
    SECRET_KEY = os.environ.get('SECRET_KEY')
    SESSION_COOKIE_SECURE = True

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEBUG = True
    UPLOAD_FOLDER = os.path.join(os.getcwd(), 'tests', 'test_data', 'uploads')
    RESULTS_FOLDER = os.path.join(os.getcwd(), 'tests', 'test_data', 'results')
    MODELS_FOLDER = os.path.join(os.getcwd(), 'tests', 'test_data', 'models')

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
