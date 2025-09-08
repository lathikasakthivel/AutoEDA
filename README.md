# AutoEDA - Deep Learning Powered EDA Assistant

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)](https://tensorflow.org/)
[![Flask](https://img.shields.io/badge/Flask-2.3+-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

AutoEDA is a comprehensive, deep learning-powered assistant that automates Exploratory Data Analysis (EDA) for sequential datasets such as time series or user activity logs. Leveraging Recurrent Neural Networks (RNNs), particularly LSTM autoencoders, the system detects anomalies, identifies key features using attention mechanisms, and optionally generates synthetic data for bias analysis.

## 🚀 Features

### Core Capabilities
- **Automated EDA**: Complete exploratory data analysis without writing code
- **Deep Learning**: LSTM autoencoders for pattern recognition and anomaly detection
- **Attention Mechanisms**: Identify key temporal features and dependencies
- **Anomaly Detection**: Multi-method ensemble approach for robust outlier detection
- **Synthetic Data Generation**: AI-powered data synthesis for testing and bias analysis
- **Interactive Visualizations**: Rich, interactive plots and dashboards

### Technical Features
- **Multi-format Support**: CSV, XLSX, JSON, Parquet files
- **Data Preprocessing**: Automatic cleaning, imputation, and encoding
- **Sequence Preparation**: Intelligent handling of time series and sequential data
- **Model Persistence**: Save and load trained models
- **Scalable Architecture**: Flask-based web interface with RESTful API
- **Real-time Processing**: Live analysis with progress tracking

## 🏗️ Architecture

```
AutoEDA/
├── app/                    # Flask application
│   ├── models/            # Deep learning models
│   │   ├── lstm_autoencoder.py      # LSTM Autoencoder implementation
│   │   └── attention_mechanism.py   # Attention mechanism
│   ├── utils/             # Utility functions
│   │   ├── data_processor.py        # Data loading and preprocessing
│   │   ├── anomaly_detector.py      # Anomaly detection algorithms
│   │   ├── synthetic_generator.py   # Synthetic data generation
│   │   └── visualization.py         # Interactive visualizations
│   ├── api/               # API endpoints
│   ├── static/            # Frontend assets and plots
│   ├── templates/         # HTML templates
│   └── config/            # Configuration files
├── data/                  # Data storage
├── models/                # Saved model files
├── results/               # Analysis results
├── sample_data/           # Generated sample datasets
├── app.py                 # Main Flask application
├── run.py                 # Startup script
├── demo.py                # Demo script
├── test_autoeda.py        # System tests
└── requirements.txt       # Python dependencies
```

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- 8GB+ RAM (recommended for large datasets)
- GPU support (optional, for faster training)

### Quick Start

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd AutoEDA
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the startup script**
   ```bash
   python run.py
   ```

The startup script will:
- Check Python version and dependencies
- Create necessary directories
- Generate sample datasets
- Run system tests
- Start the web application

### Manual Installation

If you prefer manual setup:

1. **Install core dependencies**
   ```bash
   pip install flask flask-cors numpy pandas scikit-learn tensorflow matplotlib seaborn plotly
   ```

2. **Install additional dependencies**
   ```bash
   pip install openpyxl pyarrow joblib python-dotenv
   ```

3. **Create directories**
   ```bash
   mkdir -p app/static/plots app/data app/models/saved app/results sample_data
   ```

4. **Run tests**
   ```bash
   python test_autoeda.py
   ```

5. **Start application**
   ```bash
   python app.py
   ```

## 🎯 Usage

### Web Interface

1. **Access the application**
   - Open your browser and go to `http://localhost:5000`
   - You'll see the modern, intuitive AutoEDA interface

2. **Upload your dataset**
   - Click the upload area or drag and drop your file
   - Supported formats: CSV, XLSX, JSON, Parquet
   - Maximum file size: 100MB

3. **Run analysis**
   - Click "Start Analysis" to begin the EDA pipeline
   - Watch real-time progress updates
   - The system will automatically:
     - Preprocess your data
     - Train LSTM autoencoders
     - Detect anomalies
     - Calculate feature importance
     - Generate synthetic data
     - Create interactive visualizations

4. **View results**
   - Explore comprehensive analysis results
   - Interact with generated visualizations
   - Download complete reports

### API Usage

AutoEDA provides a RESTful API for programmatic access:

```python
import requests

# Upload file
with open('data.csv', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:5000/api/upload', files=files)
    result = response.json()
    filename = result['filename']

# Run analysis
analysis_data = {'filename': filename}
response = requests.post('http://localhost:5000/api/analyze', json=analysis_data)
analysis_result = response.json()

# Get results
response = requests.get(f'http://localhost:5000/api/results/{filename}_results.json')
results = response.json()
```

### Command Line

Run the demo script to test the system:

```bash
python demo.py
```

This will:
- Generate sample time series data
- Run the complete analysis pipeline
- Test all system components
- Generate visualizations
- Save results to `demo_results.json`

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
FLASK_ENV=development
FLASK_DEBUG=1
UPLOAD_FOLDER=app/data
RESULTS_FOLDER=app/results
MODELS_FOLDER=app/models/saved
MAX_CONTENT_LENGTH=104857600
```

### Model Parameters

Adjust deep learning parameters in `app/config/config.py`:

```python
# LSTM Autoencoder
LSTM_ENCODING_DIM = 32
LSTM_UNITS = [64, 32]
LSTM_EPOCHS = 50
LSTM_BATCH_SIZE = 32

# Attention Mechanism
ATTENTION_NUM_HEADS = 8
ATTENTION_KEY_DIM = 64

# Anomaly Detection
ANOMALY_THRESHOLD = 0.95
ANOMALY_METHODS = ['reconstruction', 'statistical', 'isolation_forest']
```

## 📊 Sample Data

The system includes a comprehensive sample data generator:

```bash
python sample_data_generator.py
```

This creates:
- **Time Series Data**: 2000 samples with trends, seasonality, and anomalies
- **User Activity Data**: 200 users over 60 days with behavioral patterns
- **Sensor Data**: 15 sensors with realistic readings and anomalies
- **Financial Data**: 1000 trading days with technical indicators
- **Missing Value Variants**: Datasets with controlled missing data

## 🧪 Testing

### Run System Tests

```bash
python test_autoeda.py
```

Tests cover:
- Module imports
- Data processing
- LSTM autoencoder functionality
- Attention mechanism
- Anomaly detection
- Synthetic data generation
- Visualization generation

### Run Demo

```bash
python demo.py
```

The demo script provides a complete end-to-end test of the system.

## 🚀 Deployment

### Development

```bash
python run.py
```

### Production

```bash
# Using Gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app

# Using Docker (Dockerfile provided)
docker build -t autoeda .
docker run -p 5000:5000 autoeda
```

### Environment Configuration

```bash
export FLASK_ENV=production
export FLASK_DEBUG=0
export UPLOAD_FOLDER=/path/to/uploads
export RESULTS_FOLDER=/path/to/results
```

## 📈 Performance

### Optimization Tips

1. **GPU Acceleration**: Install TensorFlow with GPU support
2. **Batch Processing**: Adjust batch sizes based on available memory
3. **Model Caching**: Save trained models for reuse
4. **Data Streaming**: Process large datasets in chunks

### Scaling

- **Horizontal**: Deploy multiple instances behind a load balancer
- **Vertical**: Increase server resources for larger datasets
- **Caching**: Implement Redis for session and result caching
- **Async Processing**: Use Celery for background analysis tasks

## 🔍 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   pip install -r requirements.txt
   python -c "import tensorflow; print('OK')"
   ```

2. **Memory Issues**
   - Reduce batch size in config
   - Process data in smaller chunks
   - Use data streaming

3. **GPU Issues**
   ```bash
   pip install tensorflow-gpu
   nvidia-smi  # Check GPU status
   ```

4. **Port Conflicts**
   ```bash
   # Change port in app.py or run.py
   app.run(host='0.0.0.0', port=5001)
   ```

### Debug Mode

```bash
export FLASK_DEBUG=1
python app.py
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

### Development Setup

```bash
git clone <your-fork>
cd AutoEDA
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development dependencies
pre-commit install  # Git hooks
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- TensorFlow team for the deep learning framework
- Flask team for the web framework
- Plotly team for interactive visualizations
- Scikit-learn team for machine learning utilities

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **Documentation**: [Wiki](https://github.com/your-repo/wiki)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/discussions)

## 🎯 Roadmap

- [ ] Real-time streaming analysis
- [ ] Advanced anomaly detection algorithms
- [ ] Multi-language support
- [ ] Cloud deployment templates
- [ ] Mobile application
- [ ] Advanced visualization options
- [ ] Model interpretability tools
- [ ] Automated hyperparameter tuning

---

**AutoEDA** - Making Data Science Accessible to Everyone 🚀
