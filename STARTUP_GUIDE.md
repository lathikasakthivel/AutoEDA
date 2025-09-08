# AutoEDA Startup Guide 🚀

This guide will walk you through setting up and running the AutoEDA system step by step.

## 🎯 Quick Start (5 minutes)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the System
```bash
python run.py
```

### 3. Open Your Browser
Navigate to `http://localhost:5000`

That's it! 🎉

---

## 🔧 Detailed Setup

### Prerequisites Check

First, ensure you have the right environment:

```bash
# Check Python version (3.8+ required)
python --version

# Check pip
pip --version

# Check available memory (8GB+ recommended)
# On Windows: Task Manager > Performance > Memory
# On Linux/Mac: free -h
```

### Step-by-Step Installation

#### 1. Clone and Navigate
```bash
git clone <your-repo-url>
cd AutoEDA
```

#### 2. Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python -m venv venv
source venv/bin/activate
```

#### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

**Note**: This may take 5-10 minutes depending on your internet connection.

#### 4. Verify Installation
```bash
python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__} installed')"
python -c "import flask; print(f'Flask {flask.__version__} installed')"
```

---

## 🧪 Testing the System

### Run System Tests
```bash
python test_autoeda.py
```

**Expected Output:**
```
AutoEDA System Test Suite
========================================
Testing imports...
✓ All modules imported successfully

Testing Configuration...
✓ Default config loaded - Debug mode: True
✓ Development config loaded - Debug mode: True

Testing DataProcessor...
✓ DataProcessor test passed - Input shape: (100, 3), Output shape: (100, 3)

Testing LSTM Autoencoder...
✓ LSTM Autoencoder model built successfully

Testing Attention Mechanism...
✓ Attention Mechanism model built successfully

Testing Anomaly Detector...
✓ Anomaly Detector test passed

Testing Synthetic Data Generator...
✓ Synthetic Data Generator test passed

Testing Visualization...
✓ Visualization Generator test passed

========================================
Test Results: 8/8 tests passed
🎉 All tests passed! AutoEDA system is ready to use.
```

### Run Demo
```bash
python demo.py
```

This will:
- Generate sample time series data
- Run the complete analysis pipeline
- Test all system components
- Generate visualizations
- Save results to `demo_results.json`

---

## 🚀 Running the Application

### Option 1: Automated Startup (Recommended)
```bash
python run.py
```

This script will:
- ✅ Check Python version
- ✅ Verify dependencies
- ✅ Create necessary directories
- ✅ Generate sample data
- ✅ Run system tests
- ✅ Start the web application

### Option 2: Manual Startup
```bash
# Create directories
mkdir -p app/static/plots app/data app/models/saved app/results sample_data

# Generate sample data
python sample_data_generator.py

# Start application
python app.py
```

---

## 🌐 Using the Web Interface

### 1. Access the Application
- Open your browser
- Navigate to `http://localhost:5000`
- You'll see the modern AutoEDA interface

### 2. Upload Your Dataset
- Click the upload area or drag and drop
- Supported formats: CSV, XLSX, JSON, Parquet
- Maximum file size: 100MB

### 3. Run Analysis
- Click "Start Analysis"
- Watch real-time progress updates
- The system automatically:
  - Preprocesses your data
  - Trains LSTM autoencoders
  - Detects anomalies
  - Calculates feature importance
  - Generates synthetic data
  - Creates visualizations

### 4. Explore Results
- View comprehensive analysis results
- Interact with generated visualizations
- Download complete reports

---

## 📊 Sample Data

### Generate Sample Datasets
```bash
python sample_data_generator.py
```

This creates:
- **Time Series Data**: 2000 samples with trends, seasonality, anomalies
- **User Activity Data**: 200 users over 60 days
- **Sensor Data**: 15 sensors with realistic readings
- **Financial Data**: 1000 trading days with indicators
- **Missing Value Variants**: Controlled missing data for testing

### Use Your Own Data
- Any CSV/Excel file with time series or sequential data
- Ensure your data has a timestamp column (if time series)
- The system automatically detects data types and handles preprocessing

---

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
Adjust deep learning settings in `app/config/config.py`:

```python
# LSTM Autoencoder
LSTM_ENCODING_DIM = 32        # Encoding dimension
LSTM_UNITS = [64, 32]         # LSTM layer sizes
LSTM_EPOCHS = 50              # Training epochs
LSTM_BATCH_SIZE = 32          # Batch size

# Attention Mechanism
ATTENTION_NUM_HEADS = 8       # Number of attention heads
ATTENTION_KEY_DIM = 64        # Key dimension

# Anomaly Detection
ANOMALY_THRESHOLD = 0.95      # Anomaly threshold
ANOMALY_METHODS = ['reconstruction', 'statistical', 'isolation_forest']
```

---

## 🚨 Troubleshooting

### Common Issues & Solutions

#### 1. Import Errors
```bash
# Reinstall dependencies
pip uninstall -r requirements.txt -y
pip install -r requirements.txt
```

#### 2. Memory Issues
- Reduce batch size in config
- Process smaller datasets
- Close other applications

#### 3. Port Conflicts
```bash
# Change port in app.py
app.run(host='0.0.0.0', port=5001)
```

#### 4. GPU Issues
```bash
# Check GPU status
nvidia-smi

# Install GPU version
pip install tensorflow-gpu
```

#### 5. File Permission Issues
```bash
# On Linux/Mac
chmod +x run.py
chmod +x demo.py
```

### Debug Mode
```bash
export FLASK_DEBUG=1
python app.py
```

---

## 📱 API Usage

### RESTful Endpoints

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

### Available Endpoints
- `POST /api/upload` - Upload dataset
- `POST /api/analyze` - Run EDA analysis
- `GET /api/results/<filename>` - Get analysis results
- `GET /api/download/<filename>` - Download results
- `GET /api/health` - Health check

---

## 🚀 Production Deployment

### Using Gunicorn
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Using Docker
```bash
# Build image
docker build -t autoeda .

# Run container
docker run -p 5000:5000 autoeda
```

### Environment Configuration
```bash
export FLASK_ENV=production
export FLASK_DEBUG=0
export UPLOAD_FOLDER=/path/to/uploads
export RESULTS_FOLDER=/path/to/results
```

---

## 📚 Next Steps

### 1. Explore the Interface
- Upload a sample dataset
- Run the analysis pipeline
- Explore generated visualizations

### 2. Customize for Your Use Case
- Adjust model parameters
- Modify visualization styles
- Add custom preprocessing steps

### 3. Scale Up
- Process larger datasets
- Deploy to production
- Integrate with existing systems

### 4. Contribute
- Report bugs
- Suggest features
- Submit pull requests

---

## 🆘 Getting Help

### Documentation
- **README.md** - Comprehensive project overview
- **This Guide** - Step-by-step setup instructions
- **Code Comments** - Inline documentation

### Support Channels
- **GitHub Issues** - Bug reports and feature requests
- **GitHub Discussions** - Questions and community support
- **Wiki** - Detailed documentation and tutorials

### Common Questions

**Q: How long does analysis take?**
A: Depends on dataset size and complexity. Small datasets (1K rows): 1-2 minutes. Large datasets (100K+ rows): 10-30 minutes.

**Q: Can I use my own deep learning models?**
A: Yes! The system is modular. You can replace the LSTM autoencoder with custom models in `app/models/`.

**Q: What data formats are supported?**
A: CSV, XLSX, JSON, and Parquet files. The system automatically detects and handles different formats.

**Q: How do I customize visualizations?**
A: Modify the `VisualizationGenerator` class in `app/utils/visualization.py` to add custom plots and styles.

---

## 🎉 Congratulations!

You've successfully set up AutoEDA! The system is now ready to:

- 🔍 Analyze your sequential datasets
- 🧠 Train deep learning models
- 📊 Generate interactive visualizations
- 🚨 Detect anomalies automatically
- 🎲 Generate synthetic data
- 📈 Provide comprehensive insights

**Happy Data Analysis! 🚀**

---

*Need help? Check the troubleshooting section above or open a GitHub issue.*
