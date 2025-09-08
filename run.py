#!/usr/bin/env python3
"""
AutoEDA Startup Script

This script sets up the environment and runs the AutoEDA application.
It handles environment setup, dependency checking, and application startup.
"""

import os
import sys
import subprocess
import importlib.util
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✓ Python version: {sys.version.split()[0]}")
    return True

def check_dependencies():
    """Check if required dependencies are installed"""
    print("\nChecking dependencies...")
    
    required_packages = [
        'flask', 'numpy', 'pandas', 'sklearn', 'tensorflow',
        'matplotlib', 'seaborn', 'plotly', 'scipy', 'joblib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            importlib.util.find_spec(package)
            print(f"✓ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} - Missing")
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Please install missing packages with: pip install -r requirements.txt")
        return False
    
    print("✓ All required packages are installed")
    return True

def create_directories():
    """Create necessary directories if they don't exist"""
    print("\nCreating directories...")
    
    directories = [
        'app/static/plots',
        'app/data',
        'app/models/saved',
        'app/results',
        'sample_data'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created/verified: {directory}")

def generate_sample_data():
    """Generate sample data for testing"""
    print("\nGenerating sample data...")
    
    try:
        from sample_data_generator import SampleDataGenerator
        
        generator = SampleDataGenerator()
        generator.generate_all_datasets()
        print("✓ Sample data generated successfully")
        return True
    except Exception as e:
        print(f"⚠ Warning: Could not generate sample data: {e}")
        print("You can still use your own datasets")
        return False

def run_tests():
    """Run system tests"""
    print("\nRunning system tests...")
    
    try:
        result = subprocess.run([sys.executable, 'test_autoeda.py'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ System tests passed")
            return True
        else:
            print("⚠ Some tests failed, but continuing...")
            print("Test output:", result.stdout)
            return False
    except Exception as e:
        print(f"⚠ Could not run tests: {e}")
        return False


    
def start_application():
    """Start the AutoEDA application"""
    print("\n🚀 Starting AutoEDA application...")
    print("=" * 50)
    print("AutoEDA - Deep Learning Powered EDA Assistant")
    print("=" * 50)
    print("Access the application at: http://localhost:5000")
    print("Press Ctrl+C to stop the application")
    print("=" * 50)
    
    try:
        # Just run app.py directly (since it has the Flask app defined)
        subprocess.run([sys.executable, 'app.py'])
    except KeyboardInterrupt:
        print("\n\n👋 AutoEDA application stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting application: {e}")

def main():
    """Main function"""
    print("AutoEDA Startup Script")
    print("=" * 30)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check dependencies
    if not check_dependencies():
        print("\nPlease install missing dependencies and try again:")
        print("pip install -r requirements.txt")
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Generate sample data
    generate_sample_data()
    
    # Run tests (optional)
    run_tests()
    
    # Start application
    start_application()

if __name__ == "__main__":
    main()
