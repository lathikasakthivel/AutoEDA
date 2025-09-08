#!/usr/bin/env python3
"""
Sample Data Generator for AutoEDA

This script generates synthetic sequential datasets for testing the AutoEDA system.
It creates various types of time series data with different characteristics:
- Normal time series with trends and seasonality
- Anomalous data points
- Categorical variables
- Missing values
- Different data distributions

Usage:
    python sample_data_generator.py
"""

import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta
import random

class SampleDataGenerator:
    """Generates synthetic sequential datasets for testing AutoEDA"""
    
    def __init__(self, output_dir="sample_data"):
        self.output_dir = output_dir
        self._create_output_dir()
        
    def _create_output_dir(self):
        """Create output directory if it doesn't exist"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Created output directory: {self.output_dir}")
    
    def generate_time_series_data(self, n_samples=1000, n_features=5, include_anomalies=True):
        """
        Generate synthetic time series data with trends, seasonality, and anomalies
        
        Args:
            n_samples (int): Number of time points
            n_features (int): Number of features
            include_anomalies (bool): Whether to include anomalous data points
            
        Returns:
            pd.DataFrame: Generated time series data
        """
        print(f"Generating time series data: {n_samples} samples, {n_features} features")
        
        # Generate time index
        start_date = datetime(2020, 1, 1)
        dates = [start_date + timedelta(hours=i) for i in range(n_samples)]
        
        # Initialize data dictionary
        data = {'timestamp': dates}
        
        # Generate features with different characteristics
        for i in range(n_features):
            feature_name = f'feature_{i+1}'
            
            # Base trend (linear + polynomial)
            trend = np.linspace(0, 10, n_samples) + 0.01 * np.arange(n_samples)**2
            
            # Seasonality (daily and weekly patterns)
            daily_pattern = 2 * np.sin(2 * np.pi * np.arange(n_samples) / 24)
            weekly_pattern = 1.5 * np.sin(2 * np.pi * np.arange(n_samples) / (24 * 7))
            
            # Random noise
            noise = np.random.normal(0, 0.5, n_samples)
            
            # Combine components
            values = trend + daily_pattern + weekly_pattern + noise
            
            # Add anomalies if requested
            if include_anomalies:
                n_anomalies = int(0.05 * n_samples)  # 5% anomalies
                anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)
                anomaly_values = np.random.normal(0, 3, n_anomalies)  # Larger noise
                values[anomaly_indices] += anomaly_values
            
            data[feature_name] = values
        
        # Create DataFrame
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        
        return df
    
    def generate_user_activity_data(self, n_users=100, n_days=30):
        """
        Generate synthetic user activity data
        
        Args:
            n_users (int): Number of users
            n_days (int): Number of days of activity
            
        Returns:
            pd.DataFrame: Generated user activity data
        """
        print(f"Generating user activity data: {n_users} users, {n_days} days")
        
        data = []
        start_date = datetime(2020, 1, 1)
        
        for user_id in range(1, n_users + 1):
            for day in range(n_days):
                current_date = start_date + timedelta(days=day)
                
                # Generate daily activity metrics
                login_count = np.random.poisson(3)  # Average 3 logins per day
                session_duration = np.random.exponential(30)  # Average 30 minutes
                pages_viewed = np.random.poisson(15)  # Average 15 pages
                actions_performed = np.random.poisson(25)  # Average 25 actions
                
                # Add some weekly patterns
                if current_date.weekday() < 5:  # Weekdays
                    login_count = int(login_count * 1.5)
                    session_duration *= 1.3
                
                # Add some anomalies (5% chance)
                if np.random.random() < 0.05:
                    login_count = int(login_count * 3)
                    session_duration *= 2
                
                data.append({
                    'user_id': user_id,
                    'date': current_date,
                    'login_count': max(0, login_count),
                    'session_duration_minutes': max(0, session_duration),
                    'pages_viewed': max(0, pages_viewed),
                    'actions_performed': max(0, actions_performed),
                    'is_weekend': current_date.weekday() >= 5
                })
        
        df = pd.DataFrame(data)
        return df
    
    def generate_sensor_data(self, n_sensors=10, n_readings=5000):
        """
        Generate synthetic sensor data with different sensor types
        
        Args:
            n_sensors (int): Number of sensors
            n_readings (int): Number of readings per sensor
            
        Returns:
            pd.DataFrame: Generated sensor data
        """
        print(f"Generating sensor data: {n_sensors} sensors, {n_readings} readings each")
        
        data = []
        start_time = datetime(2020, 1, 1)
        
        sensor_types = ['temperature', 'pressure', 'humidity', 'vibration', 'current']
        
        for sensor_id in range(1, n_sensors + 1):
            sensor_type = sensor_types[sensor_id % len(sensor_types)]
            
            for reading in range(n_readings):
                current_time = start_time + timedelta(minutes=reading)
                
                # Generate base values based on sensor type
                if sensor_type == 'temperature':
                    base_value = 20 + 10 * np.sin(2 * np.pi * reading / (24 * 60))  # Daily cycle
                    noise = np.random.normal(0, 2)
                elif sensor_type == 'pressure':
                    base_value = 1013 + 20 * np.sin(2 * np.pi * reading / (24 * 60))
                    noise = np.random.normal(0, 5)
                elif sensor_type == 'humidity':
                    base_value = 50 + 20 * np.sin(2 * np.pi * reading / (24 * 60))
                    noise = np.random.normal(0, 3)
                elif sensor_type == 'vibration':
                    base_value = 0.1 + 0.05 * np.sin(2 * np.pi * reading / 60)  # Hourly cycle
                    noise = np.random.normal(0, 0.02)
                else:  # current
                    base_value = 5 + 2 * np.sin(2 * np.pi * reading / (24 * 60))
                    noise = np.random.normal(0, 0.5)
                
                # Add some anomalies
                if np.random.random() < 0.02:  # 2% anomalies
                    noise *= 10
                
                value = base_value + noise
                
                data.append({
                    'timestamp': current_time,
                    'sensor_id': sensor_id,
                    'sensor_type': sensor_type,
                    'value': value,
                    'unit': self._get_sensor_unit(sensor_type)
                })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df
    
    def _get_sensor_unit(self, sensor_type):
        """Get appropriate unit for sensor type"""
        units = {
            'temperature': '°C',
            'pressure': 'hPa',
            'humidity': '%',
            'vibration': 'g',
            'current': 'A'
        }
        return units.get(sensor_type, '')
    
    def generate_financial_data(self, n_days=500):
        """
        Generate synthetic financial time series data
        
        Args:
            n_days (int): Number of trading days
            
        Returns:
            pd.DataFrame: Generated financial data
        """
        print(f"Generating financial data: {n_days} trading days")
        
        # Generate dates (excluding weekends)
        start_date = datetime(2020, 1, 1)
        dates = []
        current_date = start_date
        
        while len(dates) < n_days:
            if current_date.weekday() < 5:  # Monday to Friday
                dates.append(current_date)
            current_date += timedelta(days=1)
        
        # Generate price data using random walk
        np.random.seed(42)  # For reproducibility
        returns = np.random.normal(0.001, 0.02, n_days)  # Daily returns
        prices = 100 * np.exp(np.cumsum(returns))  # Starting price $100
        
        # Generate volume data
        volumes = np.random.lognormal(10, 0.5, n_days)
        
        # Generate technical indicators
        sma_20 = pd.Series(prices).rolling(20).mean().values
        sma_50 = pd.Series(prices).rolling(50).mean().values
        
        # Add some anomalies (market events)
        anomaly_indices = np.random.choice(n_days, int(0.03 * n_days), replace=False)
        for idx in anomaly_indices:
            prices[idx] *= np.random.choice([0.8, 1.2])  # 20% price movement
            volumes[idx] *= 3  # 3x volume
        
        data = {
            'date': dates,
            'close_price': prices,
            'volume': volumes,
            'sma_20': sma_20,
            'sma_50': sma_50
        }
        
        df = pd.DataFrame(data)
        df.set_index('date', inplace=True)
        return df
    
    def add_missing_values(self, df, missing_ratio=0.05):
        """
        Add missing values to the dataset
        
        Args:
            df (pd.DataFrame): Input DataFrame
            missing_ratio (float): Ratio of values to make missing
            
        Returns:
            pd.DataFrame: DataFrame with missing values
        """
        print(f"Adding {missing_ratio*100}% missing values")
        
        df_with_missing = df.copy()
        n_missing = int(df.size * missing_ratio)
        
        # Randomly select positions to make missing
        rows = np.random.choice(df.shape[0], n_missing)
        cols = np.random.choice(df.shape[1], n_missing)
        
        for i in range(n_missing):
            df_with_missing.iloc[rows[i], cols[i]] = np.nan
        
        return df_with_missing
    
    def save_dataset(self, df, filename, format='csv'):
        """
        Save dataset to file
        
        Args:
            df (pd.DataFrame): DataFrame to save
            filename (str): Output filename
            format (str): Output format ('csv', 'xlsx', 'json', 'parquet')
        """
        filepath = os.path.join(self.output_dir, filename)
        
        if format == 'csv':
            df.to_csv(filepath)
        elif format == 'xlsx':
            df.to_excel(filepath, engine='openpyxl')
        elif format == 'json':
            df.to_json(filepath, orient='records')
        elif format == 'parquet':
            df.to_parquet(filepath)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"Saved dataset: {filepath}")
    
    def generate_all_datasets(self):
        """Generate all sample datasets"""
        print("Generating all sample datasets...")
        
        # 1. Time series data
        ts_data = self.generate_time_series_data(n_samples=2000, n_features=8)
        self.save_dataset(ts_data, 'time_series_data.csv')
        
        # 2. User activity data
        user_data = self.generate_user_activity_data(n_users=200, n_days=60)
        self.save_dataset(user_data, 'user_activity_data.csv')
        
        # 3. Sensor data
        sensor_data = self.generate_sensor_data(n_sensors=15, n_readings=3000)
        self.save_dataset(sensor_data, 'sensor_data.csv')
        
        # 4. Financial data
        financial_data = self.generate_financial_data(n_days=1000)
        self.save_dataset(financial_data, 'financial_data.csv')
        
        # 5. Time series with missing values
        ts_missing = self.add_missing_values(ts_data, missing_ratio=0.08)
        self.save_dataset(ts_missing, 'time_series_with_missing.csv')
        
        # 6. User activity with missing values
        user_missing = self.add_missing_values(user_data, missing_ratio=0.06)
        self.save_dataset(user_missing, 'user_activity_with_missing.csv')
        
        print("\nAll datasets generated successfully!")
        print(f"Output directory: {os.path.abspath(self.output_dir)}")
        
        # Print dataset summaries
        print("\nDataset summaries:")
        datasets = {
            'time_series_data.csv': ts_data,
            'user_activity_data.csv': user_data,
            'sensor_data.csv': sensor_data,
            'financial_data.csv': financial_data,
            'time_series_with_missing.csv': ts_missing,
            'user_activity_with_missing.csv': user_missing
        }
        
        for filename, df in datasets.items():
            print(f"\n{filename}:")
            print(f"  Shape: {df.shape}")
            print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
            if df.isnull().any().any():
                missing_counts = df.isnull().sum()
                print(f"  Missing values: {missing_counts[missing_counts > 0].to_dict()}")

def main():
    """Main function to generate sample datasets"""
    print("AutoEDA Sample Data Generator")
    print("=" * 40)
    
    # Create generator
    generator = SampleDataGenerator()
    
    # Generate all datasets
    generator.generate_all_datasets()
    
    print("\nSample data generation completed!")
    print("You can now use these datasets to test the AutoEDA system.")

if __name__ == "__main__":
    main()
