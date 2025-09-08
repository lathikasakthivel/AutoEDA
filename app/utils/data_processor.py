"""
Data processing utilities for AutoEDA
"""

import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import json
import os
from typing import Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Data processing utilities for AutoEDA
    
    Handles:
    - Data loading from various formats
    - Data preprocessing and cleaning
    - Feature engineering for sequential data
    - Data validation and quality checks
    - Sequence preparation for LSTM models
    """
    
    def __init__(self):
        """Initialize DataProcessor"""
        self.scalers = {}
        self.label_encoders = {}
        self.imputers = {}
        self.data_info = {}
        
        logger.info("DataProcessor initialized")
    
    def process_upload(self, filepath: str) -> Dict[str, Any]:
        """
        Process uploaded file and extract basic information
        
        Args:
            filepath (str): Path to uploaded file
            
        Returns:
            dict: Basic data information
        """
        try:
            # Load data
            df = self.load_data(filepath)
            
            # Extract basic information
            data_info = self.get_data_summary(df)
            
            # Store data info
            self.data_info[filepath] = data_info
            
            logger.info(f"File processed successfully: {filepath}")
            return data_info
            
        except Exception as e:
            logger.error(f"Error processing upload: {str(e)}")
            raise
    
    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load data from various file formats
        
        Args:
            filepath (str): Path to data file
            
        Returns:
            pd.DataFrame: Loaded data
        """
        try:
            file_ext = os.path.splitext(filepath)[1].lower()
            
            if file_ext == '.csv':
                df = pd.read_csv(filepath)
            elif file_ext in ['.xlsx', '.xls']:
                df = pd.read_excel(filepath)
            elif file_ext == '.json':
                df = pd.read_json(filepath)
            elif file_ext == '.parquet':
                df = pd.read_parquet(filepath)
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
            
            logger.info(f"Data loaded from {filepath}: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data from {filepath}: {str(e)}")
            raise
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive data summary
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            dict: Data summary information
        """
        try:
            summary = {
                'shape': df.shape,
                'columns': list(df.columns),
                'dtypes': df.dtypes.to_dict(),
                'missing_values': df.isnull().sum().to_dict(),
                'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
                'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
                'categorical_columns': df.select_dtypes(include=['object', 'category']).columns.tolist(),
                'datetime_columns': [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])],
                'memory_usage': df.memory_usage(deep=True).sum(),
                'duplicates': df.duplicated().sum(),
                'unique_counts': {col: df[col].nunique() for col in df.columns}
            }

            # Add basic statistics for numeric columns
            if summary['numeric_columns']:
                summary['numeric_stats'] = df[summary['numeric_columns']].describe().to_dict()

            # Add basic statistics for categorical columns
            if summary['categorical_columns']:
                summary['categorical_stats'] = {
                    col: df[col].value_counts().head(10).to_dict()
                    for col in summary['categorical_columns']
                }

            # Ensure JSON-serializable output
            summary_serializable = self._make_json_serializable(summary)

            logger.info(f"Data summary generated for {df.shape}")
            return summary_serializable

        except Exception as e:
            logger.error(f"Error generating data summary: {str(e)}")
            raise

    @staticmethod
    def _make_json_serializable(obj: Any) -> Any:
        """Recursively convert pandas/numpy types to JSON-serializable Python types."""
        try:
            import math
        except Exception:
            math = None

        if isinstance(obj, dict):
            return {str(k): DataProcessor._make_json_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, set)):
            return [DataProcessor._make_json_serializable(v) for v in obj]

        # Handle pandas and numpy scalar types
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            # Convert NaN to None
            value = float(obj)
            if value != value:  # NaN check
                return None
            return value
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.ndarray,)):
            return DataProcessor._make_json_serializable(obj.tolist())

        # pandas specific types
        if pd.isna(obj):
            return None
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        if isinstance(obj, pd.Timedelta):
            return obj.isoformat()
        # dtypes like Int64Dtype, object dtype, etc.
        try:
            from pandas.api.extensions import ExtensionDtype
            if isinstance(obj, (np.dtype, ExtensionDtype)):
                return str(obj)
        except Exception:
            if isinstance(obj, np.dtype):
                return str(obj)

        # Fallback for other objects (e.g., Python primitives)
        try:
            json.dumps(obj)
            return obj
        except Exception:
            return str(obj)
    
    def preprocess_data(self, df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess data for LSTM analysis
        
        Args:
            df (pd.DataFrame): Input dataframe
            config (dict): Preprocessing configuration
            
        Returns:
            dict: Preprocessed data and metadata
        """
        try:
            logger.info("Starting data preprocessing...")
            
            # Clean data
            df_clean = self._clean_data(df, config)
            
            # Handle missing values
            df_imputed = self._handle_missing_values(df_clean, config)
            
            # Encode categorical variables
            df_encoded = self._encode_categorical_variables(df_imputed, config)
            
            # Scale numeric variables
            df_scaled = self._scale_numeric_variables(df_encoded, config)
            
            # Prepare sequences for LSTM
            sequences = self._prepare_sequences(df_scaled, config)
            
            # Split data
            train_data, validation_data = self._split_data(sequences, config)
            
            processed_data = {
                'data': sequences,
                'train_data': train_data,
                'validation_data': validation_data,
                'input_shape': sequences.shape[1:],
                'feature_names': df_scaled.columns.tolist(),
                'preprocessing_info': {
                    'scalers': {k: str(v) for k, v in self.scalers.items()},
                    'encoders': {k: str(v) for k, v in self.label_encoders.items()},
                    'imputers': {k: str(v) for k, v in self.imputers.items()}
                }
            }
            
            logger.info("Data preprocessing completed successfully")
            return processed_data
            
        except Exception as e:
            logger.error(f"Error in data preprocessing: {str(e)}")
            raise
    
    def _clean_data(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Clean the input data"""
        df_clean = df.copy()
        
        # Remove duplicates
        if config.get('remove_duplicates', True):
            initial_rows = len(df_clean)
            df_clean = df_clean.drop_duplicates()
            removed_rows = initial_rows - len(df_clean)
            if removed_rows > 0:
                logger.info(f"Removed {removed_rows} duplicate rows")
        
        # Remove rows with too many missing values
        max_missing_ratio = config.get('max_missing_ratio', 0.5)
        if max_missing_ratio < 1.0:
            initial_rows = len(df_clean)
            df_clean = df_clean.dropna(thresh=int(len(df_clean.columns) * (1 - max_missing_ratio)))
            removed_rows = initial_rows - len(df_clean)
            if removed_rows > 0:
                logger.info(f"Removed {removed_rows} rows with too many missing values")
        
        # Remove columns with too many missing values
        max_col_missing_ratio = config.get('max_col_missing_ratio', 0.8)
        if max_col_missing_ratio < 1.0:
            initial_cols = len(df_clean.columns)
            df_clean = df_clean.dropna(axis=1, thresh=int(len(df_clean) * (1 - max_col_missing_ratio)))
            removed_cols = initial_cols - len(df_clean.columns)
            if removed_cols > 0:
                logger.info(f"Removed {removed_cols} columns with too many missing values")
        
        return df_clean
    
    def _handle_missing_values(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Handle missing values in the data"""
        df_imputed = df.copy()
        
        # Handle numeric columns
        numeric_cols = df_imputed.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            strategy = config.get('numeric_imputation_strategy', 'mean')
            imputer = SimpleImputer(strategy=strategy)
            df_imputed[numeric_cols] = imputer.fit_transform(df_imputed[numeric_cols])
            self.imputers['numeric'] = imputer
            
            logger.info(f"Imputed missing values in {len(numeric_cols)} numeric columns using {strategy}")
        
        # Handle categorical columns
        categorical_cols = df_imputed.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            strategy = config.get('categorical_imputation_strategy', 'most_frequent')
            imputer = SimpleImputer(strategy=strategy)
            df_imputed[categorical_cols] = imputer.fit_transform(df_imputed[categorical_cols])
            self.imputers['categorical'] = imputer
            
            logger.info(f"Imputed missing values in {len(categorical_cols)} categorical columns using {strategy}")
        
        return df_imputed
    
    def _encode_categorical_variables(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Encode categorical variables"""
        df_encoded = df.copy()
        
        categorical_cols = df_encoded.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            if df_encoded[col].nunique() <= config.get('max_categories', 100):
                # Use label encoding for columns with reasonable number of categories
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                self.label_encoders[col] = le
                
                logger.info(f"Label encoded column: {col}")
            else:
                # For high-cardinality columns, consider other strategies
                logger.warning(f"High cardinality column {col} ({df_encoded[col].nunique()} unique values)")
        
        return df_encoded
    
    def _scale_numeric_variables(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Scale numeric variables"""
        df_scaled = df.copy()
        
        numeric_cols = df_scaled.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 0:
            scaling_method = config.get('scaling_method', 'standard')
            
            if scaling_method == 'standard':
                scaler = StandardScaler()
            elif scaling_method == 'minmax':
                scaler = MinMaxScaler()
            else:
                scaler = StandardScaler()
            
            df_scaled[numeric_cols] = scaler.fit_transform(df_scaled[numeric_cols])
            self.scalers['numeric'] = scaler
            
            logger.info(f"Scaled {len(numeric_cols)} numeric columns using {scaling_method} scaling")
        
        return df_scaled
    
    def _prepare_sequences(self, df: pd.DataFrame, config: Dict[str, Any]) -> np.ndarray:
        """Prepare sequences for LSTM analysis"""
        # Convert to numpy array
        data = df.values
        
        # Get sequence parameters
        sequence_length = config.get('sequence_length', 10)
        step_size = config.get('step_size', 1)
        
        # Create sequences
        sequences = []
        for i in range(0, len(data) - sequence_length + 1, step_size):
            sequence = data[i:i + sequence_length]
            sequences.append(sequence)
        
        sequences = np.array(sequences)
        
        logger.info(f"Created {len(sequences)} sequences with length {sequence_length}")
        return sequences
    
    def _split_data(self, sequences: np.ndarray, config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """Split data into training and validation sets"""
        validation_split = config.get('validation_split', 0.2)
        split_idx = int(len(sequences) * (1 - validation_split))
        
        train_data = sequences[:split_idx]
        validation_data = sequences[split_idx:]
        
        logger.info(f"Split data: {len(train_data)} training, {len(validation_data)} validation")
        return train_data, validation_data
    
    def detect_data_types(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Automatically detect data types for each column
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            dict: Column name to detected type mapping
        """
        data_types = {}
        
        for col in df.columns:
            # Check if it's datetime
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                data_types[col] = 'datetime'
            # Check if it's numeric
            elif pd.api.types.is_numeric_dtype(df[col]):
                data_types[col] = 'numeric'
            # Check if it's categorical
            elif df[col].nunique() / len(df) < 0.1:  # Less than 10% unique values
                data_types[col] = 'categorical'
            # Check if it's text
            elif df[col].dtype == 'object':
                data_types[col] = 'text'
            else:
                data_types[col] = 'unknown'
        
        return data_types
    
    def validate_sequential_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate if data is suitable for sequential analysis
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            dict: Validation results
        """
        validation_results = {
            'is_sequential': False,
            'has_temporal_order': False,
            'recommended_sequence_length': None,
            'warnings': [],
            'recommendations': []
        }
        
        # Check if data has temporal order
        if len(df) > 1:
            # Check if there's a datetime column
            datetime_cols = df.select_dtypes(include=['datetime64']).columns
            if len(datetime_cols) > 0:
                validation_results['has_temporal_order'] = True
                validation_results['is_sequential'] = True
                validation_results['recommendations'].append("Data has temporal order - suitable for time series analysis")
            else:
                # Check if data might be sequential based on index
                if df.index.is_monotonic_increasing:
                    validation_results['has_temporal_order'] = True
                    validation_results['is_sequential'] = True
                    validation_results['recommendations'].append("Data index suggests sequential order")
                else:
                    validation_results['warnings'].append("No clear temporal order detected")
        
        # Recommend sequence length
        if validation_results['is_sequential']:
            recommended_length = min(50, max(10, len(df) // 20))
            validation_results['recommended_sequence_length'] = recommended_length
            validation_results['recommendations'].append(f"Recommended sequence length: {recommended_length}")
        
        return validation_results
    
    def export_preprocessing_pipeline(self, filepath: str) -> None:
        """
        Export preprocessing pipeline for reuse
        
        Args:
            filepath (str): Path to save pipeline
        """
        try:
            pipeline = {
                'scalers': {k: str(v) for k, v in self.scalers.items()},
                'encoders': {k: str(v) for k, v in self.label_encoders.items()},
                'imputers': {k: str(v) for k, v in self.imputers.items()},
                'data_info': self.data_info
            }
            
            with open(filepath, 'w') as f:
                json.dump(pipeline, f, indent=2, default=str)
            
            logger.info(f"Preprocessing pipeline exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Error exporting preprocessing pipeline: {str(e)}")
            raise
    
    def load_preprocessing_pipeline(self, filepath: str) -> None:
        """
        Load preprocessing pipeline from file
        
        Args:
            filepath (str): Path to pipeline file
        """
        try:
            with open(filepath, 'r') as f:
                pipeline = json.load(f)
            
            # Note: This is a simplified loading - in production you'd need more sophisticated
            # serialization/deserialization for sklearn objects
            
            self.data_info = pipeline.get('data_info', {})
            
            logger.info(f"Preprocessing pipeline loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading preprocessing pipeline: {str(e)}")
            raise
