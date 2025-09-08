"""
Anomaly detection utilities for AutoEDA
"""

import numpy as np
import pandas as pd
import logging
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from typing import Dict, Any, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class AnomalyDetector:
    """
    Anomaly detection for sequential data using multiple methods
    
    Implements:
    - LSTM reconstruction error-based detection
    - Statistical outlier detection
    - Isolation Forest
    - Local Outlier Factor
    - Elliptic Envelope
    - Ensemble methods for robust detection
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize AnomalyDetector
        
        Args:
            config (dict): Configuration dictionary
        """
        self.config = config or {}
        self.detection_methods = {}
        self.anomaly_scores = {}
        self.thresholds = {}
        
        # Set default configuration
        self._set_default_config()
        
        logger.info("AnomalyDetector initialized")
    
    def _set_default_config(self):
        """Set default configuration if not provided"""
        defaults = {
            'reconstruction_threshold': 0.1,
            'statistical_threshold': 3.0,  # Standard deviations
            'isolation_forest_contamination': 0.1,
            'lof_contamination': 0.1,
            'elliptic_contamination': 0.1,
            'ensemble_voting_threshold': 0.5,
            'window_size': 10,
            'min_anomaly_score': 0.05,
            'use_ensemble': True,
            'methods': ['reconstruction', 'statistical', 'isolation_forest', 'lof']
        }
        
        for key, value in defaults.items():
            if key not in self.config:
                self.config[key] = value
    
    def detect_anomalies(self, model, data: np.ndarray, 
                        threshold: Optional[float] = None,
                        method: str = 'ensemble') -> Dict[str, Any]:
        """
        Detect anomalies in sequential data
        
        Args:
            model: Trained LSTM autoencoder model
            data (np.ndarray): Input data with shape (samples, timesteps, features)
            threshold (float): Anomaly threshold
            method (str): Detection method to use
            
        Returns:
            dict: Anomaly detection results
        """
        try:
            logger.info(f"Starting anomaly detection using method: {method}")
            
            if threshold is None:
                threshold = self.config['reconstruction_threshold']
            
            results = {}
            
            if method == 'ensemble' and self.config['use_ensemble']:
                results = self._ensemble_detection(model, data, threshold)
            elif method == 'reconstruction':
                results = self._reconstruction_based_detection(model, data, threshold)
            elif method == 'statistical':
                results = self._statistical_detection(data, threshold)
            elif method == 'isolation_forest':
                results = self._isolation_forest_detection(data)
            elif method == 'lof':
                results = self._lof_detection(data)
            elif method == 'elliptic_envelope':
                results = self._elliptic_envelope_detection(data)
            else:
                raise ValueError(f"Unknown detection method: {method}")
            
            # Store results
            self.anomaly_scores[method] = results
            
            logger.info(f"Anomaly detection completed. Found {results['n_anomalies']} anomalies")
            return results
            
        except Exception as e:
            logger.error(f"Error in anomaly detection: {str(e)}")
            raise
    
    def _ensemble_detection(self, model, data: np.ndarray, 
                           threshold: float) -> Dict[str, Any]:
        """Ensemble anomaly detection using multiple methods"""
        try:
            # Get individual method results
            methods_results = {}
            
            # Reconstruction-based detection
            if 'reconstruction' in self.config['methods']:
                methods_results['reconstruction'] = self._reconstruction_based_detection(
                    model, data, threshold
                )
            
            # Statistical detection
            if 'statistical' in self.config['methods']:
                methods_results['statistical'] = self._statistical_detection(
                    data, self.config['statistical_threshold']
                )
            
            # Isolation Forest
            if 'isolation_forest' in self.config['methods']:
                methods_results['isolation_forest'] = self._isolation_forest_detection(data)
            
            # Local Outlier Factor
            if 'lof' in self.config['methods']:
                methods_results['lof'] = self._lof_detection(data)
            
            # Combine results using voting
            ensemble_results = self._combine_detection_results(methods_results)
            
            return ensemble_results
            
        except Exception as e:
            logger.error(f"Error in ensemble detection: {str(e)}")
            raise
    
    def _reconstruction_based_detection(self, model, data: np.ndarray, 
                                      threshold: float) -> Dict[str, Any]:
        """Anomaly detection based on LSTM reconstruction error"""
        try:
            # Get reconstruction error
            reconstruction_error = model.get_reconstruction_error(data)
            
            # Calculate threshold (can be adaptive)
            if isinstance(threshold, str) and threshold == 'adaptive':
                threshold = np.percentile(reconstruction_error, 95)
            
            # Detect anomalies
            anomaly_mask = reconstruction_error > threshold
            anomaly_indices = np.where(anomaly_mask)[0]
            
            # Calculate anomaly scores (normalized)
            anomaly_scores = (reconstruction_error - np.min(reconstruction_error)) / \
                           (np.max(reconstruction_error) - np.min(reconstruction_error))
            
            results = {
                'method': 'reconstruction',
                'anomaly_mask': anomaly_mask,
                'anomaly_indices': anomaly_indices.tolist(),
                'anomaly_scores': anomaly_scores.tolist(),
                'reconstruction_error': reconstruction_error.tolist(),
                'threshold': float(threshold),
                'n_anomalies': int(np.sum(anomaly_mask)),
                'anomaly_ratio': float(np.mean(anomaly_mask))
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error in reconstruction-based detection: {str(e)}")
            raise
    
    def _statistical_detection(self, data: np.ndarray, 
                              threshold: float) -> Dict[str, Any]:
        """Statistical outlier detection using Z-score"""
        try:
            # Reshape data for statistical analysis
            samples, timesteps, features = data.shape
            reshaped_data = data.reshape(-1, features)
            
            # Calculate Z-scores for each feature
            z_scores = np.abs((reshaped_data - np.mean(reshaped_data, axis=0)) / 
                             np.std(reshaped_data, axis=0))
            
            # Detect outliers based on Z-score threshold
            outlier_mask = np.any(z_scores > threshold, axis=1)
            outlier_indices = np.where(outlier_mask)[0]
            
            # Calculate anomaly scores
            max_z_score = np.max(z_scores, axis=1)
            anomaly_scores = np.clip(max_z_score / threshold, 0, 1)
            
            # Reshape back to original dimensions
            outlier_mask = outlier_mask.reshape(samples, timesteps)
            anomaly_scores = anomaly_scores.reshape(samples, timesteps)
            
            results = {
                'method': 'statistical',
                'anomaly_mask': outlier_mask.tolist(),
                'anomaly_indices': outlier_indices.tolist(),
                'anomaly_scores': anomaly_scores.tolist(),
                'z_scores': z_scores.reshape(samples, timesteps, features).tolist(),
                'threshold': float(threshold),
                'n_anomalies': int(np.sum(outlier_mask)),
                'anomaly_ratio': float(np.mean(outlier_mask))
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error in statistical detection: {str(e)}")
            raise
    
    def _isolation_forest_detection(self, data: np.ndarray) -> Dict[str, Any]:
        """Anomaly detection using Isolation Forest"""
        try:
            # Reshape data for Isolation Forest
            samples, timesteps, features = data.shape
            reshaped_data = data.reshape(-1, features)
            
            # Initialize and fit Isolation Forest
            iso_forest = IsolationForest(
                contamination=self.config['isolation_forest_contamination'],
                random_state=42
            )
            
            # Fit and predict
            iso_forest.fit(reshaped_data)
            predictions = iso_forest.predict(reshaped_data)
            scores = iso_forest.score_samples(reshaped_data)
            
            # Convert predictions to anomaly mask (-1 for anomaly, 1 for normal)
            anomaly_mask = (predictions == -1).reshape(samples, timesteps)
            anomaly_indices = np.where(anomaly_mask.flatten())[0]
            
            # Normalize scores to [0, 1] range
            anomaly_scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
            anomaly_scores = anomaly_scores.reshape(samples, timesteps)
            
            results = {
                'method': 'isolation_forest',
                'anomaly_mask': anomaly_mask.tolist(),
                'anomaly_indices': anomaly_indices.tolist(),
                'anomaly_scores': anomaly_scores.tolist(),
                'raw_scores': scores.reshape(samples, timesteps).tolist(),
                'n_anomalies': int(np.sum(anomaly_mask)),
                'anomaly_ratio': float(np.mean(anomaly_mask))
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error in Isolation Forest detection: {str(e)}")
            raise
    
    def _lof_detection(self, data: np.ndarray) -> Dict[str, Any]:
        """Anomaly detection using Local Outlier Factor"""
        try:
            # Reshape data for LOF
            samples, timesteps, features = data.shape
            reshaped_data = data.reshape(-1, features)
            
            # Initialize and fit LOF
            lof = LocalOutlierFactor(
                contamination=self.config['lof_contamination'],
                n_neighbors=min(20, len(reshaped_data) // 10),
                novelty=False
            )
            
            # Fit and predict
            predictions = lof.fit_predict(reshaped_data)
            scores = lof.negative_outlier_factor_
            
            # Convert predictions to anomaly mask (-1 for anomaly, 1 for normal)
            anomaly_mask = (predictions == -1).reshape(samples, timesteps)
            anomaly_indices = np.where(anomaly_mask.flatten())[0]
            
            # Normalize scores to [0, 1] range
            anomaly_scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
            anomaly_scores = anomaly_scores.reshape(samples, timesteps)
            
            results = {
                'method': 'lof',
                'anomaly_mask': anomaly_mask.tolist(),
                'anomaly_indices': anomaly_indices.tolist(),
                'anomaly_scores': anomaly_scores.tolist(),
                'raw_scores': scores.reshape(samples, timesteps).tolist(),
                'n_anomalies': int(np.sum(anomaly_mask)),
                'anomaly_ratio': float(np.mean(anomaly_mask))
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error in LOF detection: {str(e)}")
            raise
    
    def _elliptic_envelope_detection(self, data: np.ndarray) -> Dict[str, Any]:
        """Anomaly detection using Elliptic Envelope"""
        try:
            # Reshape data for Elliptic Envelope
            samples, timesteps, features = data.shape
            reshaped_data = data.reshape(-1, features)
            
            # Initialize and fit Elliptic Envelope
            elliptic = EllipticEnvelope(
                contamination=self.config['elliptic_contamination'],
                random_state=42
            )
            
            # Fit and predict
            elliptic.fit(reshaped_data)
            predictions = elliptic.predict(reshaped_data)
            scores = elliptic.score_samples(reshaped_data)
            
            # Convert predictions to anomaly mask (-1 for anomaly, 1 for normal)
            anomaly_mask = (predictions == -1).reshape(samples, timesteps)
            anomaly_indices = np.where(anomaly_mask.flatten())[0]
            
            # Normalize scores to [0, 1] range
            anomaly_scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
            anomaly_scores = anomaly_scores.reshape(samples, timesteps)
            
            results = {
                'method': 'elliptic_envelope',
                'anomaly_mask': anomaly_mask.tolist(),
                'anomaly_indices': anomaly_indices.tolist(),
                'anomaly_scores': anomaly_scores.tolist(),
                'raw_scores': scores.reshape(samples, timesteps).tolist(),
                'n_anomalies': int(np.sum(anomaly_mask)),
                'anomaly_ratio': float(np.mean(anomaly_mask))
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error in Elliptic Envelope detection: {str(e)}")
            raise
    
    def _combine_detection_results(self, methods_results: Dict[str, Any]) -> Dict[str, Any]:
        """Combine results from multiple detection methods using voting"""
        try:
            # Get all anomaly masks
            anomaly_masks = []
            anomaly_scores = []
            
            for method, results in methods_results.items():
                if 'anomaly_mask' in results:
                    anomaly_masks.append(np.array(results['anomaly_mask']))
                if 'anomaly_scores' in results:
                    anomaly_scores.append(np.array(results['anomaly_scores']))
            
            if not anomaly_masks:
                raise ValueError("No valid detection results to combine")
            
            # Convert to numpy arrays
            anomaly_masks = np.array(anomaly_masks)
            anomaly_scores = np.array(anomaly_scores)
            
            # Voting-based ensemble
            voting_threshold = self.config['ensemble_voting_threshold']
            ensemble_mask = np.mean(anomaly_masks, axis=0) >= voting_threshold
            
            # Average anomaly scores
            ensemble_scores = np.mean(anomaly_scores, axis=0)
            
            # Get anomaly indices
            anomaly_indices = np.where(ensemble_mask.flatten())[0]
            
            # Calculate ensemble statistics
            n_anomalies = int(np.sum(ensemble_mask))
            anomaly_ratio = float(np.mean(ensemble_mask))
            
            # Calculate method agreement
            method_agreement = np.mean(anomaly_masks, axis=0)
            
            results = {
                'method': 'ensemble',
                'anomaly_mask': ensemble_mask.tolist(),
                'anomaly_indices': anomaly_indices.tolist(),
                'anomaly_scores': ensemble_scores.tolist(),
                'method_agreement': method_agreement.tolist(),
                'n_anomalies': n_anomalies,
                'anomaly_ratio': anomaly_ratio,
                'methods_used': list(methods_results.keys()),
                'voting_threshold': voting_threshold
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error combining detection results: {str(e)}")
            raise
    
    def get_anomaly_summary(self, detection_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate summary statistics for anomaly detection results
        
        Args:
            detection_results (dict): Results from anomaly detection
            
        Returns:
            dict: Summary statistics
        """
        try:
            summary = {
                'method': detection_results.get('method', 'unknown'),
                'total_samples': len(detection_results.get('anomaly_mask', [])),
                'n_anomalies': detection_results.get('n_anomalies', 0),
                'anomaly_ratio': detection_results.get('anomaly_ratio', 0.0),
                'threshold': detection_results.get('threshold', None),
                'anomaly_distribution': {
                    'min_score': float(np.min(detection_results.get('anomaly_scores', [0]))),
                    'max_score': float(np.max(detection_results.get('anomaly_scores', [0]))),
                    'mean_score': float(np.mean(detection_results.get('anomaly_scores', [0]))),
                    'std_score': float(np.std(detection_results.get('anomaly_scores', [0])))
                }
            }
            
            # Add temporal analysis if available
            if 'anomaly_mask' in detection_results:
                anomaly_mask = np.array(detection_results['anomaly_mask'])
                if len(anomaly_mask.shape) > 1:
                    # Temporal distribution
                    temporal_anomalies = np.sum(anomaly_mask, axis=0)
                    summary['temporal_distribution'] = {
                        'max_anomalies_per_timestep': int(np.max(temporal_anomalies)),
                        'min_anomalies_per_timestep': int(np.min(temporal_anomalies)),
                        'mean_anomalies_per_timestep': float(np.mean(temporal_anomalies))
                    }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating anomaly summary: {str(e)}")
            raise
    
    def visualize_anomalies(self, data: np.ndarray, 
                           detection_results: Dict[str, Any],
                           sample_idx: int = 0) -> Dict[str, Any]:
        """
        Create visualization data for anomaly detection results
        
        Args:
            data (np.ndarray): Original data
            detection_results (dict): Anomaly detection results
            sample_idx (int): Sample index to visualize
            
        Returns:
            dict: Visualization data
        """
        try:
            if sample_idx >= data.shape[0]:
                sample_idx = 0
            
            # Extract data for the sample
            sample_data = data[sample_idx]
            sample_anomaly_mask = np.array(detection_results['anomaly_mask'])[sample_idx]
            sample_anomaly_scores = np.array(detection_results['anomaly_scores'])[sample_idx]
            
            # Create time series plot with anomalies highlighted
            time_series_data = {
                'x': list(range(len(sample_data))),
                'y': sample_data.mean(axis=1).tolist(),  # Average across features
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': 'Data'
            }
            
            # Highlight anomalies
            anomaly_indices = np.where(sample_anomaly_mask)[0]
            if len(anomaly_indices) > 0:
                anomaly_data = {
                    'x': anomaly_indices.tolist(),
                    'y': [sample_data[i].mean() for i in anomaly_indices],
                    'type': 'scatter',
                    'mode': 'markers',
                    'name': 'Anomalies',
                    'marker': {'color': 'red', 'size': 10}
                }
            else:
                anomaly_data = None
            
            # Anomaly scores plot
            scores_data = {
                'x': list(range(len(sample_anomaly_scores))),
                'y': sample_anomaly_scores.tolist(),
                'type': 'scatter',
                'mode': 'lines',
                'name': 'Anomaly Score',
                'yaxis': 'y2'
            }
            
            visualization = {
                'time_series': time_series_data,
                'anomalies': anomaly_data,
                'anomaly_scores': scores_data,
                'sample_idx': sample_idx,
                'n_anomalies': int(np.sum(sample_anomaly_mask))
            }
            
            return visualization
            
        except Exception as e:
            logger.error(f"Error creating anomaly visualization: {str(e)}")
            raise
    
    def export_results(self, detection_results: Dict[str, Any], 
                      filepath: str) -> None:
        """
        Export anomaly detection results to file
        
        Args:
            detection_results (dict): Detection results
            filepath (str): Path to save results
        """
        try:
            # Convert numpy arrays to lists for JSON serialization
            export_data = {}
            for key, value in detection_results.items():
                if isinstance(value, np.ndarray):
                    export_data[key] = value.tolist()
                else:
                    export_data[key] = value
            
            # Save to file
            if filepath.endswith('.json'):
                import json
                with open(filepath, 'w') as f:
                    json.dump(export_data, f, indent=2)
            elif filepath.endswith('.csv'):
                # Save anomaly mask as CSV
                anomaly_df = pd.DataFrame(detection_results['anomaly_mask'])
                anomaly_df.to_csv(filepath, index=False)
            else:
                raise ValueError("Unsupported file format. Use .json or .csv")
            
            logger.info(f"Anomaly detection results exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Error exporting results: {str(e)}")
            raise
