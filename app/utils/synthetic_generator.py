"""
Synthetic data generation utilities for AutoEDA
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy import stats

logger = logging.getLogger(__name__)

class SyntheticDataGenerator:
    """
    Synthetic data generation using trained LSTM autoencoder
    
    Implements:
    - Latent space sampling
    - Noise injection for diversity
    - Conditional generation
    - Quality assessment
    - Bias analysis
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize SyntheticDataGenerator
        
        Args:
            config (dict): Configuration dictionary
        """
        self.config = config or {}
        self.generated_data = {}
        self.quality_metrics = {}
        
        # Set default configuration
        self._set_default_config()
        
        logger.info("SyntheticDataGenerator initialized")
    
    def _set_default_config(self):
        """Set default configuration if not provided"""
        defaults = {
            'noise_level': 0.05,
            'temperature': 1.0,
            'diversity_factor': 0.1,
            'min_quality_threshold': 0.7,
            'max_generation_attempts': 10,
            'use_conditional_generation': False,
            'interpolation_steps': 5,
            'augmentation_factor': 2.0
        }
        
        for key, value in defaults.items():
            if key not in self.config:
                self.config[key] = value
    
    def generate_data(self, model, original_data: np.ndarray, 
                     n_samples: int = 1000,
                     method: str = 'latent_sampling',
                     **kwargs) -> np.ndarray:
        """
        Generate synthetic data using the trained model
        
        Args:
            model: Trained LSTM autoencoder model
            original_data (np.ndarray): Original training data
            n_samples (int): Number of synthetic samples to generate
            method (str): Generation method to use
            **kwargs: Additional arguments for specific methods
            
        Returns:
            np.ndarray: Generated synthetic data
        """
        try:
            logger.info(f"Starting synthetic data generation using method: {method}")
            
            if method == 'latent_sampling':
                synthetic_data = self._latent_sampling_generation(
                    model, original_data, n_samples, **kwargs
                )
            elif method == 'noise_injection':
                synthetic_data = self._noise_injection_generation(
                    model, original_data, n_samples, **kwargs
                )
            elif method == 'interpolation':
                synthetic_data = self._interpolation_generation(
                    model, original_data, n_samples, **kwargs
                )
            elif method == 'conditional':
                synthetic_data = self._conditional_generation(
                    model, original_data, n_samples, **kwargs
                )
            else:
                raise ValueError(f"Unknown generation method: {method}")
            
            # Quality assessment
            quality_score = self._assess_quality(original_data, synthetic_data)
            
            # Store results
            self.generated_data[method] = {
                'data': synthetic_data,
                'n_samples': n_samples,
                'quality_score': quality_score,
                'method': method
            }
            
            logger.info(f"Synthetic data generation completed. Quality score: {quality_score:.3f}")
            return synthetic_data
            
        except Exception as e:
            logger.error(f"Error in synthetic data generation: {str(e)}")
            raise
    
    def _latent_sampling_generation(self, model, original_data: np.ndarray, 
                                   n_samples: int, **kwargs) -> np.ndarray:
        """Generate synthetic data by sampling from latent space"""
        try:
            # Encode original data to get latent representations
            latent_representations = model.encode(original_data)
            
            # Get latent space statistics
            latent_mean = np.mean(latent_representations, axis=0)
            latent_std = np.std(latent_representations, axis=0)
            
            # Sample from latent space
            noise_level = kwargs.get('noise_level', self.config['noise_level'])
            temperature = kwargs.get('temperature', self.config['temperature'])
            
            # Generate random samples in latent space
            random_latent = np.random.normal(
                loc=latent_mean,
                scale=latent_std * temperature * (1 + noise_level),
                size=(n_samples, latent_representations.shape[-1])
            )
            
            # Add diversity through controlled randomness
            diversity_factor = kwargs.get('diversity_factor', self.config['diversity_factor'])
            random_latent += np.random.normal(
                loc=0,
                scale=latent_std * diversity_factor,
                size=random_latent.shape
            )
            
            # Decode to generate synthetic data
            synthetic_data = model.decode(random_latent)
            
            # Ensure proper shape
            if len(synthetic_data.shape) == 2:
                # If decoder output is 2D, reshape to 3D
                synthetic_data = synthetic_data.reshape(n_samples, -1, original_data.shape[-1])
            
            return synthetic_data
            
        except Exception as e:
            logger.error(f"Error in latent sampling generation: {str(e)}")
            raise
    
    def _noise_injection_generation(self, model, original_data: np.ndarray, 
                                   n_samples: int, **kwargs) -> np.ndarray:
        """Generate synthetic data by injecting noise into original data"""
        try:
            # Select random samples from original data
            n_original = len(original_data)
            indices = np.random.choice(n_original, size=n_samples, replace=True)
            selected_data = original_data[indices].copy()
            
            # Inject noise
            noise_level = kwargs.get('noise_level', self.config['noise_level'])
            noise = np.random.normal(
                loc=0,
                scale=noise_level,
                size=selected_data.shape
            )
            
            # Add noise to data
            noisy_data = selected_data + noise
            
            # Reconstruct using the model to ensure consistency
            synthetic_data = model.reconstruct(noisy_data)
            
            return synthetic_data
            
        except Exception as e:
            logger.error(f"Error in noise injection generation: {str(e)}")
            raise
    
    def _interpolation_generation(self, model, original_data: np.ndarray, 
                                 n_samples: int, **kwargs) -> np.ndarray:
        """Generate synthetic data by interpolating between original samples"""
        try:
            n_original = len(original_data)
            interpolation_steps = kwargs.get('interpolation_steps', self.config['interpolation_steps'])
            
            # Calculate how many interpolations we can create
            n_interpolations = n_original // 2
            n_per_interpolation = n_samples // n_interpolations
            
            synthetic_samples = []
            
            for i in range(0, n_original - 1, 2):
                if len(synthetic_samples) >= n_samples:
                    break
                
                # Get two consecutive samples
                sample1 = original_data[i]
                sample2 = original_data[i + 1]
                
                # Create interpolation weights
                weights = np.linspace(0, 1, n_per_interpolation + 2)[1:-1]
                
                for weight in weights:
                    # Linear interpolation in data space
                    interpolated = weight * sample1 + (1 - weight) * sample2
                    
                    # Add small noise for diversity
                    noise = np.random.normal(0, self.config['noise_level'], interpolated.shape)
                    interpolated += noise
                    
                    # Reconstruct using model
                    interpolated = interpolated.reshape(1, *interpolated.shape)
                    reconstructed = model.reconstruct(interpolated)
                    
                    synthetic_samples.append(reconstructed[0])
                    
                    if len(synthetic_samples) >= n_samples:
                        break
            
            synthetic_data = np.array(synthetic_samples[:n_samples])
            return synthetic_data
            
        except Exception as e:
            logger.error(f"Error in interpolation generation: {str(e)}")
            raise
    
    def _conditional_generation(self, model, original_data: np.ndarray, 
                               n_samples: int, **kwargs) -> np.ndarray:
        """Generate synthetic data conditioned on specific features"""
        try:
            if not self.config['use_conditional_generation']:
                logger.warning("Conditional generation not enabled, falling back to latent sampling")
                return self._latent_sampling_generation(model, original_data, n_samples, **kwargs)
            
            # This is a simplified conditional generation
            # In a full implementation, you would implement more sophisticated conditioning
            
            # For now, we'll use feature-based conditioning
            feature_conditions = kwargs.get('feature_conditions', {})
            
            if not feature_conditions:
                # No conditions specified, use latent sampling
                return self._latent_sampling_generation(model, original_data, n_samples, **kwargs)
            
            # Generate base synthetic data
            base_synthetic = self._latent_sampling_generation(model, original_data, n_samples, **kwargs)
            
            # Apply feature conditions
            synthetic_data = self._apply_feature_conditions(base_synthetic, feature_conditions)
            
            return synthetic_data
            
        except Exception as e:
            logger.error(f"Error in conditional generation: {str(e)}")
            raise
    
    def _apply_feature_conditions(self, synthetic_data: np.ndarray, 
                                 conditions: Dict[str, Any]) -> np.ndarray:
        """Apply feature-based conditions to synthetic data"""
        try:
            modified_data = synthetic_data.copy()
            
            for feature_name, condition in conditions.items():
                if isinstance(condition, dict):
                    if 'range' in condition:
                        min_val, max_val = condition['range']
                        feature_idx = int(feature_name) if feature_name.isdigit() else 0
                        modified_data[:, :, feature_idx] = np.clip(
                            modified_data[:, :, feature_idx], min_val, max_val
                        )
                    elif 'distribution' in condition:
                        # Apply specific distribution
                        dist_type = condition['distribution']
                        if dist_type == 'normal':
                            mean, std = condition.get('params', [0, 1])
                            feature_idx = int(feature_name) if feature_name.isdigit() else 0
                            modified_data[:, :, feature_idx] = np.random.normal(mean, std, modified_data[:, :, feature_idx].shape)
            
            return modified_data
            
        except Exception as e:
            logger.error(f"Error applying feature conditions: {str(e)}")
            raise
    
    def _assess_quality(self, original_data: np.ndarray, 
                        synthetic_data: np.ndarray) -> float:
        """Assess the quality of generated synthetic data"""
        try:
            # Calculate various quality metrics
            
            # 1. Statistical similarity
            orig_mean = np.mean(original_data, axis=(0, 1))
            orig_std = np.std(original_data, axis=(0, 1))
            synth_mean = np.mean(synthetic_data, axis=(0, 1))
            synth_std = np.std(synthetic_data, axis=(0, 1))
            
            mean_similarity = 1 - np.mean(np.abs(orig_mean - synth_mean) / (np.abs(orig_mean) + 1e-8))
            std_similarity = 1 - np.mean(np.abs(orig_std - synth_std) / (np.abs(orig_std) + 1e-8))
            
            # 2. Distribution similarity (using KL divergence approximation)
            distribution_similarity = self._calculate_distribution_similarity(original_data, synthetic_data)
            
            # 3. Temporal pattern preservation
            temporal_similarity = self._calculate_temporal_similarity(original_data, synthetic_data)
            
            # 4. Feature correlation preservation
            correlation_similarity = self._calculate_correlation_similarity(original_data, synthetic_data)
            
            # Combine metrics
            quality_score = np.mean([
                mean_similarity,
                std_similarity,
                distribution_similarity,
                temporal_similarity,
                correlation_similarity
            ])
            
            # Store quality metrics
            self.quality_metrics = {
                'mean_similarity': float(mean_similarity),
                'std_similarity': float(std_similarity),
                'distribution_similarity': float(distribution_similarity),
                'temporal_similarity': float(temporal_similarity),
                'correlation_similarity': float(correlation_similarity),
                'overall_quality': float(quality_score)
            }
            
            return quality_score
            
        except Exception as e:
            logger.error(f"Error assessing quality: {str(e)}")
            return 0.0
    
    def _calculate_distribution_similarity(self, original_data: np.ndarray, 
                                         synthetic_data: np.ndarray) -> float:
        """Calculate distribution similarity between original and synthetic data"""
        try:
            # Flatten data for distribution analysis
            orig_flat = original_data.reshape(-1)
            synth_flat = synthetic_data.reshape(-1)
            
            # Calculate histogram similarity
            bins = np.linspace(min(orig_flat.min(), synth_flat.min()),
                             max(orig_flat.max(), synth_flat.max()), 50)
            
            orig_hist, _ = np.histogram(orig_flat, bins=bins, density=True)
            synth_hist, _ = np.histogram(synth_flat, bins=bins, density=True)
            
            # Normalize histograms
            orig_hist = orig_hist / (np.sum(orig_hist) + 1e-8)
            synth_hist = synth_hist / (np.sum(synth_hist) + 1e-8)
            
            # Calculate histogram intersection
            intersection = np.minimum(orig_hist, synth_hist)
            similarity = np.sum(intersection)
            
            return similarity
            
        except Exception as e:
            logger.error(f"Error calculating distribution similarity: {str(e)}")
            return 0.0
    
    def _calculate_temporal_similarity(self, original_data: np.ndarray, 
                                      synthetic_data: np.ndarray) -> float:
        """Calculate temporal pattern similarity"""
        try:
            # Calculate autocorrelation for temporal patterns
            def autocorr(x, max_lag=10):
                acf = []
                for lag in range(1, min(max_lag + 1, len(x))):
                    corr = np.corrcoef(x[:-lag], x[lag:])[0, 1]
                    acf.append(corr if not np.isnan(corr) else 0)
                return np.array(acf)
            
            # Calculate autocorrelation for each sample
            orig_acf = np.array([autocorr(orig.mean(axis=1)) for orig in original_data])
            synth_acf = np.array([autocorr(synth.mean(axis=1)) for synth in synthetic_data])
            
            # Compare autocorrelation patterns
            acf_similarity = 1 - np.mean(np.abs(orig_acf - synth_acf))
            
            return max(0, acf_similarity)
            
        except Exception as e:
            logger.error(f"Error calculating temporal similarity: {str(e)}")
            return 0.0
    
    def _calculate_correlation_similarity(self, original_data: np.ndarray, 
                                         synthetic_data: np.ndarray) -> float:
        """Calculate feature correlation similarity"""
        try:
            # Calculate correlation matrices
            orig_reshaped = original_data.reshape(-1, original_data.shape[-1])
            synth_reshaped = synthetic_data.reshape(-1, synthetic_data.shape[-1])
            
            orig_corr = np.corrcoef(orig_reshaped.T)
            synth_corr = np.corrcoef(synth_reshaped.T)
            
            # Compare correlation matrices
            corr_diff = np.abs(orig_corr - synth_corr)
            correlation_similarity = 1 - np.mean(corr_diff)
            
            return max(0, correlation_similarity)
            
        except Exception as e:
            logger.error(f"Error calculating correlation similarity: {str(e)}")
            return 0.0
    
    def analyze_bias(self, original_data: np.ndarray, 
                     synthetic_data: np.ndarray,
                     feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Analyze potential bias in synthetic data generation
        
        Args:
            original_data (np.ndarray): Original training data
            synthetic_data (np.ndarray): Generated synthetic data
            feature_names (list): Names of features for analysis
            
        Returns:
            dict: Bias analysis results
        """
        try:
            bias_analysis = {
                'feature_bias': {},
                'distribution_shift': {},
                'representation_gaps': {},
                'overall_bias_score': 0.0
            }
            
            # Analyze each feature
            n_features = original_data.shape[-1]
            feature_names = feature_names or [f"feature_{i}" for i in range(n_features)]
            
            for i, feature_name in enumerate(feature_names):
                orig_feature = original_data[:, :, i].flatten()
                synth_feature = synthetic_data[:, :, i].flatten()
                
                # Calculate feature-specific bias metrics
                feature_bias = self._calculate_feature_bias(orig_feature, synth_feature)
                bias_analysis['feature_bias'][feature_name] = feature_bias
            
            # Calculate overall bias score
            bias_scores = [bias['bias_score'] for bias in bias_analysis['feature_bias'].values()]
            bias_analysis['overall_bias_score'] = np.mean(bias_scores)
            
            return bias_analysis
            
        except Exception as e:
            logger.error(f"Error in bias analysis: {str(e)}")
            raise
    
    def _calculate_feature_bias(self, orig_feature: np.ndarray, 
                               synth_feature: np.ndarray) -> Dict[str, float]:
        """Calculate bias metrics for a specific feature"""
        try:
            # Statistical bias
            mean_bias = np.abs(np.mean(orig_feature) - np.mean(synth_feature))
            std_bias = np.abs(np.std(orig_feature) - np.std(synth_feature))
            
            # Distribution bias (using KS test)
            ks_statistic, p_value = stats.ks_2samp(orig_feature, synth_feature)
            
            # Quantile bias
            quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
            orig_quantiles = np.quantile(orig_feature, quantiles)
            synth_quantiles = np.quantile(synth_feature, quantiles)
            quantile_bias = np.mean(np.abs(orig_quantiles - synth_quantiles))
            
            # Overall bias score (lower is better)
            bias_score = np.mean([
                mean_bias / (np.abs(np.mean(orig_feature)) + 1e-8),
                std_bias / (np.abs(np.std(orig_feature)) + 1e-8),
                ks_statistic,
                quantile_bias / (np.abs(np.mean(orig_feature)) + 1e-8)
            ])
            
            return {
                'mean_bias': float(mean_bias),
                'std_bias': float(std_bias),
                'ks_statistic': float(ks_statistic),
                'ks_p_value': float(p_value),
                'quantile_bias': float(quantile_bias),
                'bias_score': float(bias_score)
            }
            
        except Exception as e:
            logger.error(f"Error calculating feature bias: {str(e)}")
            return {'bias_score': 1.0}
    
    def get_generation_summary(self) -> Dict[str, Any]:
        """Get summary of all generated synthetic data"""
        summary = {
            'total_generations': len(self.generated_data),
            'methods_used': list(self.generated_data.keys()),
            'quality_scores': {},
            'total_samples': 0
        }
        
        for method, data_info in self.generated_data.items():
            summary['quality_scores'][method] = data_info['quality_score']
            summary['total_samples'] += data_info['n_samples']
        
        return summary
    
    def export_synthetic_data(self, method: str, filepath: str) -> None:
        """Export generated synthetic data to file"""
        try:
            if method not in self.generated_data:
                raise ValueError(f"No synthetic data found for method: {method}")
            
            data_info = self.generated_data[method]
            synthetic_data = data_info['data']
            
            # Save to file
            if filepath.endswith('.npy'):
                np.save(filepath, synthetic_data)
            elif filepath.endswith('.csv'):
                # Flatten and save as CSV
                flattened = synthetic_data.reshape(-1, synthetic_data.shape[-1])
                df = pd.DataFrame(flattened)
                df.to_csv(filepath, index=False)
            else:
                raise ValueError("Unsupported file format. Use .npy or .csv")
            
            logger.info(f"Synthetic data exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Error exporting synthetic data: {str(e)}")
            raise
