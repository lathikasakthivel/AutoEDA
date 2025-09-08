"""
Attention Mechanism for LSTM Autoencoder
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
import logging

logger = logging.getLogger(__name__)

class AttentionMechanism:
    """
    Multi-head attention mechanism for identifying important temporal features
    
    This class implements:
    - Multi-head self-attention
    - Temporal attention weights
    - Feature importance scoring
    - Interpretable attention visualization
    """
    
    def __init__(self, encoder_model, config=None):
        """
        Initialize attention mechanism
        
        Args:
            encoder_model: Trained encoder model
            config (dict): Configuration dictionary
        """
        self.encoder_model = encoder_model
        self.config = config or {}
        self.attention_model = None
        
        # Set default configuration
        self._set_default_config()
        
        # Build attention model
        self._build_attention_model()
        
        logger.info("Attention mechanism initialized successfully")
    
    def _set_default_config(self):
        """Set default configuration if not provided"""
        defaults = {
            'attention_heads': 4,
            'attention_dim': 32,
            'dropout_rate': 0.1,
            'use_positional_encoding': True,
            'normalize_attention': True
        }
        
        for key, value in defaults.items():
            if key not in self.config:
                self.config[key] = value
    
    def _build_attention_model(self):
        """Build the attention model architecture"""
        try:
            # Get encoder output shape
            encoder_output_shape = self.encoder_model.output_shape
            
            # Input layer
            input_layer = layers.Input(shape=encoder_output_shape[1:])
            
            # Multi-head attention
            attention_output = self._multi_head_attention(input_layer)
            
            # Global average pooling to get feature importance
            feature_importance = layers.GlobalAveragePooling1D()(attention_output)
            
            # Create attention model
            self.attention_model = Model(
                inputs=input_layer,
                outputs=[attention_output, feature_importance],
                name='attention_mechanism'
            )
            
            logger.info("Attention model built successfully")
            
        except Exception as e:
            logger.error(f"Error building attention model: {str(e)}")
            raise
    
    def _multi_head_attention(self, input_tensor):
        """Implement multi-head self-attention"""
        # Linear transformations for query, key, value
        query = layers.Dense(self.config['attention_dim'])(input_tensor)
        key = layers.Dense(self.config['attention_dim'])(input_tensor)
        value = layers.Dense(self.config['attention_dim'])(input_tensor)
        
        # Split into multiple heads
        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)
        
        # Scaled dot-product attention
        attention_scores = tf.matmul(query, key, transpose_b=True)
        attention_scores = attention_scores / tf.math.sqrt(tf.cast(self.config['attention_dim'], tf.float32))
        
        # Apply softmax to get attention weights
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)
        
        # Apply dropout
        attention_weights = layers.Dropout(self.config['dropout_rate'])(attention_weights)
        
        # Apply attention weights to values
        attention_output = tf.matmul(attention_weights, value)
        
        # Concatenate heads
        attention_output = self._combine_heads(attention_output)
        
        # Final linear transformation
        attention_output = layers.Dense(input_tensor.shape[-1])(attention_output)
        
        # Add residual connection
        attention_output = layers.Add()([input_tensor, attention_output])
        
        # Layer normalization
        if self.config['normalize_attention']:
            attention_output = layers.LayerNormalization()(attention_output)
        
        return attention_output
    
    def _split_heads(self, tensor):
        """Split tensor into multiple attention heads"""
        batch_size = tf.shape(tensor)[0]
        seq_length = tf.shape(tensor)[1]
        depth = tensor.shape[-1]
        
        # Reshape to (batch_size, seq_length, num_heads, depth_per_head)
        tensor = tf.reshape(tensor, [batch_size, seq_length, self.config['attention_heads'], depth // self.config['attention_heads']])
        
        # Transpose to (batch_size, num_heads, seq_length, depth_per_head)
        tensor = tf.transpose(tensor, [0, 2, 1, 3])
        
        return tensor
    
    def _combine_heads(self, tensor):
        """Combine multiple attention heads back into single tensor"""
        batch_size = tf.shape(tensor)[0]
        seq_length = tf.shape(tensor)[2]
        depth = tensor.shape[-1]
        
        # Transpose back to (batch_size, seq_length, num_heads, depth_per_head)
        tensor = tf.transpose(tensor, [0, 2, 1, 3])
        
        # Reshape to (batch_size, seq_length, total_depth)
        tensor = tf.reshape(tensor, [batch_size, seq_length, depth * self.config['attention_heads']])
        
        return tensor
    
    def get_attention_weights(self, data):
        """
        Get attention weights for input data
        
        Args:
            data (np.ndarray): Input data with shape (samples, timesteps, features)
            
        Returns:
            np.ndarray: Attention weights with shape (samples, timesteps, timesteps)
        """
        try:
            if self.attention_model is None:
                raise ValueError("Attention model not built")
            
            # Get attention weights from the model
            attention_output, feature_importance = self.attention_model.predict(data)
            
            # Calculate attention weights (simplified approach)
            # In a full implementation, you would extract the actual attention weights
            # from the multi-head attention layer
            
            # For now, we'll use a correlation-based approach
            attention_weights = self._calculate_correlation_weights(data)
            
            logger.info(f"Attention weights calculated for data shape: {data.shape}")
            return attention_weights
            
        except Exception as e:
            logger.error(f"Error calculating attention weights: {str(e)}")
            raise
    
    def _calculate_correlation_weights(self, data):
        """
        Calculate correlation-based attention weights
        
        Args:
            data (np.ndarray): Input data
            
        Returns:
            np.ndarray: Correlation-based attention weights
        """
        # Reshape data to (samples * timesteps, features)
        samples, timesteps, features = data.shape
        reshaped_data = data.reshape(-1, features)
        
        # Calculate correlation matrix
        correlation_matrix = np.corrcoef(reshaped_data.T)
        
        # Create attention weights based on correlations
        attention_weights = np.zeros((samples, timesteps, timesteps))
        
        for i in range(samples):
            # Use correlation matrix to create temporal attention weights
            for t1 in range(timesteps):
                for t2 in range(timesteps):
                    if t1 == t2:
                        attention_weights[i, t1, t2] = 1.0  # Self-attention
                    else:
                        # Use feature correlation as attention weight
                        feature_corr = np.mean(np.abs(correlation_matrix))
                        attention_weights[i, t1, t2] = feature_corr
        
        return attention_weights
    
    def get_feature_importance(self, data):
        """
        Get feature importance scores
        
        Args:
            data (np.ndarray): Input data
            
        Returns:
            np.ndarray: Feature importance scores
        """
        try:
            if self.attention_model is None:
                raise ValueError("Attention model not built")
            
            # Get feature importance from attention model
            _, feature_importance = self.attention_model.predict(data)
            
            # Normalize importance scores
            feature_importance = self._normalize_importance_scores(feature_importance)
            
            return feature_importance
            
        except Exception as e:
            logger.error(f"Error calculating feature importance: {str(e)}")
            raise
    
    def _normalize_importance_scores(self, importance_scores):
        """Normalize importance scores to [0, 1] range"""
        min_score = np.min(importance_scores)
        max_score = np.max(importance_scores)
        
        if max_score > min_score:
            normalized_scores = (importance_scores - min_score) / (max_score - min_score)
        else:
            normalized_scores = np.ones_like(importance_scores)
        
        return normalized_scores
    
    def get_temporal_attention(self, data, window_size=10):
        """
        Get temporal attention patterns
        
        Args:
            data (np.ndarray): Input data
            window_size (int): Size of temporal window
            
        Returns:
            dict: Temporal attention patterns
        """
        try:
            attention_weights = self.get_attention_weights(data)
            
            # Calculate temporal patterns
            temporal_patterns = {
                'short_term': self._calculate_short_term_attention(attention_weights, window_size),
                'long_term': self._calculate_long_term_attention(attention_weights, window_size),
                'seasonal': self._calculate_seasonal_attention(attention_weights)
            }
            
            return temporal_patterns
            
        except Exception as e:
            logger.error(f"Error calculating temporal attention: {str(e)}")
            raise
    
    def _calculate_short_term_attention(self, attention_weights, window_size):
        """Calculate short-term temporal attention patterns"""
        samples, timesteps, _ = attention_weights.shape
        
        short_term_attention = np.zeros((samples, timesteps))
        
        for i in range(samples):
            for t in range(timesteps):
                # Calculate attention within window
                start_idx = max(0, t - window_size // 2)
                end_idx = min(timesteps, t + window_size // 2 + 1)
                
                window_attention = attention_weights[i, t, start_idx:end_idx]
                short_term_attention[i, t] = np.mean(window_attention)
        
        return short_term_attention
    
    def _calculate_long_term_attention(self, attention_weights, window_size):
        """Calculate long-term temporal attention patterns"""
        samples, timesteps, _ = attention_weights.shape
        
        long_term_attention = np.zeros((samples, timesteps))
        
        for i in range(samples):
            for t in range(timesteps):
                # Calculate attention outside window (long-term dependencies)
                start_idx = max(0, t - window_size // 2)
                end_idx = min(timesteps, t + window_size // 2 + 1)
                
                # Long-term attention is the complement of short-term
                long_term_weights = np.concatenate([
                    attention_weights[i, t, :start_idx],
                    attention_weights[i, t, end_idx:]
                ])
                
                if len(long_term_weights) > 0:
                    long_term_attention[i, t] = np.mean(long_term_weights)
        
        return long_term_attention
    
    def _calculate_seasonal_attention(self, attention_weights):
        """Calculate seasonal temporal attention patterns"""
        samples, timesteps, _ = attention_weights.shape
        
        # Simple seasonal pattern detection
        seasonal_attention = np.zeros((samples, timesteps))
        
        for i in range(samples):
            # Calculate autocorrelation for seasonal patterns
            for lag in range(1, min(50, timesteps // 2)):
                for t in range(lag, timesteps):
                    seasonal_attention[i, t] += attention_weights[i, t, t - lag]
        
        # Normalize seasonal attention
        seasonal_attention = self._normalize_importance_scores(seasonal_attention)
        
        return seasonal_attention
    
    def visualize_attention(self, attention_weights, sample_idx=0):
        """
        Create attention visualization
        
        Args:
            attention_weights (np.ndarray): Attention weights
            sample_idx (int): Sample index to visualize
            
        Returns:
            dict: Visualization data
        """
        try:
            if sample_idx >= attention_weights.shape[0]:
                sample_idx = 0
            
            # Extract attention weights for the sample
            sample_attention = attention_weights[sample_idx]
            
            # Create heatmap data
            heatmap_data = {
                'z': sample_attention.tolist(),
                'x': list(range(sample_attention.shape[1])),
                'y': list(range(sample_attention.shape[0])),
                'type': 'heatmap',
                'colorscale': 'Viridis'
            }
            
            # Create temporal attention line plot
            temporal_attention = np.mean(sample_attention, axis=1)
            line_data = {
                'x': list(range(len(temporal_attention))),
                'y': temporal_attention.tolist(),
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': 'Temporal Attention'
            }
            
            visualization = {
                'heatmap': heatmap_data,
                'temporal_line': line_data,
                'sample_idx': sample_idx
            }
            
            return visualization
            
        except Exception as e:
            logger.error(f"Error creating attention visualization: {str(e)}")
            raise
    
    def get_attention_summary(self, data):
        """
        Get comprehensive attention summary
        
        Args:
            data (np.ndarray): Input data
            
        Returns:
            dict: Attention summary statistics
        """
        try:
            attention_weights = self.get_attention_weights(data)
            feature_importance = self.get_feature_importance(data)
            temporal_patterns = self.get_temporal_attention(data)
            
            summary = {
                'attention_weights_shape': attention_weights.shape,
                'feature_importance_shape': feature_importance.shape,
                'temporal_patterns': {
                    'short_term_shape': temporal_patterns['short_term'].shape,
                    'long_term_shape': temporal_patterns['long_term'].shape,
                    'seasonal_shape': temporal_patterns['seasonal'].shape
                },
                'statistics': {
                    'mean_attention': float(np.mean(attention_weights)),
                    'std_attention': float(np.std(attention_weights)),
                    'max_attention': float(np.max(attention_weights)),
                    'min_attention': float(np.min(attention_weights))
                }
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error creating attention summary: {str(e)}")
            raise
