"""
LSTM Autoencoder for sequential data analysis
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers, callbacks
from tensorflow.keras.models import load_model
import logging

logger = logging.getLogger(__name__)

class LSTMAutoencoder:
    """
    LSTM Autoencoder for unsupervised learning of sequential patterns
    
    This model consists of:
    - Encoder: LSTM layers that compress sequential data into latent representation
    - Decoder: LSTM layers that reconstruct the original sequence from latent space
    - Attention mechanism integration for interpretability
    """
    
    def __init__(self, input_shape, config=None):
        """
        Initialize LSTM Autoencoder
        
        Args:
            input_shape (tuple): Shape of input data (timesteps, features)
            config (dict): Configuration dictionary for model architecture
        """
        self.input_shape = input_shape
        self.config = config or {}
        self.encoder = None
        self.decoder = None
        self.autoencoder = None
        self.is_trained = False
        
        # Set default configuration
        self._set_default_config()
        
        # Build model
        self._build_model()
        
        logger.info(f"LSTM Autoencoder initialized with input shape: {input_shape}")
    
    def _set_default_config(self):
        """Set default configuration if not provided"""
        defaults = {
            'encoder_units': [64, 32, 16],
            'decoder_units': [16, 32, 64],
            'dropout_rate': 0.2,
            'recurrent_dropout': 0.1,
            'activation': 'tanh',
            'recurrent_activation': 'sigmoid',
            'return_sequences': True,
            'return_state': False,
            'learning_rate': 0.001,
            'loss': 'mse',
            'optimizer': 'adam'
        }
        
        for key, value in defaults.items():
            if key not in self.config:
                self.config[key] = value
    
    def _build_model(self):
        """Build the LSTM autoencoder architecture"""
        try:
            # Input layer
            input_layer = layers.Input(shape=self.input_shape)
            
            # Encoder
            self.encoder = self._build_encoder(input_layer)
            
            # Decoder
            self.decoder = self._build_decoder(self.encoder.output)
            
            # Autoencoder (encoder + decoder)
            self.autoencoder = Model(inputs=input_layer, outputs=self.decoder)
            
            # Compile model
            self._compile_model()
            
            logger.info("LSTM Autoencoder model built successfully")
            
        except Exception as e:
            logger.error(f"Error building model: {str(e)}")
            raise
    
    def _build_encoder(self, input_layer):
        """Build the encoder part of the autoencoder"""
        x = input_layer
        
        # Build encoder layers
        for i, units in enumerate(self.config['encoder_units']):
            return_sequences = i < len(self.config['encoder_units']) - 1
            
            x = layers.LSTM(
                units=units,
                return_sequences=return_sequences,
                return_state=False,
                dropout=self.config['dropout_rate'],
                recurrent_dropout=self.config['recurrent_dropout'],
                activation=self.config['activation'],
                recurrent_activation=self.config['recurrent_activation'],
                name=f'encoder_lstm_{i}'
            )(x)
            
            # Add dropout after each LSTM layer
            if return_sequences:
                x = layers.Dropout(self.config['dropout_rate'])(x)
        
        # Create encoder model
        encoder = Model(inputs=input_layer, outputs=x, name='encoder')
        return encoder
    
    def _build_decoder(self, encoder_output):
        """Build the decoder part of the autoencoder"""
        # Get the shape of encoder output
        if len(encoder_output.shape) == 2:
            # If encoder output is 2D, reshape to 3D for LSTM
            latent_dim = encoder_output.shape[-1]
            x = layers.RepeatVector(self.input_shape[0])(encoder_output)
        else:
            x = encoder_output
        
        # Build decoder layers
        for i, units in enumerate(self.config['decoder_units']):
            # All decoder layers should return sequences for full reconstruction
            x = layers.LSTM(
                units=units,
                return_sequences=True,
                return_state=False,
                dropout=self.config['dropout_rate'],
                recurrent_dropout=self.config['recurrent_dropout'],
                activation=self.config['activation'],
                recurrent_activation=self.config['recurrent_activation'],
                name=f'decoder_lstm_{i}'
            )(x)
            x = layers.Dropout(self.config['dropout_rate'])(x)
        
        # Output layer: TimeDistributed Dense for each timestep
        output_layer = layers.TimeDistributed(
            layers.Dense(
                units=self.input_shape[1],
                activation='linear',
                name='output_dense'
            )
        )(x)
        
        return output_layer
    
    def _compile_model(self):
        """Compile the autoencoder model"""
        # Set optimizer
        if self.config['optimizer'] == 'adam':
            optimizer = optimizers.Adam(learning_rate=self.config['learning_rate'])
        elif self.config['optimizer'] == 'rmsprop':
            optimizer = optimizers.RMSprop(learning_rate=self.config['learning_rate'])
        else:
            optimizer = optimizers.Adam(learning_rate=self.config['learning_rate'])
        
        # Compile model
        self.autoencoder.compile(
            optimizer=optimizer,
            loss=self.config['loss'],
            metrics=['mae']
        )
        
        logger.info(f"Model compiled with optimizer: {self.config['optimizer']}, "
                   f"learning rate: {self.config['learning_rate']}")
    
    def train(self, data, epochs=50, batch_size=32, validation_split=0.2, 
              callbacks_list=None, verbose=1):
        """
        Train the autoencoder model
        
        Args:
            data (np.ndarray): Training data with shape (samples, timesteps, features)
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            validation_split (float): Fraction of data to use for validation
            callbacks_list (list): List of Keras callbacks
            verbose (int): Verbosity level
            
        Returns:
            dict: Training history
        """
        try:
            logger.info(f"Starting training with {epochs} epochs, batch size {batch_size}")
            
            # Prepare callbacks
            if callbacks_list is None:
                callbacks_list = self._get_default_callbacks()
            
            # Train model
            history = self.autoencoder.fit(
                data, data,  # Autoencoder learns to reconstruct input
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=callbacks_list,
                verbose=verbose,
                shuffle=True
            )
            
            self.is_trained = True
            logger.info("Training completed successfully")
            
            return history.history
            
        except Exception as e:
            logger.error(f"Training error: {str(e)}")
            raise
    
    def _get_default_callbacks(self):
        """Get default training callbacks"""
        callbacks_list = [
            # Early stopping to prevent overfitting
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Reduce learning rate when plateau is reached
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            
            # Model checkpoint to save best model
            callbacks.ModelCheckpoint(
                'best_autoencoder.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        return callbacks_list
    
    def encode(self, data):
        """
        Encode data using the trained encoder
        
        Args:
            data (np.ndarray): Input data
            
        Returns:
            np.ndarray: Encoded representation
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before encoding")
        
        return self.encoder.predict(data)
    
    def decode(self, encoded_data):
        """
        Decode data using the trained decoder
        
        Args:
            encoded_data (np.ndarray): Encoded data
            
        Returns:
            np.ndarray: Decoded data
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before decoding")
        
        # Reshape if necessary
        if len(encoded_data.shape) == 2:
            encoded_data = encoded_data.reshape(1, -1)
        
        return self.decoder.predict(encoded_data)
    
    def reconstruct(self, data):
        """
        Reconstruct data using the full autoencoder
        
        Args:
            data (np.ndarray): Input data
            
        Returns:
            np.ndarray: Reconstructed data
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before reconstruction")
        
        return self.autoencoder.predict(data)
    
    def get_reconstruction_error(self, data):
        """
        Calculate reconstruction error for anomaly detection
        
        Args:
            data (np.ndarray): Input data
            
        Returns:
            np.ndarray: Reconstruction error for each sample
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before calculating reconstruction error")
        
        reconstructed = self.reconstruct(data)
        error = np.mean(np.square(data - reconstructed), axis=(1, 2))
        return error
    
    def save(self, filepath):
        """Save the trained model"""
        try:
            self.autoencoder.save(filepath)
            logger.info(f"Model saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    @classmethod
    def load(cls, filepath, config=None):
        """Load a trained model"""
        try:
            # Load the autoencoder
            autoencoder = load_model(filepath)
            
            # Create instance
            instance = cls.__new__(cls)
            instance.autoencoder = autoencoder
            
            # Extract encoder and decoder
            instance.encoder = Model(
                inputs=autoencoder.input,
                outputs=autoencoder.layers[1].output
            )
            
            # Set other attributes
            instance.input_shape = autoencoder.input_shape[1:]
            instance.config = config or {}
            instance.is_trained = True
            
            logger.info(f"Model loaded from {filepath}")
            return instance
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def summary(self):
        """Print model summary"""
        if self.autoencoder:
            self.autoencoder.summary()
        else:
            logger.warning("Model not built yet")
    
    def get_model_info(self):
        """Get information about the model architecture"""
        if not self.autoencoder:
            return {"error": "Model not built yet"}
        
        info = {
            "input_shape": self.input_shape,
            "total_params": self.autoencoder.count_params(),
            "encoder_layers": len(self.encoder.layers),
            "decoder_layers": len(self.decoder.layers) if self.decoder else 0,
            "config": self.config
        }
        
        return info
