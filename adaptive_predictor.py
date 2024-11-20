import torch
import torch.nn as nn
from typing import Dict, List
import yaml
from datetime import datetime
import numpy as np
import requests
import os

class AdaptivePredictor(nn.Module):
    def __init__(self, config_path: str = 'src/config/model_config.yaml'):
        super().__init__()
        # Load configuration from file or use defaults
        self.config = self._load_config(config_path)
        # Initialize ensemble of models
        self.models = self._initialize_models()
        # Create learnable weights for model ensemble (initially uniform)
        self.model_weights = nn.Parameter(torch.ones(len(self.models)) / len(self.models))
        # Setup optimizer for learning model weights
        self.optimizer = torch.optim.Adam([self.model_weights], lr=self.config.get('learning_rate', 0.001))
        # Store prediction history
        self.history = []
        # Cache for last input to use in adaptation
        self.last_input = None
        # Rate at which model adapts to network consensus
        self.network_adaptation_rate = 0.1

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # Return default configuration if file not found
            return {
                "model_params": {
                    # Add your default model parameters here
                }
            }

    def _initialize_models(self) -> List[nn.Module]:
        """Initialize the ensemble of models"""
        models = []
        model_params = self.config.get('model_params', {})
        
        # Create a few different model architectures
        # This is a simple example - adjust based on your needs
        models.append(self._create_model(input_size=model_params.get('input_size', 10),
                                    hidden_size=model_params.get('hidden_size', 32),
                                    num_layers=2))
        models.append(self._create_model(input_size=model_params.get('input_size', 10),
                                    hidden_size=model_params.get('hidden_size', 64),
                                    num_layers=3))
        
        return models

    def _create_model(self, input_size: int, hidden_size: int, num_layers: int) -> nn.Module:
        """Create a single model for the ensemble"""
        layers = []
        current_size = input_size
        
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(current_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            current_size = hidden_size
        
        layers.append(nn.Linear(current_size, 1))
            
        return nn.Sequential(*layers)     
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Store input for later use in adaptation
        self.last_input = x
        predictions = []
        # Get predictions from each model in ensemble
        for model in self.models:
            predictions.append(model(x))
        
        # Compute weighted average of predictions using softmax of weights
        weighted_pred = torch.zeros_like(predictions[0])
        for w, p in zip(torch.softmax(self.model_weights, dim=0), predictions):
            weighted_pred += w * p
        
        return weighted_pred

    def adapt_to_network(self, network_inference: float, local_prediction: float):
        """Adjust model weights based on network consensus"""
        with torch.no_grad():
            # Calculate how far each model's prediction is from network consensus
            model_errors = []
            for model in self.models:
                model_pred = model(self.last_input).item()
                error = abs(network_inference - model_pred)
                model_errors.append(error)
            
            # Convert errors to weights (models with lower errors get higher weights)
            errors_tensor = torch.tensor(model_errors)
            # Add small epsilon to prevent division by zero
            inverse_errors = 1.0 / (errors_tensor + 1e-6)
            new_weights = inverse_errors / inverse_errors.sum()
            
            # Update weights using momentum-like approach
            self.model_weights.data = (
                (1 - self.network_adaptation_rate) * self.model_weights.data +
                self.network_adaptation_rate * new_weights
            )
            
            # Log adaptation details
            self.history.append({
                'timestamp': datetime.now(),
                'network_inference': network_inference,
                'local_prediction': local_prediction,
                'model_errors': model_errors,
                'weights': self.model_weights.detach().numpy()
            })

    def get_prediction_metrics(self) -> Dict:
        """Return current model performance metrics"""
        if not self.history:
            return {}
            
        latest = self.history[-1]
        return {
            'timestamp': latest['timestamp'],
            # How far local prediction deviated from network
            'network_deviation': abs(latest['network_inference'] - latest['local_prediction']),
            'model_weights': latest['weights'].tolist(),
            # Highest weight among models (indicates confidence)
            'confidence': float(torch.softmax(self.model_weights, dim=0).max())
        }

    def update_learning_rate(self, network_performance: float):
        """Adjust learning rate dynamically based on network performance"""
        # Increase/decrease learning rate based on network performance
        self.optimizer.param_groups[0]['lr'] = (
            self.config['learning_rate'] * (1.0 + network_performance)
        )

    def get_inference(self, market_data: torch.Tensor) -> Dict:
        """Make prediction and return with confidence metrics"""
        with torch.no_grad():
            prediction = self.forward(market_data)
            weights = torch.softmax(self.model_weights, dim=0)
            
            return {
                'prediction': float(prediction.item()),
                # Highest weight indicates model confidence
                'confidence': float(weights.max().item()),
                'model_weights': weights.tolist()
            }