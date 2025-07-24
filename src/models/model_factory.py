"""
Model factory for creating different architectures
"""
from tensorflow.keras import Model
from typing import Dict, Type
import logging

from .base_model import BaseModel
from .unet import UNet
from ..utils.config import ModelConfig

logger = logging.getLogger(__name__)

class ModelFactory:
    """Factory class for creating different model architectures"""
    
    _models: Dict[str, Type[BaseModel]] = {
        'unet': UNet,
        # 'unet_plus_plus': UNetPlusPlus,
        # 'attention_unet': AttentionUNet,
    }
    
    @classmethod
    def register_model(cls, name: str, model_class: Type[BaseModel]):
        """Register a new model architecture"""
        cls._models[name] = model_class
        logger.info(f"Registered model: {name}")
    
    @classmethod
    def create_model(cls, config: ModelConfig) -> Model:
        """Create model based on configuration"""
        model_name = config.name.lower()
        
        if model_name not in cls._models:
            available_models = list(cls._models.keys())
            raise ValueError(f"Unknown model: {model_name}. Available models: {available_models}")
        
        logger.info(f"Creating {model_name} model")
        
        model_class = cls._models[model_name]
        model_instance = model_class(config)
        
        return model_instance.build_model()
    
    @classmethod
    def list_available_models(cls) -> list:
        """List all available model architectures"""
        return list(cls._models.keys())
    
    @classmethod
    def get_model_info(cls, model_name: str) -> dict:
        """Get information about a specific model"""
        if model_name not in cls._models:
            raise ValueError(f"Unknown model: {model_name}")
        
        model_class = cls._models[model_name]
        
        return {
            'name': model_name,
            'class': model_class.__name__,
            'module': model_class.__module__,
            'description': model_class.__doc__ or "No description available"
        }

def create_model_from_name(model_name: str, input_shape: tuple = (128, 128, 3)) -> Model:
    """Convenience function to create model by name with default config"""
    config = ModelConfig(
        name=model_name,
        input_shape=input_shape
    )
    
    return ModelFactory.create_model(config)