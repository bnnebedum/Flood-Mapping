"""
Abstract base class for all model architectures
"""
from abc import ABC, abstractmethod
from tensorflow.keras import Model
from ..utils.config import ModelConfig

class BaseModel(ABC):
    """Abstract base class for all segmentation models"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
    
    @abstractmethod
    def build_model(self) -> Model:
        """Build and return the model"""
        pass
    
    def get_model_name(self) -> str:
        """Get model name"""
        return self.config.name
    
    def get_input_shape(self) -> tuple:
        """Get model input shape"""
        return self.config.input_shape
    
    def get_num_classes(self) -> int:
        """Get number of output classes"""
        return self.config.num_classes