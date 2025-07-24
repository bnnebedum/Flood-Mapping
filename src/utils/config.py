"""
Configuration management for flood mapping project
"""
import yaml
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List

@dataclass
class DataConfig:
    """Data configuration parameters"""
    drive_path: str = "/content/drive/MyDrive/Preprocessed-128"
    train_dir: str = "train"
    val_dir: str = "val"
    test_dir: str = "test"
    sar_subdir: str = "sar"
    flood_subdir: str = "flood"
    
    # Data parameters
    image_size: tuple = (128, 128)
    channels: int = 3
    batch_size: int = 16
    shuffle_buffer: int = 1000
    prefetch_buffer: int = 2
    
    # Augmentation
    enable_augmentation: bool = True
    rotation_range: float = 20.0
    width_shift_range: float = 0.1
    height_shift_range: float = 0.1
    horizontal_flip: bool = True
    vertical_flip: bool = True
    zoom_range: float = 0.1

@dataclass
class ModelConfig:
    """Model architecture configuration"""
    name: str = "unet"
    input_shape: tuple = (128, 128, 3)
    num_classes: int = 1
    
    # U-Net specific
    base_filters: int = 64
    depth: int = 4
    dropout_rate: float = 0.5
    batch_norm: bool = True
    activation: str = "relu"
    kernel_initializer: str = "he_normal"

@dataclass
class TrainingConfig:
    """Training configuration parameters"""
    epochs: int = 50
    initial_lr: float = 0.001
    min_lr: float = 1e-7
    patience: int = 10
    factor: float = 0.5
    
    # Loss and metrics
    loss_function: str = "binary_crossentropy"
    use_dice_loss: bool = True
    dice_weight: float = 0.5
    
    # Callbacks
    early_stopping_patience: int = 15
    save_best_only: bool = True
    save_weights_only: bool = False
    
    # Optimizer
    optimizer: str = "adam"
    beta_1: float = 0.9
    beta_2: float = 0.999
    epsilon: float = 1e-7

@dataclass
class EvaluationConfig:
    """Evaluation configuration"""
    metrics: List[str] = field(default_factory=lambda: [
        "pixel_accuracy", "dice_coefficient", "precision", "recall"
    ])
    threshold: float = 0.5
    save_predictions: bool = True
    save_visualizations: bool = True
    visualization_samples: int = 20

@dataclass
class ExperimentConfig:
    """Complete experiment configuration"""
    experiment_name: str = "unet_baseline"
    output_dir: str = "/content/drive/MyDrive/experiments"
    random_seed: int = 42
    
    # Sub-configurations
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

class ConfigManager:
    """Configuration manager for loading and saving configurations"""
    
    @staticmethod
    def load_config(config_path: str) -> ExperimentConfig:
        """Load configuration from YAML file"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return ConfigManager._dict_to_config(config_dict)
    
    @staticmethod
    def save_config(config: ExperimentConfig, config_path: str):
        """Save configuration to YAML file"""
        config_dict = ConfigManager._config_to_dict(config)
        
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    @staticmethod
    def _dict_to_config(config_dict: Dict) -> ExperimentConfig:
        """Convert dictionary to ExperimentConfig"""
        # Create sub-configs
        data_config = DataConfig(**config_dict.get('data', {}))
        model_config = ModelConfig(**config_dict.get('model', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        evaluation_config = EvaluationConfig(**config_dict.get('evaluation', {}))
        
        # Create main config
        main_config = {k: v for k, v in config_dict.items() 
                      if k not in ['data', 'model', 'training', 'evaluation']}
        
        return ExperimentConfig(
            **main_config,
            data=data_config,
            model=model_config,
            training=training_config,
            evaluation=evaluation_config
        )
    
    @staticmethod
    def _config_to_dict(config: ExperimentConfig) -> Dict:
        """Convert ExperimentConfig to dictionary"""
        return {
            'experiment_name': config.experiment_name,
            'output_dir': config.output_dir,
            'random_seed': config.random_seed,
            'data': {
                'drive_path': config.data.drive_path,
                'train_dir': config.data.train_dir,
                'val_dir': config.data.val_dir,
                'test_dir': config.data.test_dir,
                'sar_subdir': config.data.sar_subdir,
                'flood_subdir': config.data.flood_subdir,
                'image_size': list(config.data.image_size),
                'channels': config.data.channels,
                'batch_size': config.data.batch_size,
                'shuffle_buffer': config.data.shuffle_buffer,
                'prefetch_buffer': config.data.prefetch_buffer,
                'enable_augmentation': config.data.enable_augmentation,
                'rotation_range': config.data.rotation_range,
                'width_shift_range': config.data.width_shift_range,
                'height_shift_range': config.data.height_shift_range,
                'horizontal_flip': config.data.horizontal_flip,
                'vertical_flip': config.data.vertical_flip,
                'zoom_range': config.data.zoom_range
            },
            'model': {
                'name': config.model.name,
                'input_shape': list(config.model.input_shape),
                'num_classes': config.model.num_classes,
                'base_filters': config.model.base_filters,
                'depth': config.model.depth,
                'dropout_rate': config.model.dropout_rate,
                'batch_norm': config.model.batch_norm,
                'activation': config.model.activation,
                'kernel_initializer': config.model.kernel_initializer
            },
            'training': {
                'epochs': config.training.epochs,
                'initial_lr': config.training.initial_lr,
                'min_lr': config.training.min_lr,
                'patience': config.training.patience,
                'factor': config.training.factor,
                'loss_function': config.training.loss_function,
                'use_dice_loss': config.training.use_dice_loss,
                'dice_weight': config.training.dice_weight,
                'early_stopping_patience': config.training.early_stopping_patience,
                'save_best_only': config.training.save_best_only,
                'save_weights_only': config.training.save_weights_only,
                'optimizer': config.training.optimizer,
                'beta_1': config.training.beta_1,
                'beta_2': config.training.beta_2,
                'epsilon': config.training.epsilon
            },
            'evaluation': {
                'metrics': config.evaluation.metrics,
                'threshold': config.evaluation.threshold,
                'save_predictions': config.evaluation.save_predictions,
                'save_visualizations': config.evaluation.save_visualizations,
                'visualization_samples': config.evaluation.visualization_samples
            }
        }

def get_default_config(model_name: str = "unet") -> ExperimentConfig:
    """Get default configuration for specified model"""
    config = ExperimentConfig()
    config.model.name = model_name
    config.experiment_name = f"{model_name}_baseline"
    return config