"""
Main training class with callbacks and monitoring
"""
import tensorflow as tf
from tensorflow.keras import optimizers
import numpy as np
from pathlib import Path
import json
import time
import logging
from typing import Dict, Optional

from ..utils.config import ExperimentConfig
from ..models.model_factory import ModelFactory
from ..data.dataset import create_datasets
from .losses import get_loss_function, create_loss
from .metrics import get_metrics
from .callbacks import create_training_callbacks

logger = logging.getLogger(__name__)

class FloodSegmentationTrainer:
    """Main trainer class for flood segmentation models"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.model = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.history = None
        
        self.experiment_dir = Path(config.output_dir) / config.experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        self._setup_logging()
        
        tf.random.set_seed(config.random_seed)
        np.random.seed(config.random_seed)
    
    def _setup_logging(self):
        """Setup logging for the experiment"""
        log_file = self.experiment_dir / 'training.log'
        
        # Configure logger
        log_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(log_formatter)
        file_handler.setLevel(logging.INFO)
        
        logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)
    
    def setup_datasets(self):
        """Setup train, validation, and test datasets"""
        logger.info("Setting up datasets...")
        
        self.train_dataset, self.val_dataset, self.test_dataset = create_datasets(self.config.data)
        
        # Calculate steps per epoch
        train_info = self._get_dataset_info('train')
        val_info = self._get_dataset_info('val')
        
        self.steps_per_epoch = train_info['matched_pairs'] // self.config.data.batch_size
        self.validation_steps = val_info['matched_pairs'] // self.config.data.batch_size
        
        logger.info(f"Training steps per epoch: {self.steps_per_epoch}")
        logger.info(f"Validation steps: {self.validation_steps}")
        
        # Save dataset info
        dataset_info = {
            'train': train_info,
            'validation': val_info,
            'test': self._get_dataset_info('test'),
            'steps_per_epoch': self.steps_per_epoch,
            'validation_steps': self.validation_steps
        }
        
        with open(self.experiment_dir / 'dataset_info.json', 'w') as f:
            json.dump(dataset_info, f, indent=2)
    
    def _get_dataset_info(self, split: str) -> dict:
        """Get dataset information for a split"""
        from ..data.dataset import FloodDataset
        dataset_creator = FloodDataset(self.config.data)
        return dataset_creator.get_dataset_info(split)
    
    def setup_model(self):
        """Setup model architecture"""
        logger.info(f"Setting up {self.config.model.name} model...")
        
        # Create model
        self.model = ModelFactory.create_model(self.config.model)
        
        # Setup optimizer
        optimizer = self._get_optimizer()
        
        # Setup loss function
        if self.config.training.use_dice_loss:
            loss_fn = create_loss(dice_weight=self.config.training.dice_weight)
        else:
            loss_fn = get_loss_function(self.config.training.loss_function)
        
        # Setup metrics
        metrics = get_metrics()
        
        self.model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=metrics
        )
        
        logger.info(f"Model compiled with {self.model.count_params():,} parameters")
        
        # Save model summary
        with open(self.experiment_dir / 'model_summary.txt', 'w') as f:
            self.model.summary(print_fn=lambda x: f.write(x + '\n'))
    
    def _get_optimizer(self) -> optimizers.Optimizer:
        """Get optimizer based on configuration"""
        if self.config.training.optimizer.lower() == 'adam':
            return optimizers.Adam(
                learning_rate=self.config.training.initial_lr,
                beta_1=self.config.training.beta_1,
                beta_2=self.config.training.beta_2,
                epsilon=self.config.training.epsilon
            )
        elif self.config.training.optimizer.lower() == 'sgd':
            return optimizers.SGD(learning_rate=self.config.training.initial_lr)
        else:
            raise ValueError(f"Unknown optimizer: {self.config.training.optimizer}")
    
    def train(self) -> Dict:
        """Execute training loop"""
        logger.info("Starting training...")
        
        callback_list = create_training_callbacks(self.config, self.experiment_dir)
        
        start_time = time.time()
        
        self.history = self.model.fit(
            self.train_dataset,
            epochs=self.config.training.epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_data=self.val_dataset,
            validation_steps=self.validation_steps,
            callbacks=callback_list,
            verbose=1
        )
        
        # Record training time
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Save training history
        self._save_training_history(training_time)
        
        return self.history.history
    
    def _save_training_history(self, training_time: float):
        """Save training history and metadata"""
        history_data = {
            'history': self.history.history,
            'training_time_seconds': training_time,
            'epochs_completed': len(self.history.history['loss']),
            'final_metrics': {
                'train_loss': float(self.history.history['loss'][-1]),
                'val_loss': float(self.history.history['val_loss'][-1]),
                'train_pixel_accuracy': float(self.history.history['pixel_accuracy'][-1]),
                'val_pixel_accuracy': float(self.history.history['val_pixel_accuracy'][-1]),
                'train_dice_coefficient': float(self.history.history['dice_coefficient'][-1]),
                'val_dice_coefficient': float(self.history.history['val_dice_coefficient'][-1]),
                'train_precision': float(self.history.history['precision'][-1]),
                'val_precision': float(self.history.history['val_precision'][-1]),
                'train_recall': float(self.history.history['recall'][-1]),
                'val_recall': float(self.history.history['val_recall'][-1])
            }
        }
        
        with open(self.experiment_dir / 'training_history.json', 'w') as f:
            json.dump(history_data, f, indent=2)
        
        logger.info("Training history saved")
    
    def evaluate_test_set(self) -> Dict:
        """Evaluate model on test set"""
        logger.info("Evaluating on test set...")
        
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        # Load best weights if available
        best_weights_path = self.experiment_dir / 'best_model.h5'
        if best_weights_path.exists():
            logger.info("Loading best weights for evaluation")
            self.model.load_weights(str(best_weights_path))
        
        # Evaluate on test set
        test_results = self.model.evaluate(
            self.test_dataset,
            steps=self._get_dataset_info('test')['matched_pairs'] // self.config.data.batch_size,
            verbose=1,
            return_dict=True
        )
        
        logger.info("Test evaluation completed")
        logger.info(f"Test metrics: {test_results}")
        
        with open(self.experiment_dir / 'test_results.json', 'w') as f:
            json.dump(test_results, f, indent=2)
        
        return test_results
    
    def save_model(self, save_weights_only: bool = False):
        """Save trained model"""
        if self.model is None:
            raise ValueError("No model to save")
        
        if save_weights_only:
            model_path = self.experiment_dir / 'final_model_weights.h5'
            self.model.save_weights(str(model_path))
            logger.info(f"Model weights saved to {model_path}")
        else:
            model_path = self.experiment_dir / 'final_model.h5'
            self.model.save(str(model_path))
            logger.info(f"Full model saved to {model_path}")
    
    def load_model(self, model_path: Optional[str] = None):
        """Load trained model"""
        if model_path is None:
            model_path = self.experiment_dir / 'best_model.h5'
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model = tf.keras.models.load_model(str(model_path))
        logger.info(f"Model loaded from {model_path}")
    
    def run_full_experiment(self) -> Dict:
        """Run complete training experiment"""
        logger.info(f"Starting experiment: {self.config.experiment_name}")
        
        # Save configuration
        from ..utils.config import ConfigManager
        ConfigManager.save_config(self.config, self.experiment_dir / 'config.yaml')
        
        # Setup datasets and model
        self.setup_datasets()
        self.setup_model()
        
        # Train model
        training_history = self.train()
        
        # Evaluate on test set
        test_results = self.evaluate_test_set()
        
        # Save final model
        self.save_model(self.config.training.save_weights_only)
        
        # Create experiment summary
        experiment_summary = {
            'experiment_name': self.config.experiment_name,
            'model_name': self.config.model.name,
            'total_parameters': int(self.model.count_params()),
            'training_completed': True,
            'final_training_metrics': training_history,
            'test_metrics': test_results,
            'experiment_directory': str(self.experiment_dir)
        }
        
        with open(self.experiment_dir / 'experiment_summary.json', 'w') as f:
            json.dump(experiment_summary, f, indent=2)
        
        logger.info("Experiment completed successfully")
        
        return experiment_summary