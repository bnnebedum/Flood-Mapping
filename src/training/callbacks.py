"""
Custom callbacks for training monitoring and control
"""
import tensorflow as tf
from tensorflow.keras import callbacks
from pathlib import Path
import json
from typing import List
import logging

from ..utils.config import ExperimentConfig

logger = logging.getLogger(__name__)

class MetricsLogger(callbacks.Callback):
    """Custom callback to log detailed metrics during training"""
    
    def __init__(self, log_dir: Path):
        super().__init__()
        self.log_dir = log_dir
        self.epoch_metrics = []
    
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        
        # Log metrics for this epoch
        epoch_data = {
            'epoch': epoch + 1,
            'timestamp': tf.timestamp().numpy().item(),
            'metrics': {k: float(v) for k, v in logs.items()}
        }
        
        self.epoch_metrics.append(epoch_data)
        
        with open(self.log_dir / 'epoch_metrics.json', 'w') as f:
            json.dump(self.epoch_metrics, f, indent=2)
        
        # Log key metrics
        logger.info(f"Epoch {epoch + 1}: "
                   f"loss={logs.get('loss', 0):.4f}, "
                   f"val_loss={logs.get('val_loss', 0):.4f}, "
                   f"dice={logs.get('dice_coefficient', 0):.4f}, "
                   f"val_dice={logs.get('val_dice_coefficient', 0):.4f}")

class LearningRateLogger(callbacks.Callback):
    """Log learning rate changes"""
    
    def __init__(self, log_dir: Path):
        super().__init__()
        self.log_dir = log_dir
        self.lr_history = []
    
    def on_epoch_end(self, epoch, logs=None):
        current_lr = float(self.model.optimizer.learning_rate.numpy())
        
        lr_data = {
            'epoch': epoch + 1,
            'learning_rate': current_lr
        }
        
        self.lr_history.append(lr_data)
        
        # Save learning rate history
        with open(self.log_dir / 'learning_rate_history.json', 'w') as f:
            json.dump(self.lr_history, f, indent=2)

class ModelCheckpointWithMetrics(callbacks.ModelCheckpoint):
    """Enhanced model checkpoint that saves additional metrics"""
    
    def __init__(self, filepath: str, monitor: str = 'val_dice_coefficient', 
                 mode: str = 'max', save_best_only: bool = True, **kwargs):
        super().__init__(filepath, monitor=monitor, mode=mode, 
                        save_best_only=save_best_only, **kwargs)
        self.best_metrics = {}
    
    def on_epoch_end(self, epoch, logs=None):
        # Call parent method
        super().on_epoch_end(epoch, logs)
        
        if logs is None:
            logs = {}
        
        current = logs.get(self.monitor)
        if current is not None:
            if self.monitor_op(current, self.best):
                self.best_metrics = {
                    'epoch': epoch + 1,
                    'best_metric': self.monitor,
                    'best_value': float(current),
                    'all_metrics': {k: float(v) for k, v in logs.items()}
                }
                
                # Save best metrics
                metrics_file = Path(self.filepath).parent / 'best_metrics.json'
                with open(metrics_file, 'w') as f:
                    json.dump(self.best_metrics, f, indent=2)

def create_training_callbacks(config: ExperimentConfig, experiment_dir: Path) -> List[callbacks.Callback]:
    """Create list of training callbacks based on configuration"""
    callback_list = []
    
    checkpoint_path = experiment_dir / 'best_model.h5'
    model_checkpoint = ModelCheckpointWithMetrics(
        filepath=str(checkpoint_path),
        monitor='val_dice_coefficient',
        mode='max',
        save_best_only=config.training.save_best_only,
        save_weights_only=config.training.save_weights_only,
        verbose=1
    )
    callback_list.append(model_checkpoint)
    
    # Early stopping
    early_stopping = callbacks.EarlyStopping(
        monitor='val_dice_coefficient',
        mode='max',
        patience=config.training.early_stopping_patience,
        restore_best_weights=True,
        verbose=1
    )
    callback_list.append(early_stopping)
    
    # Learning rate reduction
    lr_reducer = callbacks.ReduceLROnPlateau(
        monitor='val_dice_coefficient',
        mode='max',
        factor=config.training.factor,
        patience=config.training.patience,
        min_lr=config.training.min_lr,
        verbose=1
    )
    callback_list.append(lr_reducer)
    
    # TensorBoard logging
    tensorboard_dir = experiment_dir / 'tensorboard'
    tensorboard = callbacks.TensorBoard(
        log_dir=str(tensorboard_dir),
        histogram_freq=1,
        write_graph=True,
        write_images=False,
        update_freq='epoch'
    )
    callback_list.append(tensorboard)
    
    # Custom metrics logger
    metrics_logger = MetricsLogger(experiment_dir)
    callback_list.append(metrics_logger)
    
    # Learning rate logger
    lr_logger = LearningRateLogger(experiment_dir)
    callback_list.append(lr_logger)
    
    # CSV logger for easy plotting
    csv_logger = callbacks.CSVLogger(
        filename=str(experiment_dir / 'training_log.csv'),
        separator=',',
        append=False
    )
    callback_list.append(csv_logger)
    
    logger.info(f"Created {len(callback_list)} training callbacks")
    
    return callback_list

class TrainingProgressCallback(callbacks.Callback):
    """Callback to track and report training progress"""
    
    def __init__(self, total_epochs: int, log_dir: Path):
        super().__init__()
        self.total_epochs = total_epochs
        self.log_dir = log_dir
        self.start_time = None
        
    def on_train_begin(self, logs=None):
        import time
        self.start_time = time.time()
        logger.info(f"Starting training for {self.total_epochs} epochs")
    
    def on_epoch_end(self, epoch, logs=None):
        import time
        elapsed_time = time.time() - self.start_time
        epochs_completed = epoch + 1
        avg_time_per_epoch = elapsed_time / epochs_completed
        estimated_total_time = avg_time_per_epoch * self.total_epochs
        remaining_time = estimated_total_time - elapsed_time
        
        progress = epochs_completed / self.total_epochs * 100
        
        logger.info(f"Progress: {progress:.1f}% ({epochs_completed}/{self.total_epochs}) - "
                   f"Elapsed: {elapsed_time/60:.1f}min - "
                   f"Remaining: {remaining_time/60:.1f}min")
        
        progress_info = {
            'epoch': epochs_completed,
            'total_epochs': self.total_epochs,
            'progress_percent': progress,
            'elapsed_time_minutes': elapsed_time / 60,
            'estimated_remaining_minutes': remaining_time / 60,
            'avg_time_per_epoch_seconds': avg_time_per_epoch
        }
        
        with open(self.log_dir / 'training_progress.json', 'w') as f:
            json.dump(progress_info, f, indent=2)

def create_enhanced_callbacks(config: ExperimentConfig, experiment_dir: Path) -> List[callbacks.Callback]:
    """Create enhanced callback list with additional monitoring"""
    
    callback_list = create_training_callbacks(config, experiment_dir)
    
    progress_callback = TrainingProgressCallback(config.training.epochs, experiment_dir)
    callback_list.append(progress_callback)
    
    return callback_list