"""
Custom metrics for SAR flood segmentation evaluation
"""
import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
from typing import Dict
import logging

logger = logging.getLogger(__name__)

class PixelAccuracy(tf.keras.metrics.Metric):
    """Pixel-wise accuracy metric"""
    
    def __init__(self, threshold: float = 0.5, name: str = 'pixel_accuracy', **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.total_correct = self.add_weight(name='total_correct', initializer='zeros')
        self.total_pixels = self.add_weight(name='total_pixels', initializer='zeros')
    
    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred > self.threshold, tf.float32)
        
        correct_predictions = tf.cast(tf.equal(y_true, y_pred), tf.float32)
        
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, tf.float32)
            correct_predictions = tf.multiply(correct_predictions, sample_weight)
        
        self.total_correct.assign_add(tf.reduce_sum(correct_predictions))
        self.total_pixels.assign_add(tf.reduce_sum(tf.ones_like(correct_predictions)))
    
    def result(self):
        return tf.math.divide_no_nan(self.total_correct, self.total_pixels)
    
    def reset_state(self):
        self.total_correct.assign(0)
        self.total_pixels.assign(0)

class DiceCoefficient(tf.keras.metrics.Metric):
    """Dice coefficient metric"""
    
    def __init__(self, threshold: float = 0.5, smooth: float = 1e-6, name: str = 'dice_coefficient', **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.smooth = smooth
        self.intersection_sum = self.add_weight(name='intersection_sum', initializer='zeros')
        self.union_sum = self.add_weight(name='union_sum', initializer='zeros')
    
    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred > self.threshold, tf.float32)
        
        # Flatten tensors
        y_true_flat = tf.reshape(y_true, [-1])
        y_pred_flat = tf.reshape(y_pred, [-1])
        
        # Calculate intersection and union
        intersection = tf.reduce_sum(y_true_flat * y_pred_flat)
        union = tf.reduce_sum(y_true_flat) + tf.reduce_sum(y_pred_flat)
        
        self.intersection_sum.assign_add(intersection)
        self.union_sum.assign_add(union)
    
    def result(self):
        dice = tf.math.divide_no_nan(
            2.0 * self.intersection_sum + self.smooth,
            self.union_sum + self.smooth
        )
        return dice
    
    def reset_state(self):
        self.intersection_sum.assign(0)
        self.union_sum.assign(0)

class Precision(tf.keras.metrics.Metric):
    """Precision metric"""
    
    def __init__(self, threshold: float = 0.5, name: str = 'precision', **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.true_positives = self.add_weight(name='true_positives', initializer='zeros')
        self.false_positives = self.add_weight(name='false_positives', initializer='zeros')
    
    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred > self.threshold, tf.float32)
        
        # Calculate true positives and false positives
        true_positives = tf.reduce_sum(y_true * y_pred)
        false_positives = tf.reduce_sum((1 - y_true) * y_pred)
        
        self.true_positives.assign_add(true_positives)
        self.false_positives.assign_add(false_positives)
    
    def result(self):
        return tf.math.divide_no_nan(
            self.true_positives,
            self.true_positives + self.false_positives
        )
    
    def reset_state(self):
        self.true_positives.assign(0)
        self.false_positives.assign(0)

class Recall(tf.keras.metrics.Metric):
    """Recall metric"""
    
    def __init__(self, threshold: float = 0.5, name: str = 'recall', **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.true_positives = self.add_weight(name='true_positives', initializer='zeros')
        self.false_negatives = self.add_weight(name='false_negatives', initializer='zeros')
    
    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred > self.threshold, tf.float32)
        
        # Calculate true positives and false negatives
        true_positives = tf.reduce_sum(y_true * y_pred)
        false_negatives = tf.reduce_sum(y_true * (1 - y_pred))
        
        self.true_positives.assign_add(true_positives)
        self.false_negatives.assign_add(false_negatives)
    
    def result(self):
        return tf.math.divide_no_nan(
            self.true_positives,
            self.true_positives + self.false_negatives
        )
    
    def reset_state(self):
        self.true_positives.assign(0)
        self.false_negatives.assign(0)

class MetricsCalculator:
    """Calculate all metrics for evaluation"""
    
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate all metrics: pixel accuracy, DICE, precision, recall"""
        # Convert to binary predictions
        y_pred_binary = (y_pred > self.threshold).astype(np.float32)
        y_true = y_true.astype(np.float32)
        
        # Flatten arrays
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred_binary.flatten()
        
        # Calculate confusion matrix components
        tp = np.sum(y_true_flat * y_pred_flat)
        fp = np.sum((1 - y_true_flat) * y_pred_flat)
        fn = np.sum(y_true_flat * (1 - y_pred_flat))
        tn = np.sum((1 - y_true_flat) * (1 - y_pred_flat))
        
        # Calculate metrics
        pixel_accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0
        
        dice_coefficient = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        return {
            'pixel_accuracy': float(pixel_accuracy),
            'dice_coefficient': float(dice_coefficient),
            'precision': float(precision),
            'recall': float(recall)
        }
    
    def calculate_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
        """Calculate confusion matrix components"""
        y_pred_binary = (y_pred > self.threshold).astype(np.float32)
        y_true = y_true.astype(np.float32)
        
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred_binary.flatten()
        
        tp = int(np.sum(y_true_flat * y_pred_flat))
        fp = int(np.sum((1 - y_true_flat) * y_pred_flat))
        fn = int(np.sum(y_true_flat * (1 - y_pred_flat)))
        tn = int(np.sum((1 - y_true_flat) * (1 - y_pred_flat)))
        
        return {
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'true_negatives': tn
        }

def get_metrics(threshold: float = 0.5) -> list:
    """Get metrics"""
    return [
        PixelAccuracy(threshold=threshold),
        DiceCoefficient(threshold=threshold),
        Precision(threshold=threshold),
        Recall(threshold=threshold)
    ]

def calculate_batch_metrics(y_true: tf.Tensor, y_pred: tf.Tensor, threshold: float = 0.5) -> Dict[str, tf.Tensor]:
    """Calculate metrics for a batch during training/validation"""
    y_true = tf.cast(y_true, tf.float32)
    y_pred_binary = tf.cast(y_pred > threshold, tf.float32)
    
    # Flatten tensors
    y_true_flat = tf.reshape(y_true, [-1])
    y_pred_flat = tf.reshape(y_pred_binary, [-1])
    
    # Calculate confusion matrix components
    tp = tf.reduce_sum(y_true_flat * y_pred_flat)
    fp = tf.reduce_sum((1 - y_true_flat) * y_pred_flat)
    fn = tf.reduce_sum(y_true_flat * (1 - y_pred_flat))
    tn = tf.reduce_sum((1 - y_true_flat) * (1 - y_pred_flat))
    
    # Calculate metrics
    pixel_accuracy = tf.math.divide_no_nan(tp + tn, tp + fp + fn + tn)
    dice_coefficient = tf.math.divide_no_nan(2 * tp, 2 * tp + fp + fn)
    precision = tf.math.divide_no_nan(tp, tp + fp)
    recall = tf.math.divide_no_nan(tp, tp + fn)
    
    return {
        'pixel_accuracy': pixel_accuracy,
        'dice_coefficient': dice_coefficient,
        'precision': precision,
        'recall': recall
    }