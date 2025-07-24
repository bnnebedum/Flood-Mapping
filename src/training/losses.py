"""
Custom loss functions for SAR flood segmentation
"""
import tensorflow as tf
from tensorflow.keras import backend as K
import logging

logger = logging.getLogger(__name__)

class DiceLoss:
    """Dice loss for segmentation tasks"""
    
    def __init__(self, smooth: float = 1e-6):
        self.smooth = smooth
    
    def __call__(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Calculate Dice loss"""
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # Flatten tensors
        y_true_flat = tf.reshape(y_true, [-1])
        y_pred_flat = tf.reshape(y_pred, [-1])
        
        # Calculate intersection and union
        intersection = tf.reduce_sum(y_true_flat * y_pred_flat)
        union = tf.reduce_sum(y_true_flat) + tf.reduce_sum(y_pred_flat)
        
        # Calculate Dice coefficient
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # Return Dice loss (1 - Dice coefficient)
        return 1.0 - dice

class FocalLoss:
    """Focal loss for addressing class imbalance"""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        self.alpha = alpha
        self.gamma = gamma
    
    def __call__(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Calculate Focal loss"""
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # Clip predictions to prevent log(0)
        epsilon = K.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        # Calculate cross entropy
        ce_loss = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        
        # Calculate p_t
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        
        # Calculate alpha_t
        alpha_t = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        
        # Calculate focal weight
        focal_weight = alpha_t * tf.pow(1 - p_t, self.gamma)
        
        # Calculate focal loss
        focal_loss = focal_weight * ce_loss
        
        return tf.reduce_mean(focal_loss)

class TverskyLoss:
    """Tversky loss for handling class imbalance"""
    
    def __init__(self, alpha: float = 0.7, beta: float = 0.3, smooth: float = 1e-6):
        self.alpha = alpha  # Weight for false positives
        self.beta = beta    # Weight for false negatives
        self.smooth = smooth
    
    def __call__(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Calculate Tversky loss"""
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # Flatten tensors
        y_true_flat = tf.reshape(y_true, [-1])
        y_pred_flat = tf.reshape(y_pred, [-1])
        
        # Calculate true positives, false positives, false negatives
        true_pos = tf.reduce_sum(y_true_flat * y_pred_flat)
        false_neg = tf.reduce_sum(y_true_flat * (1 - y_pred_flat))
        false_pos = tf.reduce_sum((1 - y_true_flat) * y_pred_flat)
        
        tversky = (true_pos + self.smooth) / (true_pos + self.alpha * false_pos + self.beta * false_neg + self.smooth)
        
        return 1.0 - tversky

class CombinedLoss:
    """Combined loss function using multiple loss components"""
    
    def __init__(self, 
                 bce_weight: float = 0.5,
                 dice_weight: float = 0.5,
                 focal_weight: float = 0.0,
                 tversky_weight: float = 0.0):
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.tversky_weight = tversky_weight
        
        # Initialize loss functions
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss()
        self.tversky_loss = TverskyLoss()
        
        # Validate weights
        total_weight = bce_weight + dice_weight + focal_weight + tversky_weight
        if abs(total_weight - 1.0) > 1e-6:
            logger.warning(f"Loss weights sum to {total_weight}, not 1.0")
    
    def __call__(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Calculate combined loss"""
        total_loss = 0.0
        
        # Binary cross-entropy
        if self.bce_weight > 0:
            bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
            total_loss += self.bce_weight * tf.reduce_mean(bce)
        
        # Dice loss
        if self.dice_weight > 0:
            dice = self.dice_loss(y_true, y_pred)
            total_loss += self.dice_weight * dice
        
        # Focal loss
        if self.focal_weight > 0:
            focal = self.focal_loss(y_true, y_pred)
            total_loss += self.focal_weight * focal
        
        # Tversky loss
        if self.tversky_weight > 0:
            tversky = self.tversky_loss(y_true, y_pred)
            total_loss += self.tversky_weight * tversky
        
        return total_loss

def get_loss_function(loss_name: str, **kwargs) -> tf.keras.losses.Loss:
    """Factory function to get loss function by name"""
    loss_functions = {
        'binary_crossentropy': tf.keras.losses.BinaryCrossentropy(),
        'dice': DiceLoss(**kwargs),
        'focal': FocalLoss(**kwargs),
        'tversky': TverskyLoss(**kwargs),
        'combined': CombinedLoss(**kwargs)
    }
    
    if loss_name not in loss_functions:
        raise ValueError(f"Unknown loss function: {loss_name}")
    
    return loss_functions[loss_name]

def create_loss(dice_weight: float = 0.5) -> CombinedLoss:
    """Create the loss function"""
    return CombinedLoss(
        bce_weight=1.0 - dice_weight,
        dice_weight=dice_weight,
        focal_weight=0.0,
        tversky_weight=0.0
    )