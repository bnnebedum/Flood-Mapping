"""
U-Net implementation for SAR flood segmentation
"""
import tensorflow as tf
from tensorflow.keras import layers, Model
from typing import Tuple
import logging

from .base_model import BaseModel
from ..utils.config import ModelConfig

logger = logging.getLogger(__name__)

class UNet(BaseModel):
    """U-Net architecture for semantic segmentation"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.config = config
    
    def _conv_block(self, inputs: tf.Tensor, filters: int, name: str) -> tf.Tensor:
        """Convolutional block with two conv layers"""
        x = layers.Conv2D(
            filters, 
            3, 
            padding='same', 
            activation=self.config.activation,
            kernel_initializer=self.config.kernel_initializer,
            name=f"{name}_conv1"
        )(inputs)
        
        if self.config.batch_norm:
            x = layers.BatchNormalization(name=f"{name}_bn1")(x)
        
        x = layers.Conv2D(
            filters, 
            3, 
            padding='same', 
            activation=self.config.activation,
            kernel_initializer=self.config.kernel_initializer,
            name=f"{name}_conv2"
        )(x)
        
        if self.config.batch_norm:
            x = layers.BatchNormalization(name=f"{name}_bn2")(x)
        
        return x
    
    def _encoder_block(self, inputs: tf.Tensor, filters: int, name: str) -> Tuple[tf.Tensor, tf.Tensor]:
        """Encoder block with conv block and max pooling"""
        conv = self._conv_block(inputs, filters, name)
        pool = layers.MaxPooling2D(2, name=f"{name}_pool")(conv)
        return conv, pool
    
    def _decoder_block(self, inputs: tf.Tensor, skip_connection: tf.Tensor, filters: int, name: str) -> tf.Tensor:
        """Decoder block with upsampling and skip connection"""
        # Upsampling
        up = layers.Conv2DTranspose(
            filters, 
            2, 
            strides=2, 
            padding='same',
            kernel_initializer=self.config.kernel_initializer,
            name=f"{name}_upsample"
        )(inputs)
        
        # Skip connection
        concat = layers.Concatenate(name=f"{name}_concat")([up, skip_connection])
        
        # Convolutional block
        conv = self._conv_block(concat, filters, name)
        
        return conv
    
    def _bottleneck(self, inputs: tf.Tensor, filters: int) -> tf.Tensor:
        """Bottleneck layer with dropout"""
        x = self._conv_block(inputs, filters, "bottleneck")
        if self.config.dropout_rate > 0:
            x = layers.Dropout(self.config.dropout_rate, name="bottleneck_dropout")(x)
        return x
    
    def build_model(self) -> Model:
        """Build complete U-Net model"""
        inputs = layers.Input(shape=self.config.input_shape, name="input")
        
        # Calculate filter sizes for each level
        base_filters = self.config.base_filters
        filter_sizes = [base_filters * (2 ** i) for i in range(self.config.depth)]
        
        # Encoder path
        encoder_outputs = []
        x = inputs
        
        for i, filters in enumerate(filter_sizes):
            conv, pool = self._encoder_block(x, filters, f"encoder_{i+1}")
            encoder_outputs.append(conv)
            x = pool
        
        # Bottleneck
        bottleneck_filters = filter_sizes[-1] * 2
        x = self._bottleneck(x, bottleneck_filters)
        
        # Decoder path
        for i in range(self.config.depth - 1, -1, -1):
            filters = filter_sizes[i]
            skip_connection = encoder_outputs[i]
            x = self._decoder_block(x, skip_connection, filters, f"decoder_{i+1}")
        
        # Output layer
        outputs = layers.Conv2D(
            self.config.num_classes,
            1,
            activation='sigmoid',
            kernel_initializer=self.config.kernel_initializer,
            name="output"
        )(x)
        
        model = Model(inputs=inputs, outputs=outputs, name="unet")
        
        logger.info(f"Built U-Net model with {model.count_params():,} parameters")
        
        return model
    
    def get_model_summary(self) -> str:
        """Get model architecture summary"""
        model = self.build_model()
        return model.summary()

class UNetBuilder:
    """Builder class for U-Net with different configurations"""
    
    @staticmethod
    def build_standard_unet(input_shape: Tuple[int, int, int] = (128, 128, 3)) -> Model:
        """Build standard U-Net"""
        config = ModelConfig(
            name="unet_standard",
            input_shape=input_shape,
            base_filters=64,
            depth=4,
            dropout_rate=0.5,
            batch_norm=True
        )
        
        unet = UNet(config)
        return unet.build_model()
    
    @staticmethod
    def build_lightweight_unet(input_shape: Tuple[int, int, int] = (128, 128, 3)) -> Model:
        """Build lightweight U-Net for faster training"""
        config = ModelConfig(
            name="unet_lightweight",
            input_shape=input_shape,
            base_filters=32,
            depth=3,
            dropout_rate=0.3,
            batch_norm=True
        )
        
        unet = UNet(config)
        return unet.build_model()
    
    @staticmethod
    def build_deep_unet(input_shape: Tuple[int, int, int] = (128, 128, 3)) -> Model:
        """Build deeper U-Net for complex patterns"""
        config = ModelConfig(
            name="unet_deep",
            input_shape=input_shape,
            base_filters=64,
            depth=5,
            dropout_rate=0.5,
            batch_norm=True
        )
        
        unet = UNet(config)
        return unet.build_model()