"""
TensorFlow dataset creation with augmentation for SAR flood mapping
"""
import tensorflow as tf
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
import rasterio
import logging

from ..utils.config import DataConfig

logger = logging.getLogger(__name__)

class FloodDataset:
    """Dataset class for SAR flood mapping with proper augmentation"""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.normalization_stats = self._load_normalization_stats()
    
    def _load_normalization_stats(self) -> Optional[dict]:
        """Load normalization statistics from preprocessing"""
        # Check in metadata folder first
        stats_path = Path(self.config.drive_path) / "metadata" / "normalization_stats.json"
        if not stats_path.exists():
            # Fallback: check in root of Preprocessed-128
            stats_path = Path(self.config.drive_path) / "normalization_stats.json"
        
        if stats_path.exists():
            import json
            with open(stats_path, 'r') as f:
                return json.load(f)
        else:
            logger.warning("Normalization stats not found. Using default values.")
            return None
    
    def _normalize_sar(self, sar_image: tf.Tensor) -> tf.Tensor:
        """Apply normalization to SAR image using preprocessing statistics"""
        if self.normalization_stats is not None:
            means = tf.constant(self.normalization_stats['mean'], dtype=tf.float32)
            stds = tf.constant(self.normalization_stats['std'], dtype=tf.float32)
            
            # Convert to float and normalize to [0, 1]
            sar_image = tf.cast(sar_image, tf.float32) / 255.0
            
            # Apply log transform (consistent with preprocessing)
            sar_image = tf.math.log1p(sar_image)
            
            # Standardize using preprocessing stats
            means_log = tf.math.log1p(means / 255.0)
            stds_log = stds / 255.0  # Approximate std for log transform
            
            sar_image = (sar_image - means_log) / (stds_log + 1e-8)
        else:
            # Fallback normalization
            sar_image = tf.cast(sar_image, tf.float32) / 255.0
            sar_image = (sar_image - 0.5) / 0.5  # Normalize to [-1, 1]
        
        return sar_image
    
    def _normalize_mask(self, mask: tf.Tensor) -> tf.Tensor:
        """Normalize flood mask to binary [0, 1]"""
        mask = tf.cast(mask, tf.float32)
        mask = tf.where(mask > 0, 1.0, 0.0)
        return mask
    
    def _load_tiff_image_py(self, path_bytes, channels):
        """Load TIFF image using rasterio - pure Python function"""
        path_str = path_bytes.numpy().decode('utf-8') if hasattr(path_bytes, 'numpy') else str(path_bytes)
        
        try:
            with rasterio.open(path_str) as src:
                if channels == 3:
                    # Read all 3 channels for SAR
                    image = src.read([1, 2, 3])
                    image = np.transpose(image, (1, 2, 0))  # HWC format
                else:
                    # Read single channel for mask
                    image = src.read(1)
                    image = np.expand_dims(image, axis=-1)  # Add channel dimension
                
                return image.astype(np.float32)
        
        except Exception as e:
            logger.error(f"Error loading {path_str}: {e}")
            # Return zeros if loading fails
            if channels == 3:
                return np.zeros((self.config.image_size[0], self.config.image_size[1], 3), dtype=np.float32)
            else:
                return np.zeros((self.config.image_size[0], self.config.image_size[1], 1), dtype=np.float32)
    
    def _load_image_pair(self, sar_path: tf.Tensor, flood_path: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Load SAR and flood mask image pair"""
        
        sar_image = tf.py_function(
            func=self._load_tiff_image_py,
            inp=[sar_path, 3],  # 3 channels for SAR
            Tout=tf.float32
        )
        sar_image.set_shape([self.config.image_size[0], self.config.image_size[1], 3])
        
        flood_mask = tf.py_function(
            func=self._load_tiff_image_py,
            inp=[flood_path, 1],  # 1 channel for mask
            Tout=tf.float32
        )
        flood_mask.set_shape([self.config.image_size[0], self.config.image_size[1], 1])
        
        return sar_image, flood_mask
    
    def _augment_pair(self, sar_image: tf.Tensor, flood_mask: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Apply augmentation to SAR image and flood mask pair"""
        if not self.config.enable_augmentation:
            return sar_image, flood_mask
        
        # Concatenate for synchronized augmentation
        combined = tf.concat([sar_image, flood_mask], axis=-1)
        
        if self.config.horizontal_flip:
            combined = tf.image.random_flip_left_right(combined)
        
        if self.config.vertical_flip:
            combined = tf.image.random_flip_up_down(combined)
        
        # Random rotation (90 degree increments to avoid interpolation issues)
        if self.config.rotation_range > 0:
            k = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
            combined = tf.image.rot90(combined, k=k)
        
        # Random crop and resize for shift and zoom effects
        if self.config.width_shift_range > 0 or self.config.height_shift_range > 0 or self.config.zoom_range > 0:
            # Calculate crop size for zoom effect
            zoom_factor = tf.random.uniform([], 1.0 - self.config.zoom_range, 1.0 + self.config.zoom_range)
            crop_height = tf.cast(tf.cast(self.config.image_size[0], tf.float32) / zoom_factor, tf.int32)
            crop_width = tf.cast(tf.cast(self.config.image_size[1], tf.float32) / zoom_factor, tf.int32)
            
            # Ensure crop size doesn't exceed image size
            crop_height = tf.minimum(crop_height, self.config.image_size[0])
            crop_width = tf.minimum(crop_width, self.config.image_size[1])
            
            # Random crop
            combined = tf.image.random_crop(combined, [crop_height, crop_width, 4])
            # Resize back to original size
            combined = tf.image.resize(combined, [self.config.image_size[0], self.config.image_size[1]])
        
        # Split back into SAR and mask
        sar_image = combined[:, :, :3]
        flood_mask = combined[:, :, 3:]
        
        return sar_image, flood_mask
    
    def _process_pair(self, sar_path: tf.Tensor, flood_path: tf.Tensor, training: bool = False) -> Tuple[tf.Tensor, tf.Tensor]:
        """Process a single SAR-flood pair"""
        # Load images
        sar_image, flood_mask = self._load_image_pair(sar_path, flood_path)
        
        # Apply augmentation if training
        if training:
            sar_image, flood_mask = self._augment_pair(sar_image, flood_mask)
        
        # Normalize
        sar_image = self._normalize_sar(sar_image)
        flood_mask = self._normalize_mask(flood_mask)
        
        return sar_image, flood_mask
    
    def create_dataset(self, split: str, training: bool = False) -> tf.data.Dataset:
        """Create TensorFlow dataset for specified split"""
        # Get file paths
        sar_dir = Path(self.config.drive_path) / split / self.config.sar_subdir
        flood_dir = Path(self.config.drive_path) / split / self.config.flood_subdir
        
        sar_files = sorted(list(sar_dir.glob("*.tif")))
        flood_files = sorted(list(flood_dir.glob("*.tif")))
        
        if len(sar_files) == 0 or len(flood_files) == 0:
            raise ValueError(f"No files found in {split} split")
        
        # Match SAR and flood files
        matched_pairs = self._match_file_pairs(sar_files, flood_files)
        
        if len(matched_pairs) == 0:
            raise ValueError(f"No matching SAR-flood pairs found in {split} split")
        
        logger.info(f"Found {len(matched_pairs)} matched pairs in {split} split")
        
        # Create dataset from file paths
        sar_paths = [str(pair[0]) for pair in matched_pairs]
        flood_paths = [str(pair[1]) for pair in matched_pairs]
        
        dataset = tf.data.Dataset.from_tensor_slices((sar_paths, flood_paths))
        
        # Apply processing with proper error handling
        dataset = dataset.map(
            lambda sar_path, flood_path: tf.py_function(
                func=self._process_pair_wrapper,
                inp=[sar_path, flood_path, training],
                Tout=[tf.float32, tf.float32]
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Set shapes explicitly
        dataset = dataset.map(lambda x, y: (
            tf.ensure_shape(x, [self.config.image_size[0], self.config.image_size[1], 3]),
            tf.ensure_shape(y, [self.config.image_size[0], self.config.image_size[1], 1])
        ))
        
        # Configure dataset
        if training:
            dataset = dataset.shuffle(self.config.shuffle_buffer)
            dataset = dataset.repeat()
        
        dataset = dataset.batch(self.config.batch_size)
        dataset = dataset.prefetch(self.config.prefetch_buffer)
        
        return dataset
    
    def _process_pair_wrapper(self, sar_path, flood_path, training):
        """Wrapper for _process_pair to work with tf.py_function"""
        # Load images using rasterio
        sar_image = self._load_tiff_image_py(sar_path, 3)
        flood_mask = self._load_tiff_image_py(flood_path, 1)
        
        # Convert to tensors
        sar_tensor = tf.constant(sar_image)
        flood_tensor = tf.constant(flood_mask)
        
        # Apply augmentation if training
        if training:
            sar_tensor, flood_tensor = self._augment_pair(sar_tensor, flood_tensor)
        
        # Normalize
        sar_tensor = self._normalize_sar(sar_tensor)
        flood_tensor = self._normalize_mask(flood_tensor)
        
        return sar_tensor, flood_tensor
    
    def _match_file_pairs(self, sar_files: List[Path], flood_files: List[Path]) -> List[Tuple[Path, Path]]:
        """Match SAR and flood files based on naming convention"""
        matched_pairs = []
        
        for sar_file in sar_files:
            sar_base = sar_file.stem
            for prefix in ["prep_", "processed_", ""]:
                if sar_base.startswith(prefix):
                    sar_base = sar_base[len(prefix):]
                    break
            
            for flood_file in flood_files:
                flood_base = flood_file.stem
                for prefix in ["prep_", "processed_", ""]:
                    if flood_base.startswith(prefix):
                        flood_base = flood_base[len(prefix):]
                        break
                
                flood_base = flood_base.replace("_flood", "")
                
                # Check for match
                if sar_base == flood_base:
                    matched_pairs.append((sar_file, flood_file))
                    break
                elif (len(sar_base) > 10 and len(flood_base) > 10 and 
                      (sar_base in flood_base or flood_base in sar_base)):
                    matched_pairs.append((sar_file, flood_file))
                    break
        
        return matched_pairs
    
    def get_dataset_info(self, split: str) -> dict:
        """Get information about dataset split"""
        sar_dir = Path(self.config.drive_path) / split / self.config.sar_subdir
        flood_dir = Path(self.config.drive_path) / split / self.config.flood_subdir
        
        sar_files = list(sar_dir.glob("*.tif"))
        flood_files = list(flood_dir.glob("*.tif"))
        matched_pairs = self._match_file_pairs(sar_files, flood_files)
        
        return {
            'split': split,
            'total_sar_files': len(sar_files),
            'total_flood_files': len(flood_files),
            'matched_pairs': len(matched_pairs),
            'sar_dir': str(sar_dir),
            'flood_dir': str(flood_dir)
        }

def create_datasets(config: DataConfig) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Create train, validation, and test datasets"""
    dataset_creator = FloodDataset(config)
    
    train_dataset = dataset_creator.create_dataset('train', training=True)
    val_dataset = dataset_creator.create_dataset('val', training=False)
    test_dataset = dataset_creator.create_dataset('test', training=False)
    
    return train_dataset, val_dataset, test_dataset