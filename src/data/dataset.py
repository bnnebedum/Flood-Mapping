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
        stats_path = Path(self.config.drive_path) / "metadata" / "normalization_stats.json"
        if not stats_path.exists():
            stats_path = Path(self.config.drive_path) / "normalization_stats.json"
        
        if stats_path.exists():
            import json
            with open(stats_path, 'r') as f:
                return json.load(f)
        else:
            logger.warning("Normalization stats not found. Using default values.")
            return None
    
    def _process_pair_numpy(self, sar_path_bytes, flood_path_bytes, training_flag):
        """Process SAR-flood pair using NumPy"""
        # Decode paths
        sar_path = sar_path_bytes.numpy().decode('utf-8') if hasattr(sar_path_bytes, 'numpy') else str(sar_path_bytes)
        flood_path = flood_path_bytes.numpy().decode('utf-8') if hasattr(flood_path_bytes, 'numpy') else str(flood_path_bytes)
        training = training_flag.numpy() if hasattr(training_flag, 'numpy') else bool(training_flag)
        
        try:
            # Load SAR image
            with rasterio.open(sar_path) as src:
                sar_data = src.read([1, 2, 3])
                sar_data = np.transpose(sar_data, (1, 2, 0)).astype(np.float32)
            
            # Load flood mask
            with rasterio.open(flood_path) as src:
                flood_data = src.read(1)
                flood_data = np.expand_dims(flood_data, axis=-1).astype(np.float32)
            
            # Ensure correct size
            target_size = (self.config.image_size[0], self.config.image_size[1])
            if sar_data.shape[:2] != target_size:
                from scipy.ndimage import zoom
                zoom_factors = (target_size[0] / sar_data.shape[0], 
                               target_size[1] / sar_data.shape[1], 1)
                sar_data = zoom(sar_data, zoom_factors, order=1)
            
            if flood_data.shape[:2] != target_size:
                from scipy.ndimage import zoom
                zoom_factors = (target_size[0] / flood_data.shape[0], 
                               target_size[1] / flood_data.shape[1], 1)
                flood_data = zoom(flood_data, zoom_factors, order=0)
            
            # Apply simple augmentation with NumPy if training
            if training:
                if self.config.horizontal_flip and np.random.random() > 0.5:
                    sar_data = np.fliplr(sar_data)
                    flood_data = np.fliplr(flood_data)
                
                if self.config.vertical_flip and np.random.random() > 0.5:
                    sar_data = np.flipud(sar_data)
                    flood_data = np.flipud(flood_data)
                
                # Random 90-degree rotation
                if self.config.rotation_range > 0 and np.random.random() > 0.5:
                    k = np.random.randint(0, 4)
                    sar_data = np.rot90(sar_data, k)
                    flood_data = np.rot90(flood_data, k)
            
            # Normalize SAR using NumPy
            if self.normalization_stats is not None:
                means = np.array(self.normalization_stats['mean'], dtype=np.float32)
                stds = np.array(self.normalization_stats['std'], dtype=np.float32)
                
                # Normalize to [0, 1]
                sar_data = sar_data / 255.0
                
                # Apply log transform
                sar_data = np.log1p(sar_data)
                
                # Standardize
                means_log = np.log1p(means / 255.0)
                stds_log = stds / 255.0
                sar_data = (sar_data - means_log) / (stds_log + 1e-8)
            else:
                # Fallback normalization
                sar_data = sar_data / 255.0
                sar_data = (sar_data - 0.5) / 0.5
            
            # Normalize flood mask to binary
            flood_data = (flood_data > 0).astype(np.float32)
            
            return sar_data.astype(np.float32), flood_data.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error loading {sar_path}: {e}")
            # Return zeros if loading fails
            return (np.zeros(target_size + (3,), dtype=np.float32), 
                   np.zeros(target_size + (1,), dtype=np.float32))
    
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
                
                if sar_base == flood_base:
                    matched_pairs.append((sar_file, flood_file))
                    break
                elif (len(sar_base) > 10 and len(flood_base) > 10 and 
                      (sar_base in flood_base or flood_base in sar_base)):
                    matched_pairs.append((sar_file, flood_file))
                    break
        
        return matched_pairs
    
    def create_dataset(self, split: str, training: bool = False) -> tf.data.Dataset:
        """Create TensorFlow dataset for specified split"""
        sar_dir = Path(self.config.drive_path) / split / self.config.sar_subdir
        flood_dir = Path(self.config.drive_path) / split / self.config.flood_subdir
        
        sar_files = sorted(list(sar_dir.glob("*.tif")))
        flood_files = sorted(list(flood_dir.glob("*.tif")))
        
        if len(sar_files) == 0 or len(flood_files) == 0:
            raise ValueError(f"No files found in {split} split")
        
        matched_pairs = self._match_file_pairs(sar_files, flood_files)
        
        if len(matched_pairs) == 0:
            raise ValueError(f"No matching SAR-flood pairs found in {split} split")
        
        logger.info(f"Found {len(matched_pairs)} matched pairs in {split} split")
        
        # Create dataset from file paths
        sar_paths = [str(pair[0]) for pair in matched_pairs]
        flood_paths = [str(pair[1]) for pair in matched_pairs]
        
        dataset = tf.data.Dataset.from_tensor_slices((sar_paths, flood_paths))
        
        dataset = dataset.map(
            lambda sar_path, flood_path: tf.py_function(
                func=self._process_pair_numpy,
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