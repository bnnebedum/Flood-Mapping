"""
TensorFlow dataset creation
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
    """Dataset class for SAR flood mapping - loads preprocessed data"""
    
    def __init__(self, config: DataConfig):
        self.config = config
    
    def _load_image_pair(self, sar_path_bytes, flood_path_bytes):
        """Load SAR and flood images without training flag"""
        # Decode bytes to strings
        sar_path = sar_path_bytes.decode('utf-8')
        flood_path = flood_path_bytes.decode('utf-8')
        
        # DEBUG: Print file paths being loaded
        print(f"[DEBUG] Loading SAR: {sar_path}")
        print(f"[DEBUG] Loading Flood: {flood_path}")
        
        try:
            # Load SAR image (already preprocessed)
            with rasterio.open(sar_path) as src:
                sar_data = src.read([1, 2, 3])
                sar_data = np.transpose(sar_data, (1, 2, 0)).astype(np.float32)
            
            # Load flood mask (already preprocessed) 
            with rasterio.open(flood_path) as src:
                flood_data = src.read(1)
                flood_data = np.expand_dims(flood_data, axis=-1).astype(np.float32)
            
            # Ensure correct size
            target_size = (self.config.image_size[0], self.config.image_size[1])
            if sar_data.shape[:2] != target_size:
                logger.warning(f"SAR image shape {sar_data.shape[:2]} != expected {target_size}")
                from scipy.ndimage import zoom
                zoom_factors = (target_size[0] / sar_data.shape[0], 
                               target_size[1] / sar_data.shape[1], 1)
                sar_data = zoom(sar_data, zoom_factors, order=1)
            
            if flood_data.shape[:2] != target_size:
                logger.warning(f"Flood mask shape {flood_data.shape[:2]} != expected {target_size}")
                from scipy.ndimage import zoom
                zoom_factors = (target_size[0] / flood_data.shape[0], 
                               target_size[1] / flood_data.shape[1], 1)
                flood_data = zoom(flood_data, zoom_factors, order=0)
            
            # Ensure exact shapes and types
            sar_data = sar_data.astype(np.float32)
            flood_data = flood_data.astype(np.float32)
            
            # Verify shapes
            expected_sar_shape = target_size + (3,)
            expected_flood_shape = target_size + (1,)
            
            if sar_data.shape != expected_sar_shape:
                sar_data = np.resize(sar_data, expected_sar_shape)
                
            if flood_data.shape != expected_flood_shape:
                flood_data = np.resize(flood_data, expected_flood_shape)
            
            # DEBUG: Print loaded data info
            print(f"[DEBUG] SAR shape: {sar_data.shape}, dtype: {sar_data.dtype}, range: [{sar_data.min():.3f}, {sar_data.max():.3f}]")
            print(f"[DEBUG] Flood shape: {flood_data.shape}, dtype: {flood_data.dtype}, range: [{flood_data.min():.3f}, {flood_data.max():.3f}]")
            
            return sar_data, flood_data
            
        except Exception as e:
            logger.error(f"Error loading {sar_path}: {e}")
            # Return zeros if loading fails
            target_size = (self.config.image_size[0], self.config.image_size[1])
            return (np.zeros(target_size + (3,), dtype=np.float32), 
                   np.zeros(target_size + (1,), dtype=np.float32))
    
    def _augment_pair(self, sar_data, flood_data):
        """Apply augmentation to image pair"""
        if not self.config.enable_augmentation:
            return sar_data, flood_data
            
        # Horizontal flip
        if self.config.horizontal_flip and np.random.random() > 0.5:
            sar_data = np.fliplr(sar_data)
            flood_data = np.fliplr(flood_data)
        
        # Vertical flip
        if self.config.vertical_flip and np.random.random() > 0.5:
            sar_data = np.flipud(sar_data)
            flood_data = np.flipud(flood_data)
        
        # Random 90-degree rotation
        if self.config.rotation_range > 0 and np.random.random() > 0.5:
            k = np.random.randint(0, 4)
            sar_data = np.rot90(sar_data, k)
            flood_data = np.rot90(flood_data, k)
        
        return sar_data, flood_data
    
    def _load_training_pair(self, sar_path_bytes, flood_path_bytes):
        """Load and augment training pair"""
        sar_data, flood_data = self._load_image_pair(sar_path_bytes, flood_path_bytes)
        sar_data, flood_data = self._augment_pair(sar_data, flood_data)
        return sar_data, flood_data
    
    def _load_validation_pair(self, sar_path_bytes, flood_path_bytes):
        """Load validation pair without augmentation"""
        return self._load_image_pair(sar_path_bytes, flood_path_bytes)
    
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
    
    def _load_image_pair_direct(self, sar_path: str, flood_path: str):
        """Load SAR and flood images directly from file paths (no bytes decoding)"""
        try:
            # Load SAR image (already preprocessed)
            with rasterio.open(sar_path) as src:
                sar_data = src.read([1, 2, 3])
                sar_data = np.transpose(sar_data, (1, 2, 0)).astype(np.float32)
            
            # Load flood mask (already preprocessed) 
            with rasterio.open(flood_path) as src:
                flood_data = src.read(1)
                flood_data = np.expand_dims(flood_data, axis=-1).astype(np.float32)
            
            # Ensure correct size
            target_size = (self.config.image_size[0], self.config.image_size[1])
            if sar_data.shape[:2] != target_size:
                logger.warning(f"SAR image shape {sar_data.shape[:2]} != expected {target_size}")
                from scipy.ndimage import zoom
                zoom_factors = (target_size[0] / sar_data.shape[0], 
                               target_size[1] / sar_data.shape[1], 1)
                sar_data = zoom(sar_data, zoom_factors, order=1)
            
            if flood_data.shape[:2] != target_size:
                logger.warning(f"Flood mask shape {flood_data.shape[:2]} != expected {target_size}")
                from scipy.ndimage import zoom
                zoom_factors = (target_size[0] / flood_data.shape[0], 
                               target_size[1] / flood_data.shape[1], 1)
                flood_data = zoom(flood_data, zoom_factors, order=0)
            
            # Ensure exact shapes and types
            sar_data = sar_data.astype(np.float32)
            flood_data = flood_data.astype(np.float32)
            
            # Verify shapes
            expected_sar_shape = target_size + (3,)
            expected_flood_shape = target_size + (1,)
            
            if sar_data.shape != expected_sar_shape:
                sar_data = np.resize(sar_data, expected_sar_shape)
                
            if flood_data.shape != expected_flood_shape:
                flood_data = np.resize(flood_data, expected_flood_shape)
            
            return sar_data, flood_data
            
        except Exception as e:
            logger.error(f"Error loading {sar_path}: {e}")
            # Return zeros if loading fails
            target_size = (self.config.image_size[0], self.config.image_size[1])
            return (np.zeros(target_size + (3,), dtype=np.float32), 
                   np.zeros(target_size + (1,), dtype=np.float32))
    
    def _data_generator(self, matched_pairs, training: bool):
        """Generator function that yields numpy arrays"""
        for sar_file, flood_file in matched_pairs:
            try:
                # Call loading functions directly with file path objects
                sar_path = str(sar_file)
                flood_path = str(flood_file)
                
                # Load images directly without going through the old bytes-based functions
                sar_data, flood_data = self._load_image_pair_direct(sar_path, flood_path)
                
                # Apply augmentation if training
                if training:
                    sar_data, flood_data = self._augment_pair(sar_data, flood_data)
                
                # Ensure data is numpy arrays with correct dtype
                sar_data = np.asarray(sar_data, dtype=np.float32)
                flood_data = np.asarray(flood_data, dtype=np.float32)
                
                # Debug: print first few only to avoid spam
                if sar_file == matched_pairs[0][0]:  # Only print for first file
                    print(f"[GENERATOR DEBUG] SAR type: {type(sar_data)}, dtype: {sar_data.dtype}, shape: {sar_data.shape}")
                    print(f"[GENERATOR DEBUG] Flood type: {type(flood_data)}, dtype: {flood_data.dtype}, shape: {flood_data.shape}")
                
                yield sar_data, flood_data
            except Exception as e:
                logger.error(f"Error loading {sar_file}: {e}")
                # Return zeros if loading fails
                target_size = (self.config.image_size[0], self.config.image_size[1])
                yield (np.zeros(target_size + (3,), dtype=np.float32), 
                       np.zeros(target_size + (1,), dtype=np.float32))
    
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
        
        # DEBUG: Print first few file pairs
        print(f"[DEBUG] First 3 matched pairs in {split}:")
        for i, (sar_file, flood_file) in enumerate(matched_pairs[:3]):
            print(f"  {i+1}. SAR: {sar_file.name}")
            print(f"     Flood: {flood_file.name}")
        
        # DEBUG: Print dataset creation info
        print(f"[DEBUG] Creating dataset for {split} with training={training}")
        print(f"[DEBUG] Total paths: {len(matched_pairs)} matched pairs")
        
        # Create dataset from generator - this avoids tf.py_function completely
        dataset = tf.data.Dataset.from_generator(
            lambda: self._data_generator(matched_pairs, training),
            output_signature=(
                tf.TensorSpec(shape=(self.config.image_size[0], self.config.image_size[1], 3), dtype=tf.float32),
                tf.TensorSpec(shape=(self.config.image_size[0], self.config.image_size[1], 1), dtype=tf.float32)
            )
        )
        
        
        # Configure dataset
        if training:
            dataset = dataset.shuffle(self.config.shuffle_buffer)
            dataset = dataset.repeat()
        
        dataset = dataset.batch(self.config.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
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