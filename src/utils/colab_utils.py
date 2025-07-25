"""
Colab-specific utilities for setup and data management
"""
import os
import subprocess
import sys
from pathlib import Path
import json
import zipfile
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class ColabSetup:
    """Setup utilities for Google Colab environment"""
    
    @staticmethod
    def mount_drive():
        """Mount Google Drive in Colab"""
        try:
            from google.colab import drive
            drive.mount('/content/drive')
            logger.info("Google Drive mounted successfully")
            return True
        except ImportError:
            logger.warning("Not running in Colab environment")
            return False
        except Exception as e:
            logger.error(f"Failed to mount Google Drive: {e}")
            return False
    
    @staticmethod
    def install_requirements(requirements_file: Optional[str] = None):
        """Install required packages in Colab"""
        if requirements_file and Path(requirements_file).exists():
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_file])
            logger.info(f"Installed requirements from {requirements_file}")
        else:
            # Install essential packages
            packages = [
                "tensorflow>=2.10.0",
                "rasterio",
                "matplotlib",
                "numpy",
                "scipy",
                "scikit-learn",
                "pyyaml",
                "tqdm"
            ]
            
            for package in packages:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                logger.info(f"Installed {package}")
    
    @staticmethod
    def setup_environment():
        """Complete environment setup for Colab"""
        logger.info("Setting up Colab environment...")
        
        # Mount drive
        ColabSetup.mount_drive()
        
        # Install requirements
        ColabSetup.install_requirements()
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Check GPU availability
        import tensorflow as tf
        if tf.config.list_physical_devices('GPU'):
            logger.info("GPU detected and available")
        else:
            logger.warning("No GPU detected")
        
        logger.info("Colab environment setup complete")

class DataManager:
    """Manage data operations in Colab"""
    
    def __init__(self, drive_path: str = "/content/drive/MyDrive"):
        self.drive_path = Path(drive_path)
    
    def verify_data_structure(self, data_dir: str = "Preprocessed-128") -> Dict[str, any]:
        """Verify the preprocessed data structure"""
        data_path = self.drive_path / data_dir
        
        verification_result = {
            'data_path_exists': data_path.exists(),
            'splits': {},
            'total_files': 0,
            'issues': []
        }
        
        if not data_path.exists():
            verification_result['issues'].append(f"Data directory not found: {data_path}")
            return verification_result
        
        # Check each split
        for split in ['train', 'val', 'test']:
            split_path = data_path / split
            sar_path = split_path / 'sar'
            flood_path = split_path / 'flood'
            
            split_info = {
                'split_exists': split_path.exists(),
                'sar_dir_exists': sar_path.exists(),
                'flood_dir_exists': flood_path.exists(),
                'sar_files': 0,
                'flood_files': 0,
                'matched_pairs': 0
            }
            
            if sar_path.exists():
                sar_files = list(sar_path.glob('*.tif'))
                split_info['sar_files'] = len(sar_files)
            
            if flood_path.exists():
                flood_files = list(flood_path.glob('*.tif'))
                split_info['flood_files'] = len(flood_files)
            
            # Count matched pairs
            if sar_path.exists() and flood_path.exists():
                matched_pairs = self._count_matched_pairs(sar_path, flood_path)
                split_info['matched_pairs'] = matched_pairs
            
            verification_result['splits'][split] = split_info
            verification_result['total_files'] += split_info['sar_files'] + split_info['flood_files']
        
        # Check for metadata
        metadata_path = data_path / 'metadata'
        if metadata_path.exists():
            verification_result['metadata'] = {
                'metadata_dir_exists': True,
                'normalization_stats': (metadata_path / 'normalization_stats.json').exists(),
                'channel_analysis': (metadata_path / 'channel_analysis.json').exists(),
                'preprocessing_summary': (metadata_path / 'preprocessing_summary.json').exists()
            }
        else:
            verification_result['issues'].append("Metadata directory not found")
        
        return verification_result
    
    def _count_matched_pairs(self, sar_dir: Path, flood_dir: Path) -> int:
        """Count matched SAR-flood pairs"""
        sar_files = list(sar_dir.glob('*.tif'))
        flood_files = list(flood_dir.glob('*.tif'))
        
        matched_count = 0
        for sar_file in sar_files:
            sar_base = sar_file.stem.replace("prep_", "")
            
            for flood_file in flood_files:
                flood_base = flood_file.stem.replace("prep_", "").replace("_flood", "")
                
                if sar_base == flood_base or sar_base in flood_base or flood_base in sar_base:
                    matched_count += 1
                    break
        
        return matched_count
    
    def print_data_summary(self, data_dir: str = "Preprocessed-128"):
        """Print a summary of the data structure"""
        verification = self.verify_data_structure(data_dir)
        
        print(f"Data path exists: {verification['data_path_exists']}")
        print(f"Total files: {verification['total_files']}")
        print()
        
        for split, info in verification['splits'].items():
            print(f"{split.upper()} SPLIT:")
            print(f"  Directory exists: {info['split_exists']}")
            print(f"  SAR files: {info['sar_files']}")
            print(f"  Flood files: {info['flood_files']}")
            print(f"  Matched pairs: {info['matched_pairs']}")
            print()
        
        if 'metadata' in verification:
            print("METADATA:")
            for key, value in verification['metadata'].items():
                print(f"  {key}: {value}")
            print()
        
        if verification['issues']:
            print("ISSUES FOUND:")
            for issue in verification['issues']:
                print(f"  - {issue}")
        else:
            print("No issues found - data structure verified!")
    
    def load_preprocessing_metadata(self, data_dir: str = "Preprocessed-128") -> Dict:
        """Load preprocessing metadata"""
        metadata_path = self.drive_path / data_dir / "metadata"
        
        metadata = {}
        
        # Load normalization stats
        norm_stats_path = metadata_path / "normalization_stats.json"
        if norm_stats_path.exists():
            with open(norm_stats_path, 'r') as f:
                metadata['normalization_stats'] = json.load(f)
        else:
            logger.warning(f"Normalization stats not found at {norm_stats_path}")
        
        # Load channel analysis
        channel_analysis_path = metadata_path / "channel_analysis.json"
        if channel_analysis_path.exists():
            with open(channel_analysis_path, 'r') as f:
                metadata['channel_analysis'] = json.load(f)
        else:
            logger.warning(f"Channel analysis not found at {channel_analysis_path}")
        
        # Load preprocessing summary
        preprocessing_summary_path = metadata_path / "preprocessing_summary.json"
        if preprocessing_summary_path.exists():
            with open(preprocessing_summary_path, 'r') as f:
                metadata['preprocessing_summary'] = json.load(f)
        else:
            logger.warning(f"Preprocessing summary not found at {preprocessing_summary_path}")
        
        # If metadata directory doesn't exist, check root directory
        if not metadata_path.exists():
            logger.warning(f"Metadata directory not found: {metadata_path}")
            logger.info("Checking root directory for metadata files...")
            
            root_path = self.drive_path / data_dir
            for filename in ["normalization_stats.json", "channel_analysis.json", "preprocessing_summary.json"]:
                file_path = root_path / filename
                if file_path.exists():
                    with open(file_path, 'r') as f:
                        key = filename.replace('.json', '')
                        metadata[key] = json.load(f)
                    logger.info(f"Found {filename} in root directory")
        
        return metadata

class ExperimentManager:
    """Manage experiments and results in Colab"""
    
    def __init__(self, drive_path: str = "/content/drive/MyDrive"):
        self.drive_path = Path(drive_path)
        self.experiments_dir = self.drive_path / "experiments"
        self.experiments_dir.mkdir(exist_ok=True)
    
    def create_experiment_directory(self, experiment_name: str) -> Path:
        """Create experiment directory structure"""
        exp_dir = self.experiments_dir / experiment_name
        exp_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (exp_dir / "checkpoints").mkdir(exist_ok=True)
        (exp_dir / "logs").mkdir(exist_ok=True)
        (exp_dir / "visualizations").mkdir(exist_ok=True)
        (exp_dir / "tensorboard").mkdir(exist_ok=True)
        
        logger.info(f"Created experiment directory: {exp_dir}")
        return exp_dir
    
    def list_experiments(self) -> List[str]:
        """List all experiments"""
        if not self.experiments_dir.exists():
            return []
        
        experiments = []
        for item in self.experiments_dir.iterdir():
            if item.is_dir():
                experiments.append(item.name)
        
        return sorted(experiments)
    
    def get_experiment_summary(self, experiment_name: str) -> Dict:
        """Get summary of an experiment"""
        exp_dir = self.experiments_dir / experiment_name
        
        if not exp_dir.exists():
            return {"error": f"Experiment {experiment_name} not found"}
        
        summary = {
            "experiment_name": experiment_name,
            "experiment_path": str(exp_dir),
            "files": {}
        }
        
        # Check for key files
        key_files = [
            "config.yaml",
            "training_history.json",
            "test_results.json",
            "experiment_summary.json",
            "best_model.h5",
            "model_summary.txt"
        ]
        
        for filename in key_files:
            file_path = exp_dir / filename
            summary["files"][filename] = {
                "exists": file_path.exists(),
                "size_bytes": file_path.stat().st_size if file_path.exists() else 0
            }
        
        return summary
    
    def backup_experiment(self, experiment_name: str, backup_path: Optional[str] = None):
        """Create backup of experiment"""
        exp_dir = self.experiments_dir / experiment_name
        
        if not exp_dir.exists():
            raise ValueError(f"Experiment {experiment_name} not found")
        
        if backup_path is None:
            backup_path = self.drive_path / f"{experiment_name}_backup.zip"
        
        with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in exp_dir.rglob('*'):
                if file_path.is_file():
                    arcname = file_path.relative_to(exp_dir)
                    zipf.write(file_path, arcname)
        
        logger.info(f"Experiment backed up to: {backup_path}")

def setup_colab_environment(data_dir: str = "Preprocessed-128") -> Dict:
    """One-command setup for Colab environment"""
    setup = ColabSetup()
    data_manager = DataManager()
    
    setup.setup_environment()
    
    data_verification = data_manager.verify_data_structure(data_dir)
    
    data_manager.print_data_summary(data_dir)
    
    metadata = data_manager.load_preprocessing_metadata(data_dir)
    
    return {
        "setup_complete": True,
        "data_verification": data_verification,
        "metadata": metadata
    }