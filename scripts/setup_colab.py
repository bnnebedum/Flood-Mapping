"""
One-command setup script for Google Colab
Run this to set up everything needed for training
"""

def setup_flood_mapping_project():
    """Complete setup for flood mapping project in Colab"""
    
    # 1. Install required packages
    print("Installing required packages...")
    import subprocess
    import sys
    
    packages = [
        "rasterio>=1.3.0",
        "pyyaml>=6.0", 
        "tqdm>=4.64.0",
        "matplotlib>=3.5.0",
        "scikit-learn>=1.0.0"
    ]
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])
            print(f"  [INSTALLED] {package}")
        except:
            print(f"  [FAILED] Failed to install {package}")
    
    print("\nMounting Google Drive...")
    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=True)
        print("  [SUCCESS] Google Drive mounted")
    except Exception as e:
        print(f"  [ERROR] Failed to mount drive: {e}")
        return False
    
    # 3. Verify data structure
    print("\nVerifying data structure...")
    import os
    from pathlib import Path
    
    data_path = Path("/content/drive/MyDrive/Preprocessed-128")
    
    if not data_path.exists():
        print(f"  [ERROR] Data directory not found: {data_path}")
        print("     Please ensure your preprocessed data is uploaded to:")
        print("     /content/drive/MyDrive/Preprocessed-128/")
        return False
    
    # Check splits
    splits_ok = True
    for split in ['train', 'val', 'test']:
        split_path = data_path / split
        sar_path = split_path / 'sar'
        flood_path = split_path / 'flood'
        
        if not sar_path.exists() or not flood_path.exists():
            print(f"  [ERROR] Missing directories in {split}: sar={sar_path.exists()}, flood={flood_path.exists()}")
            splits_ok = False
        else:
            sar_files = len(list(sar_path.glob('*.tif')))
            flood_files = len(list(flood_path.glob('*.tif')))
            print(f"  [SUCCESS] {split}: {sar_files} SAR files, {flood_files} flood files")
    
    if not splits_ok:
        print("  [ERROR] Data structure validation failed")
        return False
    
    # 4. Check metadata
    print("\nChecking metadata...")
    metadata_path = data_path / 'metadata'
    if metadata_path.exists():
        required_files = ['normalization_stats.json', 'channel_analysis.json']
        for filename in required_files:
            file_path = metadata_path / filename
            if file_path.exists():
                print(f"  [SUCCESS] {filename}")
            else:
                print(f"  [WARNING] Missing {filename} (will use defaults)")
    else:
        print("  [WARNING] No metadata directory found (will use defaults)")
    
    project_path = Path("/content/Flood-Mapping")
    
    # Create directory structure
    dirs_to_create = [
        "src/utils",
        "src/data", 
        "src/models",
        "src/training",
        "src/evaluation",
        "configs",
        "notebooks",
        "experiments"
    ]
    
    for dir_path in dirs_to_create:
        full_path = project_path / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        
        # Create __init__.py files for Python packages
        if dir_path.startswith("src/"):
            init_file = full_path / "__init__.py"
            init_file.touch()
    
    print("  [SUCCESS] Project structure created")
    
    # 6. Setup environment variables
    print("\nSetting up environment...")
    import sys
    sys.path.append(str(project_path))
    
    try:
        import tensorflow as tf
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"  [SUCCESS] GPU configured: {len(gpus)} GPU(s) available")
        else:
            print("  [WARNING] No GPU detected - training will be slower")
    except Exception as e:
        print(f"  [WARNING] GPU setup issue: {e}")
    
    print("\nCreating default configuration...")
    
    config_content = f"""# U-Net Configuration for SAR Flood Segmentation
experiment_name: "unet_sar_flood_colab"
output_dir: "/content/drive/MyDrive/experiments"
random_seed: 42

data:
  drive_path: "/content/drive/MyDrive/Preprocessed-128"
  train_dir: "train"
  val_dir: "val" 
  test_dir: "test"
  sar_subdir: "sar"
  flood_subdir: "flood"
  
  image_size: [128, 128]
  channels: 3
  batch_size: 16
  shuffle_buffer: 1000
  prefetch_buffer: 2
  
  enable_augmentation: true
  rotation_range: 20.0
  width_shift_range: 0.1
  height_shift_range: 0.1
  horizontal_flip: true
  vertical_flip: true
  zoom_range: 0.1

model:
  name: "unet"
  input_shape: [128, 128, 3]
  num_classes: 1
  base_filters: 64
  depth: 4
  dropout_rate: 0.5
  batch_norm: true
  activation: "relu"
  kernel_initializer: "he_normal"

training:
  epochs: 50
  initial_lr: 0.001
  min_lr: 1e-7
  patience: 10
  factor: 0.5
  
  loss_function: "binary_crossentropy"
  use_dice_loss: true
  dice_weight: 0.5
  
  early_stopping_patience: 15
  save_best_only: true
  save_weights_only: false
  
  optimizer: "adam"
  beta_1: 0.9
  beta_2: 0.999
  epsilon: 1e-7

evaluation:
  metrics: ["pixel_accuracy", "dice_coefficient", "precision", "recall"]
  threshold: 0.5
  save_predictions: true
  save_visualizations: true
  visualization_samples: 20
"""
    
    config_file = project_path / "configs" / "unet_config.yaml"
    with open(config_file, 'w') as f:
        f.write(config_content)
    print(f"Configuration saved to {config_file}")
    
    
    try:
        import tensorflow as tf
        print(f"  [SUCCESS] TensorFlow {tf.__version__}")
    except:
        print("  [ERROR] TensorFlow not available")
        return False
    
    # Check total data size
    total_pairs = 0
    for split in ['train', 'val', 'test']:
        sar_files = len(list((data_path / split / 'sar').glob('*.tif')))
        total_pairs += sar_files
    
    print(f"  [SUCCESS] Total dataset: ~{total_pairs} tiles")
    
    if total_pairs < 100:
        print("  [WARNING] Dataset seems small - verify preprocessing completed successfully")
    
    print("\nSetup completed successfully!")
    print("="*60)
    print("Next steps:")
    print("1. Upload your project code to /content/Flood-Mapping/src/")
    print("2. Run the training notebook")
    print("3. Monitor training progress in /content/drive/MyDrive/experiments/")
    print()
    print("Quick data check:")
    print(f"  Data path: {data_path}")
    print(f"  Config path: {config_file}")
    print(f"  Project path: {project_path}")
    
    return True

def quick_check():
    """Quick check if environment is ready for training"""
    print("Quick Environment Check")
    
    checks = {
        "Google Drive": "/content/drive/MyDrive",
        "Data Directory": "/content/drive/MyDrive/Preprocessed-128", 
        "Train Data": "/content/drive/MyDrive/Preprocessed-128/train",
        "Val Data": "/content/drive/MyDrive/Preprocessed-128/val",
        "Test Data": "/content/drive/MyDrive/Preprocessed-128/test"
    }
    
    import os
    all_good = True
    
    for name, path in checks.items():
        exists = os.path.exists(path)
        status = "[SUCCESS]" if exists else "[ERROR]"
        print(f"{status} {name}: {path}")
        if not exists:
            all_good = False
    
    # Check TensorFlow and GPU
    try:
        import tensorflow as tf
        gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
        print(f"[SUCCESS] TensorFlow: {tf.__version__}")
        print(f"{'[SUCCESS]' if gpu_available else '[WARNING]'} GPU: {'Available' if gpu_available else 'Not detected'}")
    except:
        print("[ERROR] TensorFlow: Not available")
        all_good = False
    
    if all_good:
        print("Environment ready for training")
    else:
        print("Setup issues detected. Run setup_flood_mapping_project()")
    
    return all_good

success = setup_flood_mapping_project()

if success:
    print("Copy and run this code to start training:")
    print()
    print("exec(open('/content/Flood-Mapping/notebooks/training_notebook.py').read())")
else:
    print("\nSetup failed. Please check the errors above.")