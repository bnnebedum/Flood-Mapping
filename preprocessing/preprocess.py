import numpy as np
import rasterio
from pathlib import Path
import json
from scipy import ndimage
import time

# STEP 1: DATA VALIDATION AND QUALITY ASSESSMENT
def validate_data_quality(sar_tiles_dir, flood_tiles_dir):
    """
    Comprehensive data validation and quality assessment
    """
    print("=== DATA VALIDATION ===")
    
    sar_files = list(Path(sar_tiles_dir).glob('*.tif'))
    flood_files = list(Path(flood_tiles_dir).glob('*.tif'))
    
    validation_results = {
        'total_sar_tiles': len(sar_files),
        'total_flood_tiles': len(flood_files),
        'paired_tiles': 0,
        'spatial_alignment_issues': 0,
        'value_range_issues': 0,
        'valid_pairs': []
    }
    
    start_time = time.time()
    
    for i, sar_file in enumerate(sar_files):
        # Find corresponding flood tile
        flood_file = None
        for f in flood_files:
            if sar_file.stem in f.stem or f.stem in sar_file.stem:
                flood_file = f
                break
        
        if flood_file is None:
            continue
            
        try:
            # Check spatial alignment
            with rasterio.open(sar_file) as sar_src, rasterio.open(flood_file) as flood_src:
                # Bounds check
                bounds_aligned = np.allclose(sar_src.bounds, flood_src.bounds, atol=1e-6)
                
                # Size check
                size_aligned = (sar_src.width == flood_src.width and 
                              sar_src.height == flood_src.height)
                
                # Value range check
                sar_data = sar_src.read()
                flood_data = flood_src.read(1)
                
                sar_valid = (sar_data.min() >= 0) and (sar_data.max() <= 255)
                flood_binary = set(np.unique(flood_data)).issubset({0, 1}) or set(np.unique(flood_data)).issubset({0, 255})
                
                if bounds_aligned and size_aligned and sar_valid and flood_binary:
                    validation_results['valid_pairs'].append((sar_file, flood_file))
                    validation_results['paired_tiles'] += 1
                else:
                    if not bounds_aligned or not size_aligned:
                        validation_results['spatial_alignment_issues'] += 1
                    if not sar_valid or not flood_binary:
                        validation_results['value_range_issues'] += 1
        
        except Exception as e:
            print(f"Error validating {sar_file.name}: {e}")
        
        # Progress monitoring for large datasets
        if (i + 1) % 500 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            remaining = (len(sar_files) - i - 1) / rate
            print(f"Validated {i + 1}/{len(sar_files)} pairs")
            print(f"Rate: {rate:.1f} pairs/sec, ETA: {remaining/60:.1f} minutes")
    
    print(f"Valid paired tiles: {validation_results['paired_tiles']}")
    print(f"Spatial alignment issues: {validation_results['spatial_alignment_issues']}")
    print(f"Value range issues: {validation_results['value_range_issues']}")
    
    return validation_results

# STEP 2: PAULI RGB CHANNEL ANALYSIS
def analyze_pauli_channels(valid_pairs, output_dir):
    """
    Analyze which Pauli RGB channels are most effective for water detection
    """
    print("=== PAULI RGB CHANNEL ANALYSIS ===")
    
    channel_names = ['Red (HH-VV)', 'Green (HV)', 'Blue (HH+VV)']
    channel_stats = {
        'water_means': [[], [], []],
        'land_means': [[], [], []],
        'contrasts': [[], [], []],
        'separabilities': [[], [], []]
    }
    
    for sar_file, flood_file in valid_pairs[:50]:  # Sample for analysis
        with rasterio.open(sar_file) as sar_src, rasterio.open(flood_file) as flood_src:
            sar_data = sar_src.read().astype(np.float32)
            flood_mask = flood_src.read(1)
            
            # Normalize flood mask to binary
            flood_mask = (flood_mask > 0).astype(np.uint8)
            
            water_pixels = flood_mask == 1
            land_pixels = flood_mask == 0
            
            if water_pixels.sum() > 100 and land_pixels.sum() > 100:
                for channel in range(3):
                    water_vals = sar_data[channel][water_pixels]
                    land_vals = sar_data[channel][land_pixels]
                    
                    water_mean = water_vals.mean()
                    land_mean = land_vals.mean()
                    
                    # Calculate contrast ratio
                    contrast = abs(water_mean - land_mean) / (water_mean + land_mean + 1e-6)
                    
                    # Calculate separability (normalized difference)
                    separability = abs(water_mean - land_mean) / (water_vals.std() + land_vals.std() + 1e-6)
                    
                    channel_stats['water_means'][channel].append(water_mean)
                    channel_stats['land_means'][channel].append(land_mean)
                    channel_stats['contrasts'][channel].append(contrast)
                    channel_stats['separabilities'][channel].append(separability)
    
    # Calculate summary statistics
    analysis_results = {}
    for channel in range(3):
        analysis_results[channel_names[channel]] = {
            'avg_contrast': float(np.mean(channel_stats['contrasts'][channel])),
            'avg_separability': float(np.mean(channel_stats['separabilities'][channel])),
            'water_mean': float(np.mean(channel_stats['water_means'][channel])),
            'land_mean': float(np.mean(channel_stats['land_means'][channel]))
        }
        
        print(f"{channel_names[channel]}:")
        print(f"  Average contrast: {analysis_results[channel_names[channel]]['avg_contrast']:.3f}")
        print(f"  Average separability: {analysis_results[channel_names[channel]]['avg_separability']:.3f}")
    
    # Save analysis results
    with open(Path(output_dir) / 'channel_analysis.json', 'w') as f:
        json.dump(analysis_results, f, indent=2)

# STEP 3: NORMALIZATION CALCULATION
def calculate_normalization_stats(train_pairs, output_dir):
    """
    Calculate optimal normalization statistics from training data only
    """
    print("=== NORMALIZATION STATISTICS CALCULATION ===")
    
    n_pixels = [0, 0, 0]
    sum_vals = [0.0, 0.0, 0.0]
    sum_sq_vals = [0.0, 0.0, 0.0]
    percentile_samples = [[], [], []]
    
    total_tiles_processed = 0
    
    for sar_file, _ in train_pairs:
        with rasterio.open(sar_file) as src:
            data = src.read().astype(np.float32)
            
            for channel in range(3):
                # Only use non-zero pixels (exclude padding)
                valid_pixels = data[channel][data[channel] > 0]
                if len(valid_pixels) > 0:
                    # Running statistics for mean/std
                    n_pixels[channel] += len(valid_pixels)
                    sum_vals[channel] += valid_pixels.sum()
                    sum_sq_vals[channel] += (valid_pixels ** 2).sum()
                    
                    # Sample pixels for percentile calculation
                    if len(percentile_samples[channel]) < 100000:  # Keep max 100k samples
                        sample_size = min(1000, len(valid_pixels))
                        sample_indices = np.random.choice(len(valid_pixels), sample_size, replace=False)
                        percentile_samples[channel].extend(valid_pixels[sample_indices].tolist())
        
        total_tiles_processed += 1
        if total_tiles_processed % 50 == 0:
            print(f"Processed {total_tiles_processed} tiles for normalization stats")
    
    # Calculate final statistics
    normalization_stats = {
        'method': 'standardization',
        'channels': ['Red', 'Green', 'Blue'],
        'mean': [],
        'std': [],
        'percentiles': {
            '1': [],
            '99': []
        }
    }
    
    for channel in range(3):
        # Calculate mean and std from running sums
        mean_val = sum_vals[channel] / n_pixels[channel]
        variance = (sum_sq_vals[channel] / n_pixels[channel]) - (mean_val ** 2)
        std_val = np.sqrt(variance)
        
        # Calculate percentiles from samples
        p1, p99 = np.percentile(percentile_samples[channel], [1, 99])
        
        normalization_stats['mean'].append(float(mean_val))
        normalization_stats['std'].append(float(std_val))
        normalization_stats['percentiles']['1'].append(float(p1))
        normalization_stats['percentiles']['99'].append(float(p99))
        
        print(f"Channel {channel}: mean={mean_val:.2f}, std={std_val:.2f}, range=[{p1:.1f}, {p99:.1f}]")
        print(f"  Based on {n_pixels[channel]:,} pixels")
    
    # Save normalization statistics
    with open(Path(output_dir) / 'normalization_stats.json', 'w') as f:
        json.dump(normalization_stats, f, indent=2)
    
    return normalization_stats

# STEP 4: SAR DATA PREPROCESSING
def preprocess_sar_tile(sar_path, normalization_stats, output_path):
    """
    Comprehensive SAR tile preprocessing
    """
    with rasterio.open(sar_path) as src:
        data = src.read().astype(np.float32)
        profile = src.profile.copy()

    data = data / 255.0
    
    # Apply log transformation to enhance dynamic range
    data = np.log1p(data)
    
    # Clip extreme outliers (robust to outliers)
    for channel in range(3):
        p1 = normalization_stats['percentiles']['1'][channel] / 255.0
        p99 = normalization_stats['percentiles']['99'][channel] / 255.0
        p1_log = np.log1p(p1)
        p99_log = np.log1p(p99)
        
        data[channel] = np.clip(data[channel], p1_log, p99_log)
    
    # Standardization (zero mean, unit variance)
    for channel in range(3):
        valid_mask = data[channel] > 0
        if valid_mask.sum() > 0:
            # Calculate stats on log-transformed training data
            mean_log = np.log1p(normalization_stats['mean'][channel] / 255.0)
            std_log = normalization_stats['std'][channel] / 255.0  # Approximate std for log transform
            
            data[channel][valid_mask] = (data[channel][valid_mask] - mean_log) / (std_log + 1e-8)
    
    # Light Gaussian filter to reduce any remaining speckle
    for channel in range(3):
        data[channel] = ndimage.gaussian_filter(data[channel], sigma=0.5)
    
    # Update profile for float32
    profile.update(dtype=rasterio.float32, nodata=0.0)
    
    # Save preprocessed tile
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(data)
    
    return data

# STEP 5: MINIMAL FLOOD MASK PREPROCESSING
def preprocess_flood_mask_minimal(flood_path, output_path):
    """
    Minimal flood mask preprocessing - preserve NC OneMap boundaries exactly
    """
    with rasterio.open(flood_path) as src:
        data = src.read(1).astype(np.uint8)
        profile = src.profile.copy()
    
    data = (data > 0).astype(np.uint8)
    
    profile.update(dtype=rasterio.uint8, count=1, nodata=0)
    
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(data, 1)
    
    return data

# STEP 6: DATA QUALITY FILTERING
def quality_filtering(preprocessed_pairs, min_water_ratio=0.01, max_water_ratio=0.85):
    """
    Filter tiles based on data quality criteria (more lenient thresholds)
    """
    print("=== DATA QUALITY FILTERING ===")
    
    high_quality_pairs = []
    rejected_pairs = []
    
    for sar_path, flood_path in preprocessed_pairs:
        try:
            with rasterio.open(flood_path) as src:
                flood_data = src.read(1)
            
            # Calculate water ratio
            total_pixels = flood_data.size
            water_pixels = (flood_data > 0).sum()
            water_ratio = water_pixels / total_pixels
            
            criteria = {
                'sufficient_water': water_ratio >= min_water_ratio,
                'not_mostly_water': water_ratio <= max_water_ratio,
                'reasonable_size': total_pixels >= 8192
            }
            
            if all(criteria.values()):
                high_quality_pairs.append((sar_path, flood_path, water_ratio))
            else:
                rejected_pairs.append((sar_path, flood_path, water_ratio, criteria))
                
        except Exception as e:
            print(f"Error in quality filtering for {sar_path}: {e}")
            rejected_pairs.append((sar_path, flood_path, 0.0, {'error': str(e)}))
    
    print(f"High quality pairs: {len(high_quality_pairs)}")
    print(f"Rejected pairs: {len(rejected_pairs)}")
    
    # Analyze rejection reasons
    rejection_reasons = {
        'insufficient_water': 0,
        'mostly_water': 0,
        'too_small': 0,
        'processing_errors': 0
    }
    
    for _, _, water_ratio, criteria in rejected_pairs:
        if isinstance(criteria, dict):
            if 'error' in criteria:
                rejection_reasons['processing_errors'] += 1
            elif not criteria.get('sufficient_water', True):
                rejection_reasons['insufficient_water'] += 1
            elif not criteria.get('not_mostly_water', True):
                rejection_reasons['mostly_water'] += 1
            elif not criteria.get('reasonable_size', True):
                rejection_reasons['too_small'] += 1
    
    print("Rejection breakdown:")
    for reason, count in rejection_reasons.items():
        print(f"  {reason}: {count}")
    
    return high_quality_pairs, rejected_pairs

# STEP 7: SPATIAL TRAIN/VAL/TEST SPLIT
def spatial_split(high_quality_pairs, val_ratio=0.15, test_ratio=0.15, random_seed=42):
    """
    Create spatially-aware train/validation/test splits
    """
    print("=== SPATIAL DATA SPLITTING ===")
    
    # Extract tile center coordinates
    tile_locations = []
    for sar_path, flood_path, water_ratio in high_quality_pairs:
        with rasterio.open(sar_path) as src:
            bounds = src.bounds
            center_x = (bounds.left + bounds.right) / 2
            center_y = (bounds.bottom + bounds.top) / 2
            tile_locations.append((center_x, center_y, sar_path, flood_path, water_ratio))
    
    # Use geographic coordinates for spatial splitting
    from sklearn.cluster import KMeans
    
    coordinates = np.array([(x, y) for x, y, _, _, _ in tile_locations])
    
    # Create spatial clusters
    n_clusters = max(3, int(1 / (val_ratio + test_ratio)))
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_seed, n_init=10)
    cluster_labels = kmeans.fit_predict(coordinates)
    
    # Assign clusters to splits
    unique_clusters = np.unique(cluster_labels)
    np.random.seed(random_seed)
    np.random.shuffle(unique_clusters)
    
    n_test_clusters = max(1, int(len(unique_clusters) * test_ratio))
    n_val_clusters = max(1, int(len(unique_clusters) * val_ratio))
    
    test_clusters = set(unique_clusters[:n_test_clusters])
    val_clusters = set(unique_clusters[n_test_clusters:n_test_clusters + n_val_clusters])
    train_clusters = set(unique_clusters[n_test_clusters + n_val_clusters:])
    
    # Create splits
    train_pairs, val_pairs, test_pairs = [], [], []
    
    for i, (_, _, sar_path, flood_path, water_ratio) in enumerate(tile_locations):
        cluster = cluster_labels[i]
        
        if cluster in test_clusters:
            test_pairs.append((sar_path, flood_path, water_ratio))
        elif cluster in val_clusters:
            val_pairs.append((sar_path, flood_path, water_ratio))
        else:
            train_pairs.append((sar_path, flood_path, water_ratio))
    
    print(f"Train: {len(train_pairs)} pairs")
    print(f"Validation: {len(val_pairs)} pairs")
    print(f"Test: {len(test_pairs)} pairs")
    
    # Calculate water ratio statistics for each split
    for split_name, split_pairs in [('Train', train_pairs), ('Val', val_pairs), ('Test', test_pairs)]:
        water_ratios = [wr for _, _, wr in split_pairs]
        print(f"{split_name} water ratio: {np.mean(water_ratios):.3f} ± {np.std(water_ratios):.3f}")
    
    return train_pairs, val_pairs, test_pairs

# STEP 8: SAVE ORGANIZED SPLIT FOLDERS (Option 2)
def save_split_folders(train_pairs, val_pairs, test_pairs, output_dir):
    """
    Save tiles organized by train/val/test splits for easy Colab upload
    """
    import shutil
    
    print("=== ORGANIZING TILES BY SPLITS ===")
    output_path = Path(output_dir)
    
    # Create split folder structure
    for split_name in ['train', 'val', 'test']:
        (output_path / split_name / 'sar').mkdir(parents=True, exist_ok=True)
        (output_path / split_name / 'flood').mkdir(parents=True, exist_ok=True)
    
    # Copy files to appropriate split folders
    split_data = {
        'train': train_pairs,
        'val': val_pairs, 
        'test': test_pairs
    }
    
    for split_name, pairs in split_data.items():
        print(f"Organizing {split_name} split: {len(pairs)} tiles")
        
        for i, (sar_path, flood_path, _) in enumerate(pairs):
            sar_dest = output_path / split_name / 'sar' / sar_path.name
            flood_dest = output_path / split_name / 'flood' / flood_path.name
            
            shutil.copy2(sar_path, sar_dest)
            shutil.copy2(flood_path, flood_dest)
            
            if (i + 1) % 1000 == 0:
                print(f"  Copied {i + 1}/{len(pairs)} {split_name} tiles")
    
    print("All tiles organized by splits")
    
    # Create file lists for verification
    split_info = {}
    for split_name in ['train', 'val', 'test']:
        sar_files = list((output_path / split_name / 'sar').glob('*.tif'))
        flood_files = list((output_path / split_name / 'flood').glob('*.tif'))
        
        split_info[split_name] = {
            'sar_tiles': len(sar_files),
            'flood_tiles': len(flood_files),
            'first_sar_file': str(sar_files[0].name) if sar_files else None,
            'first_flood_file': str(flood_files[0].name) if flood_files else None
        }
    
    with open(output_path / 'split_organization.json', 'w') as f:
        json.dump(split_info, f, indent=2)

# STEP 9: FINAL PREPROCESSING EXECUTION WITH SPLIT FOLDERS
def preprocess(input_sar_dir, input_flood_dir, output_dir):
    """
    Execute the complete preprocessing pipeline and organize for Colab upload (Option 2)
    """
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Create temporary preprocessing directories
    temp_dir = output_path / 'temp_preprocessing'
    temp_dir.mkdir(exist_ok=True)
    (temp_dir / 'preprocessed_sar').mkdir(exist_ok=True)
    (temp_dir / 'preprocessed_flood').mkdir(exist_ok=True)
    
    validation_results = validate_data_quality(input_sar_dir, input_flood_dir)
    valid_pairs = validation_results['valid_pairs']
    
    analyze_pauli_channels(valid_pairs, output_path)
    
    temp_quality_pairs = [(sar, flood, 0.5) for sar, flood in valid_pairs]  # Temporary
    train_pairs_temp, _, _ = spatial_split(temp_quality_pairs)
    train_pairs_for_stats = [(sar, flood) for sar, flood, _ in train_pairs_temp]
    
    normalization_stats = calculate_normalization_stats(train_pairs_for_stats, output_path)
    
    preprocessed_pairs = []
    
    print(f"\nStarting preprocessing of {len(valid_pairs)} tile pairs...")
    start_time = time.time()
    
    for i, (sar_file, flood_file) in enumerate(valid_pairs):
        try:
            # Generate output paths in temp directory
            sar_output = temp_dir / 'preprocessed_sar' / f"prep_{sar_file.name}"
            flood_output = temp_dir / 'preprocessed_flood' / f"prep_{flood_file.name}"
            
            # Preprocess SAR tile
            preprocess_sar_tile(sar_file, normalization_stats, sar_output)
            
            # Minimal flood mask preprocessing (preserve boundaries)
            preprocess_flood_mask_minimal(flood_file, flood_output)
            
            preprocessed_pairs.append((sar_output, flood_output))
            
            if (i + 1) % 1000 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                remaining = (len(valid_pairs) - i - 1) / rate
                print(f"Preprocessed {i + 1}/{len(valid_pairs)} tile pairs")
                print(f"Rate: {rate:.1f} pairs/sec, ETA: {remaining/60:.1f} minutes")
                
        except Exception as e:
            print(f"Error preprocessing {sar_file.name}: {e}")
    
    print(f"Preprocessed {len(preprocessed_pairs)}/{len(valid_pairs)} tile pairs")
    
    # Step 6: Quality filtering (more lenient)
    high_quality_pairs, rejected_pairs = quality_filtering(preprocessed_pairs)
    
    # Step 7: Final split with preprocessed data
    train_pairs, val_pairs, test_pairs = spatial_split(high_quality_pairs)
    
    # Step 8: Organize tiles by splits for easy upload
    save_split_folders(train_pairs, val_pairs, test_pairs, output_path)
    
    metadata_dir = output_path / 'metadata'
    metadata_dir.mkdir(exist_ok=True)
    
    # Move metadata files to metadata folder
    import shutil
    if (output_path / 'channel_analysis.json').exists():
        shutil.move(str(output_path / 'channel_analysis.json'), str(metadata_dir / 'channel_analysis.json'))
    if (output_path / 'normalization_stats.json').exists():
        shutil.move(str(output_path / 'normalization_stats.json'), str(metadata_dir / 'normalization_stats.json'))
    
    summary = {
        'total_input_pairs': len(valid_pairs),
        'total_preprocessed_pairs': len(preprocessed_pairs),
        'high_quality_pairs': len(high_quality_pairs),
        'rejected_pairs': len(rejected_pairs),
        'final_splits': {
            'train': len(train_pairs),
            'val': len(val_pairs),
            'test': len(test_pairs)
        },
        'preprocessing_settings': {
            'flood_processing': 'minimal_boundary_preservation',
            'normalization_method': 'log_transform_standardization',
            'spatial_filtering': 'gaussian_sigma_0.5',
            'morphological_cleaning': 'none_applied',
            'boundary_preservation': 'complete',
            'quality_filtering': 'lenient_thresholds'
        },
        'quality_thresholds': {
            'min_water_ratio': 0.01,
            'max_water_ratio': 0.85,
            'min_tile_size': 8192
        },
        'folder_structure': {
            'train': f"{len(train_pairs)} tiles in train/sar/ and train/flood/",
            'val': f"{len(val_pairs)} tiles in val/sar/ and val/flood/",
            'test': f"{len(test_pairs)} tiles in test/sar/ and test/flood/",
            'metadata': "normalization_stats.json and channel_analysis.json in metadata/"
        }
    }
    
    with open(metadata_dir / 'preprocessing_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Remove temp directory
    shutil.rmtree(temp_dir)
    
    print(f"  Train: {len(train_pairs)} tiles")
    print(f"  Validation: {len(val_pairs)} tiles") 
    print(f"  Test: {len(test_pairs)} tiles")
    
    return train_pairs, val_pairs, test_pairs, normalization_stats

# MAIN
input_sar_directory = "/Users/ikennan/Downloads/SAR-128"
input_flood_directory = "/Users/ikennan/Downloads/Flood-128"  
output_directory = "/Users/ikennan/Downloads/Preprocessed-128"

train_pairs, val_pairs, test_pairs, norm_stats = preprocess(
    input_sar_directory,
    input_flood_directory, 
    output_directory
)
    
print("\nPreprocessing pipeline completed")