import os
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.enums import Resampling
import numpy as np
from pathlib import Path
import re

def extract_tile_info_from_filename(filename):
    """
    Extract tile information from filename
    Assumes format like: flight_tile_row_col.tif or similar
    Adjust regex pattern based on your naming convention
    """
    
    patterns = [
        r'.*_r(\d+)_c(\d+)\.tif',  # row/col format
        r'.*_(\d+)_(\d+)\.tif',    # generic number format
        r'.*tile_(\d+)_(\d+)\.tif' # explicit tile format
    ]
    
    for pattern in patterns:
        match = re.match(pattern, filename)
        if match:
            return int(match.group(1)), int(match.group(2))
    
    return None, None

def get_tile_bounds(tile_path):
    """
    Get the geographic bounds of a tile
    """
    with rasterio.open(tile_path) as src:
        bounds = src.bounds
        crs = src.crs
        transform = src.transform
        width = src.width
        height = src.height
        
    return {
        'bounds': bounds,
        'crs': crs,
        'transform': transform,
        'width': width,
        'height': height
    }

def reproject_flood_map_to_tile_crs(flood_map_path, target_crs):
    """
    Reproject flood map to match tile CRS if needed
    """
    with rasterio.open(flood_map_path) as src:
        if src.crs == target_crs:
            print("No reprojection needed")
            return flood_map_path
        
        # Calculate transform for reprojection
        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds
        )
        
        # Define output path
        output_path = flood_map_path.replace('.tif', '_reprojected.tif')
        
        # Reproject
        with rasterio.open(
            output_path, 'w',
            driver='GTiff',
            height=height,
            width=width,
            count=src.count,
            dtype=src.dtypes[0],
            crs=target_crs,
            transform=transform,
            compress='lzw'
        ) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=Resampling.nearest
                )
        
        return output_path

def extract_flood_tile_for_bounds(flood_map_path, bounds, output_size=(128, 128)):
    """
    Extract flood map data for specific bounds and resize to match tile size
    """
    with rasterio.open(flood_map_path) as src:
        window = from_bounds(*bounds, src.transform)
        
        flood_data = src.read(1, window=window)
        
        if flood_data.size == 0:
            return np.zeros(output_size, dtype=np.uint8)
        
        # Resize to match output size using nearest neighbor
        from scipy.ndimage import zoom
        
        if flood_data.shape != output_size:
            zoom_factors = (output_size[0] / flood_data.shape[0], 
                           output_size[1] / flood_data.shape[1])
            flood_data = zoom(flood_data, zoom_factors, order=0)
        
        return flood_data.astype(np.uint8)

def create_aligned_flood_tiles(uavsar_tiles_dir, flood_map_path, output_dir):
    """
    Main function to create flood extent tiles aligned with UAVSAR tiles
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    uavsar_files = list(Path(uavsar_tiles_dir).glob('*.tif'))
    
    if not uavsar_files:
        print("No .tif files found in UAVSAR tiles directory")
        return
    
    print(f"Found {len(uavsar_files)} UAVSAR tiles")
    
    sample_tile_info = get_tile_bounds(uavsar_files[0])
    target_crs = sample_tile_info['crs']
    
    reprojected_flood_path = reproject_flood_map_to_tile_crs(flood_map_path, target_crs)
    
    successful_tiles = 0
    failed_tiles = 0
    
    for uavsar_tile_path in uavsar_files:
        try:
            tile_info = get_tile_bounds(uavsar_tile_path)
            bounds = tile_info['bounds']
            
            flood_tile_data = extract_flood_tile_for_bounds(
                reprojected_flood_path, 
                bounds, 
                output_size=(128, 128)
            )
            
            tile_filename = uavsar_tile_path.stem
            output_filename = f"{tile_filename}_flood.tif"
            output_filepath = output_path / output_filename
            
            with rasterio.open(uavsar_tile_path) as src_template:
                with rasterio.open(
                    output_filepath, 'w',
                    driver='GTiff',
                    height=128,
                    width=128,
                    count=1,
                    dtype=rasterio.uint8,
                    crs=src_template.crs,
                    transform=src_template.transform,
                    compress='lzw'
                ) as dst:
                    dst.write(flood_tile_data, 1)
            
            successful_tiles += 1
            
        except Exception as e:
            print(f"Failed to process {uavsar_tile_path.name}: {str(e)}")
            failed_tiles += 1
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {successful_tiles} tiles")
    print(f"Failed: {failed_tiles} tiles")
    
    create_alignment_verification(uavsar_tiles_dir, output_dir)

def create_alignment_verification(uavsar_dir, flood_dir):
    """
    Create verification file to confirm tile alignment
    """
    verification = {
        'uavsar_tiles': [],
        'flood_tiles': [],
        'alignment_check': []
    }
    
    uavsar_files = list(Path(uavsar_dir).glob('*.tif'))
    _ = list(Path(flood_dir).glob('*.tif'))
    
    for uavsar_file in uavsar_files:
        flood_file = Path(flood_dir) / f"{uavsar_file.stem}_flood.tif"
        
        if flood_file.exists():
            uavsar_bounds = get_tile_bounds(uavsar_file)
            flood_bounds = get_tile_bounds(flood_file)
            
            bounds_match = (
                abs(uavsar_bounds['bounds'].left - flood_bounds['bounds'].left) < 1e-6 and
                abs(uavsar_bounds['bounds'].right - flood_bounds['bounds'].right) < 1e-6 and
                abs(uavsar_bounds['bounds'].top - flood_bounds['bounds'].top) < 1e-6 and
                abs(uavsar_bounds['bounds'].bottom - flood_bounds['bounds'].bottom) < 1e-6
            )
            
            verification['alignment_check'].append({
                'uavsar_file': uavsar_file.name,
                'flood_file': flood_file.name,
                'bounds_aligned': bounds_match,
                'uavsar_bounds': list(uavsar_bounds['bounds']),
                'flood_bounds': list(flood_bounds['bounds'])
            })
    
# MAIN
uavsar_tiles_directory = "/Users/ikennan/Downloads/32023-128-valid"
flood_map_file = "/Users/ikennan/Downloads/Florence_flood_extent/FloodExtentFlorence.tif"
output_directory = "/Users/ikennan/Downloads/32023-128-flood"

create_aligned_flood_tiles(
    uavsar_tiles_directory, 
    flood_map_file, 
    output_directory
)