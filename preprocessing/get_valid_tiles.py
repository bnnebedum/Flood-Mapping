import os
import shutil
from pathlib import Path
import numpy as np
import rasterio

def is_tile_all_black(file_path, tolerance=1e-10):
    try:
        with rasterio.open(file_path) as src:
            data = src.read()
            
            if np.all(np.abs(data) <= tolerance):
                return True
            
            if src.nodata is not None:
                if np.all(data == src.nodata):
                    return True
                    
            return False
            
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return False

def filter_black_tiles(input_folder, output_folder):
    input_path = Path(input_folder)
    
    if not input_path.exists():
        print(f"Input folder {input_folder} does not exist!")
        return
    
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Non-black tiles will be copied to: {output_folder}")
    tiff_files = list(input_path.glob('*.tif')) + list(input_path.glob('*.tiff'))
    
    if not tiff_files:
        print(f"No TIFF files found in {input_folder}")
        return
    
    print(f"Found {len(tiff_files)} TIFF files")
    
    background_tiles = []
    valid_tiles = []
    
    for i, file_path in enumerate(tiff_files):
        print(f"Processing {i+1}/{len(tiff_files)}: {file_path.name}", end=' ')
        
        if is_tile_all_black(file_path):
            background_tiles.append(file_path)
            print("(BACKGROUND)")
        else:
            valid_tiles.append(file_path)
            print("(DATA)")
    
    if valid_tiles:
        for file_path in valid_tiles:
            try:
                dest_path = output_path / file_path.name
                shutil.copy2(file_path, dest_path)
                print(f"Copied: {file_path.name}")
            except Exception as e:
                print(f"Error copying {file_path.name}: {e}")
        print(f"Total files analyzed: {len(tiff_files)}")
        print(f"Background tiles: {len(background_tiles)}")
        print(f"Valid tiles: {len(valid_tiles)}")

    else:
        print("\nNo valid tiles to copy")

filter_black_tiles("/Users/ikennan/Downloads/35303-128", "/Users/ikennan/Downloads/35303-128-valid")