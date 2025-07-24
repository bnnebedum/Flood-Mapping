"""
Data structure validation utility for Preprocessed-128 folder
"""
import os
from pathlib import Path
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class DataStructureValidator:
    """Validate the Preprocessed-128 folder structure"""
    
    def __init__(self, drive_path: str = "/content/drive/MyDrive"):
        self.drive_path = Path(drive_path)
        self.data_path = self.drive_path / "Preprocessed-128"
    
    def validate_complete_structure(self) -> Dict:
        """Validate the complete data structure"""
        validation_result = {
            "structure_valid": True,
            "issues": [],
            "splits": {},
            "metadata": {},
            "file_counts": {},
            "recommendations": []
        }
        
        # Check main directory
        if not self.data_path.exists():
            validation_result["structure_valid"] = False
            validation_result["issues"].append(f"Main data directory not found: {self.data_path}")
            return validation_result
        
        # Validate each split
        required_splits = ["train", "val", "test"]
        for split in required_splits:
            split_validation = self._validate_split(split)
            validation_result["splits"][split] = split_validation
            
            if not split_validation["valid"]:
                validation_result["structure_valid"] = False
                validation_result["issues"].extend(split_validation["issues"])
        
        # Validate metadata
        metadata_validation = self._validate_metadata()
        validation_result["metadata"] = metadata_validation
        
        # Generate recommendations
        validation_result["recommendations"] = self._generate_recommendations(validation_result)
        
        return validation_result
    
    def _validate_split(self, split_name: str) -> Dict:
        """Validate a single split directory"""
        split_path = self.data_path / split_name
        
        split_result = {
            "valid": True,
            "issues": [],
            "sar_files": 0,
            "flood_files": 0,
            "matched_pairs": 0,
            "example_files": {}
        }
        
        # Check split directory exists
        if not split_path.exists():
            split_result["valid"] = False
            split_result["issues"].append(f"Split directory missing: {split_path}")
            return split_result
        
        # Check sar and flood subdirectories
        sar_path = split_path / "sar"
        flood_path = split_path / "flood"
        
        if not sar_path.exists():
            split_result["valid"] = False
            split_result["issues"].append(f"SAR directory missing: {sar_path}")
        
        if not flood_path.exists():
            split_result["valid"] = False
            split_result["issues"].append(f"Flood directory missing: {flood_path}")
        
        if not split_result["valid"]:
            return split_result
        
        # Count files
        sar_files = list(sar_path.glob("*.tif"))
        flood_files = list(flood_path.glob("*.tif"))
        
        split_result["sar_files"] = len(sar_files)
        split_result["flood_files"] = len(flood_files)
        
        # Store example files for debugging
        if sar_files:
            split_result["example_files"]["sar_example"] = sar_files[0].name
        if flood_files:
            split_result["example_files"]["flood_example"] = flood_files[0].name
        
        # Check for matched pairs
        matched_pairs = self._count_matched_pairs(sar_files, flood_files)
        split_result["matched_pairs"] = matched_pairs
        
        # Validate file counts
        if len(sar_files) == 0:
            split_result["issues"].append(f"No SAR files found in {sar_path}")
        
        if len(flood_files) == 0:
            split_result["issues"].append(f"No flood files found in {flood_path}")
        
        if matched_pairs == 0 and len(sar_files) > 0 and len(flood_files) > 0:
            split_result["issues"].append(f"No matching SAR-flood pairs found in {split_name}")
        
        if abs(len(sar_files) - len(flood_files)) > len(sar_files) * 0.1:  # More than 10% difference
            split_result["issues"].append(f"Significant mismatch in file counts: {len(sar_files)} SAR vs {len(flood_files)} flood")
        
        return split_result
    
    def _count_matched_pairs(self, sar_files: List[Path], flood_files: List[Path]) -> int:
        """Count matched SAR-flood pairs using the same logic as dataset.py"""
        matched_count = 0
        
        for sar_file in sar_files:
            sar_base = sar_file.stem
            # Remove common prefixes
            for prefix in ["prep_", "processed_", ""]:
                if sar_base.startswith(prefix):
                    sar_base = sar_base[len(prefix):]
                    break
            
            for flood_file in flood_files:
                flood_base = flood_file.stem
                # Remove common prefixes and suffixes
                for prefix in ["prep_", "processed_", ""]:
                    if flood_base.startswith(prefix):
                        flood_base = flood_base[len(prefix):]
                        break
                
                flood_base = flood_base.replace("_flood", "")
                
                if sar_base == flood_base:
                    matched_count += 1
                    break
                elif (len(sar_base) > 10 and len(flood_base) > 10 and 
                      (sar_base in flood_base or flood_base in sar_base)):
                    matched_count += 1
                    break
        
        return matched_count
    
    def _validate_metadata(self) -> Dict:
        """Validate metadata directory and files"""
        metadata_path = self.data_path / "metadata"
        
        metadata_result = {
            "metadata_dir_exists": metadata_path.exists(),
            "files": {},
            "issues": []
        }
        
        expected_files = [
            "normalization_stats.json",
            "channel_analysis.json", 
            "preprocessing_summary.json"
        ]
        
        if metadata_path.exists():
            for filename in expected_files:
                file_path = metadata_path / filename
                metadata_result["files"][filename] = {
                    "exists": file_path.exists(),
                    "size_bytes": file_path.stat().st_size if file_path.exists() else 0
                }
                
                if not file_path.exists():
                    metadata_result["issues"].append(f"Missing metadata file: {filename}")
        else:
            metadata_result["issues"].append("Metadata directory does not exist")
            # Check if files exist in root directory
            for filename in expected_files:
                file_path = self.data_path / filename
                if file_path.exists():
                    metadata_result["files"][f"root_{filename}"] = {
                        "exists": True,
                        "size_bytes": file_path.stat().st_size
                    }
        
        return metadata_result
    
    def _generate_recommendations(self, validation_result: Dict) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []
        
        # Check for common issues
        total_matched_pairs = sum(split["matched_pairs"] for split in validation_result["splits"].values())
        
        if total_matched_pairs == 0:
            recommendations.append("No matched pairs found. Check file naming convention in preprocessing.")
        
        if total_matched_pairs < 1000:
            recommendations.append(f"Low number of matched pairs ({total_matched_pairs}). Consider checking preprocessing quality filters.")
        
        # Check metadata
        if not validation_result["metadata"]["metadata_dir_exists"]:
            recommendations.append("Create metadata directory and ensure normalization stats are available.")
        
        # Check split balance
        splits = validation_result["splits"]
        if all(split["valid"] for split in splits.values()):
            train_pairs = splits["train"]["matched_pairs"]
            val_pairs = splits["val"]["matched_pairs"]
            test_pairs = splits["test"]["matched_pairs"]
            
            total_pairs = train_pairs + val_pairs + test_pairs
            if total_pairs > 0:
                train_ratio = train_pairs / total_pairs
                val_ratio = val_pairs / total_pairs
                _ = test_pairs / total_pairs
                
                if train_ratio < 0.6:
                    recommendations.append(f"Training set seems small ({train_ratio:.1%}). Consider increasing training data.")
                
                if val_ratio < 0.1 or val_ratio > 0.3:
                    recommendations.append(f"Validation set ratio ({val_ratio:.1%}) might be suboptimal. Aim for 15-20%.")
        
        return recommendations
    
    def print_validation_report(self):
        """Print a comprehensive validation report"""
        validation = self.validate_complete_structure()
        
        print("DATA STRUCTURE VALIDATION REPORT")
        
        print(f"Overall Status: {'VALID' if validation['structure_valid'] else 'INVALID'}")
        print(f"Data Path: {self.data_path}")
        print()
        
        print("DATA STRUCTURE VALIDATION REPORT")
        
        print("Overall Status: {'VALID' if validation['structure_valid'] else 'INVALID'}")
        print(f"Data Path: {self.data_path}")
        print()
        
        print("SPLIT SUMMARY:")
        for split_name, split_info in validation["splits"].items():
            status = "[SUCCESS]" if split_info["valid"] else "[ERROR]"
            print(f"{split_name.upper():>6} {status}: {split_info['matched_pairs']:>4} matched pairs "
                  f"({split_info['sar_files']} SAR, {split_info['flood_files']} flood)")
        print()
        
        print("METADATA SUMMARY:")
        metadata = validation["metadata"]
        print(f"Metadata dir exists: {'[SUCCESS]' if metadata['metadata_dir_exists'] else '[ERROR]'}")
        for filename, info in metadata["files"].items():
            status = "[SUCCESS]" if info["exists"] else "[ERROR]"
            size_kb = info["size_bytes"] / 1024 if info["exists"] else 0
            print(f"{filename:>25} {status}: {size_kb:>6.1f} KB")
        print()
        
        if validation["issues"]:
            print("ISSUES FOUND:")
            for i, issue in enumerate(validation["issues"], 1):
                print(f"{i:>2}. {issue}")
            print()
        
        if validation["recommendations"]:
            print("RECOMMENDATIONS:")
            for i, rec in enumerate(validation["recommendations"], 1):
                print(f"{i:>2}. {rec}")
            print()
        
        print("EXAMPLE FILES (for debugging):")
        for split_name, split_info in validation["splits"].items():
            if "example_files" in split_info:
                examples = split_info["example_files"]
                if examples:
                    print(f"{split_name.upper()}:")
                    for file_type, filename in examples.items():
                        print(f"  {file_type}: {filename}")
        
        return validation

def quick_data_check(drive_path: str = "/content/drive/MyDrive") -> bool:
    """Quick check if data structure is ready for training"""
    validator = DataStructureValidator(drive_path)
    validation = validator.validate_complete_structure()
    
    total_pairs = sum(split["matched_pairs"] for split in validation["splits"].values())
    valid_structure = validation["structure_valid"]
    
    print(f"Quick Check: {'READY' if valid_structure and total_pairs > 0 else 'NOT READY'}")
    print(f"Total matched pairs: {total_pairs}")
    
    if not valid_structure or total_pairs == 0:
        print("Run full validation for details:")
        print("validator = DataStructureValidator()")
        print("validator.print_validation_report()")
    
    return valid_structure and total_pairs > 0