"""
Prediction visualization and error analysis for SAR flood segmentation
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import tensorflow as tf
from pathlib import Path
import json
from typing import List, Dict, Optional
import logging

from ..utils.config import EvaluationConfig

logger = logging.getLogger(__name__)

class MetricsCalculator:
    """Simple metrics calculator for visualization purposes"""
    
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate basic metrics for visualization"""
        y_true = np.asarray(y_true, dtype=np.float32)
        y_pred = np.asarray(y_pred, dtype=np.float32)
        
        y_pred_binary = (y_pred > self.threshold).astype(np.float32)
        y_true_binary = (y_true > 0.5).astype(np.float32)
        
        y_true_flat = y_true_binary.flatten()
        y_pred_flat = y_pred_binary.flatten()
        
        tp = np.sum(y_true_flat * y_pred_flat)
        fp = np.sum((1 - y_true_flat) * y_pred_flat)
        fn = np.sum(y_true_flat * (1 - y_pred_flat))
        tn = np.sum((1 - y_true_flat) * (1 - y_pred_flat))
        
        total_pixels = tp + fp + fn + tn
        pixel_accuracy = (tp + tn) / total_pixels if total_pixels > 0 else 0.0
        dice_coefficient = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        return {
            'pixel_accuracy': float(pixel_accuracy),
            'dice_coefficient': float(dice_coefficient),
            'precision': float(precision),
            'recall': float(recall)
        }
    
    def calculate_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
        """Calculate confusion matrix components"""
        y_true = np.asarray(y_true, dtype=np.float32)
        y_pred = np.asarray(y_pred, dtype=np.float32)
        
        y_pred_binary = (y_pred > self.threshold).astype(np.float32)
        y_true_binary = (y_true > 0.5).astype(np.float32)
        
        y_true_flat = y_true_binary.flatten()
        y_pred_flat = y_pred_binary.flatten()
        
        tp = int(np.sum(y_true_flat * y_pred_flat))
        fp = int(np.sum((1 - y_true_flat) * y_pred_flat))
        fn = int(np.sum(y_true_flat * (1 - y_pred_flat)))
        tn = int(np.sum((1 - y_true_flat) * (1 - y_pred_flat)))
        
        return {
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'true_negatives': tn
        }

class FloodSegmentationVisualizer:
    """Visualizer for flood segmentation predictions and error analysis"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.metrics_calc = MetricsCalculator(threshold=config.threshold)
        
        # Color maps for visualization
        self.water_cmap = ListedColormap(['white', 'blue'])
        self.error_cmap = ListedColormap(['white', 'red', 'cyan', 'green'])
        
    def visualize_prediction_comparison(self, 
                                      sar_image: np.ndarray,
                                      true_mask: np.ndarray, 
                                      pred_mask: np.ndarray,
                                      title: str = "Prediction Comparison",
                                      save_path: Optional[Path] = None) -> plt.Figure:
        """Create comparison visualization of SAR, ground truth, and prediction"""
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(title, fontsize=16)
        
        # Normalize SAR image for display (RGB)
        if sar_image.shape[-1] == 3:
            sar_display = self._normalize_sar_for_display(sar_image)
        else:
            sar_display = sar_image
        
        # Row 1: SAR RGB channels
        axes[0, 0].imshow(sar_display[:, :, 0], cmap='gray')
        axes[0, 0].set_title('SAR Red Channel (HH-VV)')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(sar_display[:, :, 1], cmap='gray')
        axes[0, 1].set_title('SAR Green Channel (HV)')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(sar_display[:, :, 2], cmap='gray')
        axes[0, 2].set_title('SAR Blue Channel (HH+VV)')
        axes[0, 2].axis('off')
        
        # Row 2: Ground truth, prediction, and error analysis
        axes[1, 0].imshow(true_mask.squeeze(), cmap=self.water_cmap, vmin=0, vmax=1)
        axes[1, 0].set_title('Ground Truth')
        axes[1, 0].axis('off')
        
        pred_binary = (pred_mask > self.config.threshold).astype(np.float32)
        axes[1, 1].imshow(pred_binary.squeeze(), cmap=self.water_cmap, vmin=0, vmax=1)
        axes[1, 1].set_title('Prediction')
        axes[1, 1].axis('off')
        
        # Error analysis
        error_map = self._create_error_map(true_mask.squeeze(), pred_binary.squeeze())
        im = axes[1, 2].imshow(error_map, cmap=self.error_cmap, vmin=0, vmax=3)
        axes[1, 2].set_title('Error Analysis')
        axes[1, 2].axis('off')
        
        # Add colorbar for error map
        cbar = plt.colorbar(im, ax=axes[1, 2], shrink=0.8)
        cbar.set_ticks([0, 1, 2, 3])
        cbar.set_ticklabels(['True Negative', 'False Positive', 'False Negative', 'True Positive'])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to {save_path}")
        
        return fig
    
    def visualize_false_predictions(self,
                                  sar_image: np.ndarray,
                                  true_mask: np.ndarray,
                                  pred_mask: np.ndarray,
                                  title: str = "False Positive/Negative Analysis",
                                  save_path: Optional[Path] = None) -> plt.Figure:
        """Create visualization focusing on false positive and false negative predictions"""
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(title, fontsize=16)
        
        # Create SAR RGB composite for display
        sar_rgb = self._create_sar_rgb_composite(sar_image)
        pred_binary = (pred_mask > self.config.threshold).astype(np.float32)
        
        # Calculate false positives and false negatives
        false_positives = ((true_mask.squeeze() == 0) & (pred_binary.squeeze() == 1)).astype(np.float32)
        false_negatives = ((true_mask.squeeze() == 1) & (pred_binary.squeeze() == 0)).astype(np.float32)
        
        # Top row: SAR with overlays
        axes[0, 0].imshow(sar_rgb)
        axes[0, 0].imshow(false_positives, cmap='Reds', alpha=0.7, vmin=0, vmax=1)
        axes[0, 0].set_title('False Positives (Red Overlay)')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(sar_rgb)
        axes[0, 1].imshow(false_negatives, cmap='Blues', alpha=0.7, vmin=0, vmax=1)
        axes[0, 1].set_title('False Negatives (Blue Overlay)')
        axes[0, 1].axis('off')
        
        # Bottom row: Error maps
        axes[1, 0].imshow(false_positives, cmap='Reds', vmin=0, vmax=1)
        axes[1, 0].set_title('False Positives Only')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(false_negatives, cmap='Blues', vmin=0, vmax=1)
        axes[1, 1].set_title('False Negatives Only')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"False prediction visualization saved to {save_path}")
        
        return fig
    
    def create_metrics_overlay(self,
                             sar_image: np.ndarray,
                             true_mask: np.ndarray,
                             pred_mask: np.ndarray,
                             save_path: Optional[Path] = None) -> plt.Figure:
        """Create visualization with metrics overlay"""
        
        # Calculate metrics
        metrics = self.metrics_calc.calculate_metrics(true_mask, pred_mask)
        confusion_matrix = self.metrics_calc.calculate_confusion_matrix(true_mask, pred_mask)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # SAR RGB composite
        sar_rgb = self._create_sar_rgb_composite(sar_image)
        axes[0].imshow(sar_rgb)
        axes[0].set_title('SAR RGB Composite')
        axes[0].axis('off')
        
        # Ground truth vs prediction
        axes[1].imshow(sar_rgb)
        axes[1].imshow(true_mask.squeeze(), cmap='Blues', alpha=0.5, vmin=0, vmax=1)
        pred_binary = (pred_mask > self.config.threshold).astype(np.float32)
        axes[1].contour(pred_binary.squeeze(), colors='red', linewidths=2, levels=[0.5])
        axes[1].set_title('Ground Truth (Blue) vs Prediction (Red)')
        axes[1].axis('off')
        
        # Error analysis with metrics
        error_map = self._create_error_map(true_mask.squeeze(), pred_binary.squeeze())
        im = axes[2].imshow(error_map, cmap=self.error_cmap, vmin=0, vmax=3)
        axes[2].set_title('Error Analysis')
        axes[2].axis('off')
        
        # Add metrics text
        metrics_text = f"""Metrics:
Pixel Accuracy: {metrics['pixel_accuracy']:.3f}
DICE Coefficient: {metrics['dice_coefficient']:.3f}
Precision: {metrics['precision']:.3f}
Recall: {metrics['recall']:.3f}

Confusion Matrix:
TP: {confusion_matrix['true_positives']}
FP: {confusion_matrix['false_positives']}
FN: {confusion_matrix['false_negatives']}
TN: {confusion_matrix['true_negatives']}"""
        
        plt.figtext(0.02, 0.98, metrics_text, fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Metrics overlay visualization saved to {save_path}")
        
        return fig
    
    def _normalize_sar_for_display(self, sar_image: np.ndarray) -> np.ndarray:
        """Normalize SAR image for display"""
        # Handle different input ranges
        if sar_image.max() <= 1.0:
            # Already normalized
            display_image = np.clip(sar_image, 0, 1)
        else:
            # Normalize from 0-255 or other range
            display_image = sar_image - sar_image.min()
            display_image = display_image / display_image.max()
        
        # Apply gamma correction for better visualization
        display_image = np.power(display_image, 0.7)
        
        return display_image
    
    def _create_sar_rgb_composite(self, sar_image: np.ndarray) -> np.ndarray:
        """Create RGB composite from SAR channels"""
        if sar_image.shape[-1] == 3:
            rgb = self._normalize_sar_for_display(sar_image)
        else:
            # If single channel, replicate across RGB
            single_channel = self._normalize_sar_for_display(sar_image.squeeze())
            rgb = np.stack([single_channel] * 3, axis=-1)
        
        return rgb
    
    def _create_error_map(self, true_mask: np.ndarray, pred_mask: np.ndarray) -> np.ndarray:
        """Create error map: 0=TN, 1=FP, 2=FN, 3=TP"""
        error_map = np.zeros_like(true_mask, dtype=np.uint8)
        
        # True Negatives (0) - already set to 0
        # False Positives (1)
        error_map[(true_mask == 0) & (pred_mask == 1)] = 1
        # False Negatives (2)
        error_map[(true_mask == 1) & (pred_mask == 0)] = 2
        # True Positives (3)
        error_map[(true_mask == 1) & (pred_mask == 1)] = 3
        
        return error_map
    
    def generate_sample_visualizations(self,
                                     model: tf.keras.Model,
                                     dataset: tf.data.Dataset,
                                     output_dir: Path,
                                     num_samples: int = None) -> Dict:
        """Generate visualizations for sample predictions"""
        
        if num_samples is None:
            num_samples = self.config.visualization_samples
        
        output_dir = Path(output_dir) / 'visualizations'
        output_dir.mkdir(exist_ok=True)
        
        sample_metrics = []
        
        # Take samples from dataset
        sample_count = 0
        for batch_idx, (sar_batch, mask_batch) in enumerate(dataset.take(num_samples // 8 + 1)):
            # Get predictions
            pred_batch = model.predict(sar_batch, verbose=0)
            
            for i in range(sar_batch.shape[0]):
                if sample_count >= num_samples:
                    break
                
                sar_image = sar_batch[i].numpy()
                true_mask = mask_batch[i].numpy()
                pred_mask = pred_batch[i]
                
                # Calculate metrics for this sample
                metrics = self.metrics_calc.calculate_metrics(true_mask, pred_mask)
                sample_metrics.append({
                    'sample_id': f"sample_{sample_count:03d}",
                    'batch_idx': batch_idx,
                    'sample_idx': i,
                    **metrics
                })
                
                # Create visualizations
                sample_name = f"sample_{sample_count:03d}"
                
                # Comparison visualization
                comparison_path = output_dir / f"{sample_name}_comparison.png"
                self.visualize_prediction_comparison(
                    sar_image, true_mask, pred_mask,
                    title=f"Sample {sample_count} - DICE: {metrics['dice_coefficient']:.3f}",
                    save_path=comparison_path
                )
                plt.close()
                
                # False predictions visualization
                fp_fn_path = output_dir / f"{sample_name}_false_predictions.png"
                self.visualize_false_predictions(
                    sar_image, true_mask, pred_mask,
                    title=f"Sample {sample_count} - Error Analysis",
                    save_path=fp_fn_path
                )
                plt.close()
                
                sample_count += 1
            
            if sample_count >= num_samples:
                break
        
        # Save sample metrics
        with open(output_dir / 'sample_metrics.json', 'w') as f:
            json.dump(sample_metrics, f, indent=2)
        
        # Generate summary statistics
        metrics_summary = self._calculate_visualization_summary(sample_metrics)
        with open(output_dir / 'visualization_summary.json', 'w') as f:
            json.dump(metrics_summary, f, indent=2)
        
        logger.info(f"Generated {sample_count} sample visualizations in {output_dir}")
        
        return {
            'num_samples': sample_count,
            'output_directory': str(output_dir),
            'metrics_summary': metrics_summary
        }
    
    def _calculate_visualization_summary(self, sample_metrics: List[Dict]) -> Dict:
        """Calculate summary statistics from sample metrics"""
        if not sample_metrics:
            return {}
        
        metrics_arrays = {
            'pixel_accuracy': [m['pixel_accuracy'] for m in sample_metrics],
            'dice_coefficient': [m['dice_coefficient'] for m in sample_metrics],
            'precision': [m['precision'] for m in sample_metrics],
            'recall': [m['recall'] for m in sample_metrics]
        }
        
        summary = {}
        for metric_name, values in metrics_arrays.items():
            values = np.array(values)
            summary[metric_name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values))
            }
        
        return summary