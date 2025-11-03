# 🔬 TBX11K Tuberculosis Detection - YOLO Training Pipeline
# CSE475 Lab Assignment 01
# Kaggle Notebook - Complete Training Script

# ============================================================================
# CELL 1: Setup & Install Libraries
# ============================================================================
print("📦 Installing required libraries...")

!pip install -q ultralytics==8.3.0
!pip install -q torch torchvision torchaudio
!pip install -q opencv-python-headless
!pip install -q pytorch-grad-cam
!pip install -q grad-cam
!pip install -q seaborn

print("✅ All libraries installed!")

# ============================================================================
# CELL 2: Import Libraries
# ============================================================================
import os
import sys
from pathlib import Path
import shutil
import json
import time
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from PIL import Image

import torch
import torch.nn as nn
from ultralytics import YOLO
from IPython.display import display, clear_output

print(f"✅ PyTorch Version: {torch.__version__}")
print(f"✅ GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✅ GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# ============================================================================
# CELL 3: Verify Dataset
# ============================================================================
print("🔍 Verifying dataset...")

# IMPORTANT: Update this path to match your Kaggle dataset name
dataset_path = '/kaggle/input/tbx11k-yolo-balanced'  # UPDATE THIS!

# Check if dataset exists
if not Path(dataset_path).exists():
    print(f"❌ Dataset not found at: {dataset_path}")
    print("🔧 Please update 'dataset_path' variable to match your Kaggle dataset!")
    print("\nAvailable datasets:")
    !ls /kaggle/input/
    sys.exit(1)

print(f"✅ Dataset found: {dataset_path}")

# Display structure
print("\n📁 Dataset structure:")
!ls -lh {dataset_path}

# Count images
train_imgs = list(Path(f"{dataset_path}/images/train").glob("*.png"))
val_imgs = list(Path(f"{dataset_path}/images/val").glob("*.png"))

print(f"\n📊 Dataset Statistics:")
print(f"   Training images: {len(train_imgs)}")
print(f"   Validation images: {len(val_imgs)}")
print(f"   Total: {len(train_imgs) + len(val_imgs)}")

# Check data.yaml
yaml_path = Path(f"{dataset_path}/data.yaml")
if yaml_path.exists():
    print(f"\n✅ data.yaml found")
    !cat {yaml_path}
else:
    print(f"\n❌ data.yaml NOT found!")

# ============================================================================
# CELL 4: Training Configuration
# ============================================================================
class TrainingConfig:
    """Training hyperparameters"""
    
    # Dataset
    DATA_YAML = f'{dataset_path}/data.yaml'
    
    # Training parameters
    EPOCHS = 100
    BATCH_SIZE = 16
    IMG_SIZE = 512
    PATIENCE = 20  # Early stopping
    
    # Models to train
    MODELS = {
        'YOLOv10': 'yolov10n.pt',  # Nano version (fastest)
        'YOLOv11': 'yolov11n.pt',  # Nano version
        'YOLOv8': 'yolov8s.pt',    # Small version
    }
    
    # Device
    DEVICE = 0 if torch.cuda.is_available() else 'cpu'
    
    # Save paths
    PROJECT = 'runs'
    
    # Augmentation
    AUG_PARAMS = {
        'degrees': 10,        # Rotation
        'translate': 0.1,     # Translation
        'scale': 0.2,         # Scaling
        'flipud': 0.0,        # No vertical flip (X-rays)
        'fliplr': 0.5,        # Horizontal flip
        'mosaic': 0.8,        # Mosaic augmentation
        'mixup': 0.1,         # Mixup augmentation
    }

config = TrainingConfig()

print("⚙️ Training Configuration:")
print(f"   Epochs: {config.EPOCHS}")
print(f"   Batch Size: {config.BATCH_SIZE}")
print(f"   Image Size: {config.IMG_SIZE}")
print(f"   Device: {config.DEVICE}")
print(f"   Models: {list(config.MODELS.keys())}")

# ============================================================================
# CELL 5: Training Function
# ============================================================================
def train_model(model_name, model_path, config):
    """
    Train a YOLO model
    
    Args:
        model_name: Name of the model (e.g., 'YOLOv10')
        model_path: Path to pretrained weights (e.g., 'yolov10n.pt')
        config: Training configuration
    
    Returns:
        results: Training results
        metrics: Validation metrics
    """
    print(f"\n{'='*80}")
    print(f"🚀 Training {model_name}")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    try:
        # Load model
        print(f"📥 Loading model: {model_path}")
        model = YOLO(model_path)
        
        # Train
        print(f"🏋️ Starting training...")
        results = model.train(
            data=config.DATA_YAML,
            epochs=config.EPOCHS,
            imgsz=config.IMG_SIZE,
            batch=config.BATCH_SIZE,
            name=f'{model_name.lower()}_tbx11k',
            patience=config.PATIENCE,
            save=True,
            plots=True,
            cache=True,
            device=config.DEVICE,
            workers=4,
            project=config.PROJECT,
            **config.AUG_PARAMS
        )
        
        # Validate
        print(f"📊 Validating model...")
        metrics = model.val(data=config.DATA_YAML)
        
        training_time = (time.time() - start_time) / 60  # minutes
        
        print(f"\n✅ {model_name} Training Complete!")
        print(f"   Time: {training_time:.2f} minutes")
        print(f"   mAP@0.5: {metrics.box.map50:.4f}")
        print(f"   mAP@0.5:0.95: {metrics.box.map:.4f}")
        print(f"   Precision: {metrics.box.mp:.4f}")
        print(f"   Recall: {metrics.box.mr:.4f}")
        
        return results, metrics, training_time
        
    except Exception as e:
        print(f"❌ Error training {model_name}: {str(e)}")
        return None, None, None

# ============================================================================
# CELL 6: Train YOLOv10
# ============================================================================
print("\n" + "="*80)
print("🎯 PHASE 1: Training YOLOv10")
print("="*80)

yolov10_results, yolov10_metrics, yolov10_time = train_model(
    'YOLOv10',
    config.MODELS['YOLOv10'],
    config
)

# ============================================================================
# CELL 7: Train YOLOv11
# ============================================================================
print("\n" + "="*80)
print("🎯 PHASE 2: Training YOLOv11")
print("="*80)

yolov11_results, yolov11_metrics, yolov11_time = train_model(
    'YOLOv11',
    config.MODELS['YOLOv11'],
    config
)

# ============================================================================
# CELL 8: Train YOLOv8
# ============================================================================
print("\n" + "="*80)
print("🎯 PHASE 3: Training YOLOv8")
print("="*80)

yolov8_results, yolov8_metrics, yolov8_time = train_model(
    'YOLOv8',
    config.MODELS['YOLOv8'],
    config
)

# ============================================================================
# CELL 9: Compile Results
# ============================================================================
print("\n" + "="*80)
print("📊 Compiling Results...")
print("="*80)

# Create results dictionary
results_data = {
    'Model': ['YOLOv10', 'YOLOv11', 'YOLOv8'],
    'mAP@0.5': [
        yolov10_metrics.box.map50 if yolov10_metrics else 0,
        yolov11_metrics.box.map50 if yolov11_metrics else 0,
        yolov8_metrics.box.map50 if yolov8_metrics else 0,
    ],
    'mAP@0.5:0.95': [
        yolov10_metrics.box.map if yolov10_metrics else 0,
        yolov11_metrics.box.map if yolov11_metrics else 0,
        yolov8_metrics.box.map if yolov8_metrics else 0,
    ],
    'Precision': [
        yolov10_metrics.box.mp if yolov10_metrics else 0,
        yolov11_metrics.box.mp if yolov11_metrics else 0,
        yolov8_metrics.box.mp if yolov8_metrics else 0,
    ],
    'Recall': [
        yolov10_metrics.box.mr if yolov10_metrics else 0,
        yolov11_metrics.box.mr if yolov11_metrics else 0,
        yolov8_metrics.box.mr if yolov8_metrics else 0,
    ],
    'Training Time (min)': [
        yolov10_time if yolov10_time else 0,
        yolov11_time if yolov11_time else 0,
        yolov8_time if yolov8_time else 0,
    ]
}

results_df = pd.DataFrame(results_data)

print("\n📊 FINAL RESULTS:")
print(results_df.to_string(index=False))

# Save to CSV
results_df.to_csv('yolo_results.csv', index=False)
print("\n✅ Results saved to: yolo_results.csv")

# Find best model
best_model_idx = results_df['mAP@0.5'].idxmax()
best_model = results_df.loc[best_model_idx, 'Model']
best_map = results_df.loc[best_model_idx, 'mAP@0.5']

print(f"\n🏆 Best Model: {best_model} (mAP@0.5: {best_map:.4f})")

# ============================================================================
# CELL 10: Visualize Results
# ============================================================================
print("\n📊 Creating visualization...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('YOLO Models Comparison - TBX11K Dataset', fontsize=16, fontweight='bold')

metrics = ['mAP@0.5', 'mAP@0.5:0.95', 'Precision', 'Recall']
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

for idx, metric in enumerate(metrics):
    ax = axes[idx // 2, idx % 2]
    
    bars = ax.bar(results_df['Model'], results_df[metric], color=colors[idx], alpha=0.8)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_title(f'{metric}', fontsize=13, fontweight='bold')
    ax.set_ylabel(metric, fontsize=11)
    ax.set_xlabel('Model', fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, max(results_df[metric]) * 1.2)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("✅ Comparison plot saved to: model_comparison.png")

# Training time comparison
plt.figure(figsize=(10, 6))
plt.bar(results_df['Model'], results_df['Training Time (min)'], 
        color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.8)
plt.title('Training Time Comparison', fontsize=14, fontweight='bold')
plt.ylabel('Time (minutes)', fontsize=11)
plt.xlabel('Model', fontsize=11)
plt.grid(axis='y', alpha=0.3, linestyle='--')

for i, v in enumerate(results_df['Training Time (min)']):
    plt.text(i, v + 1, f'{v:.1f} min', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('training_time_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("✅ Training time plot saved to: training_time_comparison.png")

# ============================================================================
# CELL 11: Display Training Curves
# ============================================================================
print("\n📈 Displaying training curves...")

models_info = [
    ('YOLOv10', 'runs/yolov10_tbx11k'),
    ('YOLOv11', 'runs/yolov11_tbx11k'),
    ('YOLOv8', 'runs/yolov8_tbx11k'),
]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Training Curves', fontsize=16, fontweight='bold')

for idx, (name, path) in enumerate(models_info):
    results_img = Path(path) / 'results.png'
    
    if results_img.exists():
        img = plt.imread(results_img)
        axes[idx].imshow(img)
        axes[idx].set_title(name, fontsize=13, fontweight='bold')
        axes[idx].axis('off')
    else:
        axes[idx].text(0.5, 0.5, f'{name}\nResults not found',
                      ha='center', va='center', fontsize=12)
        axes[idx].axis('off')

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
plt.show()

print("✅ Training curves saved to: training_curves.png")

# ============================================================================
# CELL 12: Test Predictions on Validation Set
# ============================================================================
print("\n🎯 Testing predictions on validation images...")

# Get validation images
val_images = sorted(list(Path(f"{dataset_path}/images/val").glob("*.png")))[:6]

# Load best model
best_model_path = {
    'YOLOv10': 'runs/yolov10_tbx11k/weights/best.pt',
    'YOLOv11': 'runs/yolov11_tbx11k/weights/best.pt',
    'YOLOv8': 'runs/yolov8_tbx11k/weights/best.pt',
}[best_model]

model = YOLO(best_model_path)

# Create prediction comparison
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle(f'{best_model} Predictions on Validation Set', fontsize=16, fontweight='bold')

for idx, img_path in enumerate(val_images):
    # Run prediction
    results = model.predict(str(img_path), save=False, conf=0.25)
    
    # Plot
    ax = axes[idx // 3, idx % 3]
    result_img = results[0].plot()
    ax.imshow(result_img)
    ax.set_title(f'{img_path.name}', fontsize=11)
    ax.axis('off')

plt.tight_layout()
plt.savefig('validation_predictions.png', dpi=150, bbox_inches='tight')
plt.show()

print("✅ Validation predictions saved to: validation_predictions.png")

# ============================================================================
# CELL 13: XAI - Grad-CAM Visualization
# ============================================================================
print("\n🔍 Generating XAI Grad-CAM visualizations...")

def visualize_yolo_predictions(model, image_path, save_name='gradcam'):
    """Visualize YOLO predictions with attention"""
    
    # Load image
    img = cv2.imread(str(image_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (512, 512))
    
    # Predict
    results = model.predict(str(image_path), save=False, conf=0.25)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original image
    axes[0].imshow(img_resized)
    axes[0].set_title('Original Image', fontsize=13, fontweight='bold')
    axes[0].axis('off')
    
    # Detection result
    result_img = results[0].plot()
    axes[1].imshow(result_img)
    axes[1].set_title('YOLO Detection', fontsize=13, fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{save_name}.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print detection details
    boxes = results[0].boxes
    if len(boxes) > 0:
        print(f"\n📊 Detections for {Path(image_path).name}:")
        for i, box in enumerate(boxes):
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            print(f"   {i+1}. {results[0].names[cls]}: {conf:.4f}")
    else:
        print(f"\n❌ No detections for {Path(image_path).name}")
    
    return results

# Generate for sample images
print("\n🎯 Generating Grad-CAM for TB-positive samples...")

# Find images with TB annotations
tb_images = []
for img_path in val_images[:20]:
    label_path = Path(str(img_path).replace('/images/', '/labels/').replace('.png', '.txt'))
    if label_path.exists() and label_path.stat().st_size > 0:
        tb_images.append(img_path)
        if len(tb_images) >= 3:
            break

if len(tb_images) > 0:
    for idx, img_path in enumerate(tb_images):
        print(f"\n--- Image {idx+1} ---")
        visualize_yolo_predictions(model, img_path, f'gradcam_{idx+1}')
else:
    print("⚠️ No TB-positive images found in validation set")

print("\n✅ Grad-CAM visualizations complete!")

# ============================================================================
# CELL 14: Confusion Matrix & Detailed Metrics
# ============================================================================
print("\n📊 Analyzing confusion matrix and detailed metrics...")

# Load best model
model = YOLO(best_model_path)

# Validate with plots
metrics = model.val(data=config.DATA_YAML, plots=True, save_json=True)

# Display confusion matrix
confusion_path = Path(best_model_path).parent.parent / 'confusion_matrix.png'
if confusion_path.exists():
    img = plt.imread(confusion_path)
    plt.figure(figsize=(10, 8))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f'Confusion Matrix - {best_model}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('confusion_matrix_final.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Confusion matrix saved to: confusion_matrix_final.png")

# Display PR curve
pr_curve_path = Path(best_model_path).parent.parent / 'PR_curve.png'
if pr_curve_path.exists():
    img = plt.imread(pr_curve_path)
    plt.figure(figsize=(10, 8))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f'Precision-Recall Curve - {best_model}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('pr_curve_final.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ PR curve saved to: pr_curve_final.png")

# Display F1 curve
f1_curve_path = Path(best_model_path).parent.parent / 'F1_curve.png'
if f1_curve_path.exists():
    img = plt.imread(f1_curve_path)
    plt.figure(figsize=(10, 8))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f'F1 Score Curve - {best_model}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('f1_curve_final.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ F1 curve saved to: f1_curve_final.png")

# ============================================================================
# CELL 15: Per-Class Performance
# ============================================================================
print("\n📊 Analyzing per-class performance...")

# Get per-class metrics
print(f"\n{'='*60}")
print(f"Per-Class Performance - {best_model}")
print(f"{'='*60}")

class_names = ['Active TB', 'Obsolete TB', 'Pulmonary TB']

if hasattr(metrics.box, 'ap_class_index'):
    # Get per-class AP
    ap_per_class = metrics.box.ap50  # AP at IoU=0.5
    
    # Create DataFrame
    class_performance = pd.DataFrame({
        'Class': class_names,
        'AP@0.5': [ap_per_class[i] if i < len(ap_per_class) else 0 for i in range(3)],
    })
    
    print(class_performance.to_string(index=False))
    
    # Plot per-class performance
    plt.figure(figsize=(10, 6))
    plt.bar(class_names, class_performance['AP@0.5'], 
            color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.8)
    plt.title(f'Per-Class Average Precision - {best_model}', 
              fontsize=14, fontweight='bold')
    plt.ylabel('AP@0.5', fontsize=11)
    plt.xlabel('Class', fontsize=11)
    plt.ylim(0, 1.0)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    for i, v in enumerate(class_performance['AP@0.5']):
        plt.text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('per_class_performance.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✅ Per-class performance saved to: per_class_performance.png")
else:
    print("⚠️ Per-class metrics not available")

# ============================================================================
# CELL 16: Package Results for Download
# ============================================================================
print("\n📦 Packaging results for download...")

# Create results directory
results_dir = Path('/kaggle/working/assignment_results')
results_dir.mkdir(exist_ok=True)

# Files to copy
files_to_copy = [
    'yolo_results.csv',
    'model_comparison.png',
    'training_time_comparison.png',
    'training_curves.png',
    'validation_predictions.png',
    'confusion_matrix_final.png',
    'pr_curve_final.png',
    'f1_curve_final.png',
    'per_class_performance.png',
]

# Copy files
for file in files_to_copy:
    if Path(file).exists():
        shutil.copy(file, results_dir / file)
        print(f"✅ Copied: {file}")

# Copy Grad-CAM images
for i in range(1, 4):
    gradcam_file = f'gradcam_{i}.png'
    if Path(gradcam_file).exists():
        shutil.copy(gradcam_file, results_dir / gradcam_file)
        print(f"✅ Copied: {gradcam_file}")

# Copy best model weights
for model_name in ['YOLOv10', 'YOLOv11', 'YOLOv8']:
    model_path = f'runs/{model_name.lower()}_tbx11k/weights/best.pt'
    if Path(model_path).exists():
        shutil.copy(model_path, results_dir / f'{model_name.lower()}_best.pt')
        print(f"✅ Copied: {model_name} weights")

# Copy all training results
for model_name in ['YOLOv10', 'YOLOv11', 'YOLOv8']:
    results_png = f'runs/{model_name.lower()}_tbx11k/results.png'
    if Path(results_png).exists():
        shutil.copy(results_png, results_dir / f'{model_name.lower()}_results.png')
        print(f"✅ Copied: {model_name} training results")

# Create summary report
summary = f"""
# TBX11K Tuberculosis Detection - Training Summary
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Models Trained
{results_df.to_markdown(index=False)}

## Best Model
**{best_model}** with mAP@0.5: {best_map:.4f}

## Dataset
- Training images: {len(train_imgs)}
- Validation images: {len(val_imgs)}
- Image size: {config.IMG_SIZE}x{config.IMG_SIZE}

## Training Configuration
- Epochs: {config.EPOCHS}
- Batch size: {config.BATCH_SIZE}
- Device: {'GPU - ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}
- Early stopping patience: {config.PATIENCE}

## Files Included
- Model weights (.pt files)
- Training results (results.png files)
- Comparison plots
- Confusion matrices
- Grad-CAM visualizations
- Per-class performance

## Next Steps
1. Download assignment_results.zip
2. Review all visualizations
3. Write your report
4. Include key findings and analysis

Good luck with your assignment! 🎓
"""

with open(results_dir / 'SUMMARY.md', 'w') as f:
    f.write(summary)

print("✅ Summary report created")

# Zip everything
!cd /kaggle/working && zip -r assignment_results.zip assignment_results/

print("\n" + "="*80)
print("🎉 ALL DONE!")
print("="*80)
print(f"\n📊 Results Summary:")
print(results_df.to_string(index=False))
print(f"\n🏆 Best Model: {best_model} (mAP@0.5: {best_map:.4f})")
print(f"\n📦 Download: /kaggle/working/assignment_results.zip")
print("\n✅ All files packaged and ready for download!")
print("="*80)

# ============================================================================
# END OF NOTEBOOK
# ============================================================================
