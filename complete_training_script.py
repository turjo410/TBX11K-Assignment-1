"""
TBX11K Tuberculosis Detection - Complete Training & Evaluation Pipeline
CSE475 Lab Assignment 01

This is a complete, production-ready script for training and evaluating
multiple object detection models on the TBX11K dataset.

Author: Turjo Khan
Institution: East West University
Date: November 2025

Usage in Kaggle Notebook:
1. Copy each section into separate cells
2. Update DATASET_PATH in config
3. Run cells sequentially
4. Download results from /kaggle/working/

Total Cells: ~40 cells organized in 8 major sections
Estimated Runtime: 3-4 hours (mostly training)
"""

# ============================================================================
# CELL 1: Project Header (Markdown)
# ============================================================================
"""
# 🔬 TBX11K Tuberculosis Detection using Deep Learning

## CSE475 - Machine Learning Lab Assignment 01

**Dataset:** TBX11K - Tuberculosis X-ray Detection  
**Task:** Multi-class Object Detection  
**Models:** YOLO v8/v10/v11, RT-DETR  
**Classes:** Active TB, Obsolete TB, Pulmonary TB

---

### Research Objectives:
1. Train and compare state-of-the-art object detection models
2. Implement extensive data augmentation techniques
3. Perform Explainable AI (XAI) analysis using Grad-CAM
4. Generate comprehensive performance metrics and visualizations
5. Analyze class imbalance and its mitigation strategies
6. Create deployment-ready models with full documentation

**Expected Outputs:**
- 6 trained models with weights
- 30+ professional visualizations
- Comprehensive metrics and comparison tables
- XAI attention maps
- Research-quality final report

---
"""

# ============================================================================
# CELL 2: Install Required Packages (Python)
# ============================================================================
# Uncomment to install (first time only)
"""
!pip install -q ultralytics==8.3.0
!pip install -q torch torchvision torchaudio
!pip install -q opencv-python-headless
!pip install -q pytorch-grad-cam grad-cam
!pip install -q seaborn plotly scikit-learn
!pip install -q albumentations
!pip install -q timm
"""

print("✅ Installation commands ready (uncomment if needed)")

# ============================================================================
# CELL 3: Import All Libraries (Python)
# ============================================================================
# Core Libraries
import os
import sys
import json
import time
import warnings
import random
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any
import shutil

# Data Processing
import numpy as np
import pandas as pd
from scipy import stats

# Machine Learning
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_recall_fscore_support, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    accuracy_score, f1_score
)

# Visualization
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Image Processing
import cv2
from PIL import Image, ImageDraw, ImageFont
import albumentations as A

# Deep Learning
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# YOLO and Object Detection
from ultralytics import YOLO, RTDETR
from ultralytics.utils.plotting import Annotator, colors

# XAI (Explainable AI)
try:
    from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
    from pytorch_grad_cam.utils.image import show_cam_on_image
except ImportError:
    print("⚠️ pytorch_grad_cam not installed - XAI features disabled")

# Utilities
from tqdm.notebook import tqdm
from IPython.display import display, HTML, Image as IPImage, clear_output

# Configuration
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', '{:.4f}'.format)

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True

print("="*80)
print("✅ All libraries imported successfully!")
print("="*80)
print(f"PyTorch: {torch.__version__}, GPU: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print("="*80)

# ============================================================================
# CELL 4: Configuration Class (Python)
# ============================================================================
class Config:
    """Complete configuration for TBX11K training"""
    
    # ========== PATHS - UPDATE THIS! ==========
    DATASET_PATH = '/kaggle/input/tbx11k-yolo-balanced'  # ⚠️ CHANGE THIS!
    
    # Alternative for local:
    # DATASET_PATH = '/path/to/your/TBX11K/yolo_dataset_balanced_33_67'
    
    DATA_YAML = f'{DATASET_PATH}/data.yaml'
    TRAIN_IMG = f'{DATASET_PATH}/images/train'
    VAL_IMG = f'{DATASET_PATH}/images/val'
    TRAIN_LABEL = f'{DATASET_PATH}/labels/train'
    VAL_LABEL = f'{DATASET_PATH}/labels/val'
    
    # Output paths
    OUTPUT_DIR = Path('/kaggle/working')
    MODELS_DIR = OUTPUT_DIR / 'models'
    PLOTS_DIR = OUTPUT_DIR / 'plots'
    XAI_DIR = OUTPUT_DIR / 'xai'
    RESULTS_DIR = OUTPUT_DIR / 'results'
    
    # Create directories
    for d in [MODELS_DIR, PLOTS_DIR, XAI_DIR, RESULTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    
    # ========== DATASET ==========
    NUM_CLASSES = 3
    CLASS_NAMES = ['Active TB', 'Obsolete TB', 'Pulmonary TB']
    CLASS_COLORS = [(255,0,0), (0,255,255), (255,165,0)]
    
    # ========== TRAINING ==========
    IMG_SIZE = 512
    BATCH_SIZE = 16
    EPOCHS = 150
    PATIENCE = 25
    LEARNING_RATE = 0.001
    
    # ========== AUGMENTATION ==========
    AUG = {
        'degrees': 15.0,
        'translate': 0.15,
        'scale': 0.3,
        'shear': 5.0,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 0.8,
        'mixup': 0.15,
        'copy_paste': 0.1,
        'erasing': 0.4,
    }
    
    # ========== MODELS ==========
    MODELS = {
        'YOLOv8n': {'weights': 'yolov8n.pt', 'type': 'yolo'},
        'YOLOv8s': {'weights': 'yolov8s.pt', 'type': 'yolo'},
        'YOLOv8m': {'weights': 'yolov8m.pt', 'type': 'yolo'},
        'YOLOv10n': {'weights': 'yolov10n.pt', 'type': 'yolo'},
        'YOLOv11n': {'weights': 'yolov11n.pt', 'type': 'yolo'},
        'RT-DETR': {'weights': 'rtdetr-l.pt', 'type': 'rtdetr'},
    }
    
    # ========== EVALUATION ==========
    CONF_THRESH = 0.25
    IOU_THRESH = 0.45
    
    # ========== DEVICE ==========
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

config = Config()

print("⚙️  Configuration Loaded")
print(f"📁 Dataset: {config.DATASET_PATH}")
print(f"🖥️  Device: {config.DEVICE}")
print(f"🎯 Models: {list(config.MODELS.keys())}")

# ============================================================================
# CELL 5: Verify Dataset (Python)
# ============================================================================
print("🔍 Verifying dataset...")

paths = {
    'Root': config.DATASET_PATH,
    'YAML': config.DATA_YAML,
    'Train Images': config.TRAIN_IMG,
    'Train Labels': config.TRAIN_LABEL,
    'Val Images': config.VAL_IMG,
    'Val Labels': config.VAL_LABEL,
}

all_ok = True
for name, path in paths.items():
    exists = Path(path).exists()
    print(f"{'✅' if exists else '❌'} {name}: {path}")
    if not exists:
        all_ok = False

if not all_ok:
    print("\n⚠️ Some paths missing!")
    print("Available inputs:")
    if Path('/kaggle/input').exists():
        for item in Path('/kaggle/input').iterdir():
            print(f"  - {item.name}")
    print("\n➡️ Update config.DATASET_PATH!")
else:
    print("\n✅ All paths verified!")

# ============================================================================
# CELL 6: Dataset Analysis Function (Python)
# ============================================================================
def analyze_dataset(img_dir, label_dir, name='Dataset'):
    """Comprehensive dataset statistics"""
    img_dir, label_dir = Path(img_dir), Path(label_dir)
    
    images = list(img_dir.glob('*.png')) + list(img_dir.glob('*.jpg'))
    labels = list(label_dir.glob('*.txt'))
    
    print(f"\n{'='*80}")
    print(f"📊 {name} Analysis")
    print(f"{'='*80}")
    print(f"Images: {len(images)}, Labels: {len(labels)}")
    
    # Analyze labels
    class_counts = defaultdict(int)
    bbox_counts = []
    with_bbox = without_bbox = total_boxes = 0
    bbox_areas = []
    bbox_aspects = []
    
    for label_file in tqdm(labels, desc="Analyzing"):
        lines = label_file.read_text().strip().split('\n')
        lines = [l for l in lines if l]
        
        if not lines:
            without_bbox += 1
        else:
            with_bbox += 1
            bbox_counts.append(len(lines))
            
            for line in lines:
                parts = line.split()
                if len(parts) >= 5:
                    cls, x, y, w, h = int(parts[0]), *map(float, parts[1:5])
                    class_counts[cls] += 1
                    total_boxes += 1
                    bbox_areas.append(w * h)
                    bbox_aspects.append(w / h if h > 0 else 0)
    
    print(f"\n📦 Bounding Boxes:")
    print(f"   Total: {total_boxes}")
    print(f"   Images with TB: {with_bbox} ({with_bbox/len(images)*100:.1f}%)")
    print(f"   Images without: {without_bbox} ({without_bbox/len(images)*100:.1f}%)")
    
    if bbox_counts:
        print(f"   Boxes per image: {np.mean(bbox_counts):.2f} ± {np.std(bbox_counts):.2f}")
    
    print(f"\n🏷️  Classes:")
    for cls in sorted(class_counts.keys()):
        count = class_counts[cls]
        pct = count/total_boxes*100 if total_boxes > 0 else 0
        print(f"   {config.CLASS_NAMES[cls]}: {count} ({pct:.1f}%)")
    
    print(f"{'='*80}")
    
    return {
        'total_images': len(images),
        'with_bbox': with_bbox,
        'without_bbox': without_bbox,
        'total_boxes': total_boxes,
        'class_counts': dict(class_counts),
        'bbox_counts': bbox_counts,
        'bbox_areas': bbox_areas,
        'bbox_aspects': bbox_aspects,
    }

# Analyze datasets
train_stats = analyze_dataset(config.TRAIN_IMG, config.TRAIN_LABEL, 'Training')
val_stats = analyze_dataset(config.VAL_IMG, config.VAL_LABEL, 'Validation')

# ============================================================================
# CELL 7: Dataset Distribution Visualization (Python)
# ============================================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('TBX11K Dataset Analysis', fontsize=16, fontweight='bold')

# 1. Class distribution - Train
ax = axes[0, 0]
classes = [config.CLASS_NAMES[i] for i in range(3)]
counts = [train_stats['class_counts'].get(i, 0) for i in range(3)]
bars = ax.bar(classes, counts, color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.8)
ax.set_title('Training Set - Class Distribution', fontweight='bold')
ax.set_ylabel('Count')
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}', ha='center', va='bottom')

# 2. Class distribution - Val
ax = axes[0, 1]
counts = [val_stats['class_counts'].get(i, 0) for i in range(3)]
bars = ax.bar(classes, counts, color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.8)
ax.set_title('Validation Set - Class Distribution', fontweight='bold')
ax.set_ylabel('Count')
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}', ha='center', va='bottom')

# 3. Images with/without TB
ax = axes[0, 2]
categories = ['With TB', 'Without TB']
train_counts = [train_stats['with_bbox'], train_stats['without_bbox']]
val_counts = [val_stats['with_bbox'], val_stats['without_bbox']]
x = np.arange(len(categories))
width = 0.35
ax.bar(x - width/2, train_counts, width, label='Train', alpha=0.8, color='#FF6B6B')
ax.bar(x + width/2, val_counts, width, label='Val', alpha=0.8, color='#4ECDC4')
ax.set_title('TB Presence Distribution', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()
ax.set_ylabel('Count')

# 4. Bounding box count distribution
ax = axes[1, 0]
if train_stats['bbox_counts']:
    ax.hist(train_stats['bbox_counts'], bins=20, alpha=0.7, color='#FF6B6B', label='Train')
if val_stats['bbox_counts']:
    ax.hist(val_stats['bbox_counts'], bins=20, alpha=0.7, color='#4ECDC4', label='Val')
ax.set_title('Bounding Boxes per Image', fontweight='bold')
ax.set_xlabel('Number of Boxes')
ax.set_ylabel('Frequency')
ax.legend()

# 5. Bounding box area distribution
ax = axes[1, 1]
if train_stats['bbox_areas']:
    ax.hist(train_stats['bbox_areas'], bins=30, alpha=0.7, color='#FF6B6B', label='Train')
if val_stats['bbox_areas']:
    ax.hist(val_stats['bbox_areas'], bins=30, alpha=0.7, color='#4ECDC4', label='Val')
ax.set_title('Bounding Box Area Distribution', fontweight='bold')
ax.set_xlabel('Normalized Area')
ax.set_ylabel('Frequency')
ax.legend()

# 6. Bounding box aspect ratio
ax = axes[1, 2]
if train_stats['bbox_aspects']:
    ax.hist(train_stats['bbox_aspects'], bins=30, alpha=0.7, color='#FF6B6B', label='Train')
if val_stats['bbox_aspects']:
    ax.hist(val_stats['bbox_aspects'], bins=30, alpha=0.7, color='#4ECDC4', label='Val')
ax.set_title('Bounding Box Aspect Ratio', fontweight='bold')
ax.set_xlabel('Width / Height')
ax.set_ylabel('Frequency')
ax.legend()

plt.tight_layout()
plt.savefig(config.PLOTS_DIR / 'dataset_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"✅ Saved: {config.PLOTS_DIR / 'dataset_analysis.png'}")

# ============================================================================
# CELL 8: Visualize Sample Images with Annotations (Python)
# ============================================================================
def visualize_samples(img_dir, label_dir, n_samples=6, title='Samples'):
    """Visualize random annotated samples"""
    img_dir, label_dir = Path(img_dir), Path(label_dir)
    images = list(img_dir.glob('*.png')) + list(img_dir.glob('*.jpg'))
    
    # Get samples with bounding boxes
    samples_with_bbox = []
    for img_path in images:
        label_path = label_dir / f"{img_path.stem}.txt"
        if label_path.exists() and label_path.stat().st_size > 0:
            samples_with_bbox.append(img_path)
    
    # Select random samples
    if len(samples_with_bbox) >= n_samples:
        selected = random.sample(samples_with_bbox, n_samples)
    else:
        selected = samples_with_bbox + random.sample(
            [img for img in images if img not in samples_with_bbox],
            n_samples - len(samples_with_bbox)
        )
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'{title} - Annotated X-rays', fontsize=16, fontweight='bold')
    
    for idx, img_path in enumerate(selected):
        ax = axes[idx // 3, idx % 3]
        
        # Load image
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        
        # Load labels
        label_path = label_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            # Draw bounding boxes
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls, x_c, y_c, width, height = int(parts[0]), *map(float, parts[1:5])
                    
                    # Convert YOLO format to pixel coordinates
                    x1 = int((x_c - width/2) * w)
                    y1 = int((y_c - height/2) * h)
                    x2 = int((x_c + width/2) * w)
                    y2 = int((y_c + height/2) * h)
                    
                    # Draw rectangle
                    color = config.CLASS_COLORS[cls]
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    
                    # Add label
                    label_text = config.CLASS_NAMES[cls]
                    cv2.putText(img, label_text, (x1, y1-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        ax.imshow(img)
        ax.set_title(f'{img_path.stem}', fontsize=10)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(config.PLOTS_DIR / f'{title.lower()}_samples.png', 
                dpi=150, bbox_inches='tight')
    plt.show()

# Visualize training and validation samples
visualize_samples(config.TRAIN_IMG, config.TRAIN_LABEL, n_samples=6, title='Training')
visualize_samples(config.VAL_IMG, config.VAL_LABEL, n_samples=6, title='Validation')

print("✅ Sample visualizations complete")

# ============================================================================
# CELL 9: Augmentation Demonstration (Python)
# ============================================================================
def show_augmentation_effects(img_path, label_path):
    """Demonstrate augmentation effects"""
    # Load image
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Data Augmentation Examples', fontsize=16, fontweight='bold')
    
    # Original
    axes[0, 0].imshow(img)
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    # Rotation
    angle = 15
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h))
    axes[0, 1].imshow(rotated)
    axes[0, 1].set_title(f'Rotation ({angle}°)')
    axes[0, 1].axis('off')
    
    # Horizontal flip
    flipped = cv2.flip(img, 1)
    axes[0, 2].imshow(flipped)
    axes[0, 2].set_title('Horizontal Flip')
    axes[0, 2].axis('off')
    
    # Brightness
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 2] = hsv[:, :, 2] * 1.3
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    bright = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    axes[0, 3].imshow(bright)
    axes[0, 3].set_title('Brightness +30%')
    axes[0, 3].axis('off')
    
    # Contrast
    alpha = 1.3
    contrasted = cv2.convertScaleAbs(img, alpha=alpha, beta=0)
    axes[1, 0].imshow(contrasted)
    axes[1, 0].set_title('Contrast +30%')
    axes[1, 0].axis('off')
    
    # Blur
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    axes[1, 1].imshow(blurred)
    axes[1, 1].set_title('Gaussian Blur')
    axes[1, 1].axis('off')
    
    # Sharpening
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(img, -1, kernel)
    axes[1, 2].imshow(sharpened)
    axes[1, 2].set_title('Sharpening')
    axes[1, 2].axis('off')
    
    # Scaling
    scale = 0.8
    scaled = cv2.resize(img, None, fx=scale, fy=scale)
    pad_h = (h - scaled.shape[0]) // 2
    pad_w = (w - scaled.shape[1]) // 2
    scaled_padded = np.zeros_like(img)
    scaled_padded[pad_h:pad_h+scaled.shape[0], pad_w:pad_w+scaled.shape[1]] = scaled
    axes[1, 3].imshow(scaled_padded)
    axes[1, 3].set_title('Scaling (80%)')
    axes[1, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig(config.PLOTS_DIR / 'augmentation_demo.png', dpi=150, bbox_inches='tight')
    plt.show()

# Get a sample image with TB
train_images = list(Path(config.TRAIN_IMG).glob('*.png'))
sample_imgs_with_tb = []
for img in train_images:
    label_path = Path(config.TRAIN_LABEL) / f"{img.stem}.txt"
    if label_path.exists() and label_path.stat().st_size > 0:
        sample_imgs_with_tb.append(img)
        if len(sample_imgs_with_tb) >= 1:
            break

if sample_imgs_with_tb:
    show_augmentation_effects(sample_imgs_with_tb[0], 
                             Path(config.TRAIN_LABEL) / f"{sample_imgs_with_tb[0].stem}.txt")
    print("✅ Augmentation demonstration complete")
else:
    print("⚠️ No TB samples found for augmentation demo")

# Continue in next message due to length...
print("\n" + "="*80)
print("📝 NOTE: This is a partial script showing first 9 cells")
print("   Complete notebook has 40+ cells covering:")
print("   - Model training (YOLO v8/v10/v11, RT-DETR)")
print("   - Comprehensive evaluation & metrics")
print("   - XAI analysis (Grad-CAM)")
print("   - Professional visualizations")
print("   - Results export and reporting")
print("="*80)
