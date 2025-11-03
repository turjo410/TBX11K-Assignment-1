# 🎯 COMPLETE NOTEBOOK - YOLOv10, YOLOv11, YOLOv12 Training
# CSE475 Lab Assignment 01 - TBX11K Tuberculosis Detection

## ⚠️ IMPORTANT: Models to Train
- ✅ **YOLOv10** (nano, small, medium)
- ✅ **YOLOv11** (nano, small, medium) 
- ✅ **YOLOv12** (IF AVAILABLE - otherwise use YOLOv8m as latest stable)
- ✅ **RT-DETR** (Bonus - Transformer-based)

**Note:** YOLOv12 might not be released yet. If unavailable, we'll use YOLOv8m/YOLOv11m as the third model.

---

## 📋 How to Use This Notebook

1. Copy each cell below into Kaggle/Jupyter
2. Update `DATASET_PATH` in Cell 4
3. Run all cells sequentially
4. Download results from `/kaggle/working/`

**Total Cells:** 30 cells  
**Runtime:** 3-4 hours with GPU

---

# ============================================================================
# CELL 1 (Markdown): Project Title
# ============================================================================

# 🔬 TBX11K Tuberculosis Detection
## Training YOLOv10, YOLOv11, YOLOv12 for TB Detection

**Student:** Turjo Khan  
**Institution:** East West University  
**Course:** CSE475 - Machine Learning  
**Assignment:** Lab Assignment 01  
**Date:** November 2025

---

### 📋 Assignment Objectives

As per the PDF requirements:

1. ✅ **Train YOLOv10** - Latest YOLO architecture
2. ✅ **Train YOLOv11** - Newest YOLO version
3. ✅ **Train YOLOv12** - Most recent (or YOLOv8 if v12 unavailable)
4. ✅ **Extensive Data Augmentation** - Rotation, flip, color, mosaic, mixup
5. ✅ **XAI Analysis** - Grad-CAM for explainability
6. ✅ **Comprehensive Visualizations** - 40+ professional plots
7. ✅ **Model Comparison** - Detailed performance metrics
8. ✅ **Bonus Models** - RT-DETR (Transformer-based detector)

---

### 📊 Dataset: TBX11K

- **Training:** 1,797 images (33% TB+, 67% TB-)
- **Validation:** 600 images (33% TB+, 67% TB-)
- **Classes:** 3 (Active TB, Obsolete TB, Pulmonary TB)
- **Image Size:** 512x512
- **Format:** YOLO (balanced dataset)

---

### ⏱️ Expected Runtime

- Setup: 5 min
- Data Analysis: 10 min
- Training (6 models): 2-3 hours ⚡
- Evaluation: 30 min
- XAI Analysis: 20 min
- **Total: 3-4 hours**

---

**Let's begin!** 🚀

# ============================================================================
# CELL 2 (Python): Install Packages
# ============================================================================

# Install required packages (uncomment if first time)
!pip install -q ultralytics==8.3.0
!pip install -q pytorch-grad-cam grad-cam
!pip install -q albumentations
!pip install -q plotly

print("✅ Packages installed successfully!")
print("Note: If YOLOv12 is not available, we'll use YOLOv11m/YOLOv8m instead.")

# ============================================================================
# CELL 3 (Python): Import Libraries
# ============================================================================

# Core libraries
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import random
import shutil
import time
from collections import defaultdict, Counter

# Data & ML
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# Image processing
import cv2
from PIL import Image

# Deep Learning
import torch
from ultralytics import YOLO, RTDETR
from tqdm.notebook import tqdm

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Matplotlib config
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
pd.set_option('display.float_format', '{:.4f}'.format')

print("="*80)
print("✅ All libraries imported successfully!")
print("="*80)
print(f"PyTorch: {torch.__version__}")
print(f"GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print("="*80)

# ============================================================================
# CELL 4 (Python): Configuration - ⚠️ UPDATE DATASET_PATH!
# ============================================================================

class Config:
    """Complete configuration for TBX11K training"""
    
    # ========== DATASET PATHS - ⚠️ UPDATE THIS! ==========
    # For Kaggle:
    DATASET_PATH = '/kaggle/input/tbx11k-yolo-balanced'  # ⚠️ CHANGE THIS!
    
    # For Local (uncomment if running locally):
    # DATASET_PATH = '/Users/turjokhan/Study EWU CSE /10th Semester/CSE475/Assignement 1/TBX11K/yolo_dataset_balanced_33_67'
    
    # Data paths
    DATA_YAML = f'{DATASET_PATH}/data.yaml'
    TRAIN_IMG = f'{DATASET_PATH}/images/train'
    VAL_IMG = f'{DATASET_PATH}/images/val'
    TRAIN_LABEL = f'{DATASET_PATH}/labels/train'
    VAL_LABEL = f'{DATASET_PATH}/labels/val'
    
    # Output directories
    OUT = Path('/kaggle/working')  # For Kaggle
    # OUT = Path('./outputs')  # For local
    
    MODELS_DIR = OUT / 'models'
    PLOTS_DIR = OUT / 'plots'
    XAI_DIR = OUT / 'xai'
    RESULTS_DIR = OUT / 'results'
    
    # Create directories
    for d in [MODELS_DIR, PLOTS_DIR, XAI_DIR, RESULTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    
    # ========== TRAINING PARAMETERS ==========
    IMG_SIZE = 512
    BATCH_SIZE = 16
    EPOCHS = 150
    PATIENCE = 25
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    # ========== DATASET INFO ==========
    NUM_CLASSES = 3
    CLASS_NAMES = ['Active TB', 'Obsolete TB', 'Pulmonary TB']
    CLASS_COLORS = [(255,0,0), (0,255,255), (255,165,0)]
    
    # ========== MODELS TO TRAIN (AS PER PDF) ==========
    MODELS_TO_TRAIN = {
        # YOLOv10 variants
        'YOLOv10n': 'yolov10n.pt',
        'YOLOv10s': 'yolov10s.pt',
        'YOLOv10m': 'yolov10m.pt',
        
        # YOLOv11 variants
        'YOLOv11n': 'yolov11n.pt',
        'YOLOv11s': 'yolov11s.pt',
        'YOLOv11m': 'yolov11m.pt',
        
        # YOLOv12 (if available, otherwise will skip)
        'YOLOv12n': 'yolov12n.pt',
        'YOLOv12s': 'yolov12s.pt',
        'YOLOv12m': 'yolov12m.pt',
        
        # Bonus: RT-DETR (Transformer-based)
        'RT-DETR-l': 'rtdetr-l.pt',
    }
    
    # ========== AUGMENTATION CONFIG ==========
    AUGMENTATION = {
        # Geometric
        'degrees': 15.0,          # Rotation ±15°
        'translate': 0.15,        # Translation ±15%
        'scale': 0.3,             # Scaling 70-130%
        'shear': 5.0,             # Shearing ±5°
        'perspective': 0.0005,    # Perspective distortion
        
        # Color
        'hsv_h': 0.015,           # Hue
        'hsv_s': 0.7,             # Saturation
        'hsv_v': 0.4,             # Value/Brightness
        
        # Spatial
        'flipud': 0.0,            # No vertical flip (X-rays)
        'fliplr': 0.5,            # Horizontal flip 50%
        'mosaic': 0.8,            # Mosaic augmentation
        'mixup': 0.15,            # Mixup augmentation
        'copy_paste': 0.1,        # Copy-paste
        'erasing': 0.4,           # Random erasing
    }
    
    # ========== EVALUATION ==========
    CONF_THRESHOLD = 0.25
    IOU_THRESHOLD = 0.45

cfg = Config()

print("⚙️  CONFIGURATION LOADED")
print("="*80)
print(f"📁 Dataset: {cfg.DATASET_PATH}")
print(f"🖥️  Device: {cfg.DEVICE}")
print(f"🖼️  Image Size: {cfg.IMG_SIZE}x{cfg.IMG_SIZE}")
print(f"📦 Batch Size: {cfg.BATCH_SIZE}")
print(f"🔄 Epochs: {cfg.EPOCHS}")
print(f"🎯 Classes: {cfg.NUM_CLASSES}")
print(f"\n🤖 Models to train ({len(cfg.MODELS_TO_TRAIN)} models):")
for name in cfg.MODELS_TO_TRAIN.keys():
    print(f"   - {name}")
print("="*80)

# ============================================================================
# CELL 5 (Python): Verify Dataset Paths
# ============================================================================

print("🔍 Verifying dataset structure...")
print("="*80)

paths_to_check = {
    'Dataset Root': cfg.DATASET_PATH,
    'Data YAML': cfg.DATA_YAML,
    'Train Images': cfg.TRAIN_IMG,
    'Train Labels': cfg.TRAIN_LABEL,
    'Val Images': cfg.VAL_IMG,
    'Val Labels': cfg.VAL_LABEL,
}

all_exist = True
for name, path in paths_to_check.items():
    exists = Path(path).exists()
    status = "✅" if exists else "❌"
    print(f"{status} {name}")
    if not exists:
        all_exist = False

if not all_exist:
    print("\n⚠️  WARNING: Some paths don't exist!")
    print("📝 Available inputs in Kaggle:")
    !ls /kaggle/input/
    print("\n➡️  Please update cfg.DATASET_PATH in Cell 4!")
else:
    print("\n✅ All paths verified successfully!")
    
    # Count files
    train_imgs = len(list(Path(cfg.TRAIN_IMG).glob('*.png')))
    val_imgs = len(list(Path(cfg.VAL_IMG).glob('*.png')))
    print(f"\n📊 Dataset Size:")
    print(f"   Training images: {train_imgs}")
    print(f"   Validation images: {val_imgs}")
    print(f"   Total: {train_imgs + val_imgs}")

print("="*80)

# ============================================================================
# CELL 6 (Markdown): Data Analysis Section
# ============================================================================

## 📊 Section 2: Dataset Analysis & Exploration

Performing comprehensive analysis of the TBX11K dataset:
- Image and label statistics
- Class distribution
- Bounding box analysis
- Data balance verification

# ============================================================================
# CELL 7 (Python): Dataset Analysis Function
# ============================================================================

def analyze_dataset_split(img_dir, label_dir, split_name='Dataset'):
    """Comprehensive dataset analysis"""
    img_dir = Path(img_dir)
    label_dir = Path(label_dir)
    
    images = list(img_dir.glob('*.png')) + list(img_dir.glob('*.jpg'))
    labels = list(label_dir.glob('*.txt'))
    
    print(f"\n{'='*80}")
    print(f"📊 {split_name} Analysis")
    print(f"{'='*80}")
    
    # Basic counts
    print(f"Total images: {len(images)}")
    print(f"Total labels: {len(labels)}")
    
    # Analyze labels
    class_counts = defaultdict(int)
    bbox_counts = []
    with_tb = without_tb = total_boxes = 0
    bbox_areas = []
    bbox_aspects = []
    
    for label_file in tqdm(labels, desc=f"Analyzing {split_name}"):
        lines = label_file.read_text().strip().split('\n')
        lines = [l for l in lines if l]
        
        if not lines:
            without_tb += 1
        else:
            with_tb += 1
            bbox_counts.append(len(lines))
            
            for line in lines:
                parts = line.split()
                if len(parts) >= 5:
                    cls = int(parts[0])
                    x_c, y_c, w, h = map(float, parts[1:5])
                    
                    class_counts[cls] += 1
                    total_boxes += 1
                    bbox_areas.append(w * h)  # Normalized area
                    bbox_aspects.append(w / h if h > 0 else 0)
    
    # Print statistics
    print(f"\n📦 Bounding Box Statistics:")
    print(f"   Total boxes: {total_boxes}")
    print(f"   Images with TB: {with_tb} ({with_tb/len(images)*100:.2f}%)")
    print(f"   Images without TB: {without_tb} ({without_tb/len(images)*100:.2f}%)")
    print(f"   Balance ratio: 1:{without_tb/with_tb:.2f}")
    
    if bbox_counts:
        print(f"\n   Boxes per positive image:")
        print(f"      Mean: {np.mean(bbox_counts):.2f}")
        print(f"      Median: {np.median(bbox_counts):.0f}")
        print(f"      Min: {min(bbox_counts)}, Max: {max(bbox_counts)}")
    
    print(f"\n🏷️  Class Distribution:")
    for cls in sorted(class_counts.keys()):
        count = class_counts[cls]
        pct = (count / total_boxes * 100) if total_boxes > 0 else 0
        print(f"   Class {cls} ({cfg.CLASS_NAMES[cls]}): {count} ({pct:.2f}%)")
    
    if bbox_areas:
        print(f"\n📏 Bounding Box Properties:")
        print(f"   Area (normalized):")
        print(f"      Mean: {np.mean(bbox_areas):.4f}")
        print(f"      Median: {np.median(bbox_areas):.4f}")
        print(f"   Aspect Ratio (W/H):")
        print(f"      Mean: {np.mean(bbox_aspects):.2f}")
        print(f"      Median: {np.median(bbox_aspects):.2f}")
    
    print(f"{'='*80}\n")
    
    return {
        'total_images': len(images),
        'with_tb': with_tb,
        'without_tb': without_tb,
        'total_boxes': total_boxes,
        'class_counts': dict(class_counts),
        'bbox_counts': bbox_counts,
        'bbox_areas': bbox_areas,
        'bbox_aspects': bbox_aspects,
    }

# Analyze both splits
train_stats = analyze_dataset_split(cfg.TRAIN_IMG, cfg.TRAIN_LABEL, 'Training Set')
val_stats = analyze_dataset_split(cfg.VAL_IMG, cfg.VAL_LABEL, 'Validation Set')

# ============================================================================
# CELL 8 (Python): Visualize Dataset Distribution
# ============================================================================

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('TBX11K Dataset Distribution Analysis', fontsize=16, fontweight='bold')

# 1. Training class distribution
ax = axes[0, 0]
classes = cfg.CLASS_NAMES
train_counts = [train_stats['class_counts'].get(i, 0) for i in range(3)]
bars = ax.bar(range(3), train_counts, color=['#FF6B6B','#4ECDC4','#45B7D1'], alpha=0.8)
ax.set_title('Training Set - Class Distribution', fontweight='bold', fontsize=12)
ax.set_ylabel('Count')
ax.set_xticks(range(3))
ax.set_xticklabels(classes, rotation=15, ha='right')
ax.grid(axis='y', alpha=0.3)
for i, bar in enumerate(bars):
    h = bar.get_height()
    ax.text(bar.get_x()+bar.get_width()/2, h, f'{int(h)}', ha='center', va='bottom')

# 2. Validation class distribution
ax = axes[0, 1]
val_counts = [val_stats['class_counts'].get(i, 0) for i in range(3)]
bars = ax.bar(range(3), val_counts, color=['#FF6B6B','#4ECDC4','#45B7D1'], alpha=0.8)
ax.set_title('Validation Set - Class Distribution', fontweight='bold', fontsize=12)
ax.set_ylabel('Count')
ax.set_xticks(range(3))
ax.set_xticklabels(classes, rotation=15, ha='right')
ax.grid(axis='y', alpha=0.3)
for i, bar in enumerate(bars):
    h = bar.get_height()
    ax.text(bar.get_x()+bar.get_width()/2, h, f'{int(h)}', ha='center', va='bottom')

# 3. TB Presence comparison
ax = axes[0, 2]
categories = ['With TB', 'Without TB']
x = np.arange(len(categories))
w = 0.35
ax.bar(x-w/2, [train_stats['with_tb'], train_stats['without_tb']], 
       w, label='Train', color='#FF6B6B', alpha=0.8)
ax.bar(x+w/2, [val_stats['with_tb'], val_stats['without_tb']], 
       w, label='Val', color='#4ECDC4', alpha=0.8)
ax.set_title('TB Presence Distribution', fontweight='bold', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()
ax.set_ylabel('Count')
ax.grid(axis='y', alpha=0.3)

# 4. Boxes per image distribution
ax = axes[1, 0]
if train_stats['bbox_counts']:
    ax.hist(train_stats['bbox_counts'], bins=15, alpha=0.7, color='#FF6B6B', label='Train', edgecolor='black')
if val_stats['bbox_counts']:
    ax.hist(val_stats['bbox_counts'], bins=15, alpha=0.7, color='#4ECDC4', label='Val', edgecolor='black')
ax.set_title('Bounding Boxes per Image', fontweight='bold', fontsize=12)
ax.set_xlabel('Number of Boxes')
ax.set_ylabel('Frequency')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# 5. Bounding box area distribution
ax = axes[1, 1]
if train_stats['bbox_areas']:
    ax.hist(train_stats['bbox_areas'], bins=30, alpha=0.7, color='#FF6B6B', label='Train', edgecolor='black')
if val_stats['bbox_areas']:
    ax.hist(val_stats['bbox_areas'], bins=30, alpha=0.7, color='#4ECDC4', label='Val', edgecolor='black')
ax.set_title('Bounding Box Area (Normalized)', fontweight='bold', fontsize=12)
ax.set_xlabel('Area')
ax.set_ylabel('Frequency')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# 6. Aspect ratio distribution
ax = axes[1, 2]
if train_stats['bbox_aspects']:
    ax.hist(train_stats['bbox_aspects'], bins=30, alpha=0.7, color='#FF6B6B', label='Train', edgecolor='black')
if val_stats['bbox_aspects']:
    ax.hist(val_stats['bbox_aspects'], bins=30, alpha=0.7, color='#4ECDC4', label='Val', edgecolor='black')
ax.set_title('Bounding Box Aspect Ratio (W/H)', fontweight='bold', fontsize=12)
ax.set_xlabel('Aspect Ratio')
ax.set_ylabel('Frequency')
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(cfg.PLOTS_DIR / 'dataset_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"✅ Saved: {cfg.PLOTS_DIR / 'dataset_distribution.png'}")

# ============================================================================
# CELL 9 (Python): Visualize Sample Annotated Images
# ============================================================================

def show_annotated_samples(img_dir, label_dir, n=6, title='Samples'):
    """Display sample images with TB annotations"""
    img_dir = Path(img_dir)
    label_dir = Path(label_dir)
    
    images = list(img_dir.glob('*.png'))
    
    # Get images with TB
    tb_images = []
    for img in images:
        lbl = label_dir / f"{img.stem}.txt"
        if lbl.exists() and lbl.stat().st_size > 0:
            tb_images.append(img)
    
    if len(tb_images) < n:
        print(f"⚠️  Only {len(tb_images)} TB-positive images found")
        samples = tb_images
    else:
        samples = random.sample(tb_images, n)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'{title} - Annotated Chest X-rays', fontsize=16, fontweight='bold')
    
    for idx, img_path in enumerate(samples):
        ax = axes[idx//3, idx%3]
        
        # Load image
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        
        # Load and draw annotations
        lbl_path = label_dir / f"{img_path.stem}.txt"
        with open(lbl_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls = int(parts[0])
                    x_c, y_c, bw, bh = map(float, parts[1:5])
                    
                    # Convert to pixel coordinates
                    x1 = int((x_c - bw/2) * w)
                    y1 = int((y_c - bh/2) * h)
                    x2 = int((x_c + bw/2) * w)
                    y2 = int((y_c + bh/2) * h)
                    
                    # Draw
                    color = cfg.CLASS_COLORS[cls]
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                    label_text = cfg.CLASS_NAMES[cls]
                    cv2.putText(img, label_text, (x1, y1-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        ax.imshow(img)
        ax.set_title(f'{img_path.stem}', fontsize=10)
        ax.axis('off')
    
    plt.tight_layout()
    save_path = cfg.PLOTS_DIR / f'{title.lower().replace(" ", "_")}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✅ Saved: {save_path}")

# Visualize samples
show_annotated_samples(cfg.TRAIN_IMG, cfg.TRAIN_LABEL, n=6, title='Training Samples')
show_annotated_samples(cfg.VAL_IMG, cfg.VAL_LABEL, n=6, title='Validation Samples')

# ============================================================================
# CELL 10 (Markdown): Training Section
# ============================================================================

## 🚀 Section 3: Model Training

Training multiple YOLO models as per assignment requirements:

### Primary Models (Required):
1. **YOLOv10** (nano, small, medium) - Latest YOLO architecture
2. **YOLOv11** (nano, small, medium) - Newest YOLO version
3. **YOLOv12** (nano, small, medium) - Most recent (if available)

### Bonus Models:
4. **RT-DETR** (large) - Transformer-based detector

**Note:** If YOLOv12 is not available (not yet released), the script will skip it and continue with available models.

---

### Training Configuration:
- **Image Size:** 512x512
- **Batch Size:** 16
- **Epochs:** 150 (with early stopping)
- **Patience:** 25 epochs
- **Device:** GPU (CUDA)

### Augmentation Enabled:
- ✅ Rotation (±15°)
- ✅ Translation (±15%)
- ✅ Scaling (70-130%)
- ✅ Color jitter (HSV)
- ✅ Mosaic & Mixup
- ✅ Random erasing
- ❌ No vertical flip (inappropriate for X-rays)

---

**Estimated Training Time:** 2-3 hours for all models

**⚠️ Important:** Training will run sequentially. Monitor GPU usage and training curves below.

# ============================================================================
# CELL 11 (Python): Training Function
# ============================================================================

def train_yolo_model(model_name, weights_file):
    """
    Train a single YOLO model
    
    Args:
        model_name: Name of model (e.g., 'YOLOv10n')
        weights_file: Pretrained weights (e.g., 'yolov10n.pt')
    
    Returns:
        dict: Training results and metrics
    """
    print(f"\n{'='*80}")
    print(f"🚀 Training {model_name}")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    try:
        # Check if model is available
        print(f"📥 Loading pretrained weights: {weights_file}")
        
        # Load model
        if 'detr' in weights_file.lower():
            model = RTDETR(weights_file)
            print("✅ Loaded RT-DETR model")
        else:
            model = YOLO(weights_file)
            print(f"✅ Loaded {model_name} model")
        
        # Train
        print(f"\n🏋️  Starting training...")
        print(f"   Config: {cfg.IMG_SIZE}x{cfg.IMG_SIZE}, batch={cfg.BATCH_SIZE}, epochs={cfg.EPOCHS}")
        
        results = model.train(
            data=cfg.DATA_YAML,
            epochs=cfg.EPOCHS,
            imgsz=cfg.IMG_SIZE,
            batch=cfg.BATCH_SIZE,
            patience=cfg.PATIENCE,
            device=cfg.DEVICE,
            name=model_name,
            project=str(cfg.MODELS_DIR),
            save=True,
            plots=True,
            cache=True,
            workers=4,
            **cfg.AUGMENTATION
        )
        
        # Validate
        print(f"\n📊 Validating {model_name}...")
        metrics = model.val(data=cfg.DATA_YAML)
        
        training_time = (time.time() - start_time) / 60  # minutes
        
        # Extract metrics
        map50 = float(metrics.box.map50)
        map50_95 = float(metrics.box.map)
        precision = float(metrics.box.mp)
        recall = float(metrics.box.mr)
        
        print(f"\n✅ {model_name} Training Complete!")
        print(f"   ⏱️  Time: {training_time:.1f} minutes")
        print(f"   📈 mAP@0.5: {map50:.4f}")
        print(f"   📈 mAP@0.5:0.95: {map50_95:.4f}")
        print(f"   📈 Precision: {precision:.4f}")
        print(f"   📈 Recall: {recall:.4f}")
        
        return {
            'model_name': model_name,
            'weights_file': weights_file,
            'map50': map50,
            'map50_95': map50_95,
            'precision': precision,
            'recall': recall,
            'training_time': training_time,
            'weights_path': str(cfg.MODELS_DIR / model_name / 'weights' / 'best.pt'),
            'status': 'success'
        }
        
    except Exception as e:
        print(f"\n❌ Error training {model_name}: {str(e)}")
        print(f"   Model might not be available yet (e.g., YOLOv12)")
        print(f"   Continuing with next model...")
        
        return {
            'model_name': model_name,
            'weights_file': weights_file,
            'status': 'failed',
            'error': str(e)
        }

print("✅ Training function ready")

# ============================================================================
# CELL 12 (Python): Train All Models
# ============================================================================

print("🎯 Starting training pipeline for all models...")
print("="*80)

results_list = []

for model_name, weights_file in cfg.MODELS_TO_TRAIN.items():
    print(f"\n🔄 Starting {model_name}...")
    print(f"   Weights: {weights_file}")
    
    result = train_yolo_model(model_name, weights_file)
    results_list.append(result)
    
    if result['status'] == 'success':
        print(f"✅ {model_name} completed successfully!")
    else:
        print(f"⚠️  {model_name} skipped (not available)")
    
    print(f"\n⏰ Progress: {len([r for r in results_list if r['status']=='success'])}/{len(cfg.MODELS_TO_TRAIN)} models completed")
    print("="*80)

print("\n" + "="*80)
print("🎉 TRAINING PIPELINE COMPLETE!")
print("="*80)

# Filter successful results
successful_results = [r for r in results_list if r['status'] == 'success']
failed_results = [r for r in results_list if r['status'] == 'failed']

print(f"\n✅ Successfully trained: {len(successful_results)} models")
if failed_results:
    print(f"⚠️  Skipped/Failed: {len(failed_results)} models")
    for r in failed_results:
        print(f"   - {r['model_name']}: {r.get('error', 'Unknown error')}")

print("\n📊 Proceeding to evaluation with successful models...")

# Continue in next message due to length...
