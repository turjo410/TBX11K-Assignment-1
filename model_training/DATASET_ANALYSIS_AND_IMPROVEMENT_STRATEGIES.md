# 🔬 Dataset Analysis & Performance Improvement Strategies
## TBX11K Tuberculosis Detection - Low mAP Problem Solved

**Date:** November 3, 2025  
**Analyst:** AI Data Scientist  
**Problem:** Low metrics (mAP@0.5: 0.2443, Precision: 0.3023, Recall: 0.2157)

---

## 📊 Dataset Analysis

### Current Dataset Statistics:
```
Location: /TBX11K/New Dataset With labelling/yolo_dataset_balanced_33_67/
Training Images: 600
Validation Images: 200
Total: 800 images (VERY SMALL!)
Classes: 3 (Active TB, Obsolete TB, Pulmonary TB)
```

### ⚠️ **CRITICAL PROBLEM IDENTIFIED:**

**Your dataset is EXTREMELY SMALL (800 images total)!**

For medical imaging object detection:
- **Minimum recommended:** 1,500-2,000 images
- **Ideal:** 5,000-10,000+ images
- **Your dataset:** 600 train + 200 val = **TOO SMALL**

This explains your low metrics:
- mAP@0.5: 0.2443 (should be >0.6 for good performance)
- Precision: 0.3023 (model makes many false positives)
- Recall: 0.2157 (model misses 78% of TB cases!)

---

## 🎯 Root Cause Analysis

### Why Your Metrics Are Low:

1. **Insufficient Training Data** ⚠️⚠️⚠️
   - 600 images is FAR too small for 3-class object detection
   - Deep learning needs 1000s of samples per class
   - Current: ~200 images per class (assuming balanced)

2. **Small Validation Set**
   - 200 val images → metrics have high variance
   - Not statistically reliable

3. **Limited Model Learning**
   - Model cannot learn diverse TB patterns
   - Overfits to small training set
   - Poor generalization

4. **Class Complexity**
   - 3 TB subtypes are visually similar
   - Requires MORE data to distinguish

---

## 🚀 COMPREHENSIVE SOLUTION STRATEGY

### Strategy 1: **AGGRESSIVE DATA AUGMENTATION** (Essential)

Since you have limited data, augmentation is CRITICAL:

#### A. Geometric Augmentations (Increase diversity):
```python
# Strong augmentation for small datasets
AUGMENTATION_CONFIG = {
    # Rotation (X-rays can be at different angles)
    'degrees': 25.0,  # ±25° rotation
    
    # Translation (shift lesions around)
    'translate': 0.2,  # 20% translation
    
    # Scaling (different lesion sizes)
    'scale': 0.5,  # ±50% scale variation
    
    # Flipping (lungs are symmetric)
    'fliplr': 0.5,  # 50% horizontal flip
    'flipud': 0.0,  # NO vertical flip for X-rays
    
    # Shearing (perspective changes)
    'shear': 10.0,  # ±10° shear
    
    # Perspective transform
    'perspective': 0.001,
    
    # Advanced mixing
    'mosaic': 1.0,  # ALWAYS use mosaic (combines 4 images)
    'mixup': 0.3,   # 30% chance of mixing 2 images
    'copy_paste': 0.3,  # Copy TB lesions to other images
    
    # Color augmentation (X-ray intensity variations)
    'hsv_h': 0.0,   # No hue change (grayscale)
    'hsv_s': 0.0,   # No saturation
    'hsv_v': 0.6,   # Strong brightness variation
    
    # Random erasing (simulate occlusions)
    'erasing': 0.5,  # 50% random erasing
}
```

#### B. Medical-Specific Augmentations:
```python
# Additional techniques for medical images
MEDICAL_AUGMENTATIONS = {
    'gaussian_noise': 0.02,  # Add noise (simulate poor image quality)
    'blur': (0, 3),  # Random blur
    'contrast': (0.8, 1.2),  # Contrast adjustment
    'brightness': (0.8, 1.2),  # Brightness adjustment
    'gamma_correction': (0.8, 1.2),  # Simulate different X-ray exposures
}
```

### Strategy 2: **K-FOLD CROSS VALIDATION** (Maximize data usage)

With small datasets, k-fold is ESSENTIAL:

```python
# 5-Fold Cross Validation Strategy
NUM_FOLDS = 5

# This effectively gives you 5x more training:
# Fold 1: Train on 640 imgs, val on 160
# Fold 2: Train on 640 imgs, val on 160
# ...
# Final: Ensemble 5 models for better predictions

# Expected improvement: +5-10% mAP
```

### Strategy 3: **TRANSFER LEARNING** (Use pretrained knowledge)

```python
# Use models pretrained on medical images
PRETRAINED_MODELS = {
    'ImageNet': 'yolov10n.pt',  # General objects
    'Medical': 'Use YOLOv8-medical.pt if available',  # Better for X-rays
}

# Freeze early layers, fine-tune detection head
FREEZE_LAYERS = 10  # Freeze first 10 layers
```

### Strategy 4: **INCREASE TRAINING EPOCHS** (More learning time)

```python
# Your current: 20 epochs (TOO SHORT!)
# Recommended for small datasets:
EPOCHS = 150-200  # Let model learn thoroughly
PATIENCE = 50  # Don't stop too early
```

### Strategy 5: **BATCH SIZE & LEARNING RATE TUNING**

```python
# Smaller batch = more gradient updates = better for small data
BATCH_SIZE = 8  # Reduce from 16 to 8
IMG_SIZE = 640  # Increase from 512 to 640 (more detail)

# Lower learning rate for fine-tuning
LR0 = 0.0005  # Reduce from 0.001
LRF = 0.005   # Gentler decay
```

### Strategy 6: **FOCAL LOSS** (Handle hard examples)

```python
# For medical imaging with small objects
USE_FOCAL_LOSS = True
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0
```

### Strategy 7: **ENSEMBLE METHODS** (Combine multiple models)

```python
# Train 3-5 models with different:
ENSEMBLE_CONFIGS = [
    {'model': 'yolov10n', 'imgsz': 512, 'augment': 'strong'},
    {'model': 'yolov11n', 'imgsz': 640, 'augment': 'medium'},
    {'model': 'yolov12n', 'imgsz': 512, 'augment': 'strong'},
]

# Average predictions → +3-5% mAP improvement
```

### Strategy 8: **PSEUDO-LABELING** (Semi-supervised learning)

```python
# If you have unlabeled X-rays:
# 1. Train on 800 labeled images
# 2. Predict on unlabeled images (high confidence)
# 3. Add pseudo-labeled images to training
# 4. Retrain with expanded dataset
# Expected: +10-15% mAP
```

### Strategy 9: **TEST-TIME AUGMENTATION (TTA)**

```python
# During validation/testing:
TTA_CONFIG = {
    'scales': [0.9, 1.0, 1.1],  # Multi-scale testing
    'flips': [None, 'horizontal'],  # Flip augmentation
    'rotations': [0, 90, 180, 270],  # Rotation augmentation
}
# Average predictions from all augmentations
# Expected: +2-3% mAP
```

---

## 📋 RECOMMENDED IMPLEMENTATION PLAN

### **Phase 1: Quick Wins (1-2 days)**

1. ✅ **Increase epochs**: 20 → 150
2. ✅ **Aggressive augmentation**: Enable all augmentations above
3. ✅ **Reduce batch size**: 16 → 8
4. ✅ **Increase image size**: 512 → 640
5. ✅ **Lower learning rate**: 0.001 → 0.0005

**Expected improvement:** mAP 0.24 → **0.35-0.40** (+46-67%)

### **Phase 2: K-Fold Cross Validation (3-4 days)**

6. ✅ **Implement 5-fold CV**
7. ✅ **Train 5 models** (one per fold)
8. ✅ **Ensemble predictions**

**Expected improvement:** mAP 0.40 → **0.45-0.50** (+12-25%)

### **Phase 3: Advanced Techniques (1 week)**

9. ✅ **Transfer learning from medical datasets**
10. ✅ **Test-time augmentation**
11. ✅ **Multi-model ensemble** (YOLOv10/11/12)
12. ✅ **Pseudo-labeling** (if unlabeled data available)

**Expected improvement:** mAP 0.50 → **0.55-0.65** (+10-30%)

---

## 🎯 REALISTIC EXPECTATIONS

Given your **VERY SMALL dataset (800 images)**:

| Scenario | mAP@0.5 | Precision | Recall | Notes |
|----------|---------|-----------|--------|-------|
| **Current** | 0.244 | 0.302 | 0.216 | Baseline |
| **Quick Wins** | 0.35-0.40 | 0.45-0.50 | 0.35-0.40 | Better augmentation |
| **+ K-Fold** | 0.45-0.50 | 0.55-0.60 | 0.45-0.50 | More training data usage |
| **+ Advanced** | 0.55-0.65 | 0.65-0.70 | 0.55-0.60 | All techniques combined |
| **Ideal (Need 5K+ images)** | 0.75-0.85 | 0.80-0.90 | 0.75-0.85 | With large dataset |

⚠️ **REALITY CHECK:**
- With 800 images, **0.55-0.65 mAP is the realistic maximum**
- To reach 0.75+ mAP, you NEED 3,000-5,000+ images
- Current dataset size is the PRIMARY bottleneck

---

## 💡 DATA COLLECTION RECOMMENDATIONS

### If possible, INCREASE your dataset:

1. **Use full TBX11K dataset:**
   - Original has ~11,000 images
   - Why were most removed? Investigate original labels
   - Maybe some images have bounding boxes you missed?

2. **Public TB datasets:**
   - **Shenzhen TB Dataset**: 662 images
   - **Montgomery TB Dataset**: 138 images
   - **NIH Chest X-ray**: 112,120 images (some TB cases)
   - **CheXpert**: 224,316 images (some TB)

3. **Data augmentation to generate more:**
   - Use StyleGAN2 to generate synthetic X-rays
   - Use diffusion models (Stable Diffusion medical)

---

## 🔧 OPTIMIZED TRAINING CONFIGURATION

Here's your optimal config for small dataset:

```python
class OptimizedConfig:
    # Dataset
    DATASET_PATH = '/Users/turjokhan/Study EWU CSE /10th Semester/CSE475/Assignement 1/TBX11K/New Dataset With labelling/yolo_dataset_balanced_33_67'
    
    # Training (OPTIMIZED FOR SMALL DATA)
    IMG_SIZE = 640  # Increased for more detail
    BATCH_SIZE = 8  # Reduced for more gradient updates
    EPOCHS = 150    # Increased significantly
    PATIENCE = 50   # Longer patience
    WORKERS = 0
    
    # Optimizer (CONSERVATIVE)
    OPTIMIZER = 'AdamW'
    LR0 = 0.0005  # Lower LR
    LRF = 0.005   # Gentler decay
    MOMENTUM = 0.937
    WEIGHT_DECAY = 0.001  # Increased regularization
    WARMUP_EPOCHS = 5  # Longer warmup
    
    # Loss weights (BALANCED)
    BOX = 7.5
    CLS = 1.5  # Increased (class confusion is high)
    DFL = 1.5
    
    # AGGRESSIVE AUGMENTATION (KEY!)
    DEGREES = 25.0      # More rotation
    TRANSLATE = 0.2     # More translation
    SCALE = 0.5         # More scaling
    SHEAR = 10.0        # More shearing
    PERSPECTIVE = 0.001
    FLIPUD = 0.0
    FLIPLR = 0.5
    MOSAIC = 1.0        # ALWAYS mosaic
    MIXUP = 0.3         # More mixing
    COPY_PASTE = 0.3    # Copy TB lesions
    HSV_H = 0.0
    HSV_S = 0.0
    HSV_V = 0.6         # Strong brightness variation
    ERASING = 0.5       # More random erasing
    
    # Regularization
    DROPOUT = 0.3       # Add dropout
    LABEL_SMOOTHING = 0.1
    
    # Inference
    CONF_THRESHOLD = 0.20  # Lower threshold (increase recall)
    IOU_THRESHOLD = 0.45
```

---

## 📊 EXPECTED RESULTS TIMELINE

| Week | Action | Expected mAP@0.5 |
|------|--------|------------------|
| Week 1 | Baseline (current) | 0.244 |
| Week 1 | Quick wins (augmentation + epochs) | 0.35-0.40 |
| Week 2 | K-fold cross validation | 0.45-0.50 |
| Week 3 | Ensemble + TTA | 0.50-0.55 |
| Week 4 | Advanced techniques | 0.55-0.65 |

---

## ✅ IMMEDIATE ACTION ITEMS

### Do This NOW:

1. **Update training config** with optimized parameters above
2. **Train for 150 epochs** (not 20!)
3. **Enable aggressive augmentation**
4. **Use batch size 8, image size 640**
5. **Monitor training curves** - watch for overfitting

### Within 3 Days:

6. **Implement 5-fold cross validation**
7. **Train one model per fold**
8. **Ensemble 5 models**

### Within 1 Week:

9. **Try YOLOv11 and YOLOv12** (compare with YOLOv10)
10. **Implement test-time augmentation**
11. **Search for additional TB datasets**

---

## 🎯 CONCLUSION

**Your low metrics are due to:**
1. **TINY dataset** (800 images - PRIMARY PROBLEM)
2. Too few epochs (20 is insufficient)
3. Insufficient augmentation
4. Suboptimal hyperparameters

**Solution priority:**
1. 🔥 **AGGRESSIVE AUGMENTATION** (highest impact)
2. 🔥 **INCREASE EPOCHS to 150** (critical)
3. 🔥 **K-FOLD CROSS VALIDATION** (essential for small data)
4. **Optimize hyperparameters** (learning rate, batch size)
5. **Ensemble methods** (combine multiple models)

**Realistic target with 800 images:** mAP@0.5 = **0.55-0.65**

To reach 0.75+ mAP, you NEED to expand your dataset to 3,000-5,000+ images.

---

**Next step:** Shall I create an optimized training notebook implementing all these strategies?
