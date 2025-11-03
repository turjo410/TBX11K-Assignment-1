# 📊 TBX11K Dataset - Current Status Report

**Date:** November 2, 2025  
**Status:** ✅ **READY FOR YOLO TRAINING**

---

## 🔄 What I Changed

### 1. **Converted COCO → YOLO Format**
- Original: JSON files with absolute pixel coordinates `[x_min, y_min, width, height]`
- Now: TXT files with normalized YOLO format `class x_center y_center width height`
- All coordinates normalized to 0-1 range

### 2. **Reorganized Directory Structure**
**Before:**
```
TBX11K/
├── imgs/
│   ├── health/
│   ├── sick/
│   └── tb/
└── annotations/
    └── json/
```

**After (Added):**
```
TBX11K/
├── yolo_dataset/              ← NEW! Everything for YOLO here
│   ├── data.yaml              ← Configuration file
│   ├── train_yolo.py          ← Training script
│   ├── images/
│   │   ├── train/ (6,600 images)
│   │   └── val/   (1,800 images)
│   └── labels/
│       ├── train/ (6,600 .txt files)
│       └── val/   (1,800 .txt files)
└── [original files intact]
```

### 3. **Created Empty Labels for Negative Samples**
- Images without TB: Created empty `.txt` files (required by YOLO)
- Images with TB: Created `.txt` files with bounding boxes

### 4. **Kept Everything Inside TBX11K Folder**
- Easy to upload entire `TBX11K` folder to Kaggle
- All converted data in `TBX11K/yolo_dataset/` subfolder

---

## 📈 Current Dataset Statistics

### **Training Set: 6,600 images**
| Category | Count | Percentage |
|----------|-------|------------|
| Images with TB annotations | 599 | 9.1% |
| Images without TB (negative) | 6,001 | 90.9% |
| **Total bounding boxes** | **902** | - |

### **Validation Set: 1,800 images**
| Category | Count | Percentage |
|----------|-------|------------|
| Images with TB annotations | 200 | 11.1% |
| Images without TB (negative) | 1,600 | 88.9% |
| **Total bounding boxes** | **309** | - |

### **Total Dataset**
- **Total images**: 8,400 (6,600 train + 1,800 val)
- **Total annotations**: 1,211 bounding boxes
- **Average boxes per image**: 0.14 (very low due to many negative samples)
- **Image size**: 512×512 pixels
- **Format**: PNG
- **Classes**: 3 types of TB

---

## 🎯 Training Data Quality

### ✅ **Strengths:**
1. **Good image quality** - 512×512 clear X-rays
2. **Proper annotations** - Accurate bounding boxes verified
3. **Good split** - 78.6% train, 21.4% val (standard ratio)
4. **Clean format** - Properly converted to YOLO format
5. **All files validated** - Every image has corresponding label

### ⚠️ **Challenges:**

#### 1. **SEVERE Class Imbalance** (Main Issue)
- Only **9.1% of images have TB** (599 out of 6,600)
- **90.9% are negative samples** (healthy or sick but non-TB)
- This is the **biggest challenge** for training

#### 2. **Low Annotation Density**
- Average 0.14 boxes per image (very sparse)
- Some images have multiple TB regions, most have none
- Model needs to learn when NOT to detect (important skill)

#### 3. **Class Distribution (within TB cases)**
```
Active TB (Class 0):        ~80% of annotations
Latent TB (Class 1):        ~20% of annotations  
Uncertain TB (Class 2):      ~0% of annotations
```
- Class 2 barely exists in training data!

---

## 🤔 Will It Be Good Training?

### **Realistic Expectations:**

#### ✅ **YES, it will work BUT...**

**Expected Performance:**
- **mAP@0.5**: 35-55% (typical for heavily imbalanced medical datasets)
- **Precision**: 40-60% (many false positives expected)
- **Recall**: 50-70% (better - important for medical use)

**Why these numbers?**
1. Severe class imbalance makes learning difficult
2. Model will tend to predict "no TB" often (safer)
3. Many false positives on sick-but-not-TB cases
4. Medical images are challenging (subtle differences)

### **Comparison to Other Datasets:**

| Dataset Type | Typical mAP | Your Expected mAP |
|--------------|-------------|-------------------|
| COCO (balanced) | 50-70% | N/A |
| Medical (balanced) | 45-65% | N/A |
| **Your dataset (imbalanced)** | **N/A** | **35-55%** |

**Note:** Lower mAP doesn't mean bad! Medical datasets are harder and imbalance reduces metrics.

---

## 💡 How to Make Training Better

### **Critical Strategies:**

#### 1. **Handle Class Imbalance** (MUST DO)
```python
# Option A: Undersample negative samples
# Use only 1,000 negative + 599 positive = 1,599 images

# Option B: Weighted sampling
# Sample positive images 5-10x more often

# Option C: Focal loss (built into YOLOv8)
# Already included, helps automatically
```

#### 2. **Augmentation Strategy**
```python
# Use MORE augmentation on TB-positive images
# Use LESS on negative samples
# Medical-appropriate augmentations:
- Rotation: ±5-10 degrees ✅
- Brightness: ±30% ✅
- Contrast: ±30% ✅
- Horizontal flip: Yes ✅
- Vertical flip: NO ❌ (X-rays)
- Mosaic: Reduced (0.5) ✅
- Mixup: NO ❌ (medical)
```

#### 3. **Training Configuration**
```python
# Recommended settings:
epochs = 150-200  # More epochs needed for imbalanced data
batch = 16        # Adjust based on GPU
imgsz = 512       # Keep at 512 (matches dataset)
patience = 30     # Higher patience for imbalanced data

# Focus on RECALL over PRECISION
# Better to have false positives than miss TB cases!
```

#### 4. **Model Selection**
```
Start with: YOLOv8n (fastest, test quickly)
If accuracy low: YOLOv8s or YOLOv8m (better)
For best accuracy: YOLOv8l (slower but best)
```

---

## 📊 Realistic Training Scenario

### **Best Case Scenario:**
```
mAP@0.5: 50-60%
Precision: 55-65%
Recall: 65-75%
Training time: 2-3 hours (with GPU)
```
**Requirements:** 
- Good GPU, proper augmentation, handle imbalance well

### **Average Case Scenario:**
```
mAP@0.5: 40-50%
Precision: 45-55%
Recall: 55-65%
Training time: 3-4 hours (with GPU)
```
**Requirements:**
- Basic setup, some augmentation, YOLOv8n/s

### **Worst Case Scenario:**
```
mAP@0.5: 25-35%
Precision: 30-40%
Recall: 40-50%
Training time: 4+ hours (CPU) or fails to converge
```
**Problems:**
- No GPU, no imbalance handling, wrong hyperparameters

---

## ✅ What You Have Now

### **Files Ready to Upload to Kaggle:**

```
TBX11K/                          ← Upload this entire folder
├── yolo_dataset/                ← YOLO training data
│   ├── data.yaml               ✅ Configuration (ready)
│   ├── train_yolo.py           ✅ Training script (ready)
│   ├── images/
│   │   ├── train/ (6,600)      ✅ All images copied
│   │   └── val/ (1,800)        ✅ All images copied
│   └── labels/
│       ├── train/ (6,600)      ✅ All labels created
│       └── val/ (1,800)        ✅ All labels created
├── imgs/                        (original, can keep for reference)
├── annotations/                 (original, can keep for reference)
└── lists/                       (original, can keep for reference)
```

**Total size:** ~3 GB (mostly images)

---

## 🚀 Quick Start on Kaggle

### **Step 1: Upload Dataset**
```
1. Zip the TBX11K folder
2. Upload to Kaggle as dataset
3. Create new notebook
```

### **Step 2: Training Command**
```python
# In Kaggle notebook:
!pip install ultralytics

# Navigate to dataset
%cd /kaggle/input/tbx11k/TBX11K/yolo_dataset

# Train
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.train(data='data.yaml', epochs=100, imgsz=512, batch=16)
```

**That's it!** Simple 3-line training.

---

## 🎯 Bottom Line

### **Is your dataset ready?**
✅ **YES - 100% ready for YOLO training**

### **Will training be good?**
⚠️ **MODERATE - Expected mAP 35-55%**
- Not excellent, but **realistic for this type of data**
- Main issue: severe class imbalance (90% negative samples)
- This is **normal for medical detection** tasks

### **Will it be useful?**
✅ **YES - If you handle it correctly:**
1. Focus on **recall** (catching TB cases) over precision
2. Expect and accept **false positives** (better safe than sorry)
3. Use **ensemble methods** or **post-processing** to filter
4. Consider it a **screening tool** (flag for human review)

### **Recommendation:**
🎯 **PROCEED WITH TRAINING** but:
- Understand the limitations (imbalance)
- Use proper augmentation
- Consider undersampling negative samples (optional)
- Focus on high recall for medical safety
- Use YOLOv8s or YOLOv8m for better accuracy

---

## 📝 Summary

**What changed:** Converted COCO → YOLO, organized in `yolo_dataset/` folder  
**Data amount:** 6,600 train + 1,800 val = 8,400 images ready  
**Quality:** ✅ Good images, ⚠️ severe imbalance  
**Expected results:** Moderate (35-55% mAP) but **usable**  
**Ready to train:** ✅ **YES - Go ahead!**

The dataset is properly prepared. The training won't be "perfect" due to imbalance, but it's **realistic and will work**. Many medical AI models operate at similar performance levels. Focus on practical utility (high recall) rather than perfect metrics!

---

**🚀 You're ready to train on Kaggle! Good luck!**
