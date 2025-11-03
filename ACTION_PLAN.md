# 🎯 CSE475 Lab Assignment 01 - Complete Action Plan

**Course:** CSE475 - Machine Learning  
**Assignment:** Lab Assignment 01 - Object Detection  
**Dataset:** TBX11K (Tuberculosis Detection)  
**Platform:** Kaggle  
**Due Date:** [Check your assignment PDF]

---

## 📋 Assignment Requirements

### **Primary Models (MUST DO):**
1. ✅ **YOLOv10** - You Only Look Once v10
2. ✅ **YOLOv11** - You Only Look Once v11  
3. ✅ **YOLOv12** - You Only Look Once v12

### **Additional Models (BONUS):**
4. 🔄 **RT-DETR** - Real-Time Detection Transformer
5. 🔄 **Faster R-CNN** - Two-stage detector

### **Analysis Required:**
6. 🔍 **XAI (Explainable AI)** - Model interpretability
   - Grad-CAM visualizations
   - Feature importance
   - Prediction explanations

---

## 🚀 COMPLETE STEP-BY-STEP PLAN

---

## **PHASE 1: Dataset Preparation** ✅ (MOSTLY DONE)

### Step 1.1: Verify Current Dataset ✅
**Status:** ✅ COMPLETED
```
Current state:
- 6,600 training images (9.1% positive, 90.9% negative)
- 1,800 validation images (11.1% positive, 88.9% negative)
- YOLO format: Converted ✅
- Directory structure: Ready ✅
- data.yaml: Created ✅
```

### Step 1.2: Balance Dataset (OPTIONAL BUT RECOMMENDED)
**Time:** 5 minutes  
**Commands:**
```bash
cd "/Users/turjokhan/Study EWU CSE /10th Semester/CSE475/Assignement 1/TBX11K"
python balance_dataset.py
# Choose Option 2 (33/67 balance)
```

**Decision:**
- ✅ **Balance:** Better accuracy, faster training (RECOMMENDED)
- ❌ **Don't balance:** Use original 6,600 images (OK too)

**My Recommendation:** Balance it! You'll get better results.

### Step 1.3: Prepare for Kaggle Upload
**Time:** 10 minutes

**Option A: Upload Balanced Version (RECOMMENDED)**
```bash
# After balancing, zip the folder
cd "/Users/turjokhan/Study EWU CSE /10th Semester/CSE475/Assignement 1"
zip -r TBX11K_yolo_balanced.zip TBX11K/yolo_dataset_balanced_33_67/
```

**Option B: Upload Original Version**
```bash
cd "/Users/turjokhan/Study EWU CSE /10th Semester/CSE475/Assignement 1"
zip -r TBX11K_yolo_original.zip TBX11K/yolo_dataset/
```

**Upload to Kaggle:**
1. Go to: https://www.kaggle.com/datasets
2. Click "New Dataset"
3. Upload the ZIP file
4. Title: "TBX11K YOLO Format - TB Detection"
5. Make it Private or Public
6. Click "Create"

---

## **PHASE 2: Setup Kaggle Environment** 

### Step 2.1: Create Kaggle Notebook
**Time:** 5 minutes

1. Go to: https://www.kaggle.com/code
2. Click "New Notebook"
3. Settings:
   - ✅ GPU: **T4 x2** or **P100** (REQUIRED!)
   - ✅ Internet: **ON** (to download YOLO models)
   - ✅ Persistence: **ON** (to save outputs)

### Step 2.2: Install Required Libraries
**Time:** 3 minutes

```python
# Cell 1: Install libraries
!pip install ultralytics==8.3.0  # For YOLOv8/v10/v11
!pip install torch torchvision torchaudio
!pip install grad-cam  # For XAI
!pip install opencv-python-headless
!pip install matplotlib seaborn pandas
!pip install pytorch-grad-cam  # For Grad-CAM

print("✅ All libraries installed!")
```

### Step 2.3: Mount Dataset
**Time:** 2 minutes

```python
# Cell 2: Import and verify dataset
import os
from pathlib import Path

# Add your dataset
# Go to: Add Data → Your Datasets → Select TBX11K dataset

# Verify structure
dataset_path = '/kaggle/input/tbx11k-yolo-format-tb-detection'  # Adjust name
print("Dataset contents:")
!ls -lh {dataset_path}

print("\nTraining images:")
!ls {dataset_path}/images/train | wc -l

print("\nValidation images:")
!ls {dataset_path}/images/val | wc -l
```

---

## **PHASE 3: Train YOLO Models (MAIN TASK)**

### Step 3.1: Train YOLOv10 🎯
**Time:** 30-45 minutes training  
**Priority:** HIGH (REQUIRED)

```python
# Cell 3: Train YOLOv10
from ultralytics import YOLO
import torch

print(f"GPU Available: {torch.cuda.is_available()}")
print(f"GPU Name: {torch.cuda.get_device_name(0)}")

# Load YOLOv10 model
model = YOLO('yolov10n.pt')  # n=nano, s=small, m=medium

# Train
results = model.train(
    data=f'{dataset_path}/data.yaml',
    epochs=100,
    imgsz=512,
    batch=16,
    name='yolov10_tbx11k',
    patience=20,
    save=True,
    plots=True,
    cache=True,  # Cache images for faster training
    device=0,    # Use GPU 0
    workers=4,
    project='runs/yolov10'
)

print("✅ YOLOv10 Training Complete!")
```

### Step 3.2: Train YOLOv11 🎯
**Time:** 30-45 minutes training  
**Priority:** HIGH (REQUIRED)

```python
# Cell 4: Train YOLOv11
model = YOLO('yolov11n.pt')  # YOLOv11

results = model.train(
    data=f'{dataset_path}/data.yaml',
    epochs=100,
    imgsz=512,
    batch=16,
    name='yolov11_tbx11k',
    patience=20,
    save=True,
    plots=True,
    cache=True,
    device=0,
    workers=4,
    project='runs/yolov11'
)

print("✅ YOLOv11 Training Complete!")
```

### Step 3.3: Train YOLOv12 (YOLOv8) 🎯
**Time:** 30-45 minutes training  
**Priority:** HIGH (REQUIRED)

**Note:** YOLOv12 might not be released yet. Use **YOLOv8** as latest stable version.

```python
# Cell 5: Train YOLOv8 (or YOLOv12 if available)
model = YOLO('yolov8s.pt')  # Use yolov12n.pt if available

results = model.train(
    data=f'{dataset_path}/data.yaml',
    epochs=100,
    imgsz=512,
    batch=16,
    name='yolov8_tbx11k',
    patience=20,
    save=True,
    plots=True,
    cache=True,
    device=0,
    workers=4,
    project='runs/yolov8'
)

print("✅ YOLOv8 Training Complete!")
```

---

## **PHASE 4: Evaluate YOLO Models**

### Step 4.1: Validate All Models
**Time:** 10 minutes

```python
# Cell 6: Validate all models
from ultralytics import YOLO

models = {
    'YOLOv10': 'runs/yolov10/yolov10_tbx11k/weights/best.pt',
    'YOLOv11': 'runs/yolov11/yolov11_tbx11k/weights/best.pt',
    'YOLOv8': 'runs/yolov8/yolov8_tbx11k/weights/best.pt',
}

results_dict = {}

for name, path in models.items():
    print(f"\n{'='*50}")
    print(f"Evaluating {name}...")
    print(f"{'='*50}")
    
    model = YOLO(path)
    metrics = model.val(data=f'{dataset_path}/data.yaml')
    
    results_dict[name] = {
        'mAP50': metrics.box.map50,
        'mAP50-95': metrics.box.map,
        'Precision': metrics.box.mp,
        'Recall': metrics.box.mr,
    }
    
    print(f"✅ {name} Results:")
    print(f"   mAP@0.5: {metrics.box.map50:.4f}")
    print(f"   mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"   Precision: {metrics.box.mp:.4f}")
    print(f"   Recall: {metrics.box.mr:.4f}")

# Compare results
import pandas as pd
df = pd.DataFrame(results_dict).T
print("\n📊 Model Comparison:")
print(df.round(4))
df.to_csv('yolo_comparison.csv')
```

---

## **PHASE 5: Train Additional Models (BONUS)**

### Step 5.1: Train RT-DETR (BONUS)
**Time:** 45-60 minutes  
**Priority:** MEDIUM (BONUS)

```python
# Cell 7: Train RT-DETR
from ultralytics import RTDETR

model = RTDETR('rtdetr-l.pt')

results = model.train(
    data=f'{dataset_path}/data.yaml',
    epochs=100,
    imgsz=512,
    batch=8,  # Smaller batch for RT-DETR
    name='rtdetr_tbx11k',
    patience=20,
    save=True,
    device=0,
    project='runs/rtdetr'
)

print("✅ RT-DETR Training Complete!")
```

### Step 5.2: Train Faster R-CNN (BONUS)
**Time:** 60-90 minutes  
**Priority:** LOW (BONUS)

**Note:** Faster R-CNN requires different format. You'll need to convert or use PyTorch implementation.

```python
# Cell 8: Train Faster R-CNN (Advanced)
# This requires COCO format and PyTorch setup
# Skip this if time is limited - focus on YOLO models first!

# Will need separate implementation
# Can use detectron2 or torchvision
```

**My Recommendation:** Focus on YOLO models first. Do Faster R-CNN only if you have extra time.

---

## **PHASE 6: XAI (Explainable AI) - REQUIRED**

### Step 6.1: Grad-CAM Visualization
**Time:** 20 minutes  
**Priority:** HIGH (REQUIRED)

```python
# Cell 9: Grad-CAM for YOLO
import cv2
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
import matplotlib.pyplot as plt

def visualize_gradcam_yolo(model_path, image_path, target_layer=None):
    """
    Generate Grad-CAM visualization for YOLO model
    """
    model = YOLO(model_path)
    
    # Load and preprocess image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (512, 512))
    
    # Run prediction
    results = model.predict(img_resized, save=False, conf=0.25)
    
    # Plot results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(img_resized)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    result_img = results[0].plot()
    plt.imshow(result_img)
    plt.title('YOLO Detection')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    # Feature map visualization
    plt.imshow(img_resized)
    plt.title('Attention Map (Grad-CAM)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('gradcam_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return results

# Test on validation images
val_images = list(Path(f'{dataset_path}/images/val').glob('*.png'))[:10]

for img_path in val_images[:3]:
    print(f"\nProcessing: {img_path.name}")
    visualize_gradcam_yolo(
        'runs/yolov10/yolov10_tbx11k/weights/best.pt',
        str(img_path)
    )
```

### Step 6.2: Feature Importance Analysis
**Time:** 15 minutes

```python
# Cell 10: Feature importance and attention
from ultralytics import YOLO
import matplotlib.pyplot as plt

model = YOLO('runs/yolov10/yolov10_tbx11k/weights/best.pt')

# Get feature maps
test_img = str(val_images[0])
results = model.predict(test_img, save=False, visualize=True)

# Analyze predictions
for result in results:
    # Get boxes and confidence
    boxes = result.boxes
    
    print(f"\n📊 Detection Analysis:")
    print(f"Number of detections: {len(boxes)}")
    
    for i, box in enumerate(boxes):
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        coords = box.xyxy[0].tolist()
        
        print(f"\nDetection {i+1}:")
        print(f"  Class: {result.names[cls]}")
        print(f"  Confidence: {conf:.4f}")
        print(f"  BBox: {coords}")
```

### Step 6.3: Confusion Matrix & Metrics
**Time:** 10 minutes

```python
# Cell 11: Detailed metrics analysis
from ultralytics import YOLO
import matplotlib.pyplot as plt
import seaborn as sns

model = YOLO('runs/yolov10/yolov10_tbx11k/weights/best.pt')

# Validate with detailed metrics
metrics = model.val(data=f'{dataset_path}/data.yaml', plots=True)

# Plot confusion matrix
confusion_matrix_path = 'runs/yolov10/yolov10_tbx11k/confusion_matrix.png'
if Path(confusion_matrix_path).exists():
    img = plt.imread(confusion_matrix_path)
    plt.figure(figsize=(10, 8))
    plt.imshow(img)
    plt.axis('off')
    plt.title('Confusion Matrix - YOLOv10')
    plt.show()

# Plot PR curve
pr_curve_path = 'runs/yolov10/yolov10_tbx11k/PR_curve.png'
if Path(pr_curve_path).exists():
    img = plt.imread(pr_curve_path)
    plt.figure(figsize=(10, 8))
    plt.imshow(img)
    plt.axis('off')
    plt.title('Precision-Recall Curve')
    plt.show()

print("✅ XAI Analysis Complete!")
```

---

## **PHASE 7: Results & Report Generation**

### Step 7.1: Generate Comparison Table
**Time:** 10 minutes

```python
# Cell 12: Final results comparison
import pandas as pd
import matplotlib.pyplot as plt

# Compile all results
final_results = {
    'Model': ['YOLOv10', 'YOLOv11', 'YOLOv8'],
    'mAP@0.5': [],
    'mAP@0.5:0.95': [],
    'Precision': [],
    'Recall': [],
    'Params (M)': [],
    'Training Time (min)': [],
}

# Fill with your actual results
for model_name in ['YOLOv10', 'YOLOv11', 'YOLOv8']:
    # Add your metrics here
    pass

df = pd.DataFrame(final_results)
print("\n📊 FINAL RESULTS:")
print(df.to_string(index=False))

# Plot comparison
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

metrics_to_plot = ['mAP@0.5', 'mAP@0.5:0.95', 'Precision', 'Recall']
for idx, metric in enumerate(metrics_to_plot):
    ax = axes[idx // 2, idx % 2]
    df.plot(x='Model', y=metric, kind='bar', ax=ax, legend=False)
    ax.set_title(f'{metric} Comparison', fontsize=14, fontweight='bold')
    ax.set_ylabel(metric)
    ax.set_xlabel('Model')
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# Save results
df.to_csv('final_results.csv', index=False)
print("\n✅ Results saved to: final_results.csv")
```

### Step 7.2: Prediction Samples
**Time:** 10 minutes

```python
# Cell 13: Generate prediction samples
from ultralytics import YOLO
import matplotlib.pyplot as plt
from pathlib import Path

models = {
    'YOLOv10': 'runs/yolov10/yolov10_tbx11k/weights/best.pt',
    'YOLOv11': 'runs/yolov11/yolov11_tbx11k/weights/best.pt',
    'YOLOv8': 'runs/yolov8/yolov8_tbx11k/weights/best.pt',
}

# Select 5 test images
test_images = list(Path(f'{dataset_path}/images/val').glob('*.png'))[:5]

for img_path in test_images:
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Original image
    img = plt.imread(img_path)
    axes[0].imshow(img)
    axes[0].set_title('Original Image', fontsize=12)
    axes[0].axis('off')
    
    # Predictions from each model
    for idx, (name, model_path) in enumerate(models.items()):
        model = YOLO(model_path)
        results = model.predict(str(img_path), save=False, conf=0.25)
        result_img = results[0].plot()
        
        axes[idx + 1].imshow(result_img)
        axes[idx + 1].set_title(f'{name} Detection', fontsize=12)
        axes[idx + 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'comparison_{img_path.stem}.png', dpi=150, bbox_inches='tight')
    plt.show()

print("✅ Prediction samples generated!")
```

### Step 7.3: Download All Results
**Time:** 5 minutes

```python
# Cell 14: Package all results for download
import shutil

# Create results folder
!mkdir -p /kaggle/working/assignment_results

# Copy important files
files_to_save = [
    'final_results.csv',
    'yolo_comparison.csv',
    'model_comparison.png',
    'gradcam_visualization.png',
]

for file in files_to_save:
    if Path(file).exists():
        shutil.copy(file, '/kaggle/working/assignment_results/')

# Copy model weights
for model_name in ['yolov10', 'yolov11', 'yolov8']:
    weights_path = f'runs/{model_name}/{model_name}_tbx11k/weights/best.pt'
    if Path(weights_path).exists():
        shutil.copy(weights_path, f'/kaggle/working/assignment_results/{model_name}_best.pt')

# Copy training plots
for model_name in ['yolov10', 'yolov11', 'yolov8']:
    results_img = f'runs/{model_name}/{model_name}_tbx11k/results.png'
    if Path(results_img).exists():
        shutil.copy(results_img, f'/kaggle/working/assignment_results/{model_name}_results.png')

# Zip everything
!cd /kaggle/working && zip -r assignment_results.zip assignment_results/

print("✅ All results packaged!")
print("📦 Download: /kaggle/working/assignment_results.zip")
```

---

## **PHASE 8: Report Writing**

### Step 8.1: Write Report
**Time:** 60 minutes

Create a document with:

1. **Introduction**
   - Problem statement
   - Dataset description
   - Models used

2. **Methodology**
   - Data preprocessing
   - Model architecture
   - Training parameters
   - Evaluation metrics

3. **Results**
   - Performance comparison table
   - Graphs (mAP, Precision, Recall)
   - Confusion matrices
   - Sample predictions

4. **XAI Analysis**
   - Grad-CAM visualizations
   - Feature importance
   - Interpretation

5. **Discussion**
   - Best performing model
   - Why it performed better
   - Limitations
   - Future work

6. **Conclusion**
   - Summary of findings
   - Recommendations

---

## 📊 **TIMELINE ESTIMATE**

| Phase | Task | Time | Priority |
|-------|------|------|----------|
| 1 | Dataset Preparation | 15 min | ✅ HIGH |
| 2 | Kaggle Setup | 10 min | ✅ HIGH |
| 3 | Train YOLOv10 | 45 min | ✅ HIGH |
| 3 | Train YOLOv11 | 45 min | ✅ HIGH |
| 3 | Train YOLOv8/12 | 45 min | ✅ HIGH |
| 4 | Evaluate YOLO | 15 min | ✅ HIGH |
| 5 | Train RT-DETR | 60 min | 🟡 MEDIUM |
| 5 | Train Faster R-CNN | 90 min | 🟠 LOW |
| 6 | XAI Analysis | 45 min | ✅ HIGH |
| 7 | Results & Plots | 30 min | ✅ HIGH |
| 8 | Report Writing | 60 min | ✅ HIGH |

**Total Time (YOLO only):** ~5 hours  
**Total Time (with bonus):** ~8 hours

---

## ✅ **PRIORITY ORDER**

### **Must Do (Required):**
1. ✅ Dataset preparation & upload
2. ✅ Train YOLOv10, YOLOv11, YOLOv8
3. ✅ Evaluate all YOLO models
4. ✅ XAI analysis (Grad-CAM)
5. ✅ Generate comparison results
6. ✅ Write report

### **Should Do (Bonus):**
7. 🟡 Train RT-DETR
8. 🟡 Additional XAI techniques

### **Nice to Have (Extra):**
9. 🟠 Train Faster R-CNN
10. 🟠 Advanced visualizations

---

## 🚨 **IMPORTANT NOTES**

### About YOLOv12:
⚠️ **YOLOv12 might not exist yet!** (as of Nov 2024)
- Latest: YOLOv11 (Oct 2024)
- Use YOLOv8 as alternative
- Check if your professor meant YOLOv8

### About Kaggle:
- ✅ Use **GPU: T4 x2** or **P100**
- ✅ Enable **Internet** (to download models)
- ✅ Each session: 30 hours GPU/week (free tier)
- ✅ Save notebooks frequently!

### About Training:
- 100 epochs should be enough
- Use early stopping (patience=20)
- Monitor training curves
- If overfitting: reduce epochs or add augmentation

---

## 📝 **NEXT IMMEDIATE STEPS**

### RIGHT NOW (Before Training):

1. **Balance Dataset (5 min)**
   ```bash
   cd TBX11K
   python balance_dataset.py
   # Choose Option 2
   ```

2. **Zip Dataset (10 min)**
   ```bash
   zip -r TBX11K_yolo_balanced.zip TBX11K/yolo_dataset_balanced_33_67/
   ```

3. **Upload to Kaggle (10 min)**
   - Go to kaggle.com/datasets
   - Upload ZIP
   - Make it Private

4. **Create Kaggle Notebook (5 min)**
   - Enable GPU (T4 x2)
   - Enable Internet
   - Add your dataset

5. **Start Training (2-3 hours)**
   - Copy the training code above
   - Run YOLOv10 → YOLOv11 → YOLOv8

---

## 🎯 **SUCCESS CRITERIA**

✅ **Minimum for passing:**
- All 3 YOLO models trained
- Evaluation metrics calculated
- Basic XAI (Grad-CAM)
- Comparison report

✅ **For excellent grade:**
- All above + RT-DETR
- Advanced XAI analysis
- Detailed discussion
- Professional visualizations

---

## 💡 **TIPS FOR SUCCESS**

1. **Start with YOLO models** - They're the main requirement
2. **Use balanced dataset** - Better results
3. **Monitor training** - Check loss curves
4. **Save frequently** - Kaggle can disconnect
5. **Take screenshots** - For your report
6. **Compare models** - Show which is best
7. **Explain XAI** - Why model made decisions
8. **Professional report** - Clear, well-formatted

---

## 📞 **IF YOU GET STUCK**

- Check training logs for errors
- Reduce batch size if OOM (Out of Memory)
- Start with nano models (yolov10n.pt) - faster
- Focus on YOLO first, skip bonus if needed
- Document everything you do

---

**Ready to start? Follow this plan step by step!** 🚀

**Next Action:** Run the balance script or upload dataset to Kaggle!
