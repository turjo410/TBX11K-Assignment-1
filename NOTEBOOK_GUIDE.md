# 🔬 TBX11K Complete Research Notebook - Quick Start Guide

This notebook contains **18 comprehensive sections** for professional tuberculosis detection research using state-of-the-art object detection models.

## 📋 Notebook Sections Overview:

### Part 1: Setup & Data Analysis (Cells 1-30)
1. **Project Overview & Introduction** - Research objectives and methodology
2. **Library Installation & Imports** - All required packages
3. **Configuration Setup** - Paths, hyperparameters, augmentation settings
4. **Dataset Loading & Verification** - Path checking and structure validation
5. **Exploratory Data Analysis (EDA)** - Comprehensive dataset statistics
6. **Data Distribution Visualizations** - Class balance, bbox analysis
7. **Sample Visualization** - Display annotated X-ray samples

### Part 2: Data Preprocessing & Augmentation (Cells 31-45)
8. **Augmentation Strategy** - Define extensive augmentation pipeline
9. **Augmentation Visualization** - Show before/after augmented samples
10. **Data Loader Setup** - Prepare training pipelines
11. **Class Imbalance Analysis** - Detailed imbalance study

### Part 3: Model Training (Cells 46-75)
12. **YOLOv8 Variants Training** - Train YOLOv8n, YOLOv8s, YOLOv8m
13. **YOLOv10 Training** - Latest YOLO architecture
14. **YOLOv11 Training** - Newest YOLO version
15. **RT-DETR Training** - Transformer-based detector
16. **Training Progress Monitoring** - Real-time tracking

### Part 4: Evaluation & Analysis (Cells 76-110)
17. **Model Evaluation** - Comprehensive metrics for all models
18. **Confusion Matrices** - Per-model confusion analysis
19. **Precision-Recall Curves** - PR curves for all models
20. **ROC Curves & AUC** - ROC analysis per class
21. **F1-Score Analysis** - F1 curves and optimal thresholds
22. **Per-Class Performance** - Detailed class-wise metrics
23. **Model Comparison** - Side-by-side comparison charts

### Part 5: XAI & Interpretability (Cells 111-125)
24. **Grad-CAM Visualization** - Attention maps for TB detection
25. **Feature Importance Analysis** - What the model focuses on
26. **Prediction Confidence Analysis** - Confidence distribution
27. **Error Analysis** - Analyze misclassifications
28. **Failure Case Study** - Deep dive into failure modes

### Part 6: Advanced Visualizations (Cells 126-145)
29. **Training Curves** - Loss and metric plots over epochs
30. **Interactive Plots** - Plotly interactive visualizations
31. **Bounding Box Statistics** - Spatial distribution analysis
32. **Augmentation Impact Study** - With vs without augmentation
33. **Inference Speed Comparison** - FPS and latency analysis

### Part 7: Results & Deployment (Cells 146-160)
34. **Final Results Summary** - Comprehensive results table
35. **Best Model Selection** - Select optimal model
36. **Model Export** - Save weights and configurations
37. **Deployment Preparation** - ONNX export, optimization
38. **Prediction Pipeline** - End-to-end inference demo
39. **Report Generation** - Auto-generate research report

---

## 🚀 How to Use This Notebook:

### Step 1: Update Configuration (Cell 5)
```python
# In the Config class, update:
DATASET_PATH = '/kaggle/input/your-dataset-name'  # Change this!
```

### Step 2: Run All Cells Sequentially
- **Cells 1-10**: Setup and data loading (5 minutes)
- **Cells 11-45**: EDA and augmentation setup (10 minutes)
- **Cells 46-75**: Model training (2-3 hours) ⏰
- **Cells 76-145**: Evaluation and visualization (30 minutes)
- **Cells 146-160**: Results and export (15 minutes)

**Total Time: ~4 hours** (mostly training)

### Step 3: Download Results
All outputs saved to `/kaggle/working/` or `./outputs/`:
- ✅ Trained model weights (.pt files)
- ✅ Training curves and metrics
- ✅ Confusion matrices and PR curves
- ✅ XAI visualizations
- ✅ Comparison charts
- ✅ Final research report

---

## 📊 Expected Outputs:

### Models Trained:
1. ✅ YOLOv8n (Nano) - Fast baseline
2. ✅ YOLOv8s (Small) - Balanced
3. ✅ YOLOv8m (Medium) - High accuracy
4. ✅ YOLOv10n (Nano) - Latest architecture
5. ✅ YOLOv11n (Nano) - Newest version
6. ✅ RT-DETR (Large) - Transformer-based

### Visualizations Generated:
- 📊 30+ professional research-quality plots
- 📈 Training curves for all models
- 🎯 Confusion matrices (6 models)
- 📉 PR curves, ROC curves, F1 curves
- 🔍 Grad-CAM attention maps
- 📊 Comparison bar charts, radar plots
- 🖼️ Annotated prediction samples

### Metrics Computed:
- mAP@0.5, mAP@0.5:0.95
- Precision, Recall, F1-Score (per class)
- Training time, Inference speed (FPS)
- Model size, Parameter count
- Confusion matrices, ROC-AUC scores

---

## ⚙️ Training Configuration:

**Extensive Augmentation Enabled:**
- ✅ Rotation (±15°)
- ✅ Translation (±15%)
- ✅ Scaling (70%-130%)
- ✅ Horizontal flip (50%)
- ✅ Color jitter (HSV)
- ✅ Mosaic & Mixup
- ✅ Random erasing
- ❌ Vertical flip (disabled for X-rays)

**Hardware Requirements:**
- GPU: T4 / P100 / V100 (16GB+ VRAM)
- RAM: 16GB+
- Storage: 5GB for outputs

---

## 🎓 Assignment Completion Checklist:

### ✅ Required Tasks (from PDF):
- [x] Train YOLOv8 variants (n, s, m)
- [x] Train YOLOv10
- [x] Train YOLOv11
- [x] Train RT-DETR (bonus)
- [x] Implement extensive augmentation
- [x] XAI analysis (Grad-CAM)
- [x] Model comparison
- [x] Professional visualizations
- [x] Comprehensive evaluation metrics
- [x] Research-quality documentation

### ✅ Bonus Features:
- [x] Interactive Plotly visualizations
- [x] Per-class performance analysis
- [x] Error analysis and failure cases
- [x] Inference speed benchmarking
- [x] Model export (ONNX)
- [x] Auto-generated research report
- [x] Augmentation impact study
- [x] Statistical significance tests

---

## 📝 Tips for Success:

1. **Check GPU**: Make sure GPU is enabled in Kaggle settings
2. **Monitor Training**: Watch for overfitting in training curves
3. **Patience**: Training takes 2-3 hours for all models
4. **Save Often**: Notebooks can disconnect - save progress
5. **Adjust Batch Size**: If OOM error, reduce batch_size in Config
6. **Review Plots**: Check all visualizations for insights
7. **Document**: Take screenshots for your report

---

## 🚨 Troubleshooting:

**Q: Dataset not found?**
```python
# Check available datasets
!ls /kaggle/input/

# Update Config.DATASET_PATH accordingly
```

**Q: Out of memory?**
```python
# In Config class, reduce:
BATCH_SIZE = 8  # instead of 16
IMG_SIZE = 416  # instead of 512
```

**Q: Training too slow?**
```python
# Use nano models:
MODELS_TO_TRAIN = {'YOLOv8n': {...}, 'YOLOv10n': {...}}
EPOCHS = 50  # instead of 150
```

**Q: Need faster results?**
```python
# Quick mode:
EPOCHS = 30
PATIENCE = 10
# Train only YOLOv8n and YOLOv11n
```

---

## 📦 Final Deliverables:

After running all cells, you'll have:

### 📁 `/kaggle/working/` folder contains:
```
working/
├── models/
│   ├── yolov8n_best.pt
│   ├── yolov8s_best.pt
│   ├── yolov8m_best.pt
│   ├── yolov10n_best.pt
│   ├── yolov11n_best.pt
│   └── rtdetr_best.pt
├── plots/
│   ├── class_distribution.png
│   ├── confusion_matrices.png
│   ├── pr_curves.png
│   ├── roc_curves.png
│   ├── f1_curves.png
│   ├── training_curves.png
│   ├── model_comparison.png
│   └── ... (30+ plots)
├── xai_analysis/
│   ├── gradcam_sample_1.png
│   ├── gradcam_sample_2.png
│   └── feature_importance.png
├── results/
│   ├── metrics_summary.csv
│   ├── per_class_performance.csv
│   ├── model_comparison.csv
│   └── final_report.md
└── logs/
    └── training_logs.txt
```

**Download this entire folder for your assignment!**

---

## 🎯 Next Steps:

1. ✅ Run this notebook completely
2. ✅ Download all outputs
3. ✅ Write your report using generated visualizations
4. ✅ Include best model weights
5. ✅ Submit with confidence!

---

**🎓 Good luck with your assignment!**

**This notebook provides everything needed for a comprehensive, publication-quality research project on tuberculosis detection using state-of-the-art deep learning models.**

---

*Created for CSE475 Lab Assignment 01 - East West University*  
*Author: Turjo Khan*  
*Date: November 2025*
