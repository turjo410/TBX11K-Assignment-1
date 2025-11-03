# TBX11K Dataset Analysis for YOLO Training

## 📊 Dataset Overview

The TBX11K dataset is a tuberculosis detection dataset containing **11,200 X-ray images** (512x512) with bounding box annotations for TB detection.

### Dataset Split:
- **Training**: 6,600 images
- **Validation**: 1,800 images  
- **Testing**: 2,800 images

### Image Distribution:
```
├── health/    : 3,800 images (Healthy X-rays)
├── sick/      : 3,800 images (Sick but Non-TB)
├── tb/        : 800 images (TB cases with annotations)
├── test/      : 3,302 images (Test set)
└── extra/     : 2 images (Additional datasets)
```

### Categories:
1. **ActiveTuberculosis** (Category ID: 1)
2. **ObsoletePulmonaryTuberculosis** (Latent TB, Category ID: 2)
3. **PulmonaryTuberculosis** (Uncertain TB, Category ID: 3)

---

## ✅ Current Status

### What's Working:
1. ✅ **Annotations Available**: COCO-style JSON annotations are present
2. ✅ **XML Annotations**: Pascal VOC format XML files exist for TB images
3. ✅ **Images Organized**: Images are properly organized in folders
4. ✅ **Training Data**: 6,600 images with 902 bounding box annotations

### Annotation Statistics:
- **Training Set (TBX11K_train.json)**:
  - Total Images: 6,600
  - Total Annotations: 902 bounding boxes
  - Average: ~0.14 annotations per image (most images are negative samples)

---

## ⚠️ Issues for YOLO Training

### 🔴 **CRITICAL ISSUES:**

### 1. **Annotation Format Incompatibility**
- ❌ Current format: **COCO JSON** (absolute bounding boxes)
- ✅ Required format: **YOLO TXT** (normalized coordinates)

**YOLO Format Required:**
```
<class_id> <x_center> <y_center> <width> <height>
```
All values normalized to [0, 1]

**Current Format (COCO):**
```json
{
  "bbox": [x_min, y_min, width, height],  // Absolute pixels
  "category_id": 1
}
```

### 2. **Directory Structure**
- ❌ Current structure is **NOT** YOLO-compatible
- ✅ YOLO expects:
```
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

### 3. **Class Imbalance**
- Only **~13.7% of training images** have TB annotations (902/6600)
- **86.3% are negative samples** (healthy or sick but non-TB)
- This severe imbalance needs attention during training

### 4. **Missing Label Files**
- No `.txt` label files exist for YOLO
- Need to convert COCO JSON → YOLO TXT format

---

## 🔧 **REQUIRED ACTIONS TO TRAIN YOLO**

### Step 1: Convert COCO to YOLO Format

**Create a conversion script:**

```python
import json
import os
from pathlib import Path

def convert_coco_to_yolo(json_file, img_width=512, img_height=512):
    """
    Convert COCO format to YOLO format
    COCO: [x_min, y_min, width, height] (absolute)
    YOLO: [class_id, x_center, y_center, width, height] (normalized)
    """
    with open(json_file) as f:
        data = json.load(f)
    
    # Create image_id to annotations mapping
    img_annotations = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in img_annotations:
            img_annotations[img_id] = []
        img_annotations[img_id].append(ann)
    
    # Convert each image
    labels_dir = Path('labels/train')  # or val/test
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    for img in data['images']:
        img_id = img['id']
        img_name = img['file_name'].replace('.png', '.txt')
        
        # Get annotations for this image
        anns = img_annotations.get(img_id, [])
        
        label_file = labels_dir / img_name
        
        with open(label_file, 'w') as f:
            for ann in anns:
                # COCO format: [x_min, y_min, width, height]
                x_min, y_min, width, height = ann['bbox']
                
                # Convert to YOLO format
                x_center = (x_min + width / 2) / img_width
                y_center = (y_min + height / 2) / img_height
                norm_width = width / img_width
                norm_height = height / img_height
                
                # YOLO uses 0-indexed classes
                class_id = ann['category_id'] - 1  # Convert 1,2,3 to 0,1,2
                
                # Write YOLO format
                f.write(f"{class_id} {x_center} {y_center} {norm_width} {norm_height}\n")
```

### Step 2: Reorganize Directory Structure

**Create proper YOLO structure:**

```bash
mkdir -p dataset/images/{train,val,test}
mkdir -p dataset/labels/{train,val,test}

# Copy images according to train/val/test splits
# Use the .txt list files to determine which images go where
```

### Step 3: Create YOLO Dataset Configuration

**Create `data.yaml`:**

```yaml
# data.yaml
train: ../dataset/images/train
val: ../dataset/images/val
test: ../dataset/images/test

nc: 3  # number of classes
names: ['ActiveTuberculosis', 'ObsoletePulmonaryTuberculosis', 'PulmonaryTuberculosis']

# Optional: class weights for imbalanced dataset
# class_weights: [1.0, 1.0, 1.0]
```

### Step 4: Handle Class Imbalance

**Strategies:**
1. **Weighted Loss**: Apply class weights inversely proportional to frequency
2. **Augmentation**: Heavy augmentation on TB-positive images
3. **Focal Loss**: Use focal loss to focus on hard examples
4. **Sampling**: Oversample TB-positive images during training

### Step 5: Create Empty Label Files

For images without annotations (negative samples), create **empty .txt files**:

```python
# For images in healthy/ and sick/ folders without annotations
import os
from pathlib import Path

def create_empty_labels(image_dir, label_dir):
    """Create empty label files for negative samples"""
    label_dir = Path(label_dir)
    label_dir.mkdir(parents=True, exist_ok=True)
    
    for img_file in Path(image_dir).glob('*.png'):
        label_file = label_dir / (img_file.stem + '.txt')
        if not label_file.exists():
            label_file.touch()  # Create empty file
```

---

## 📝 **COMPLETE CONVERSION SCRIPT**

Here's a complete script you can use:

```python
#!/usr/bin/env python3
"""
TBX11K to YOLO Converter
Converts TBX11K dataset to YOLO format for training
"""

import json
import shutil
from pathlib import Path
from tqdm import tqdm

class TBX11KtoYOLO:
    def __init__(self, dataset_root, output_root):
        self.dataset_root = Path(dataset_root)
        self.output_root = Path(output_root)
        self.img_size = 512  # TBX11K images are 512x512
        
    def convert_bbox_coco_to_yolo(self, bbox, img_w, img_h):
        """Convert COCO bbox to YOLO format"""
        x_min, y_min, width, height = bbox
        
        x_center = (x_min + width / 2) / img_w
        y_center = (y_min + height / 2) / img_h
        norm_width = width / img_w
        norm_height = height / img_h
        
        return x_center, y_center, norm_width, norm_height
    
    def process_split(self, json_file, split_name):
        """Process one split (train/val/test)"""
        print(f"\n{'='*50}")
        print(f"Processing {split_name} split...")
        print(f"{'='*50}")
        
        # Create directories
        img_out_dir = self.output_root / 'images' / split_name
        lbl_out_dir = self.output_root / 'labels' / split_name
        img_out_dir.mkdir(parents=True, exist_ok=True)
        lbl_out_dir.mkdir(parents=True, exist_ok=True)
        
        # Load JSON
        with open(json_file) as f:
            data = json.load(f)
        
        # Create image_id to annotations mapping
        img_annotations = {}
        for ann in data['annotations']:
            img_id = ann['image_id']
            if img_id not in img_annotations:
                img_annotations[img_id] = []
            img_annotations[img_id].append(ann)
        
        # Process each image
        stats = {'total': 0, 'with_annotations': 0, 'without_annotations': 0}
        
        for img in tqdm(data['images'], desc=f'{split_name}'):
            img_id = img['id']
            img_name = img['file_name']
            img_path = self.dataset_root / 'imgs' / img_name
            
            # Copy image
            if img_path.exists():
                shutil.copy2(img_path, img_out_dir / Path(img_name).name)
                stats['total'] += 1
            else:
                print(f"Warning: {img_path} not found!")
                continue
            
            # Create label file
            label_name = Path(img_name).stem + '.txt'
            label_path = lbl_out_dir / label_name
            
            anns = img_annotations.get(img_id, [])
            
            with open(label_path, 'w') as f:
                if anns:
                    stats['with_annotations'] += 1
                    for ann in anns:
                        bbox = ann['bbox']
                        x_c, y_c, w, h = self.convert_bbox_coco_to_yolo(
                            bbox, self.img_size, self.img_size
                        )
                        
                        # YOLO uses 0-indexed classes
                        class_id = ann['category_id'] - 1
                        
                        f.write(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")
                else:
                    stats['without_annotations'] += 1
                    # Empty file for negative samples
                    pass
        
        print(f"\n{split_name.upper()} Statistics:")
        print(f"  Total images: {stats['total']}")
        print(f"  With annotations: {stats['with_annotations']} ({stats['with_annotations']/stats['total']*100:.1f}%)")
        print(f"  Without annotations: {stats['without_annotations']} ({stats['without_annotations']/stats['total']*100:.1f}%)")
        
    def convert(self):
        """Convert all splits"""
        print("Starting TBX11K to YOLO conversion...")
        
        # Process each split
        splits = [
            ('TBX11K_train.json', 'train'),
            ('TBX11K_val.json', 'val'),
            # ('all_test.json', 'test')  # Uncomment if you have test labels
        ]
        
        for json_file, split_name in splits:
            json_path = self.dataset_root / 'annotations' / 'json' / json_file
            if json_path.exists():
                self.process_split(json_path, split_name)
            else:
                print(f"Warning: {json_path} not found! Skipping {split_name}...")
        
        # Create data.yaml
        self.create_yaml()
        
        print("\n✅ Conversion complete!")
        print(f"📁 Output directory: {self.output_root}")
        
    def create_yaml(self):
        """Create data.yaml for YOLO"""
        yaml_content = f"""# TBX11K Dataset Configuration for YOLO
path: {self.output_root.absolute()}
train: images/train
val: images/val
test: images/test  # optional

# Classes
nc: 3
names:
  0: ActiveTuberculosis
  1: ObsoletePulmonaryTuberculosis
  2: PulmonaryTuberculosis

# Training hyperparameters for imbalanced dataset
# Consider using:
# - Higher weight_decay
# - Focal loss
# - Class weights
# - More augmentation on positive samples
"""
        
        yaml_path = self.output_root / 'data.yaml'
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        
        print(f"\n📝 Created data.yaml at: {yaml_path}")

if __name__ == '__main__':
    # Configure paths
    DATASET_ROOT = Path('/Users/turjokhan/Study EWU CSE /10th Semester/CSE475/Assignement 1/TBX11K')
    OUTPUT_ROOT = Path('/Users/turjokhan/Study EWU CSE /10th Semester/CSE475/Assignement 1/TBX11K_YOLO')
    
    # Convert
    converter = TBX11KtoYOLO(DATASET_ROOT, OUTPUT_ROOT)
    converter.convert()
    
    print("\n🎯 Next Steps:")
    print("1. Verify converted dataset structure")
    print("2. Train YOLO model with:")
    print("   yolo train data=data.yaml model=yolov8n.pt epochs=100 imgsz=512")
    print("3. Consider class weights for imbalanced data")
    print("4. Use heavy augmentation on TB-positive samples")
```

---

## 🎯 **Training Recommendations**

### 1. **Model Selection**
- Start with **YOLOv8n or YOLOv8s** (small, fast)
- For better accuracy: **YOLOv8m or YOLOv8l**

### 2. **Training Configuration**

```python
from ultralytics import YOLO

# Load model
model = YOLO('yolov8n.pt')

# Train with optimized settings
results = model.train(
    data='data.yaml',
    epochs=100,
    imgsz=512,  # Match TBX11K image size
    batch=16,
    patience=20,
    
    # Augmentation (important for medical images)
    mosaic=0.5,  # Reduce mosaic for medical images
    mixup=0.0,   # Avoid mixup for medical images
    hsv_h=0.01,  # Minimal hue shift
    hsv_s=0.5,   # Some saturation
    hsv_v=0.3,   # Some brightness
    degrees=5,   # Small rotation
    translate=0.1,
    scale=0.3,
    fliplr=0.5,  # Horizontal flip OK
    flipud=0.0,  # No vertical flip for X-rays
    
    # Loss weights for imbalanced data
    cls=1.0,
    box=7.5,
    dfl=1.5,
)
```

### 3. **Handling Imbalance**

```python
# Option 1: Undersample negative samples
# Use only a portion of healthy/sick images

# Option 2: Focal loss (built into YOLOv8)
# Already included in YOLO

# Option 3: Custom sampling
# Oversample TB-positive images in dataloader
```

---

## 📊 **Expected Results**

Based on the dataset characteristics:
- **mAP@0.5**: 40-60% (due to class imbalance)
- **Precision**: May be low due to false positives on healthy samples
- **Recall**: Should improve with proper augmentation

---

## ✅ **Summary**

### **Is the dataset ready for YOLO?**
❌ **NO** - Requires conversion

### **What needs to be done?**
1. ✅ Convert COCO JSON → YOLO TXT format
2. ✅ Reorganize directory structure
3. ✅ Create data.yaml configuration
4. ✅ Handle class imbalance
5. ✅ Create empty labels for negative samples

### **Is the dataset good quality?**
✅ **YES** - Well-organized, properly annotated, good resolution (512x512)

### **Main Challenge:**
⚠️ **Severe class imbalance** (86% negative samples) - needs careful handling during training

---

## 🚀 **Quick Start Command**

After running the conversion script:

```bash
# Install ultralytics
pip install ultralytics

# Train YOLOv8
yolo train data=TBX11K_YOLO/data.yaml model=yolov8n.pt epochs=100 imgsz=512 batch=16
```

---

**Ready to proceed? Run the conversion script above and start training! 🎯**
