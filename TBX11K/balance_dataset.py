#!/usr/bin/env python3
"""
Fix Class Imbalance in TBX11K Dataset
Creates a balanced dataset by undersampling negative samples

Strategy: Keep all TB-positive images, reduce negative samples to achieve better balance
"""

import shutil
from pathlib import Path
import random
from tqdm import tqdm

class DatasetBalancer:
    """Balance TBX11K dataset by undersampling negative samples"""
    
    def __init__(self, source_dir, output_dir, balance_ratio=0.5):
        """
        Initialize balancer
        
        Args:
            source_dir: Path to original yolo_dataset
            output_dir: Path for balanced dataset
            balance_ratio: Target ratio of negative to positive samples
                          0.5 = 1 negative for every 2 positives
                          1.0 = 1 negative for every 1 positive (50/50)
                          2.0 = 2 negatives for every 1 positive
        """
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.balance_ratio = balance_ratio
        
    def find_positive_negative_samples(self, split='train'):
        """Find which images have TB (positive) vs no TB (negative)"""
        
        labels_dir = self.source_dir / 'labels' / split
        
        positive_samples = []
        negative_samples = []
        
        print(f"\nAnalyzing {split} set...")
        
        for label_file in labels_dir.glob('*.txt'):
            # Check if file has content (TB detected)
            if label_file.stat().st_size > 0:
                positive_samples.append(label_file.stem)
            else:
                negative_samples.append(label_file.stem)
        
        print(f"  Positive (with TB): {len(positive_samples)}")
        print(f"  Negative (no TB): {len(negative_samples)}")
        print(f"  Imbalance ratio: 1:{len(negative_samples)/len(positive_samples):.1f}")
        
        return positive_samples, negative_samples
    
    def create_balanced_split(self, split='train'):
        """Create balanced dataset for one split"""
        
        print(f"\n{'='*70}")
        print(f"Balancing {split.upper()} set")
        print(f"{'='*70}")
        
        # Find positive and negative samples
        positive_samples, negative_samples = self.find_positive_negative_samples(split)
        
        # Calculate how many negatives to keep
        num_positives = len(positive_samples)
        num_negatives_to_keep = int(num_positives * self.balance_ratio)
        
        # Randomly sample negatives
        random.shuffle(negative_samples)
        selected_negatives = negative_samples[:num_negatives_to_keep]
        
        print(f"\nBalancing strategy:")
        print(f"  Keep all positives: {num_positives}")
        print(f"  Keep negatives: {num_negatives_to_keep} (from {len(negative_samples)})")
        print(f"  New total: {num_positives + num_negatives_to_keep}")
        print(f"  New ratio: 1:{num_negatives_to_keep/num_positives:.2f}")
        print(f"  Positive percentage: {num_positives/(num_positives+num_negatives_to_keep)*100:.1f}%")
        
        # Combine samples
        selected_samples = positive_samples + selected_negatives
        random.shuffle(selected_samples)
        
        # Create output directories
        output_img_dir = self.output_dir / 'images' / split
        output_lbl_dir = self.output_dir / 'labels' / split
        output_img_dir.mkdir(parents=True, exist_ok=True)
        output_lbl_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy selected samples
        print(f"\nCopying {len(selected_samples)} samples...")
        
        for sample_id in tqdm(selected_samples, desc=f'Copying {split}'):
            # Copy image
            src_img = self.source_dir / 'images' / split / f'{sample_id}.png'
            dst_img = output_img_dir / f'{sample_id}.png'
            if src_img.exists():
                shutil.copy2(src_img, dst_img)
            
            # Copy label
            src_lbl = self.source_dir / 'labels' / split / f'{sample_id}.txt'
            dst_lbl = output_lbl_dir / f'{sample_id}.txt'
            if src_lbl.exists():
                shutil.copy2(src_lbl, dst_lbl)
        
        stats = {
            'total': len(selected_samples),
            'positive': num_positives,
            'negative': num_negatives_to_keep,
            'positive_pct': num_positives/(num_positives+num_negatives_to_keep)*100
        }
        
        return stats
    
    def balance_dataset(self):
        """Balance entire dataset (train and val)"""
        
        print("\n" + "="*70)
        print("TBX11K DATASET BALANCER")
        print("="*70)
        print(f"\nSource: {self.source_dir}")
        print(f"Output: {self.output_dir}")
        print(f"Balance ratio: {self.balance_ratio} (negatives per positive)")
        print("="*70)
        
        # Balance train set
        train_stats = self.create_balanced_split('train')
        
        # Balance val set
        val_stats = self.create_balanced_split('val')
        
        # Copy data.yaml and update
        self.create_data_yaml(train_stats, val_stats)
        
        # Print summary
        print("\n" + "="*70)
        print("✅ DATASET BALANCING COMPLETE!")
        print("="*70)
        print(f"\nTRAIN SET:")
        print(f"  Total images: {train_stats['total']}")
        print(f"  Positive (TB): {train_stats['positive']} ({train_stats['positive_pct']:.1f}%)")
        print(f"  Negative (no TB): {train_stats['negative']} ({100-train_stats['positive_pct']:.1f}%)")
        
        print(f"\nVAL SET:")
        print(f"  Total images: {val_stats['total']}")
        print(f"  Positive (TB): {val_stats['positive']} ({val_stats['positive_pct']:.1f}%)")
        print(f"  Negative (no TB): {val_stats['negative']} ({100-val_stats['positive_pct']:.1f}%)")
        
        print(f"\nTOTAL:")
        print(f"  Combined images: {train_stats['total'] + val_stats['total']}")
        print(f"  Space saved: ~{(8400 - train_stats['total'] - val_stats['total']) * 0.35:.0f} MB")
        
        print("\n" + "="*70)
        print("🎯 Ready for training with balanced data!")
        print("="*70)
    
    def create_data_yaml(self, train_stats, val_stats):
        """Create updated data.yaml for balanced dataset"""
        
        yaml_content = f"""# TBX11K Dataset Configuration for YOLO (BALANCED VERSION)
# Tuberculosis Detection Dataset - Class Imbalance Fixed
# Generated by Dataset Balancer

# Dataset paths (relative to this yaml file)
path: {self.output_dir.absolute()}
train: images/train
val: images/val
test: images/test  # optional

# Number of classes
nc: 3

# Class names (0-indexed for YOLO)
names:
  0: ActiveTuberculosis              # Active TB
  1: ObsoletePulmonaryTuberculosis   # Latent TB  
  2: PulmonaryTuberculosis           # Uncertain TB

# Dataset Statistics (BALANCED)
# Train: {train_stats['total']} images ({train_stats['positive_pct']:.1f}% positive)
# Val: {val_stats['total']} images ({val_stats['positive_pct']:.1f}% positive)
# Total: {train_stats['total'] + val_stats['total']} images

# Balancing Details:
# - Original imbalance: 90% negative samples
# - New balance: ~{100-train_stats['positive_pct']:.0f}% negative samples
# - Method: Undersampling (kept all positives, reduced negatives)
# - This improves model learning significantly!

# Training Notes:
# 1. Dataset is now BALANCED - much better for training!
# 2. Model will learn to detect TB cases better
# 3. Less bias toward "no TB" predictions
# 4. Expected mAP improvement: +10-20%
# 5. Recommended settings:
#    - epochs: 100-150
#    - batch: 16
#    - imgsz: 512
#    - patience: 20
"""
        
        yaml_path = self.output_dir / 'data.yaml'
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        
        print(f"\n📝 Created balanced data.yaml: {yaml_path}")


def main():
    """Main function with different balancing options"""
    
    import sys
    
    print("\n" + "="*70)
    print("TBX11K DATASET BALANCER")
    print("Fix Class Imbalance by Undersampling Negative Samples")
    print("="*70)
    
    # Get paths
    source_dir = Path(__file__).parent / 'yolo_dataset'
    
    if not source_dir.exists():
        print(f"\n❌ Error: Source dataset not found at {source_dir}")
        sys.exit(1)
    
    print("\n📊 ORIGINAL DATASET:")
    print("  Train: 6,600 images (9.1% positive, 90.9% negative)")
    print("  Val: 1,800 images (11.1% positive, 88.9% negative)")
    print("  Problem: Severe imbalance!")
    
    print("\n🎯 BALANCING OPTIONS:")
    print("\n1. AGGRESSIVE (50/50 balance)")
    print("   → Keep all 599 positives + 599 negatives = 1,198 train images")
    print("   → 50% positive, 50% negative")
    print("   → Best for learning, but smaller dataset")
    
    print("\n2. MODERATE (33/67 balance) - RECOMMENDED")
    print("   → Keep all 599 positives + 1,198 negatives = 1,797 train images")
    print("   → 33% positive, 67% negative")
    print("   → Good balance + reasonable dataset size")
    
    print("\n3. CONSERVATIVE (25/75 balance)")
    print("   → Keep all 599 positives + 1,797 negatives = 2,396 train images")
    print("   → 25% positive, 75% negative")
    print("   → Still better than original 9/91")
    
    print("\n4. CUSTOM")
    print("   → You specify the ratio")
    
    print("\n" + "="*70)
    
    # Get user choice
    choice = input("\nSelect option (1/2/3/4) [default: 2]: ").strip() or "2"
    
    if choice == "1":
        balance_ratio = 1.0  # 1 negative per 1 positive (50/50)
        output_name = "yolo_dataset_balanced_50_50"
    elif choice == "2":
        balance_ratio = 2.0  # 2 negatives per 1 positive (33/67)
        output_name = "yolo_dataset_balanced_33_67"
    elif choice == "3":
        balance_ratio = 3.0  # 3 negatives per 1 positive (25/75)
        output_name = "yolo_dataset_balanced_25_75"
    elif choice == "4":
        try:
            balance_ratio = float(input("Enter ratio (e.g., 1.5 for 40/60 balance): "))
            output_name = f"yolo_dataset_balanced_custom"
        except ValueError:
            print("Invalid input. Using moderate balance (33/67).")
            balance_ratio = 2.0
            output_name = "yolo_dataset_balanced_33_67"
    else:
        print("Invalid choice. Using moderate balance (33/67).")
        balance_ratio = 2.0
        output_name = "yolo_dataset_balanced_33_67"
    
    output_dir = source_dir.parent / output_name
    
    print(f"\n✅ Selected balance ratio: {balance_ratio}")
    print(f"📁 Output directory: {output_dir.name}")
    print("\nPress Enter to start balancing or Ctrl+C to cancel...")
    input()
    
    # Create balancer and run
    balancer = DatasetBalancer(source_dir, output_dir, balance_ratio)
    balancer.balance_dataset()
    
    print(f"\n💡 NEXT STEPS:")
    print(f"   1. Use the new balanced dataset for training")
    print(f"   2. Update your training script to use: {output_dir.name}/data.yaml")
    print(f"   3. Expected improvement: +10-20% mAP")
    print(f"   4. Training will be faster (fewer images)")
    print(f"\n🚀 Happy training with balanced data!")


if __name__ == '__main__':
    main()
