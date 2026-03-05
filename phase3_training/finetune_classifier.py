"""
Fine-tune Engagement Classifier
================================
Fine-tunes the existing OUC-CGE trained model with clean, individually-labeled data
to fix the group-level label noise issue.

Usage:
    python finetune_classifier.py --data ../finetune_dataset
    python finetune_classifier.py --data ../finetune_dataset --epochs 30 --lr 0.0005
"""

import sys
sys.path.append('..')

import argparse
from pathlib import Path
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description='Fine-tune engagement classifier')
    
    parser.add_argument('--data', type=str, required=True,
                        help='Path to fine-tune dataset (from relabel_tool.py)')
    parser.add_argument('--base-model', type=str, 
                        default='runs/engagement_organized_full/weights/best.pt',
                        help='Base model to fine-tune')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs (default: 20)')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate (default: 0.0001, small for fine-tuning)')
    parser.add_argument('--batch', type=int, default=64,
                        help='Batch size (default: 64)')
    parser.add_argument('--freeze', type=int, default=10,
                        help='Number of layers to freeze (default: 10)')
    parser.add_argument('--imgsz', type=int, default=224,
                        help='Image size (default: 224)')
    parser.add_argument('--patience', type=int, default=5,
                        help='Early stopping patience (default: 5)')
    parser.add_argument('--name', type=str, default='engagement_finetuned_v1',
                        help='Run name')
    parser.add_argument('--device', type=str, default='0',
                        help='Device (default: 0 for GPU)')
    
    args = parser.parse_args()
    
    # Validate paths
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"❌ Dataset not found: {data_path}")
        print("   Run relabel_tool.py first to create the fine-tune dataset!")
        return
    
    # Check dataset structure
    print("=" * 60)
    print("🔧 FINE-TUNING ENGAGEMENT CLASSIFIER")
    print("=" * 60)
    
    for split in ['train', 'val', 'test']:
        split_dir = data_path / split
        if split_dir.exists():
            total = 0
            for cat in ['high', 'medium', 'low']:
                cat_dir = split_dir / cat
                if cat_dir.exists():
                    count = len(list(cat_dir.glob('*.jpg')))
                    total += count
            print(f"  {split}: {total} crops")
        else:
            print(f"  ⚠ {split}/ not found")
    
    print(f"\n  Base model: {args.base_model}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Frozen layers: {args.freeze}")
    print("=" * 60)
    
    # Load and fine-tune
    print(f"\n📦 Loading base model: {args.base_model}")
    model = YOLO(args.base_model)
    
    print(f"\n🚀 Starting fine-tuning...\n")
    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        lr0=args.lr,
        lrf=0.01,               # Final LR = lr0 * lrf
        freeze=args.freeze,      # Freeze early layers
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        optimizer='AdamW',
        warmup_epochs=2,
        cos_lr=True,
        amp=True,
        device=args.device,
        name=args.name,
        exist_ok=True,
        verbose=True,
        # Augmentations (milder for fine-tuning)
        hsv_h=0.01,
        hsv_s=0.3,
        hsv_v=0.3,
        degrees=5,
        translate=0.05,
        scale=0.2,
        flipud=0.0,
        fliplr=0.5,
    )
    
    print(f"\n{'='*60}")
    print(f"✅ FINE-TUNING COMPLETE!")
    print(f"{'='*60}")
    print(f"  Best model: runs/{args.name}/weights/best.pt")
    print(f"  Last model: runs/{args.name}/weights/last.pt")
    print(f"\n  Copy best model to production:")
    print(f"    copy runs\\{args.name}\\weights\\best.pt ..\\models\\engagement_finetuned.pt")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
