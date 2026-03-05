"""
Smart Relabeling Tool for Fine-tuning Dataset
==============================================
Helps you review and relabel person crops from OUC-CGE dataset.

Features:
- Smart mode: Uses existing model to find suspicious crops first
- Random mode: Samples randomly from all folders
- Keyboard shortcuts for quick labeling
- Progress tracking with resume capability
- Auto-splits into train/val/test (80/10/10)

Usage:
    # Smart mode (recommended) - shows suspicious crops first
    python relabel_tool.py --smart --model ../phase3_training/runs/engagement_organized_full/weights/best.pt

    # Random mode - samples randomly
    python relabel_tool.py

    # Custom target count
    python relabel_tool.py --smart --target 1000
"""

import sys
sys.path.append('..')

import cv2
import numpy as np
from pathlib import Path
import argparse
import json
import random
import shutil
from datetime import datetime


def load_model_for_smart_mode(model_path):
    """Load YOLO model for smart prediction"""
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
        print(f"✅ Model loaded: {model_path}")
        return model
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None


def get_all_crops(crops_root):
    """Get all crop files organized by folder label"""
    crops_root = Path(crops_root)
    all_crops = []
    
    for split in ['train', 'val', 'test']:
        for category in ['high', 'medium', 'low']:
            category_dir = crops_root / split / category
            if category_dir.exists():
                for img_path in category_dir.glob('*.jpg'):
                    all_crops.append({
                        'path': str(img_path),
                        'folder_label': category,
                        'split': split
                    })
    
    return all_crops


def find_suspicious_crops(crops, model, sample_size=5000, batch_size=32):
    """Use model to find crops where prediction disagrees with folder label"""
    print(f"\n🔍 Scanning {sample_size} crops for suspicious labels...")
    
    # Sample a subset to scan
    if len(crops) > sample_size:
        sampled = random.sample(crops, sample_size)
    else:
        sampled = crops
    
    # Map class indices to names
    class_names = {0: 'high', 1: 'low', 2: 'medium'}  # From data.yaml
    
    suspicious = []
    agreeing = []
    
    for i, crop_info in enumerate(sampled):
        if (i + 1) % 200 == 0:
            print(f"  Scanned {i+1}/{len(sampled)} | Found {len(suspicious)} suspicious")
        
        try:
            result = model(crop_info['path'], verbose=False)
            pred_idx = int(result[0].probs.top1)
            pred_conf = float(result[0].probs.top1conf)
            pred_name = class_names.get(pred_idx, 'unknown')
            
            crop_info['predicted_label'] = pred_name
            crop_info['prediction_confidence'] = pred_conf
            
            # Suspicious = prediction disagrees with folder AND confidence is high
            if pred_name != crop_info['folder_label'] and pred_conf > 0.7:
                crop_info['suspicion_score'] = pred_conf
                suspicious.append(crop_info)
            else:
                agreeing.append(crop_info)
                
        except Exception as e:
            continue
    
    # Sort suspicious by confidence (most suspicious first)
    suspicious.sort(key=lambda x: x['suspicion_score'], reverse=True)
    
    print(f"\n📊 Scan Results:")
    print(f"  Total scanned: {len(sampled)}")
    print(f"  Suspicious (label mismatch): {len(suspicious)} ({len(suspicious)/len(sampled)*100:.1f}%)")
    print(f"  Agreeing: {len(agreeing)}")
    
    return suspicious, agreeing


def create_display_image(img, crop_info, progress, total, is_smart=False):
    """Create display image with info overlay"""
    # Resize for better visibility
    display_h = 500
    h, w = img.shape[:2]
    scale = display_h / h
    display_w = int(w * scale)
    display = cv2.resize(img, (display_w, display_h))
    
    # Create info panel
    panel_width = 450
    total_width = display_w + panel_width
    canvas = np.zeros((max(display_h, 550), total_width, 3), dtype=np.uint8)
    canvas[:, :, :] = 30  # Dark background
    
    # Place image
    canvas[:display_h, :display_w] = display
    
    # Info panel
    x_start = display_w + 15
    y = 30
    line_h = 30
    
    # Title
    cv2.putText(canvas, "RELABELING TOOL", (x_start, y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
    y += line_h + 10
    
    # Progress
    cv2.putText(canvas, f"Progress: {progress}/{total}", (x_start, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    y += line_h
    
    # Progress bar
    bar_width = panel_width - 30
    bar_height = 15
    progress_ratio = progress / max(total, 1)
    cv2.rectangle(canvas, (x_start, y), (x_start + bar_width, y + bar_height), (60, 60, 60), -1)
    cv2.rectangle(canvas, (x_start, y), (x_start + int(bar_width * progress_ratio), y + bar_height), (0, 200, 100), -1)
    y += bar_height + 20
    
    # Folder label
    folder_color = {'high': (0, 255, 0), 'medium': (0, 165, 255), 'low': (0, 0, 255)}
    label = crop_info['folder_label']
    cv2.putText(canvas, f"Folder label: {label.upper()}", (x_start, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, folder_color.get(label, (200, 200, 200)), 2)
    y += line_h
    
    # Smart mode info
    if is_smart and 'predicted_label' in crop_info:
        pred = crop_info['predicted_label']
        conf = crop_info.get('prediction_confidence', 0)
        cv2.putText(canvas, f"Model predicts: {pred.upper()} ({conf:.0%})", (x_start, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, folder_color.get(pred, (200, 200, 200)), 2)
        y += line_h
        
        if pred != label:
            cv2.putText(canvas, ">>> MISMATCH! <<<", (x_start, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            y += line_h
    
    y += 20
    
    # Instructions
    cv2.putText(canvas, "--- KEYBOARD ---", (x_start, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1)
    y += line_h + 5
    
    instructions = [
        ("H", "= Label HIGH (engaged)", (0, 255, 0)),
        ("M", "= Label MEDIUM", (0, 165, 255)),
        ("L", "= Label LOW (disengaged)", (0, 0, 255)),
        ("S", "= SKIP (not sure)", (150, 150, 150)),
        ("Z", "= UNDO last label", (255, 255, 0)),
        ("Q", "= QUIT & save", (255, 100, 100)),
    ]
    
    for key, desc, color in instructions:
        cv2.putText(canvas, f"[{key}]  {desc}", (x_start, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
        y += line_h - 2
    
    y += 15
    
    # Source file
    filename = Path(crop_info['path']).name
    if len(filename) > 35:
        filename = filename[:32] + "..."
    cv2.putText(canvas, f"File: {filename}", (x_start, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)
    
    return canvas


def save_progress(progress_file, labeled_data, stats):
    """Save progress to JSON file"""
    progress = {
        'timestamp': datetime.now().isoformat(),
        'stats': stats,
        'labeled': labeled_data
    }
    with open(progress_file, 'w', encoding='utf-8') as f:
        json.dump(progress, f, indent=2, ensure_ascii=False)


def load_progress(progress_file):
    """Load previous progress"""
    try:
        with open(progress_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return None


def copy_labeled_crops(labeled_data, output_root):
    """Copy labeled crops to output directory with train/val/test split"""
    output_root = Path(output_root)
    
    # Separate by assigned label
    by_label = {'high': [], 'medium': [], 'low': []}
    for item in labeled_data:
        label = item['assigned_label']
        if label in by_label:
            by_label[label].append(item)
    
    print(f"\n📦 Organizing labeled crops...")
    print(f"  High: {len(by_label['high'])}")
    print(f"  Medium: {len(by_label['medium'])}")
    print(f"  Low: {len(by_label['low'])}")
    
    total_copied = 0
    
    for label, items in by_label.items():
        if not items:
            continue
            
        # Shuffle for random split
        random.shuffle(items)
        
        n = len(items)
        n_train = max(1, int(n * 0.8))
        n_val = max(1, int(n * 0.1))
        # Rest goes to test
        
        splits = {
            'train': items[:n_train],
            'val': items[n_train:n_train + n_val],
            'test': items[n_train + n_val:]
        }
        
        for split_name, split_items in splits.items():
            split_dir = output_root / split_name / label
            split_dir.mkdir(parents=True, exist_ok=True)
            
            for item in split_items:
                src = Path(item['path'])
                if src.exists():
                    dst = split_dir / src.name
                    # Avoid name conflicts
                    counter = 1
                    while dst.exists():
                        dst = split_dir / f"{src.stem}_{counter}{src.suffix}"
                        counter += 1
                    shutil.copy2(str(src), str(dst))
                    total_copied += 1
    
    print(f"\n✅ Copied {total_copied} crops to {output_root}")
    
    # Print final structure
    print(f"\n📁 Output structure:")
    for split in ['train', 'val', 'test']:
        split_dir = output_root / split
        if split_dir.exists():
            print(f"  {split}/")
            for cat in ['high', 'medium', 'low']:
                cat_dir = split_dir / cat
                if cat_dir.exists():
                    count = len(list(cat_dir.glob('*.jpg')))
                    print(f"    {cat}/  ({count} crops)")


def run_labeling(crops_to_label, output_root, progress_file, is_smart=False):
    """Main labeling loop"""
    output_root = Path(output_root)
    
    # Load previous progress
    prev_progress = load_progress(progress_file)
    labeled_data = []
    already_labeled_paths = set()
    
    if prev_progress:
        labeled_data = prev_progress.get('labeled', [])
        already_labeled_paths = {item['path'] for item in labeled_data}
        stats = prev_progress.get('stats', {'high': 0, 'medium': 0, 'low': 0, 'skipped': 0})
        print(f"\n📂 Resuming from previous session!")
        print(f"   Already labeled: {len(labeled_data)} crops")
        print(f"   H:{stats['high']} M:{stats['medium']} L:{stats['low']} Skip:{stats['skipped']}")
    else:
        stats = {'high': 0, 'medium': 0, 'low': 0, 'skipped': 0}
    
    # Filter out already labeled
    remaining = [c for c in crops_to_label if c['path'] not in already_labeled_paths]
    total = len(remaining)
    
    if total == 0:
        print("✅ All crops already labeled!")
        return labeled_data, stats
    
    print(f"\n🎯 {total} crops to label")
    print(f"   Press H/M/L to label, S to skip, Z to undo, Q to quit\n")
    
    cv2.namedWindow("Relabeling Tool", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Relabeling Tool", 950, 550)
    
    i = 0
    history = []  # For undo
    
    while i < total:
        crop_info = remaining[i]
        
        # Load image
        img = cv2.imread(crop_info['path'])
        if img is None:
            i += 1
            continue
        
        # Create display
        display = create_display_image(img, crop_info, i + 1, total, is_smart)
        cv2.imshow("Relabeling Tool", display)
        
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord('h') or key == ord('H'):
            label = 'high'
        elif key == ord('m') or key == ord('M'):
            label = 'medium'
        elif key == ord('l') or key == ord('L'):
            label = 'low'
        elif key == ord('s') or key == ord('S'):
            label = 'skip'
        elif key == ord('z') or key == ord('Z'):
            # Undo
            if history:
                last = history.pop()
                labeled_data.remove(last)
                old_label = last['assigned_label']
                if old_label in stats:
                    stats[old_label] -= 1
                i = max(0, i - 1)
                print(f"  ↩ Undo: {Path(last['path']).name}")
            continue
        elif key == ord('q') or key == ord('Q') or key == 27:  # Q or Escape
            print("\n💾 Saving progress...")
            break
        else:
            continue
        
        if label == 'skip':
            stats['skipped'] += 1
            print(f"  ⏭ Skip [{i+1}/{total}]")
        else:
            entry = {
                'path': crop_info['path'],
                'folder_label': crop_info['folder_label'],
                'assigned_label': label,
                'timestamp': datetime.now().isoformat()
            }
            if 'predicted_label' in crop_info:
                entry['predicted_label'] = crop_info['predicted_label']
                entry['prediction_confidence'] = crop_info['prediction_confidence']
            
            labeled_data.append(entry)
            history.append(entry)
            stats[label] += 1
            
            changed = " ⚠ CHANGED!" if label != crop_info['folder_label'] else ""
            print(f"  ✓ {label.upper():6s} [{i+1}/{total}] (was: {crop_info['folder_label']}){changed}")
        
        # Auto-save every 25 labels
        if (i + 1) % 25 == 0:
            save_progress(progress_file, labeled_data, stats)
            print(f"  💾 Auto-saved! (H:{stats['high']} M:{stats['medium']} L:{stats['low']})")
        
        i += 1
    
    cv2.destroyAllWindows()
    
    # Final save
    save_progress(progress_file, labeled_data, stats)
    
    print(f"\n{'='*60}")
    print(f"📊 LABELING SUMMARY")
    print(f"{'='*60}")
    print(f"  Total labeled:  {len(labeled_data)}")
    print(f"  High:           {stats['high']}")
    print(f"  Medium:         {stats['medium']}")
    print(f"  Low:            {stats['low']}")
    print(f"  Skipped:        {stats['skipped']}")
    
    # Count label changes
    changed = sum(1 for item in labeled_data if item['assigned_label'] != item['folder_label'])
    print(f"  Labels changed: {changed} ({changed/max(len(labeled_data),1)*100:.1f}%)")
    print(f"{'='*60}")
    
    return labeled_data, stats


def main():
    parser = argparse.ArgumentParser(
        description='Smart Relabeling Tool for Fine-tuning Dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Smart mode (recommended) - finds suspicious crops first
  python relabel_tool.py --smart
  
  # Random mode
  python relabel_tool.py
  
  # Custom target and output
  python relabel_tool.py --smart --target 1000 --output ../finetune_dataset
        """
    )
    
    parser.add_argument('--crops-root', type=str, 
                        default='person_crops_organized',
                        help='Root directory of person crops (default: person_crops_organized)')
    
    parser.add_argument('--output', type=str, 
                        default='../finetune_dataset',
                        help='Output directory for relabeled crops (default: ../finetune_dataset)')
    
    parser.add_argument('--model', type=str,
                        default='../phase3_training/runs/engagement_organized_full/weights/best.pt',
                        help='Path to trained model (for smart mode)')
    
    parser.add_argument('--smart', action='store_true',
                        help='Use model to find suspicious crops first')
    
    parser.add_argument('--target', type=int, default=600,
                        help='Target number of crops to label (default: 600)')
    
    parser.add_argument('--scan-size', type=int, default=5000,
                        help='Number of crops to scan in smart mode (default: 5000)')
    
    parser.add_argument('--no-copy', action='store_true',
                        help='Skip copying files at the end (just save progress)')
    
    args = parser.parse_args()
    
    crops_root = Path(args.crops_root)
    output_root = Path(args.output)
    progress_file = output_root / 'labeling_progress.json'
    output_root.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("🏷️  SMART RELABELING TOOL")
    print("=" * 60)
    print(f"  Crops root:  {crops_root}")
    print(f"  Output:      {output_root}")
    print(f"  Target:      {args.target} crops")
    print(f"  Mode:        {'🧠 SMART (model-assisted)' if args.smart else '🎲 RANDOM'}")
    print("=" * 60)
    
    # Get all crops
    all_crops = get_all_crops(crops_root)
    print(f"\n📊 Found {len(all_crops):,} total crops")
    
    if not all_crops:
        print("❌ No crops found! Check --crops-root path")
        return
    
    # Prepare crops to label
    if args.smart:
        model = load_model_for_smart_mode(args.model)
        if model is None:
            print("⚠ Falling back to random mode")
            crops_to_label = random.sample(all_crops, min(args.target * 2, len(all_crops)))
        else:
            suspicious, agreeing = find_suspicious_crops(all_crops, model, args.scan_size)
            
            # Mix: suspicious first, then some agreeing ones for balance
            n_suspicious = min(len(suspicious), args.target // 2)
            n_agreeing = args.target - n_suspicious
            agreeing_sample = random.sample(agreeing, min(n_agreeing, len(agreeing)))
            
            crops_to_label = suspicious[:n_suspicious] + agreeing_sample
            print(f"\n🎯 Prepared {len(crops_to_label)} crops to label:")
            print(f"   Suspicious: {n_suspicious} (review these carefully!)")
            print(f"   Agreeing:   {len(agreeing_sample)} (for balance)")
    else:
        # Random sampling - balanced across categories
        by_category = {'high': [], 'medium': [], 'low': []}
        for crop in all_crops:
            by_category[crop['folder_label']].append(crop)
        
        per_category = args.target // 3
        crops_to_label = []
        for cat, crops in by_category.items():
            sample = random.sample(crops, min(per_category, len(crops)))
            crops_to_label.extend(sample)
        
        random.shuffle(crops_to_label)
        print(f"\n🎯 Sampled {len(crops_to_label)} crops (balanced across categories)")
    
    # Run labeling
    print("\n🚀 Starting labeling tool...")
    print("  Window will open shortly. Use keyboard to label.\n")
    
    labeled_data, stats = run_labeling(crops_to_label, output_root, progress_file, args.smart)
    
    # Copy to output
    if labeled_data and not args.no_copy:
        print("\n📦 Copying labeled crops to output directory...")
        copy_labeled_crops(labeled_data, output_root)
        
        print(f"\n✅ DONE! Fine-tune dataset ready at: {output_root}")
        print(f"\n📋 Next step - fine-tune your model:")
        print(f"   cd ../phase3_training")
        print(f"   python train_classifier.py \\")
        print(f"     --data \"{output_root}\" \\")
        print(f"     --model ../phase3_training/runs/engagement_organized_full/weights/best.pt \\")
        print(f"     --epochs 20 \\")
        print(f"     --name engagement_finetuned_v1")
    elif not labeled_data:
        print("\n⚠ No crops were labeled.")
    
    print(f"\n💾 Progress saved to: {progress_file}")
    print("   Run again to resume where you left off!")


if __name__ == "__main__":
    main()
