"""
Phase 1 POC: Main Runner
Complete pipeline for testing person tracking and engagement scoring
"""

import sys
sys.path.append('..')

import os
from pathlib import Path
import argparse
import time

import config
from detect_track import PersonTracker
from visualizer import visualize_from_tracking_data, create_comparison_video
from utils.logger import setup_logger, ExperimentLogger
from utils.video_utils import get_video_info


def run_poc_single_video(video_path, output_dir, visualize=True, comparison=True):
    """
    Run POC on a single video
    
    Args:
        video_path: Path to video file
        output_dir: Output directory
        visualize: Create visualization video
        comparison: Create comparison video
    
    Returns:
        Results dictionary
    """
    logger = setup_logger("poc_runner")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    video_name = Path(video_path).stem
    
    logger.info("=" * 80)
    logger.info(f"PHASE 1 POC - Processing: {video_name}")
    logger.info("=" * 80)
    
    # Get video info
    video_info = get_video_info(video_path)
    logger.info(f"Video: {video_info['width']}x{video_info['height']}, "
                f"{video_info['fps']:.1f} FPS, {video_info['duration']:.1f}s")
    
    # Initialize tracker
    start_time = time.time()
    
    tracker = PersonTracker(
        detection_model=config.DETECTION_MODEL,
        pose_model=config.POSE_MODEL,
        tracker_config=config.TRACKER_CONFIG,
        conf_threshold=config.CONF_THRESHOLD,
        device=config.TRAINING_CONFIG['device']
    )
    
    # Process video (no visualization yet, just tracking)
    logger.info("\n--- Step 1: Detection & Tracking ---")
    df = tracker.process_video(
        video_path,
        output_path=None,  # Don't save raw video yet
        save_video=False
    )
    
    tracking_time = time.time() - start_time
    logger.info(f"Tracking completed in {tracking_time:.1f}s")
    
    # Save tracking results
    logger.info("\n--- Step 2: Saving Results ---")
    tracker.save_results(output_dir, video_name)
    
    # Get summary
    summary = tracker.get_summary()
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("ENGAGEMENT SUMMARY")
    logger.info("=" * 80)
    
    class_summary = summary['class']
    logger.info(f"\nClass Statistics:")
    logger.info(f"  Students detected: {class_summary['num_students']}")
    logger.info(f"  Total observations: {class_summary['total_observations']}")
    logger.info(f"  Mean engagement: {class_summary['mean_score']:.3f}")
    logger.info(f"  Median engagement: {class_summary['median_score']:.3f}")
    
    logger.info(f"\nEngagement Distribution:")
    for level in ['high', 'medium', 'low']:
        pct = class_summary['level_percentages'][level]
        count = class_summary['level_counts'][level]
        logger.info(f"  {level.upper()}: {pct:.1f}% ({count} observations)")
    
    # Per-student summary
    if summary['students']:
        logger.info(f"\nPer-Student Engagement:")
        for student_id, student_data in sorted(summary['students'].items()):
            logger.info(f"  Student {student_id}: "
                       f"{student_data['dominant_level'].upper()} "
                       f"(score: {student_data['mean_score']:.3f}, "
                       f"n={student_data['total_observations']})")
    
    # Visualization
    if visualize:
        logger.info("\n--- Step 3: Creating Visualization ---")
        vis_start = time.time()
        
        tracking_csv = output_dir / f"{video_name}_tracking.csv"
        output_vis = output_dir / f"{video_name}_visualized.mp4"
        
        visualize_from_tracking_data(
            video_path,
            tracking_csv,
            output_vis
        )
        
        vis_time = time.time() - vis_start
        logger.info(f"Visualization completed in {vis_time:.1f}s")
        logger.info(f"Output saved: {output_vis}")
        
        # Comparison video
        if comparison:
            logger.info("\n--- Step 4: Creating Comparison Video ---")
            comp_start = time.time()
            
            output_comp = output_dir / f"{video_name}_comparison.mp4"
            create_comparison_video(
                video_path,
                tracking_csv,
                output_comp
            )
            
            comp_time = time.time() - comp_start
            logger.info(f"Comparison completed in {comp_time:.1f}s")
            logger.info(f"Output saved: {output_comp}")
    
    total_time = time.time() - start_time
    
    logger.info("\n" + "=" * 80)
    logger.info(f"POC COMPLETE - Total time: {total_time:.1f}s")
    logger.info("=" * 80)
    
    # Return results
    results = {
        'video_name': video_name,
        'video_info': video_info,
        'summary': summary,
        'tracking_time': tracking_time,
        'total_time': total_time,
        'output_dir': str(output_dir)
    }
    
    return results


def run_poc_batch(video_list, output_base_dir, visualize=True):
    """
    Run POC on multiple videos
    
    Args:
        video_list: List of video paths
        output_base_dir: Base output directory
        visualize: Create visualizations
    
    Returns:
        List of results
    """
    logger = setup_logger("poc_batch")
    
    logger.info("=" * 80)
    logger.info(f"PHASE 1 POC - Batch Processing ({len(video_list)} videos)")
    logger.info("=" * 80)
    
    all_results = []
    
    for i, video_path in enumerate(video_list, 1):
        logger.info(f"\n\n{'='*80}")
        logger.info(f"VIDEO {i}/{len(video_list)}: {Path(video_path).name}")
        logger.info(f"{'='*80}\n")
        
        try:
            # Create output directory for this video
            video_name = Path(video_path).stem
            output_dir = Path(output_base_dir) / video_name
            
            # Run POC
            results = run_poc_single_video(
                video_path,
                output_dir,
                visualize=visualize,
                comparison=False  # Skip comparison for batch
            )
            
            all_results.append(results)
            
        except Exception as e:
            logger.error(f"Error processing {video_path}: {e}")
            import traceback
            traceback.print_exc()
    
    # Aggregate summary
    logger.info("\n\n" + "=" * 80)
    logger.info("BATCH PROCESSING SUMMARY")
    logger.info("=" * 80)
    
    logger.info(f"\nTotal videos processed: {len(all_results)}/{len(video_list)}")
    
    if all_results:
        avg_students = sum(r['summary']['class']['num_students'] 
                          for r in all_results) / len(all_results)
        avg_engagement = sum(r['summary']['class']['mean_score'] 
                            for r in all_results) / len(all_results)
        
        logger.info(f"Average students per video: {avg_students:.1f}")
        logger.info(f"Average engagement score: {avg_engagement:.3f}")
        
        # Aggregate engagement levels
        total_high = sum(r['summary']['class']['level_counts']['high'] 
                        for r in all_results)
        total_medium = sum(r['summary']['class']['level_counts']['medium'] 
                          for r in all_results)
        total_low = sum(r['summary']['class']['level_counts']['low'] 
                       for r in all_results)
        total_obs = total_high + total_medium + total_low
        
        logger.info(f"\nOverall Engagement Distribution:")
        logger.info(f"  HIGH: {total_high/total_obs*100:.1f}%")
        logger.info(f"  MEDIUM: {total_medium/total_obs*100:.1f}%")
        logger.info(f"  LOW: {total_low/total_obs*100:.1f}%")
    
    return all_results


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Phase 1 POC: Person Tracking & Engagement Scoring'
    )
    
    parser.add_argument(
        '--video',
        type=str,
        help='Path to single video file'
    )
    
    parser.add_argument(
        '--video-dir',
        type=str,
        help='Directory containing videos'
    )
    
    parser.add_argument(
        '--category',
        type=str,
        choices=['high', 'med', 'low'],
        help='Video category (if using default dataset structure)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory'
    )
    
    parser.add_argument(
        '--no-viz',
        action='store_true',
        help='Skip visualization'
    )
    
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Process multiple videos'
    )
    
    args = parser.parse_args()
    
    # Determine output directory
    if args.output:
        output_dir = args.output
    else:
        output_dir = config.POC_RESULTS_DIR
    
    # Single video
    if args.video:
        run_poc_single_video(
            args.video,
            output_dir,
            visualize=not args.no_viz,
            comparison=True
        )
    
    # Video directory
    elif args.video_dir:
        video_files = []
        for ext in ['.mp4', '.avi', '.mov', '.mkv']:
            video_files.extend(Path(args.video_dir).glob(f'*{ext}'))
        
        if not video_files:
            print(f"No videos found in {args.video_dir}")
            return
        
        run_poc_batch(
            video_files,
            output_dir,
            visualize=not args.no_viz
        )
    
    # Use default dataset
    elif args.category:
        videos = config.get_video_list(args.category)
        
        if not videos:
            print(f"No videos found for category: {args.category}")
            return
        
        video_paths = [v['path'] for v in videos]
        
        run_poc_batch(
            video_paths,
            output_dir,
            visualize=not args.no_viz
        )
    
    else:
        parser.print_help()
        print("\nExamples:")
        print("  # Single video")
        print("  python run_poc.py --video path/to/video.mp4")
        print("\n  # Video directory")
        print("  python run_poc.py --video-dir path/to/videos/")
        print("\n  # Category from dataset")
        print("  python run_poc.py --category high")


if __name__ == "__main__":
    main()