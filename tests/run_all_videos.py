"""
tests/run_all_videos.py

Runs the complete video inference pipeline on all five test videos:
- test1.mp4
- test2.mp4
- test3.mp4
- test4.mp4 (already processed, will load cached or re-run if needed)
- test5.mp4

Collects full benchmark metrics and outputs summary JSON to outputs/data/video_benchmarks.json.
"""

import os
import sys
import json
import time

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(repo_root, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from cs229.video_pipeline import process_video

def main():
    videos = ['test1.mp4', 'test2.mp4', 'test3.mp4', 'test4.mp4', 'test5.mp4']
    all_benchmarks = {}

    benchmarks_json_path = os.path.join(repo_root, 'outputs', 'data', 'video_benchmarks.json')

    for v in videos:
        print(f"\nProcessing {v} using MSMF backend with deduplication...")
        res = process_video(
            video_name=v,
            output_dir=os.path.join(repo_root, 'outputs', 'videos'),
            csv_dir=os.path.join(repo_root, 'outputs', 'videos'),
            screenshot_dir=os.path.join(repo_root, 'outputs', 'screenshots'),
            backend='msmf'
        )
        all_benchmarks[v] = res

    # Save benchmark summary
    with open(benchmarks_json_path, 'w', encoding='utf-8') as f:
        json.dump(all_benchmarks, f, indent=2)

    print("\n" + "=" * 80)
    print("ALL FIVE VIDEOS PROCESSED SUCCESSFULLY!")
    print("=" * 80)
    for v, b in all_benchmarks.items():
        print(f"  {v:10s}: {b['frames']} frames, {b['avg_fps']:.2f} FPS ({b['latency_ms']:.1f} ms), size: {b['video_size_mb']:.2f} MB, CSV: {b['csv_size_kb']:.1f} KB")

if __name__ == '__main__':
    main()

