import os
import cv2
import numpy as np

def inspect_video_msmf(vname):
    path = os.path.join("input", "video", vname)
    if not os.path.exists(path):
        path = os.path.join("input", "videos", vname)
    
    cap = cv2.VideoCapture(path, cv2.CAP_MSMF)
    fps = cap.get(cv2.CAP_PROP_FPS)
    header_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    frames_info = []
    prev_frame = None
    duplicates = []
    
    idx = 0
    while True:
        pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
        pos_frames = cap.get(cv2.CAP_PROP_POS_FRAMES)
        ok, frame = cap.read()
        if not ok:
            break
        
        # Check duplicate
        is_dup = False
        max_diff = 0
        if prev_frame is not None:
            diff = np.abs(frame.astype(np.int16) - prev_frame.astype(np.int16))
            max_diff = int(np.max(diff))
            if max_diff == 0:
                is_dup = True
                duplicates.append(idx)
        
        # Mean intensity to verify non-empty
        mean_val = float(np.mean(frame))
        
        frames_info.append({
            'idx': idx,
            'pos_msec': pos_msec,
            'pos_frames': pos_frames,
            'is_dup': is_dup,
            'max_diff_from_prev': max_diff,
            'mean_val': mean_val
        })
        
        prev_frame = frame.copy()
        idx += 1
        
    cap.release()
    
    print(f"==================================================", flush=True)
    print(f"VIDEO: {vname}", flush=True)
    print(f"  Dimensions: {width}x{height}, FPS: {fps:.2f}", flush=True)
    print(f"  Header Frame Count: {header_count}", flush=True)
    print(f"  Total MSMF Decoded: {idx}", flush=True)
    print(f"  Duplicate Frames Count: {len(duplicates)} (indices: {duplicates})", flush=True)
    
    if len(frames_info) > 0:
        first_f = frames_info[0]
        last_f = frames_info[-1]
        print(f"  First Frame pos_msec: {first_f['pos_msec']:.2f}, pos_frames: {first_f['pos_frames']}", flush=True)
        print(f"  Last Frame pos_msec:  {last_f['pos_msec']:.2f}, pos_frames: {last_f['pos_frames']}", flush=True)
        print(f"  Duration from last msec: {last_f['pos_msec'] / 1000.0:.3f} s", flush=True)
        print(f"  Duration from count/fps: {idx / fps:.3f} s", flush=True)
        
        # Inspect tail frames (last 6 frames)
        print(f"  Tail Frames Details (last 6 frames):", flush=True)
        for fi in frames_info[-6:]:
            print(f"    Frame {fi['idx']}: pos_msec={fi['pos_msec']:.2f}, pos_frames={fi['pos_frames']}, "
                  f"max_diff_from_prev={fi['max_diff_from_prev']}, is_dup={fi['is_dup']}, mean={fi['mean_val']:.2f}", flush=True)

if __name__ == "__main__":
    for v in ["test1.mp4", "test2.mp4", "test3.mp4", "test4.mp4", "test5.mp4"]:
        inspect_video_msmf(v)

