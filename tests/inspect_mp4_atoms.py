"""
tests/inspect_mp4_atoms.py

Inspects binary MP4 box atoms (mvhd, mdhd, stts, stsz, stco) to extract exact
metadata directly from the container:
- timescale, duration
- total sample count (frames) declared in stsz (sample size box)
- total sample count declared in stts (time to sample box)
- compare against OpenCV CAP_PROP_FRAME_COUNT and actual cv2.read() decodable frames.
"""

import os
import struct
import cv2

def parse_mp4_boxes(filepath):
    with open(filepath, 'rb') as f:
        data = f.read()

    # Search for 'mvhd' (movie header)
    idx_mvhd = data.find(b'mvhd')
    mvhd_info = {}
    if idx_mvhd != -1:
        ver = data[idx_mvhd + 4]
        if ver == 0:
            timescale, duration = struct.unpack('>II', data[idx_mvhd + 16 : idx_mvhd + 24])
        else:
            timescale = struct.unpack('>I', data[idx_mvhd + 24 : idx_mvhd + 28])[0]
            duration = struct.unpack('>Q', data[idx_mvhd + 28 : idx_mvhd + 36])[0]
        mvhd_info = {
            'timescale': timescale,
            'duration': duration,
            'duration_s': duration / timescale
        }

    # Search for 'stts' (time to sample)
    idx_stts = data.find(b'stts')
    stts_info = {}
    if idx_stts != -1:
        entry_count = struct.unpack('>I', data[idx_stts + 8 : idx_stts + 12])[0]
        entries = []
        for i in range(entry_count):
            start = idx_stts + 12 + i * 8
            scount, sdelta = struct.unpack('>II', data[start : start + 8])
            entries.append((scount, sdelta))
        stts_info = {
            'entry_count': entry_count,
            'total_samples': sum(e[0] for e in entries),
            'entries': entries
        }

    # Search for 'stsz' (sample size box)
    idx_stsz = data.find(b'stsz')
    stsz_info = {}
    if idx_stsz != -1:
        def_size, sample_count = struct.unpack('>II', data[idx_stsz + 8 : idx_stsz + 16])
        stsz_info = {
            'default_size': def_size,
            'sample_count': sample_count
        }

    return {
        'filesize_bytes': len(data),
        'mvhd': mvhd_info,
        'stts': stts_info,
        'stsz': stsz_info
    }

def main():
    print("=" * 80)
    print("MP4 CONTAINER ATOM BOX INSPECTION (INDEPENDENT OF OPENCV)")
    print("=" * 80)

    video_files = ['test1.mp4', 'test2.mp4', 'test3.mp4', 'test4.mp4', 'test5.mp4']
    for v in video_files:
        src_path = os.path.join('input', 'video', v)
        boxes = parse_mp4_boxes(src_path)

        cap = cv2.VideoCapture(src_path)
        cv_fps = cap.get(cv2.CAP_PROP_FPS)
        cv_header_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        actual_decoded = 0
        decode_errors = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame is None or frame.size == 0:
                decode_errors += 1
            else:
                actual_decoded += 1
        cap.release()

        print(f"\nVideo: {v} ({boxes['filesize_bytes']:,} bytes)")
        print(f"  Container mvhd duration:      {boxes['mvhd'].get('duration_s', 0):.3f}s (timescale={boxes['mvhd'].get('timescale', 0)})")
        print(f"  Container stsz sample count:  {boxes['stsz'].get('sample_count', 0)} samples")
        print(f"  Container stts sample count:  {boxes['stts'].get('total_samples', 0)} samples")
        print(f"  OpenCV CAP_PROP_FPS:          {cv_fps:.2f} fps")
        print(f"  OpenCV CAP_PROP_FRAME_COUNT:  {cv_header_count:.0f} frames")
        print(f"  OpenCV actual decoded frames: {actual_decoded} frames")
        print(f"  Corrupted/Empty frames read:  {decode_errors}")
        print(f"  Difference (stsz - decoded):  {boxes['stsz'].get('sample_count', 0) - actual_decoded} frames")

if __name__ == '__main__':
    main()

