"""
src/cs229/video_pipeline.py

Comprehensive Video Inference, Visualization, and Telemetry Pipeline for FruitFly-ML.

Executes the complete four-stage machine learning pipeline on test videos:
1. Video decoding and chamber masking
2. Contour extraction and FlyCount (Decision Tree)
3. Male / Female sex identification (StandardScaler + LogisticRegression)
4. Male & Female body orientation (HOG + PCA-15 + LogisticRegression)
5. Male left & right wing angle estimation (HOG + PCA-40 + LinearRegression)
6. Rich visual HUD rendering (contours, vectors, text, FPS, timestamps)
7. Frame-accurate CSV trajectory logging
8. Automatic screenshot extraction for key behavioral states
"""

import os
import sys
import time
import csv
from argparse import ArgumentParser
import cv2
import numpy as np

# Ensure cs229 package is accessible
repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
src_dir = os.path.join(repo_root, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from cs229.files import read_video, get_file, get_dir
from cs229.image import img_to_mask, bound_point
from cs229.contour import find_contours
from cs229.patch import crop_to_contour
from cs229.load_data_wing import male_fly_patch
from cs229.train_is_fly import IsFlyPredictor
from cs229.train_id import IdPredictor
from cs229.train_orientation import PosePredictor
from cs229.train_wing import WingPredictor

# Visual Styling Palette (BGR)
COLOR_MALE = (255, 200, 0)       # Vivid Cyan / Azure
COLOR_FEMALE = (180, 50, 255)    # Vivid Magenta / Coral
COLOR_OVERLAP = (0, 165, 255)    # Bright Amber / Orange
COLOR_WING = (0, 255, 255)       # Bright Yellow
COLOR_HEAD = (0, 255, 100)       # Emerald Green
COLOR_TEXT_BG = (20, 20, 20)     # Dark Charcoal
COLOR_WHITE = (255, 255, 255)

def arrow_from_point(img, point, length, angle, color, thickness=4):
    """Draw an arrow from point along angle (radians) in image coordinates."""
    ax = point[0] + length * np.cos(angle)
    ay = point[1] - length * np.sin(angle)
    tip = bound_point((ax, ay), img)
    cv2.arrowedLine(img, (int(point[0]), int(point[1])), tip, color, thickness, tipLength=0.25)
    return tip


def draw_hud(img, frame_num, total_frames, timestamp_s, fps_proc,
             orig_fps, status, fly_count, m_info, f_info):
    """Render a resolution-aware information HUD in the top-left corner."""

    height, width = img.shape[:2]

    # Scale HUD according to input resolution.
    scale = min(width / 1530.0, height / 1530.0)
    scale = max(scale, 0.30)

    margin = max(5, int(40 * scale))
    card_x, card_y = margin, margin

    card_w = min(int(760 * scale), width - 2 * margin)
    card_h = min(int(190 * scale), height - 2 * margin)

    # Use smaller font and spacing for smaller videos.
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.30, 0.9 * scale)
    thickness = max(1, int(2 * scale))

    overlay = img.copy()
    cv2.rectangle(
        overlay,
        (card_x, card_y),
        (card_x + card_w, card_y + card_h),
        COLOR_TEXT_BG, -1
    )
    cv2.rectangle(
        overlay,
        (card_x, card_y),
        (card_x + card_w, card_y + card_h),
        (80, 80, 80), thickness
    )
    cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)

    # Adapt vertical spacing to the available card height.
    line_gap = max(18, int(35 * scale))
    text_x = card_x + max(5, int(20 * scale))
    text_y = card_y + max(15, int(27 * scale))

    def draw_text(text, y, color, size=1.0, weight=None):
        actual_scale = max(0.28, font_scale * size)
        actual_thickness = thickness if weight is None else weight

        # Truncate text if it would exceed the card width.
        max_width = max(20, card_w - (text_x - card_x) - 5)
        while text and cv2.getTextSize(
            text, font, actual_scale, actual_thickness
        )[0][0] > max_width:
            text = text[:-1]

        cv2.putText(
            img, text, (text_x, y), font,
            actual_scale, color, actual_thickness, cv2.LINE_AA
        )

    title = "FruitFly-ML Behavioral Analysis"
    draw_text(title, text_y, COLOR_WHITE, 1.0)

    time_str = f"{int(timestamp_s // 60):02d}:{timestamp_s % 60:05.2f}"
    frame_text = (
        f"Frame: {frame_num:04d}/{total_frames:04d} | "
        f"Time: {time_str} | FPS: {fps_proc:.1f}"
    )
    draw_text(frame_text, text_y + line_gap, (200, 200, 200), 0.75)

    status_color = (
        COLOR_HEAD if "SEPARATED" in status
        else COLOR_OVERLAP if "OVERLAP" in status
        else (150, 150, 150)
    )

    draw_text(
        f"Status: {status} | Flies: {fly_count}",
        text_y + 2 * line_gap,
        status_color, 0.8
    )

    pose_text = "Male/Female Poses: "
    pose_text += "Available" if m_info or f_info else "Unavailable"

    draw_text(
        pose_text,
        text_y + 3 * line_gap,
        (0, 180, 255), 0.75
    )

def process_video(
    video_name="test4.mp4",
    output_dir="outputs/videos",
    csv_dir="outputs/videos",
    screenshot_dir="outputs/screenshots",
    max_frames=None,
    backend="msmf",
    no_display=True
):
    print("=" * 80)
    print(f"RUNNING VIDEO INFERENCE PIPELINE: {video_name} (Backend: {backend})")
    print("=" * 80)

    # 1. Open Video
    cap, props = read_video(video_name, backend=backend)
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  Input Video: {video_name} ({props.width}x{props.height}, {props.fps:.2f} fps, {total_video_frames} declared frames)")

    mask = None

    # 2. Output Video Writer
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(screenshot_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(video_name))[0]
    out_video_path = os.path.join(output_dir, f"{base_name}_annotated.mp4")
    out_csv_path = os.path.join(csv_dir, f"{base_name}_trajectory.csv")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_video_path, fourcc, props.fps, (props.width, props.height))

    # 3. Load Trained Phase 4 Predictors
    print("  Loading trained Phase 4 models from models/...")
    is_fly_pred = IsFlyPredictor()
    id_pred = IdPredictor()
    pose_pred = {
        'male': PosePredictor('male'),
        'female': PosePredictor('female')
    }
    wing_pred = WingPredictor()

    # 4. Prepare CSV Logging
    csv_file = open(out_csv_path, 'w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "frame_number",
        "timestamp_sec",
        "detected_fly_count",
        "male_x",
        "male_y",
        "female_x",
        "female_y",
        "male_orient_deg",
        "female_orient_deg",
        "male_wing_left_deg",
        "male_wing_right_deg",
        "status"
    ])

    # Benchmarking state
    processed_count = 0
    skipped_duplicates = 0
    prev_raw_frame = None
    t_start = time.perf_counter()
    rolling_fps = props.fps
    saved_screenshots = {
        'separated': False,
        'overlapping': False,
        'wings_active': False
    }

    try:
        while True:
            if max_frames and processed_count >= max_frames:
                break

            t_frame_start = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                break

            # Filter duplicate frame artifacts produced by decoder PTS padding / extrapolation
            if prev_raw_frame is not None and np.array_equal(frame, prev_raw_frame):
                skipped_duplicates += 1
                continue
            prev_raw_frame = frame.copy()

            img_gray = frame[:, :, 0]
            if mask is None:
                mask = img_to_mask(img_gray)

            processed_count += 1
            timestamp_s = (processed_count - 1) / props.fps
            out = frame.copy()

            # Step 1: Detect contours
            contours = find_contours(img_gray, mask=mask, type='core')

            # Step 2: Classify contours with FlyCount
            contours_by_label = {'neither': [], 'one': [], 'both': []}
            for c in contours:
                lbl = is_fly_pred.predict(c)
                contours_by_label[lbl].append(c)

            n_one = len(contours_by_label['one'])
            n_both = len(contours_by_label['both'])

            # Defaults for telemetry
            status = "UNKNOWN"
            fly_count = n_one + (n_both * 2)
            male_info = None
            female_info = None

            # Case A: Two separate flies detected
            if n_one == 2 and n_both == 0:
                status = "TWO_FLIES_SEPARATED"
                c1, c2 = contours_by_label['one'][0], contours_by_label['one'][1]
                p1, p2 = crop_to_contour(img_gray, c1), crop_to_contour(img_gray, c2)

                id_label = id_pred.predict(c1, c2, p1, p2)
                if id_label == 'mf':
                    m_contour, m_patch = c1, p1
                    f_contour, f_patch = c2, p2
                else:
                    m_contour, m_patch = c2, p2
                    f_contour, f_patch = c1, p1

                # Male pose
                (m_cx, m_cy), m_angle = pose_pred['male'].predict(m_patch)
                m_deg = float(np.degrees(m_angle))
                male_info = {
                    'cx': float(m_cx), 'cy': float(m_cy),
                    'angle': float(m_angle), 'angle_deg': m_deg,
                    'contour': m_contour, 'patch': m_patch
                }

                # Female pose
                (f_cx, f_cy), f_angle = pose_pred['female'].predict(f_patch)
                f_deg = float(np.degrees(f_angle))
                female_info = {
                    'cx': float(f_cx), 'cy': float(f_cy),
                    'angle': float(f_angle), 'angle_deg': f_deg,
                    'contour': f_contour, 'patch': f_patch
                }

                # Male wing angle estimation
                patch_m_wings = male_fly_patch(img_gray, mask, (m_cx, m_cy), m_angle)
                if patch_m_wings is not None:
                    w_right_rad, w_left_rad = wing_pred.predict(patch_m_wings)
                    if w_right_rad is not None and w_left_rad is not None:
                        w_right_deg = float(np.degrees(w_right_rad))
                        w_left_deg = float(np.degrees(w_left_rad))
                        male_info['wing_right_rad'] = float(w_right_rad)
                        male_info['wing_left_rad'] = float(w_left_rad)
                        male_info['right_deg'] = f"{w_right_deg:.1f}°"
                        male_info['left_deg'] = f"{w_left_deg:.1f}°"
                        male_info['w_r_deg_num'] = w_right_deg
                        male_info['w_l_deg_num'] = w_left_deg

                # Draw Male Visuals
                cv2.drawContours(out, [m_contour], -1, COLOR_MALE, 3)
                m_center = (int(m_cx), int(m_cy))
                cv2.circle(out, m_center, 6, COLOR_MALE, -1)
                MA_m, _ = m_patch.estimate_axes()
                tip_m = arrow_from_point(out, m_center, 0.35 * MA_m, m_angle, COLOR_MALE, thickness=4)
                cv2.circle(out, (int(tip_m[0]), int(tip_m[1])), 5, COLOR_HEAD, -1) # Head landmark

                # Abdomen landmark
                ab_x = m_cx - 0.30 * MA_m * np.cos(m_angle)
                ab_y = m_cy + 0.30 * MA_m * np.sin(m_angle)
                cv2.circle(out, (int(ab_x), int(ab_y)), 5, COLOR_MALE, -1)

                cv2.putText(out, f"MALE ({m_deg:.1f}deg)", (int(m_cx) - 60, int(m_cy) - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_MALE, 2, cv2.LINE_AA)

                # Draw Wing Vectors
                if 'wing_right_rad' in male_info:
                    ang_r = m_angle + male_info['wing_right_rad'] - np.pi
                    ang_l = m_angle - male_info['wing_left_rad'] - np.pi
                    r_tip = arrow_from_point(out, m_center, 0.40 * MA_m, ang_r, COLOR_WING, thickness=3)
                    l_tip = arrow_from_point(out, m_center, 0.40 * MA_m, ang_l, COLOR_WING, thickness=3)
                    cv2.putText(out, f"R:{male_info['w_r_deg_num']:.0f}deg", (int(r_tip[0]) + 5, int(r_tip[1])),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WING, 1, cv2.LINE_AA)
                    cv2.putText(out, f"L:{male_info['w_l_deg_num']:.0f}deg", (int(l_tip[0]) - 45, int(l_tip[1])),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WING, 1, cv2.LINE_AA)

                # Draw Female Visuals
                cv2.drawContours(out, [f_contour], -1, COLOR_FEMALE, 3)
                f_center = (int(f_cx), int(f_cy))
                cv2.circle(out, f_center, 6, COLOR_FEMALE, -1)
                MA_f, _ = f_patch.estimate_axes()
                tip_f = arrow_from_point(out, f_center, 0.35 * MA_f, f_angle, COLOR_FEMALE, thickness=4)
                cv2.circle(out, (int(tip_f[0]), int(tip_f[1])), 5, COLOR_HEAD, -1) # Head landmark

                ab_fx = f_cx - 0.30 * MA_f * np.cos(f_angle)
                ab_fy = f_cy + 0.30 * MA_f * np.sin(f_angle)
                cv2.circle(out, (int(ab_fx), int(ab_fy)), 5, COLOR_FEMALE, -1)

                cv2.putText(out, f"FEMALE ({f_deg:.1f}deg)", (int(f_cx) - 60, int(f_cy) - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_FEMALE, 2, cv2.LINE_AA)

            # Case B: Overlapping / Touching flies
            elif n_both >= 1 or (n_one == 0 and len(contours) > 0 and n_both > 0):
                status = "OVERLAPPING_FLIES"
                for bc in contours_by_label['both']:
                    cv2.drawContours(out, [bc], -1, COLOR_OVERLAP, 4)
                    patch_b = crop_to_contour(img_gray, bc)
                    cx_b, cy_b = patch_b.estimate_center(absolute=True)
                    cv2.circle(out, (int(cx_b), int(cy_b)), 7, COLOR_OVERLAP, -1)
                    cv2.putText(out, "CONTACT / OVERLAP", (int(cx_b) - 100, int(cy_b) - 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_OVERLAP, 2, cv2.LINE_AA)

            # Case C: Single fly isolated
            elif n_one == 1:
                status = "SINGLE_FLY_DETECTED"
                c1 = contours_by_label['one'][0]
                cv2.drawContours(out, [c1], -1, (200, 200, 200), 2)
                p1 = crop_to_contour(img_gray, c1)
                cx1, cy1 = p1.estimate_center(absolute=True)
                cv2.putText(out, "FLY DETECTED", (int(cx1) - 60, int(cy1) - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

            elif n_one == 0 and n_both == 0:
                status = "NO_FLIES_DETECTED"
            else:
                status = f"UNEXPECTED_CONTOURS(one={n_one},both={n_both})"

            # Calculate rolling processing FPS
            t_frame = time.perf_counter() - t_frame_start
            inst_fps = 1.0 / max(t_frame, 1e-5)
            rolling_fps = 0.9 * rolling_fps + 0.1 * inst_fps

            # Render HUD
            draw_hud(out, processed_count, total_video_frames, timestamp_s, rolling_fps, props.fps,
                     status, fly_count, male_info, female_info)

            # Write Video Frame
            writer.write(out)

            # Write CSV row (empty strings for missing values)
            m_x_val = f"{male_info['cx']:.1f}" if male_info else ""
            m_y_val = f"{male_info['cy']:.1f}" if male_info else ""
            f_x_val = f"{female_info['cx']:.1f}" if female_info else ""
            f_y_val = f"{female_info['cy']:.1f}" if female_info else ""
            m_ang_val = f"{male_info['angle_deg']:.2f}" if male_info else ""
            f_ang_val = f"{female_info['angle_deg']:.2f}" if female_info else ""
            l_w_val = f"{male_info['w_l_deg_num']:.2f}" if (male_info and 'w_l_deg_num' in male_info) else ""
            r_w_val = f"{male_info['w_r_deg_num']:.2f}" if (male_info and 'w_r_deg_num' in male_info) else ""

            csv_writer.writerow([
                processed_count,
                f"{timestamp_s:.3f}",
                fly_count,
                m_x_val, m_y_val,
                f_x_val, f_y_val,
                m_ang_val, f_ang_val,
                l_w_val, r_w_val,
                status
            ])

            # Save representative screenshots
            if not saved_screenshots['separated'] and status == "TWO_FLIES_SEPARATED":
                sc_path = os.path.join(screenshot_dir, "screenshot_two_flies_separated.jpg")
                cv2.imwrite(sc_path, out)
                saved_screenshots['separated'] = True

            if not saved_screenshots['overlapping'] and status == "OVERLAPPING_FLIES":
                sc_path = os.path.join(screenshot_dir, "screenshot_overlapping_flies.jpg")
                cv2.imwrite(sc_path, out)
                saved_screenshots['overlapping'] = True

            if not saved_screenshots['wings_active'] and male_info and 'wing_right_rad' in male_info:
                # Pick a frame with significant wing display if possible
                if male_info.get('w_r_deg_num', 0) > 15.0 or male_info.get('w_l_deg_num', 0) > 15.0:
                    sc_path = os.path.join(screenshot_dir, "screenshot_wing_angles_active.jpg")
                    cv2.imwrite(sc_path, out)
                    saved_screenshots['wings_active'] = True

            if processed_count % 100 == 0 or processed_count == total_video_frames:
                print(f"  Processed {processed_count:4d}/{total_video_frames} frames | Rolling FPS: {rolling_fps:.1f}")

    finally:
        cap.release()
        writer.release()
        csv_file.close()

    total_time = time.perf_counter() - t_start
    avg_fps = processed_count / max(total_time, 1e-5)
    avg_latency_ms = (total_time / max(processed_count, 1)) * 1000.0
    is_realtime = avg_fps >= props.fps

    video_size_mb = os.path.getsize(out_video_path) / (1024 * 1024)
    csv_size_kb = os.path.getsize(out_csv_path) / 1024

    print("\n" + "-" * 60)
    print(f"BENCHMARK RESULTS FOR {video_name} (Backend: {backend}):")
    print(f"  Total Frames Processed:    {processed_count} (Skipped {skipped_duplicates} duplicate frames)")
    print(f"  Total Processing Time:     {total_time:.2f} seconds")
    print(f"  Average Processing FPS:    {avg_fps:.2f} FPS")
    print(f"  Average Frame Latency:     {avg_latency_ms:.2f} ms/frame")
    print(f"  Original Video FPS:        {props.fps:.2f} FPS")
    print(f"  Real-Time Status:          {'FASTER THAN REAL-TIME' if is_realtime else 'SLOWER THAN REAL-TIME'} ({avg_fps / props.fps:.2f}x source speed)")
    print(f"  Output Video:              {out_video_path} ({video_size_mb:.2f} MB)")
    print(f"  Output CSV:                {out_csv_path} ({csv_size_kb:.1f} KB)")
    print("-" * 60)

    return {
        'video_name': video_name,
        'backend': backend,
        'frames': processed_count,
        'skipped_duplicates': skipped_duplicates,
        'total_time_s': total_time,
        'avg_fps': avg_fps,
        'latency_ms': avg_latency_ms,
        'source_fps': props.fps,
        'real_time_ratio': avg_fps / props.fps,
        'is_realtime': is_realtime,
        'video_path': out_video_path,
        'video_size_mb': video_size_mb,
        'csv_path': out_csv_path,
        'csv_size_kb': csv_size_kb
    }

def main():
    parser = ArgumentParser(description="FruitFly-ML Complete Video Inference Pipeline")
    parser.add_argument('-i', '--input', type=str, default='test4.mp4', help="Video filename or path")
    parser.add_argument('--all', action='store_true', help="Process all five test videos")
    parser.add_argument('--max_frames', type=int, default=None, help="Limit frames for testing")
    parser.add_argument('--backend', type=str, default='msmf', choices=['msmf', 'ffmpeg', 'auto'], help="Video capture backend")
    args = parser.parse_args()

    if args.all:
        videos = ['test1.mp4', 'test2.mp4', 'test3.mp4', 'test4.mp4', 'test5.mp4']
        results = []
        for v in videos:
            res = process_video(v, max_frames=args.max_frames, backend=args.backend)
            results.append(res)
    else:
        process_video(args.input, max_frames=args.max_frames, backend=args.backend)

if __name__ == '__main__':
    main()

