"""
tests/diagnose_wings.py

Detailed diagnosis of candidate wing frames and the exact 21 excluded frames.
"""

import os
import sys
from collections import Counter
from math import degrees
import cv2
import numpy as np

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(repo_root, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from cs229.annotation import get_annotations
from cs229.image import img_to_mask, bound_point
from cs229.contour import contour_label, find_contours, in_contour
from cs229.patch import crop_to_contour, ImagePatch, mask_from_contour
from cs229.util import angle_diff

tol_radians = 0.1
crop_size = 400

def check_male_fly_patch(img, mask, body_center, rotate_angle):
    body_center = bound_point((body_center[0], body_center[1]), img)
    fly_patch = ImagePatch(img, mask)
    fly_patch = fly_patch.recenter(crop_size, crop_size, body_center)
    contours = find_contours(fly_patch.img, fly_patch.mask, type='wings')
    fly_center = (crop_size // 2, crop_size // 2)
    for contour in sorted(contours, key=lambda x: len(x)):
        if in_contour(fly_center, contour):
            fly_mask = mask_from_contour(fly_patch.img, contour)
            fly_patch.mask = cv2.bitwise_and(fly_patch.mask, fly_mask)
            fly_patch.img = cv2.bitwise_and(fly_patch.img, fly_patch.img, mask=fly_patch.mask)
            return True
    return False

def main():
    annos = list(get_annotations())
    print(f"Total annotations in dataset: {len(annos)}")

    # All candidate frames with required landmarks: count('mw') == 2, mh, ma, mp2
    candidates = []
    missing_landmarks = []
    for a in annos:
        if a.count('mw') == 2 and a.has('mh') and a.has('ma') and a.has('mp2'):
            candidates.append(a)
        else:
            missing_landmarks.append(a)

    print(f"Candidate frames with 2 mw, mh, ma, mp2: {len(candidates)}")
    print(f"Frames missing required wing landmarks: {len(missing_landmarks)}")

    excluded = []
    passed = []

    for anno in candidates:
        img = cv2.imread(anno.image_path, 0)
        mask = img_to_mask(img)
        contours = find_contours(img, mask=mask, type='core')

        male_contour = None
        for contour in contours:
            ctype = contour_label(anno, contour)
            if ctype == 'male':
                male_contour = contour
                break

        if male_contour is None:
            is_both = any(contour_label(anno, c) == 'both' for c in contours)
            reason = "Touching flies / merged in single 'both' contour (no isolated male)" if is_both else "No male contour detected"
            excluded.append((anno.image_path, reason))
            continue

        mh = anno.get('mh')[0]
        ma = anno.get('ma')[0]
        mp2 = anno.get('mp2')[0]

        body_patch = crop_to_contour(img, male_contour)
        label_angle = np.arctan2(ma[1] - mh[1], mh[0] - ma[0])
        rotate_angle = body_patch.estimate_angle()
        diff = abs(angle_diff(rotate_angle, label_angle))

        if not (diff <= tol_radians or np.pi - tol_radians <= diff <= np.pi + tol_radians):
            reason = f"Angle ambiguity: diff={degrees(diff):.1f}° (exceeds tolerance {degrees(tol_radians):.1f}°)"
            excluded.append((anno.image_path, reason))
            continue
        elif np.pi - tol_radians <= diff <= np.pi + tol_radians:
            rotate_angle = rotate_angle + np.pi

        body_center = body_patch.estimate_center(absolute=True)
        patch_ok = check_male_fly_patch(img, mask, body_center, rotate_angle)
        if not patch_ok:
            reason = "Patch extraction failure (no wings contour contains fly center)"
            excluded.append((anno.image_path, reason))
            continue

        passed.append(anno.image_path)

    print(f"\nPassed frames: {len(passed)} -> Generates {len(passed) * 2} wing samples")
    print(f"Excluded frames from 257 candidates: {len(excluded)} -> {len(excluded) * 2} excluded samples")

    print("\nSummary of Exclusion Reasons:")
    reasons_summary = Counter([r.split(':')[0] for _, r in excluded])
    for r, count in reasons_summary.items():
        print(f"  - {r}: {count} frames ({count * 2} samples)")

    print("\nExact 21 Excluded Frames:")
    for i, (path, reason) in enumerate(excluded, 1):
        filename = os.path.basename(path)
        folder = os.path.basename(os.path.dirname(path))
        print(f"  {i:2d}. {folder}/{filename} : {reason}")

if __name__ == '__main__':
    main()

