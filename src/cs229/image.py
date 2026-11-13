import cv2
import numpy as np

def bound_point(pt, img):
    def clip(dim):
        return np.clip(np.round(pt[dim]), 0, img.shape[1-dim] - 1).astype(int)

    return (clip(0), clip(1))

def img_to_mask(img):
    rows, cols = img.shape[0], img.shape[1]

    # Create a circular mask centred within the image.
    # For rectangular images, use the largest circle that fits.
    r = min((rows - 1) // 2, (cols - 1) // 2)

    center = ((cols - 1) // 2, (rows - 1) // 2)

    mask = np.zeros((rows, cols), np.uint8)
    cv2.circle(mask, center, r, 255, -1)

    return mask