import os
import cv2
from cs229.annotation import get_annotations, Annotation
from cs229.files import get_dir

def verify_all_annotations():
    images_dir = get_dir('input', 'images')
    print(f"Images Directory: {images_dir}")
    
    subfolders = [f.path for f in os.scandir(images_dir) if f.is_dir()]
    print(f"Subfolders Found ({len(subfolders)}):")
    for s in sorted(subfolders):
        print(f"  - {os.path.basename(s)}")

    all_annos = list(get_annotations())
    print(f"\nTotal JSON files parsed: {len(all_annos)}")
    
    loaded_images = 0
    missing_images = []
    invalid_annos = []
    
    for i, anno in enumerate(all_annos):
        # Check image existence
        if not os.path.exists(anno.image_path):
            missing_images.append((anno.json_path, anno.image_path))
            continue
            
        # Test loading image with OpenCV
        img = cv2.imread(anno.image_path, 0)
        if img is None:
            missing_images.append((anno.json_path, anno.image_path))
            continue
            
        if img.shape != (1530, 1530):
            invalid_annos.append((anno.json_path, f"Unexpected shape {img.shape}"))
            continue
            
        loaded_images += 1

    print(f"Successfully loaded and verified images: {loaded_images} / {len(all_annos)}")
    print(f"Missing images: {len(missing_images)}")
    if missing_images:
        for jp, ip in missing_images[:5]:
            print(f"  JSON: {jp} -> Missing: {ip}")
            
    print(f"Invalid annotations: {len(invalid_annos)}")
    if invalid_annos:
        for jp, reason in invalid_annos[:5]:
            print(f"  JSON: {jp} -> Reason: {reason}")
            
    return len(all_annos), loaded_images, len(missing_images)

if __name__ == '__main__':
    verify_all_annotations()

