import cv2
from glob import glob
import os
from tqdm import tqdm

original_images = glob("data\\val\\*.jpg")
edited_images = glob("results\\result_*.jpg")

for paths in tqdm(zip(original_images, edited_images), desc="Generating edit maps", total=len(original_images)):
    original_image = cv2.imread(paths[0])
    edited_image = cv2.imread(paths[1])
    if original_image is None or edited_image is None:
        print(f"Warning: Could not load image(s): {paths[0]}, {paths[1]}")
        continue
    if original_image.shape != edited_image.shape:
        print(f"Warning: Image shapes do not match: {paths[0]}, {paths[1]}")
        continue
    original_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    edited_gray = cv2.cvtColor(edited_image, cv2.COLOR_BGR2GRAY)
    diff_gray = cv2.absdiff(original_gray, edited_gray)
    _, mask = cv2.threshold(diff_gray, 30, 255, cv2.THRESH_BINARY)
    cv2.imwrite(f"masks\\mask_{os.path.basename(paths[0])}", mask)