import json
import tensorflow as tf
import numpy as np
from PIL import Image
import os
from pycocotools import mask as mask_utils

def rle_to_mask(segmentation_rle, height, width):
    rle = {
        'size': [height, width],
        'counts': segmentation_rle['counts']
    }
    decoded_mask = mask_utils.decode(rle)
    return decoded_mask

def data_generator(json_file_list, image_dir, batch_size, image_size=(1024, 1024), mask_size=(256, 256), num_points=10):
    while True:
        np.random.shuffle(json_file_list)
        for json_file in json_file_list:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                continue

            annotations = data['annotations']
            image_info = data['image']
            image_path = os.path.join(image_dir, image_info['file_name'])

            if not os.path.exists(image_path):
                continue

            try:
                image = Image.open(image_path).convert('RGB')
            except Exception as e:
                print(f"Warning: Could not open image {image_path}. Skipping. Error: {e}")
                continue

            original_width, original_height = image.size
            image_resized = image.resize(image_size)
            image_array = np.array(image_resized) / 255.0

            width_scale = image_size[0] / original_width
            height_scale = image_size[1] / original_height

            batch_images = []
            batch_points = []
            batch_labels = []
            batch_masks = []

            np.random.shuffle(annotations)

            for ann in annotations:
                gt_mask_original = rle_to_mask(ann['segmentation'], image_info['height'], image_info['width'])
                
                gt_mask_resized = tf.image.resize(gt_mask_original[:, :, np.newaxis], mask_size, method='nearest')
                gt_mask_array = np.array(gt_mask_resized, dtype=np.float32)

                points = []
                labels = []

                if 'point_coords' in ann and ann['point_coords']:
                    pos_point = ann['point_coords'][0]
                    scaled_pos_point = [pos_point[0] * width_scale, pos_point[1] * height_scale]
                    points.append(scaled_pos_point)
                    labels.append(1)

                num_negative_points = num_points - len(points)
                background_coords = np.where(gt_mask_original == 0)
                
                if len(background_coords[0]) > num_negative_points:
                    indices = np.random.choice(len(background_coords[0]), size=num_negative_points, replace=False)
                    
                    for i in indices:
                        neg_y, neg_x = background_coords[0][i], background_coords[1][i]
                        scaled_neg_point = [neg_x * width_scale, neg_y * height_scale]
                        points.append(scaled_neg_point)
                        labels.append(0)

                while len(points) < num_points:
                    points.append([0, 0])
                    labels.append(-1)

                batch_images.append(image_array)
                batch_points.append(np.array(points, dtype=np.float32))
                batch_labels.append(np.array(labels, dtype=np.int32))
                batch_masks.append(gt_mask_array)

                if len(batch_images) == batch_size:
                    yield (
                        {
                            "image_input": np.array(batch_images),
                            "points_input": np.array(batch_points),
                            "labels_input": np.array(batch_labels)
                        },
                        np.array(batch_masks)
                    )
                    batch_images, batch_points, batch_labels, batch_masks = [], [], [], []