import os
import cv2
import albumentations as A
from tqdm import tqdm

# Define augmentation pipeline
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Rotate(limit=15, p=0.5),
    A.GaussNoise(p=0.2),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
    A.RandomSizedCrop(min_max_height=(256, 512), height=640, width=640, p=0.5),
    A.CLAHE(p=0.2)
], bbox_params=A.BboxParams(format='yolo', min_visibility=0.2, label_fields=['class_labels']))

# Define paths
input_image_dir = './dataset/images/val'
input_label_dir = './dataset/labels/val'
output_image_dir = './output/images/val'
output_label_dir = './output/labels/val'
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_label_dir, exist_ok=True)

# Load images
image_files = [f for f in os.listdir(input_image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

# Function to read label file
def read_label_file(label_path):
    bboxes = []
    class_labels = []
    with open(label_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            class_labels.append(int(parts[0]))
            bbox = list(map(float, parts[1:]))
            bboxes.append(bbox)
    return bboxes, class_labels

# Function to write label file
def write_label_file(label_path, bboxes, class_labels):
    with open(label_path, 'w') as file:
        for bbox, label in zip(bboxes, class_labels):
            file.write(f"{label} " + " ".join(map(str, bbox)) + "\n")

# Augment and save images and labels
for i, image_file in enumerate(tqdm(image_files)):
    image_path = os.path.join(input_image_dir, image_file)
    label_path = os.path.join(input_label_dir, os.path.splitext(image_file)[0] + '.txt')

    image = cv2.imread(image_path)
    bboxes, class_labels = read_label_file(label_path)

    for j in range(10):  # Create 10 augmentations per image
        augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
        augmented_image = augmented['image']
        augmented_bboxes = augmented['bboxes']
        augmented_class_labels = augmented['class_labels']

        # Save augmented image
        output_image_path = os.path.join(output_image_dir, f'aug_{i}_{j}.jpg')
        cv2.imwrite(output_image_path, augmented_image)

        # Save augmented labels
        output_label_path = os.path.join(output_label_dir, f'aug_{i}_{j}.txt')
        write_label_file(output_label_path, augmented_bboxes, augmented_class_labels)

print("Data augmentation completed.")