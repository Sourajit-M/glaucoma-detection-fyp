import cv2
import os
import numpy as np
from tqdm import tqdm
from src.config import IMAGE_SIZE

def preprocess_image(img_path):
    img = cv2.imread(img_path)

    if img is None:
        return None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMAGE_SIZE)

    return img

def save_image(img, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, img)

def preprocess_g1020(raw_root, processed_root):
    for split in ["training", "testing"]:
        for label in ["glaucoma", "normal"]:
            src_dir = os.path.join(raw_root, split, label)
            dst_dir = os.path.join(processed_root, split, label)

            for img_name in tqdm(os.listdir(src_dir), desc=f"G1020 {split} {label}"):
                img_path = os.path.join(src_dir, img_name)
                img = preprocess_image(img_path)

                if img is None:
                    continue

                save_path = os.path.join(dst_dir, img_name)
                save_image(img, save_path)

def preprocess_refuge(raw_root, processed_root):
    for split in ["training", "val", "testing"]:
        src_dir = os.path.join(raw_root, split, "images")
        dst_dir = os.path.join(processed_root, split)

        for img_name in tqdm(os.listdir(src_dir), desc=f"REFUGE {split}"):
            img_path = os.path.join(src_dir, img_name)
            img = preprocess_image(img_path)

            if img is None:
                continue

            save_path = os.path.join(dst_dir, img_name)
            save_image(img, save_path)