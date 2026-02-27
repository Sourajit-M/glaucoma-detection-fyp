import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.feature import local_binary_pattern



def compute_cdr_features(mask):
    """
    Compute disc area, cup area, and CDR from segmentation mask.
    REFUGE convention:
        Disc = 255
        Cup = 128
    """
    disc = (mask == 255).astype(np.uint8)
    cup = (mask == 128).astype(np.uint8)

    disc_area = np.sum(disc)
    cup_area = np.sum(cup)

    cdr = cup_area / (disc_area + 1e-6)

    return disc_area, cup_area, cdr


def compute_color_features(img):
    """
    Compute mean RGB intensities.
    """
    mean_r = img[:, :, 0].mean()
    mean_g = img[:, :, 1].mean()
    mean_b = img[:, :, 2].mean()

    return mean_r, mean_g, mean_b


def compute_lbp_feature(img, radius=2, points=8):
    """
    Compute mean Local Binary Pattern value.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    lbp = local_binary_pattern(gray, points, radius, method="uniform")
    return lbp.mean()



def load_image_and_mask(processed_root, raw_root, split, img_name):
    img_path = os.path.join(processed_root, split, img_name)
    mask_dir = os.path.join(raw_root, split, "mask")

    base_name = os.path.splitext(img_name)[0]

    possible_masks = [
        base_name + ".png",
        base_name + ".jpg",
        base_name + "_mask.png",
        base_name + "_mask.jpg"
    ]

    mask_path = None
    for m in possible_masks:
        candidate = os.path.join(mask_dir, m)
        if os.path.exists(candidate):
            mask_path = candidate
            break

    if mask_path is None:
        return None, None

    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if img is None or mask is None:
        return None, None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img, mask



def extract_features_refuge(processed_root, raw_root):
    """
    Extract engineered features from REFUGE dataset.
    Returns DataFrame.
    """
    records = []

    for split in ["training", "val", "testing"]:
        img_dir = os.path.join(processed_root, split)

        for img_name in tqdm(os.listdir(img_dir), desc=f"Extracting {split}"):
            img, mask = load_image_and_mask(
                processed_root, raw_root, split, img_name
            )

            if img is None:
                continue

            disc_area, cup_area, cdr = compute_cdr_features(mask)
            mean_r, mean_g, mean_b = compute_color_features(img)
            lbp_mean = compute_lbp_feature(img)

            # Temporary label inference
            label = 1 if "g" in img_name.lower() else 0

            records.append({
                "image": img_name,
                "split": split,
                "disc_area": disc_area,
                "cup_area": cup_area,
                "cdr": cdr,
                "mean_r": mean_r,
                "mean_g": mean_g,
                "mean_b": mean_b,
                "lbp_mean": lbp_mean,
                "label": label
            })

    return pd.DataFrame(records)