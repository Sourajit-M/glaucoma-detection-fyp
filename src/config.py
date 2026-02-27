import os

SEED = 42

IMAGE_SIZE = (224, 224)

DATA_DIR = os.path.join(os.getcwd(), "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")