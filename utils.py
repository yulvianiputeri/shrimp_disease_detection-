
import os
import numpy as np
import cv2
from PIL import Image
from skimage.feature import local_binary_pattern
import mahotas

def extract_features_single(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    radius, n_points, METHOD = 3, 24, 'uniform'
    lbp = local_binary_pattern(gray, n_points, radius, METHOD)
    hist, _ = np.histogram(lbp, bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    glcm = mahotas.features.haralick(gray).mean(axis=0)
    return np.hstack([hist, glcm])

def load_gallery_images(base_folder, label, sample_size=6):
    images = []
    folder_path = os.path.join("data_udang", base_folder)
    files = os.listdir(folder_path)[:sample_size]
    for fname in files:
        img_path = os.path.join(folder_path, fname)
        try:
            img = Image.open(img_path).convert('RGB').resize((128,128))
            images.append((img, label))
        except:
            continue
    return images
