# MultipleFiles/utils.py
import os
import numpy as np
import cv2
from PIL import Image
from skimage.feature import local_binary_pattern
import mahotas

def extract_features_single(img):
    """
    Ekstraksi fitur LBP dan Haralick (GLCM) dari satu gambar.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    radius, n_points, METHOD = 3, 24, 'uniform' 
    lbp = local_binary_pattern(gray, n_points, radius, METHOD)
    hist, _ = np.histogram(lbp, bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6) 

    glcm = mahotas.features.haralick(gray).mean(axis=0)
    
    return np.hstack([hist, glcm])

def load_gallery_images(base_folder, label, sample_size=6):
    """
    Memuat contoh gambar untuk galeri.
    """
    images = []
    folder_path = os.path.join("data_udang", base_folder)
    
    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        return []

    files = os.listdir(folder_path)

    image_files = [f for f in files if f.lower().endswith(('.jpg', '.png', '.jpeg'))][:sample_size]

    for fname in image_files:
        img_path = os.path.join(folder_path, fname)
        try:
            img = Image.open(img_path).convert('RGB').resize((128,128))
            images.append((img, label))
        except Exception as e:
            print(f"Error loading gallery image {img_path}: {e}")
            continue
    return images

