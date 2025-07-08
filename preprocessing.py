
import cv2
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from skimage.feature import local_binary_pattern
from PIL import Image
import mahotas

# Setting dasar
base_path = 'data_udang'
folders = ['1. Healthy', '2. BG', '3. WSSV', '4. WSSV_BG']
label_map = {'1. Healthy': 0, '2. BG': 1, '3. WSSV': 1, '4. WSSV_BG': 1}

def load_images():
    images = []
    labels = []
    filenames = []
    for folder in folders:
        folder_path = os.path.join(base_path, folder)
        for img_file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_file)
            try:
                img = Image.open(img_path).convert('RGB').resize((128, 128))
                images.append(np.array(img))
                labels.append(label_map[folder])
                filenames.append(img_file)
            except Exception as e:
                print(f"Error loading {img_path}: {str(e)}")
                continue
    return np.array(images), np.array(labels), filenames

def extract_features(images):
    lbp_features = []
    glcm_features = []
    radius = 3
    n_points = 8 * radius
    METHOD = 'uniform'

    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # LBP
        lbp = local_binary_pattern(gray, n_points, radius, METHOD)
        hist, _ = np.histogram(lbp, bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-6)
        lbp_features.append(hist)

        # Haralick (GLCM)
        glcm_props = mahotas.features.haralick(gray).mean(axis=0)
        glcm_features.append(glcm_props)

    return np.hstack([np.array(lbp_features), np.array(glcm_features)])

def load_and_preprocess_data():
    images, labels, _ = load_images()
    features = extract_features(images)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    n_components = min(50, features.shape[1]-1)
    pca = PCA(n_components=n_components, random_state=42)
    features_pca = pca.fit_transform(features_scaled)
    X_train, X_test, y_train, y_test = train_test_split(features_pca, labels, test_size=0.2, stratify=labels, random_state=42)
    return X_train, X_test, y_train, y_test
