import cv2
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib
from skimage.feature import local_binary_pattern
import mahotas
from PIL import Image

# Load dataset
base_path = 'data_udang'
folders = ['1. Healthy', '2. BG', '3. WSSV', '4. WSSV_BG']
label_map = {'1. Healthy': 0, '2. BG': 1, '3. WSSV': 1, '4. WSSV_BG': 1}

def load_images():
    images, labels = [], []
    for folder in folders:
        folder_path = os.path.join(base_path, folder)
        for img_file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_file)
            try:
                img = Image.open(img_path).convert('RGB').resize((128, 128))
                images.append(np.array(img))
                labels.append(label_map[folder])
            except:
                continue
    return np.array(images), np.array(labels)

def extract_features(images):
    lbp_features, glcm_features = [], []
    radius, n_points = 3, 24
    METHOD = 'uniform'
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        lbp = local_binary_pattern(gray, n_points, radius, METHOD)
        hist, _ = np.histogram(lbp, bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-6)
        lbp_features.append(hist)
        glcm = mahotas.features.haralick(gray).mean(axis=0)
        glcm_features.append(glcm)
    return np.hstack([np.array(lbp_features), np.array(glcm_features)])

images, labels = load_images()
features = extract_features(images)

# Preprocessing
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
pca = PCA(n_components=min(50, features.shape[1]-1), random_state=42)
features_pca = pca.fit_transform(features_scaled)

X_train, X_test, y_train, y_test = train_test_split(
    features_pca, labels, test_size=0.2, stratify=labels, random_state=42)

# Train models
knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
knn.fit(X_train, y_train)

svm = SVC(kernel='rbf', C=10, gamma=0.01, probability=True, random_state=42)
svm.fit(X_train, y_train)

# Save models
joblib.dump(knn, "knn_model.pkl")
joblib.dump(svm, "svm_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(pca, "pca.pkl")

# Evaluate
print("KNN accuracy:", accuracy_score(y_test, knn.predict(X_test)))
print("SVM accuracy:", accuracy_score(y_test, svm.predict(X_test)))
print("Training & save models done.")
