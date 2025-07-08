
import numpy as np
import pandas as pd
import os
from PIL import Image
import cv2
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
import joblib
import matplotlib.pyplot as plt
from utils import extract_features_single

# Load images & labels
images = []
labels = []
for folder, label_val in [("1. Healthy", 0), ("2. BG", 1), ("3. WSSV", 1), ("4. WSSV_BG", 1)]:
    path = os.path.join("data_udang", folder)
    for fname in os.listdir(path):
        img_path = os.path.join(path, fname)
        try:
            img = Image.open(img_path).convert('RGB').resize((128,128))
            images.append(np.array(img))
            labels.append(label_val)
        except:
            continue
images = np.array(images)
labels = np.array(labels)

# Extract features
features = np.array([extract_features_single(img) for img in images])

# Scale + PCA
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
pca = PCA(n_components=10)
features_pca = pca.fit_transform(features_scaled)

# Scatter plot
plt.figure(figsize=(8,6))
for i in range(len(features_pca)):
    if labels[i] == 0:
        plt.scatter(features_pca[i,0], features_pca[i,1], color='blue', label='Healthy' if i==0 else "")
    else:
        plt.scatter(features_pca[i,0], features_pca[i,1], color='red', label='Diseased' if i==len(labels)//2 else "")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("PCA Feature Space of Shrimp Dataset")
plt.legend()
plt.grid(True)
plt.savefig("pca_clusters.png")
plt.close()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features_pca, labels, test_size=0.2, random_state=42)

# Train models with calibration
knn = KNeighborsClassifier(n_neighbors=3)
knn_calib = CalibratedClassifierCV(knn, cv=3)
knn_calib.fit(X_train, y_train)

svm = SVC(kernel='rbf', probability=True)
svm_calib = CalibratedClassifierCV(svm, cv=3)
svm_calib.fit(X_train, y_train)

# Save models
joblib.dump(knn_calib, "knn_model.pkl")
joblib.dump(svm_calib, "svm_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(pca, "pca.pkl")

print("Models trained and saved with calibrated probabilities.")
print("PCA scatter plot saved as pca_clusters.png.")
