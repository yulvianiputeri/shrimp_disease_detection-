import numpy as np
import pandas as pd
import os
from PIL import Image
import cv2
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing import load_and_preprocess_data
from model import bandingkan_model
from utils import extract_features_single

print("Starting model training process...")

X_train, X_test, y_train, y_test, scaler, pca = load_and_preprocess_data()
print(f"Data loaded and preprocessed. X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

plt.figure(figsize=(10, 8))

X_pca_healthy = X_train[y_train == 0]
X_pca_diseased = X_train[y_train == 1]

plt.scatter(X_pca_healthy[:, 0], X_pca_healthy[:, 1], color='blue', label='Healthy', alpha=0.6)
plt.scatter(X_pca_diseased[:, 0], X_pca_diseased[:, 1], color='red', label='Diseased', alpha=0.6)

plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("PCA Feature Space of Shrimp Dataset (Training Data)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("pca_clusters_augmented.png")
plt.close()
print("PCA scatter plot saved as pca_clusters_augmented.png.")


print("\nInitiating model training and comparison...")
hasil_model = bandingkan_model(X_train, y_train, X_test, y_test)

knn_model = hasil_model['KNN']['model']
svm_model = hasil_model['SVM']['model']

print("\nSaving trained models and preprocessing objects...")
joblib.dump(knn_model, "knn_model.pkl")
joblib.dump(svm_model, "svm_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(pca, "pca.pkl")

joblib.dump(X_test, "X_test.pkl")
joblib.dump(y_test, "y_test.pkl")
print("X_test and y_test saved successfully for consistent evaluation.")

print("Models (knn_model.pkl, svm_model.pkl) and preprocessing objects (scaler.pkl, pca.pkl) saved successfully.")
print("Training process completed.")
